#!/usr/bin/env python3
"""Video KV-cache splice engine for Qwen3-Omni Thinker (position-preserving reuse).

Ports the A100 guide's section 5.3 recipe to a video/omni VLM:

  PREBAKE: run ONE Thinker forward over the FULL clip (all N frames + audio +
    question), use_cache=True. Keep the returned per-layer K/V (a DynamicCache)
    and the per-token 3D interleaved M-RoPE position_ids. Every frame-token's K/V
    is rotated at its ORIGINAL absolute (t,h,w) position and carries causal global
    memory of all earlier frames (incl. ones we will later drop) + cross-modal
    (audio) context. This is the global pre-baked cache.

  CACHED arm @ coverage c: build a cache holding only the token slices for
    [sink+prompt-prefix UNION selected frames UNION question/answer], kept at
    their ORIGINAL positions (gaps preserved). Because RoPE is relative, the kept
    frames are seen at their true relative distances; dropped frames are simply
    absent. No un-rotate / re-rotate -- the cached K/V is reused verbatim. Then
    append question + gold fresh and score.

  FRESH arm @ coverage c: re-encode ONLY the selected frame tokens (the model
    sees just that subset), at the SAME original positions, then question + gold.

This isolates exactly "cached-K/V (built over full clip) vs fresh-K/V (built over
the subset)" at identical positions -- the splice effect, and whether the global
+ cross-modal memory in the cache lets cached BEAT fresh (the reversal / c*).

Mandatory sanity gate (run before any curve): COV100 alpha=0 identity --
splicing the FULL cache at 100% coverage must reproduce a plain full forward's
logits to <1e-3 (fp32) / loose in bf16. If not, the M-RoPE / cache surgery is
wrong; fix before trusting anything.

This module is import-safe (no GPU work at import). The runner drives it.
"""
import inspect
import torch

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover
    DynamicCache = None


# ----------------------------------------------------------------------------
# Cache helpers -- tolerant to transformers DynamicCache layout variations.
# ----------------------------------------------------------------------------
def cache_layers(pkv):
    """Return list[(K,V)] for any of the DynamicCache layouts we may meet."""
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return list(zip(pkv.key_cache, pkv.value_cache))
    if hasattr(pkv, "layers"):
        return [(l.keys, l.values) for l in pkv.layers]
    if isinstance(pkv, (list, tuple)):
        return list(pkv)
    raise TypeError(f"unknown cache type {type(pkv)}")


def build_cache_from_layers(layers):
    """Build a DynamicCache from list[(K,V)] (each [B, kvH, S, D])."""
    if DynamicCache is None:
        return tuple(layers)
    dc = DynamicCache()
    for li, (k, v) in enumerate(layers):
        dc.update(k, v, li)
    return dc


def gather_kv(pkv, keep_idx, device=None):
    """Slice every layer's K/V to the token positions in keep_idx (LongTensor).

    keep_idx indexes the sequence dim (dim=2). Returns a new DynamicCache with
    the K/V values copied verbatim (no re-rotation -- they keep the rotation from
    their original absolute position). This is the heart of position-preserving
    reuse: we keep VALUES from the full-clip prebake but present only a subset.
    """
    out = []
    for (k, v) in cache_layers(pkv):
        idx = keep_idx.to(k.device)
        kk = k.index_select(2, idx).contiguous()
        vv = v.index_select(2, idx).contiguous()
        if device is not None:
            kk, vv = kk.to(device), vv.to(device)
        out.append((kk, vv))
    return build_cache_from_layers(out)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def gold_nll(thinker, tok, prefix_ids, prefix_pos3d, past_kv, answer_str, device):
    """Teacher-forced gold NLL of answer_str given (prefix tokens + past_kv).

    prefix_ids:  python list[int] of the NEW tokens to feed now (question, etc.)
                 that are NOT already in past_kv. May be empty if all context is
                 in the cache (then we still need at least the last real token).
    prefix_pos3d: LongTensor [3, len(prefix_ids)] M-RoPE positions for those
                 tokens (t,h,w rows), continuing past the cached positions.
    past_kv:     DynamicCache already holding the (spliced) context.
    Scores only the gold-token logits (never .float() the full vocab x T).
    """
    oid = tok(answer_str, add_special_tokens=False).input_ids
    new_ids = list(prefix_ids) + list(oid)
    n_prefix = len(prefix_ids)
    # positions for gold tokens continue the last prefix t-position (text => t=h=w)
    last_t = int(prefix_pos3d[0, -1].item()) if n_prefix else _cache_last_pos(past_kv)
    gold_pos = list(range(last_t + 1, last_t + 1 + len(oid)))
    pos_rows = []
    for r in range(3):
        row = prefix_pos3d[r].tolist() + gold_pos
        pos_rows.append(row)
    ids_t = torch.tensor([new_ids], device=device)
    pos_t = torch.tensor([pos_rows], device=device).transpose(0, 1)  # [3,1,L]
    with torch.no_grad():
        out = thinker(input_ids=ids_t, position_ids=pos_t,
                      past_key_values=past_kv, use_cache=False, return_dict=True)
    logits = out.logits[0]  # [L, vocab]
    # logits at index (n_prefix-1 .. n_prefix+len(oid)-2) predict the gold tokens
    start = n_prefix - 1 if n_prefix > 0 else 0
    rows = logits[start: start + len(oid)].float()
    lp = torch.log_softmax(rows, dim=-1)
    nll = -sum(lp[j, t].item() for j, t in enumerate(oid)) / len(oid)
    return nll


def _cache_last_pos(past_kv):
    # fallback if no prefix tokens -- caller should pass prefix_pos3d instead
    return cache_layers(past_kv)[0][0].shape[2] - 1


# ----------------------------------------------------------------------------
# Sanity gate
# ----------------------------------------------------------------------------
def identity_gate(thinker, fwd, full_pos3d, device, tol=1e-2):
    """COV100 alpha=0 identity check (multimodal-aware).

    fwd:        full multimodal forward kwargs (input_ids + pixel_values_videos +
                input_features + grids + masks ...), already on device/dtype.
                MUST embed the video/audio tokens, so features are required here.
    full_pos3d: position_ids [3, 1, T] from get_rope_index.

    1) plain full forward (with features) -> reference logits + full cache.
    2) splice: keep the full cache for all-but-last token, then feed ONLY the
       last token id (no features needed -- context is cached) with its original
       position against the spliced cache. The last logit row must match the
       reference's last row to < tol. A mismatch means the M-RoPE / cache surgery
       is wrong -- fix before trusting any curve.
    Returns (max_abs_diff, passed).
    """
    pos_t = full_pos3d.to(device)
    if pos_t.dim() == 2:           # [3,T] -> [3,1,T]
        pos_t = pos_t.unsqueeze(1)
    with torch.no_grad():
        ref = thinker(**fwd, position_ids=pos_t, use_cache=True, return_dict=True)
    ref_last = ref.logits[0, -1].float()
    full_cache = ref.past_key_values
    T = pos_t.shape[-1]
    keep = torch.arange(T - 1, device=device)
    spliced = gather_kv(full_cache, keep, device=device)
    last_id = fwd["input_ids"][:, -1:].to(device)
    last_pos = pos_t[:, :, -1:]                     # [3,1,1]
    with torch.no_grad():
        out = thinker(input_ids=last_id, position_ids=last_pos,
                      past_key_values=spliced, use_cache=False, return_dict=True)
    got_last = out.logits[0, -1].float()
    diff = (ref_last - got_last).abs().max().item()
    return diff, diff < tol


if __name__ == "__main__":
    print("omni_kv engine module -- import and drive from a runner.")
