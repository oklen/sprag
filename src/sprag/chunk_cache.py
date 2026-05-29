"""Phase 1: precompute per-chunk caches for ReAttention.

For each chunk we store:
  - input_ids: the token ids of the chunk
  - position range [A_start, A_end) in the original document
  - For each full-attn layer (3, 7, 11, 15, 19, 23):
      K_rotated: shape (chunk_len, n_kv_heads, head_dim) — post k_norm, post RoPE at position A
      V:         shape (chunk_len, n_kv_heads, head_dim)
  - chunk_repr: dense vector from an external embedding model

Storage format: one safetensors per chunk to allow lazy mmap loading.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor, nn

from .loader import FULL_ATTN_LAYERS
from .rope import build_inv_freq, shift_rope


@dataclass
class ChunkMeta:
    chunk_id: int
    a_start: int
    a_end: int
    num_tokens: int


@contextlib.contextmanager
def capture_full_attn_kv(model: nn.Module, layer_indices=FULL_ATTN_LAYERS):
    """Wrap each full-attn layer's forward to stash post-RoPE K and V into a dict.

    Returns a dict layer_idx -> {"K": tensor, "V": tensor} populated after forward.
    K shape: (bs, n_kv_heads, seq_len, head_dim).
    """
    store: dict[int, dict[str, Tensor]] = {}
    originals: dict[int, callable] = {}
    layers = model.model.layers

    def make_wrapped(layer_idx, attn_module):
        original = attn_module.forward

        def wrapped(hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, cache_position=None, **kw):
            # Replicate the Q/K/V projection + norm + RoPE so we can grab K post-RoPE,
            # then delegate to the original for the actual attention computation.
            # Simpler approach: monkey the apply_rotary call. But cleanest is to
            # call original and re-run K computation. Instead use a side-channel
            # hook by capturing inside a torch hook on apply_rotary_pos_emb.
            # Pragmatic: redo K computation here, then call original.
            cfg = attn_module
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, cfg.head_dim)

            k = cfg.k_norm(cfg.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            v = cfg.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
            # Apply RoPE only to K (q not needed for cache); reuse function with a
            # dummy q of same shape.
            _, k_rot = apply_rotary_pos_emb(k, k, cos, sin)
            store[layer_idx] = {"K": k_rot.detach().cpu(), "V": v.detach().cpu()}
            return original(hidden_states, position_embeddings=position_embeddings,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            cache_position=cache_position, **kw)

        return wrapped

    try:
        for li in layer_indices:
            attn = layers[li].self_attn
            originals[li] = attn.forward
            attn.forward = make_wrapped(li, attn)
        yield store
    finally:
        for li, fn in originals.items():
            layers[li].self_attn.forward = fn


def split_into_chunks(token_ids: Tensor, chunk_size: int = 512) -> list[ChunkMeta]:
    """Non-overlapping chunks. token_ids: 1D tensor."""
    n = token_ids.shape[0]
    chunks = []
    for i, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        chunks.append(ChunkMeta(chunk_id=i, a_start=start, a_end=end,
                                 num_tokens=end - start))
    return chunks


def build_chunk_cache(
    model,
    tokenizer,
    text: str,
    out_dir: Path,
    chunk_size: int = 512,
    embed_fn=None,
):
    """Run a single forward over the entire doc, slice & store per-chunk caches."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = split_into_chunks(tokens, chunk_size=chunk_size)
    print(f"Total tokens: {tokens.shape[0]};  {len(chunks)} chunks")

    input_ids = tokens.unsqueeze(0).to(device)  # (1, N)
    with torch.no_grad(), capture_full_attn_kv(model) as kv_store:
        out = model.model(input_ids=input_ids, use_cache=False)
    last_h = out.last_hidden_state[0]  # (N, hidden)

    chunk_text_pieces = []
    chunk_tensors: list[dict[str, Tensor]] = []
    for ch in chunks:
        sl = slice(ch.a_start, ch.a_end)
        tensors = {"input_ids": tokens[sl].clone()}
        for li in FULL_ATTN_LAYERS:
            k_full = kv_store[li]["K"][0]  # (n_kv, N, head_dim)
            v_full = kv_store[li]["V"][0]
            tensors[f"K_l{li}"] = k_full[:, sl, :].contiguous()
            tensors[f"V_l{li}"] = v_full[:, sl, :].contiguous()
        tensors["repr_mean_last"] = last_h[sl].mean(dim=0).float().cpu()
        chunk_text_pieces.append(tokenizer.decode(tokens[sl], skip_special_tokens=False))
        chunk_tensors.append(tensors)

    if embed_fn is not None:
        reprs = embed_fn(chunk_text_pieces)
        for tensors, vec in zip(chunk_tensors, reprs):
            tensors["chunk_repr"] = vec.float().cpu().contiguous()

    for ch, tensors in zip(chunks, chunk_tensors):
        save_file(tensors, str(out_dir / f"chunk_{ch.chunk_id:05d}.safetensors"))

    # If a runner already memoised this cache_dir from a previous build,
    # drop the stale RAM copies so the next runner re-reads from disk.
    try:
        from .runner import invalidate_chunk_ram
        invalidate_chunk_ram(out_dir)
    except ImportError:
        pass

    meta = {
        "chunk_size": chunk_size,
        "num_tokens": int(tokens.shape[0]),
        "num_chunks": len(chunks),
        "chunks": [
            {"id": c.chunk_id, "a_start": c.a_start, "a_end": c.a_end,
             "num_tokens": c.num_tokens, "text_preview": chunk_text_pieces[i][:120]}
            for i, c in enumerate(chunks)
        ],
        "full_attn_layers": list(FULL_ATTN_LAYERS),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(chunks)} chunks + meta.json to {out_dir}")
    return chunks, meta


def build_anchor_chunk_cache(
    model,
    tokenizer,
    text: str,
    out_dir: Path,
    chunk_size: int = 512,
    anchor_M: int = 4,
    filler_mode: str = "none",
    embed_fn=None,
):
    """Anchor-conditioned cache.

    filler_mode="none"  (anchor v2, §5m): per chunk, short forward of
        [sink_M] + [chunk]. chunk_0 is special-cased (standalone, sink IS
        its first M tokens). Build emulates "this chunk at top-1 placement".

    filler_mode="self_prev"  (multi-anchor cheap probe, §5n): per chunk,
        short forward of [sink_M] + [chunk_{i-1}] + [chunk_i]. Build emulates
        "this chunk at top-2 placement, preceded by sink + 1 other chunk".
        chunk_0: standalone (same as anchor v2).
        chunk_1: [chunk_0 + chunk_1] — chunk_0 already starts with sink, no
                 extra prepend.
    """
    if filler_mode not in ("none", "self_prev"):
        raise ValueError(f"unknown filler_mode {filler_mode!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = split_into_chunks(tokens, chunk_size=chunk_size)
    print(f"Anchor cache build: {tokens.shape[0]} tok, {len(chunks)} chunks, M={anchor_M}")

    cfg = model.config
    inv_freq = build_inv_freq(
        head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.rope_parameters.get("partial_rotary_factor", 1.0),
        rope_theta=cfg.rope_parameters["rope_theta"],
    ).to(device)
    sink_ids = tokens[:anchor_M].clone()

    chunk_text_pieces = []
    chunk_tensors_list = []

    for i, ch in enumerate(chunks):
        chunk_token_ids = tokens[ch.a_start:ch.a_end]
        # Chunk 0 starts at a_start=0, so its first M tokens *are* the sink —
        # prepending sink_ids would duplicate them and corrupt K. Build chunk 0
        # standalone in every filler_mode.
        if ch.a_start == 0:
            small_ids = chunk_token_ids.unsqueeze(0).to(device)
            slice_off = 0
            shift_delta = 0
        elif filler_mode == "self_prev":
            prev = chunks[i - 1]
            prev_token_ids = tokens[prev.a_start:prev.a_end]
            if prev.a_start == 0:
                # filler is chunk_0, which already begins with sink — no extra prepend.
                small_ids = torch.cat([prev_token_ids, chunk_token_ids], dim=0).unsqueeze(0).to(device)
                slice_off = prev.num_tokens
            else:
                small_ids = torch.cat([sink_ids, prev_token_ids, chunk_token_ids], dim=0).unsqueeze(0).to(device)
                slice_off = anchor_M + prev.num_tokens
            shift_delta = ch.a_start - slice_off
        else:
            small_ids = torch.cat([sink_ids, chunk_token_ids], dim=0).unsqueeze(0).to(device)
            slice_off = anchor_M
            shift_delta = ch.a_start - anchor_M
        with torch.no_grad(), capture_full_attn_kv(model) as kv_store:
            out = model.model(input_ids=small_ids, use_cache=False)
        last_h = out.last_hidden_state[0]

        tensors = {"input_ids": chunk_token_ids.clone()}
        for li in FULL_ATTN_LAYERS:
            k_full = kv_store[li]["K"][0]
            v_full = kv_store[li]["V"][0]
            k_chunk = k_full[:, slice_off:, :].contiguous()
            if shift_delta != 0:
                k_chunk = shift_rope(
                    k_chunk.unsqueeze(0).to(device=device, dtype=model_dtype),
                    shift_delta, inv_freq,
                ).squeeze(0).cpu().float()
            tensors[f"K_l{li}"] = k_chunk.contiguous()
            tensors[f"V_l{li}"] = v_full[:, slice_off:, :].contiguous()

        tensors["repr_mean_last"] = last_h[slice_off:].mean(dim=0).float().cpu()
        chunk_text_pieces.append(tokenizer.decode(chunk_token_ids, skip_special_tokens=False))
        chunk_tensors_list.append(tensors)

    if embed_fn is not None:
        reprs = embed_fn(chunk_text_pieces)
        for tensors, vec in zip(chunk_tensors_list, reprs):
            tensors["chunk_repr"] = vec.float().cpu().contiguous()

    for ch, tensors in zip(chunks, chunk_tensors_list):
        save_file(tensors, str(out_dir / f"chunk_{ch.chunk_id:05d}.safetensors"))

    try:
        from .runner import invalidate_chunk_ram
        invalidate_chunk_ram(out_dir)
    except ImportError:
        pass

    meta = {
        "chunk_size": chunk_size,
        "anchor_M": anchor_M,
        "anchor_conditioned": True,
        "filler_mode": filler_mode,
        "num_tokens": int(tokens.shape[0]),
        "num_chunks": len(chunks),
        "chunks": [
            {"id": c.chunk_id, "a_start": c.a_start, "a_end": c.a_end,
             "num_tokens": c.num_tokens, "text_preview": chunk_text_pieces[i][:120]}
            for i, c in enumerate(chunks)
        ],
        "full_attn_layers": list(FULL_ATTN_LAYERS),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(chunks)} anchor chunks + meta.json to {out_dir}")
    return chunks, meta


def build_random_anchor_chunk_cache(
    model,
    tokenizer,
    text: str,
    out_dir: Path,
    chunk_size: int = 512,
    anchor_M: int = 4,
    vocab_low: int = 200,
    vocab_high: int = 50000,
    seed: int = 0,
    embed_fn=None,
):
    """Per-chunk *unique* random-token anchor (§5p probe).

    For each chunk_i with id=i, sample M token ids deterministically from
    [vocab_low, vocab_high) using seed (seed, i). Build cache via short
    forward of [anchor_i + chunk_i], capture chunk K/V, Inverse-RoPE-shift
    K back to original doc position so on-disk K format matches standard
    cache. Anchor token ids are stored in meta.json per chunk so the splice
    routine can insert them fresh at assembly time.

    Goal: each chunk's drift direction in head-space should now be driven
    by its *own* random anchor, breaking the cos(sib0_drift, sib1_drift)
    ≈ 0.8 correlation we measured under shared-sink anchor v2.

    chunk_0 (a_start=0) keeps the standalone build (sink IS its first M
    tokens) and has anchor_ids=[].
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = split_into_chunks(tokens, chunk_size=chunk_size)
    print(f"Random-anchor cache build: {tokens.shape[0]} tok, "
          f"{len(chunks)} chunks, M={anchor_M}, vocab=[{vocab_low}, {vocab_high})")

    cfg = model.config
    inv_freq = build_inv_freq(
        head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.rope_parameters.get("partial_rotary_factor", 1.0),
        rope_theta=cfg.rope_parameters["rope_theta"],
    ).to(device)
    vocab_size = int(getattr(cfg, "vocab_size", vocab_high))
    hi = min(vocab_high, vocab_size - 10)
    lo = max(vocab_low, 0)

    chunk_text_pieces = []
    chunk_tensors_list = []
    anchor_ids_per_chunk: list[list[int]] = []

    for i, ch in enumerate(chunks):
        chunk_token_ids = tokens[ch.a_start:ch.a_end]
        if ch.a_start == 0:
            small_ids = chunk_token_ids.unsqueeze(0).to(device)
            slice_off = 0
            shift_delta = 0
            anchor_ids = torch.zeros(0, dtype=torch.long)
        else:
            g = torch.Generator().manual_seed(seed * 1_000_003 + ch.chunk_id)
            anchor_ids = torch.randint(lo, hi, (anchor_M,), generator=g,
                                        dtype=torch.long)
            small_ids = torch.cat(
                [anchor_ids, chunk_token_ids], dim=0
            ).unsqueeze(0).to(device)
            slice_off = anchor_M
            shift_delta = ch.a_start - anchor_M

        with torch.no_grad(), capture_full_attn_kv(model) as kv_store:
            out = model.model(input_ids=small_ids, use_cache=False)
        last_h = out.last_hidden_state[0]

        tensors = {"input_ids": chunk_token_ids.clone()}
        for li in FULL_ATTN_LAYERS:
            k_full = kv_store[li]["K"][0]
            v_full = kv_store[li]["V"][0]
            k_chunk = k_full[:, slice_off:, :].contiguous()
            if shift_delta != 0:
                k_chunk = shift_rope(
                    k_chunk.unsqueeze(0).to(device=device, dtype=model_dtype),
                    shift_delta, inv_freq,
                ).squeeze(0).cpu().float()
            tensors[f"K_l{li}"] = k_chunk.contiguous()
            tensors[f"V_l{li}"] = v_full[:, slice_off:, :].contiguous()

        tensors["repr_mean_last"] = last_h[slice_off:].mean(dim=0).float().cpu()
        chunk_text_pieces.append(tokenizer.decode(chunk_token_ids, skip_special_tokens=False))
        chunk_tensors_list.append(tensors)
        anchor_ids_per_chunk.append(anchor_ids.tolist())

    if embed_fn is not None:
        reprs = embed_fn(chunk_text_pieces)
        for tensors, vec in zip(chunk_tensors_list, reprs):
            tensors["chunk_repr"] = vec.float().cpu().contiguous()

    for ch, tensors in zip(chunks, chunk_tensors_list):
        save_file(tensors, str(out_dir / f"chunk_{ch.chunk_id:05d}.safetensors"))

    try:
        from .runner import invalidate_chunk_ram
        invalidate_chunk_ram(out_dir)
    except ImportError:
        pass

    meta = {
        "chunk_size": chunk_size,
        "anchor_M": anchor_M,
        "anchor_conditioned": True,
        "filler_mode": "random_per_chunk",
        "anchor_seed": seed,
        "vocab_low": lo, "vocab_high": hi,
        "num_tokens": int(tokens.shape[0]),
        "num_chunks": len(chunks),
        "chunks": [
            {"id": c.chunk_id, "a_start": c.a_start, "a_end": c.a_end,
             "num_tokens": c.num_tokens,
             "text_preview": chunk_text_pieces[i][:120],
             "anchor_ids": anchor_ids_per_chunk[i]}
            for i, c in enumerate(chunks)
        ],
        "full_attn_layers": list(FULL_ATTN_LAYERS),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(chunks)} random-anchor chunks + meta.json to {out_dir}")
    return chunks, meta


def build_fixed_anchor_chunk_cache(
    model,
    tokenizer,
    text: str,
    out_dir: Path,
    chunk_size: int = 512,
    anchor_M: int = 4,
    anchor_token_id: int | None = None,
    embed_fn=None,
):
    """Fixed shared anchor (the symmetric-anchor probe).

    Every chunk — INCLUDING chunk 0, no special case — is built via a short
    forward of [anchor_M copies of one FIXED, content-independent token] +
    [chunk]. K is Inverse-RoPE-shifted (a_start - anchor_M) back to the
    chunk's canonical doc position so the on-disk format matches the standard
    cache. The same fixed anchor is placed once, FRESH, at the very front of
    the assembly (no per-chunk anchors). Because the anchor sits at position 0
    with nothing before it in *both* build and use, its K/V is identical
    across the two — the build/use context is symmetric, so the top-1 chunk
    sees exactly its build-time prefix (zero cache->assembly drift).

    Contrast with build_anchor_chunk_cache (sink = the doc's first M tokens,
    content-dependent, chunk 0 special-cased standalone). Qwen3.5 has no BOS;
    default anchor token = <|endoftext|> (the doc-boundary / attention-sink
    analog). anchor_token_id is recorded in meta.json for the assembler.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    if anchor_token_id is None:
        anchor_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    anchor_token_id = int(anchor_token_id)
    anchor_ids = torch.full((anchor_M,), anchor_token_id, dtype=torch.long)

    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = split_into_chunks(tokens, chunk_size=chunk_size)
    print(f"Fixed-anchor cache build: {tokens.shape[0]} tok, {len(chunks)} chunks, "
          f"M={anchor_M}, anchor_token_id={anchor_token_id}")

    cfg = model.config
    inv_freq = build_inv_freq(
        head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.rope_parameters.get("partial_rotary_factor", 1.0),
        rope_theta=cfg.rope_parameters["rope_theta"],
    ).to(device)

    chunk_text_pieces = []
    chunk_tensors_list = []

    for ch in chunks:
        chunk_token_ids = tokens[ch.a_start:ch.a_end]
        small_ids = torch.cat([anchor_ids, chunk_token_ids], dim=0).unsqueeze(0).to(device)
        slice_off = anchor_M
        shift_delta = ch.a_start - anchor_M
        with torch.no_grad(), capture_full_attn_kv(model) as kv_store:
            out = model.model(input_ids=small_ids, use_cache=False)
        last_h = out.last_hidden_state[0]

        tensors = {"input_ids": chunk_token_ids.clone()}
        for li in FULL_ATTN_LAYERS:
            k_full = kv_store[li]["K"][0]
            v_full = kv_store[li]["V"][0]
            k_chunk = k_full[:, slice_off:, :].contiguous()
            if shift_delta != 0:
                k_chunk = shift_rope(
                    k_chunk.unsqueeze(0).to(device=device, dtype=model_dtype),
                    shift_delta, inv_freq,
                ).squeeze(0).cpu().float()
            tensors[f"K_l{li}"] = k_chunk.contiguous()
            tensors[f"V_l{li}"] = v_full[:, slice_off:, :].contiguous()

        tensors["repr_mean_last"] = last_h[slice_off:].mean(dim=0).float().cpu()
        chunk_text_pieces.append(tokenizer.decode(chunk_token_ids, skip_special_tokens=False))
        chunk_tensors_list.append(tensors)

    if embed_fn is not None:
        reprs = embed_fn(chunk_text_pieces)
        for tensors, vec in zip(chunk_tensors_list, reprs):
            tensors["chunk_repr"] = vec.float().cpu().contiguous()

    for ch, tensors in zip(chunks, chunk_tensors_list):
        save_file(tensors, str(out_dir / f"chunk_{ch.chunk_id:05d}.safetensors"))

    try:
        from .runner import invalidate_chunk_ram
        invalidate_chunk_ram(out_dir)
    except ImportError:
        pass

    meta = {
        "chunk_size": chunk_size,
        "anchor_M": anchor_M,
        "anchor_conditioned": True,
        "filler_mode": "fixed",
        "anchor_token_id": anchor_token_id,
        "num_tokens": int(tokens.shape[0]),
        "num_chunks": len(chunks),
        "chunks": [
            {"id": c.chunk_id, "a_start": c.a_start, "a_end": c.a_end,
             "num_tokens": c.num_tokens, "text_preview": chunk_text_pieces[i][:120]}
            for i, c in enumerate(chunks)
        ],
        "full_attn_layers": list(FULL_ATTN_LAYERS),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(chunks)} fixed-anchor chunks + meta.json to {out_dir}")
    return chunks, meta


def load_chunk(out_dir: Path, chunk_id: int) -> dict[str, Tensor]:
    from safetensors.torch import load_file as _load
    return _load(str(Path(out_dir) / f"chunk_{chunk_id:05d}.safetensors"))


def load_meta(out_dir: Path) -> dict:
    with open(Path(out_dir) / "meta.json") as f:
        return json.load(f)
