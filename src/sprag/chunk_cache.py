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


def load_chunk(out_dir: Path, chunk_id: int) -> dict[str, Tensor]:
    from safetensors.torch import load_file as _load
    return _load(str(Path(out_dir) / f"chunk_{chunk_id:05d}.safetensors"))


def load_meta(out_dir: Path) -> dict:
    with open(Path(out_dir) / "meta.json") as f:
        return json.load(f)
