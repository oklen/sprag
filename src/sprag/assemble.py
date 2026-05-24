"""Phase 2a: monkey-patch the 6 full-attn layers to splice cached K/V
(with Inverse-RoPE shift) at specified chunk positions in the assembled sequence.
Linear-attn layers are untouched (v1 = full re-forward).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward

from .loader import FULL_ATTN_LAYERS
from .rope import build_inv_freq, shift_rope


@dataclass
class ChunkPlacement:
    """One retrieved chunk's placement in the assembled sequence."""
    a_start: int      # original document position of first chunk token
    b_start: int      # new (assembled) position of first chunk token
    length: int       # number of tokens
    # per-layer cached tensors: layer_idx -> (K_cached, V_cached)
    # K_cached shape: (n_kv_heads, length, head_dim) — already RoPE-rotated at A
    cached: dict[int, tuple[Tensor, Tensor]] = None


def make_inv_freq_for(model) -> Tensor:
    cfg = model.config
    return build_inv_freq(
        head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.rope_parameters.get("partial_rotary_factor", 1.0),
        rope_theta=cfg.rope_parameters["rope_theta"],
    )


def _patched_attn_forward_factory(attn_module, layer_idx: int,
                                  per_layer_splice: list[tuple[int, int, Tensor, Tensor]]):
    """Return a forward replacement that overwrites K/V at chunk positions."""

    def forward(hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kw):
        cfg = attn_module
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, cfg.head_dim)

        query_states, gate = torch.chunk(
            cfg.q_proj(hidden_states).view(*input_shape, -1, cfg.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = cfg.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = cfg.k_norm(cfg.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = cfg.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Splice only during prefill (multi-token forward). During decode the
        # chunk K/V is already inside past_key_values from the prefill step.
        if key_states.shape[-2] > 1:
            for b_start, b_end, k_shift, v_shift in per_layer_splice:
                key_states[:, :, b_start:b_end, :] = k_shift.to(key_states.dtype).to(key_states.device)
                value_states[:, :, b_start:b_end, :] = v_shift.to(value_states.dtype).to(value_states.device)

        # Preserve cache update behaviour (needed for decode-stage generation).
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, cfg.layer_idx, cache_kwargs
            )

        attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            cfg.config._attn_implementation, eager_attention_forward
        )
        attn_output, _ = attn_interface(
            cfg, query_states, key_states, value_states, attention_mask,
            dropout=0.0, scaling=cfg.scaling, **kw,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = cfg.o_proj(attn_output)
        return attn_output, None

    return forward


@contextlib.contextmanager
def patched_full_attn(model, placements: Sequence[ChunkPlacement], inv_freq: Tensor | None = None):
    """Patch the 6 full-attn layers' forward to splice cached K/V (with Inverse-RoPE shift).

    placements: list of ChunkPlacement. Each placement contributes one splice per
        full-attn layer at (b_start, b_start + length).
    inv_freq: optional precomputed inv_freq for shift_rope. If None, builds from model config.
    """
    if inv_freq is None:
        inv_freq = make_inv_freq_for(model)

    # Pre-shift K for each (placement, layer) pair to avoid recomputing inside forward.
    per_layer: dict[int, list[tuple[int, int, Tensor, Tensor]]] = {li: [] for li in FULL_ATTN_LAYERS}
    for p in placements:
        delta = p.b_start - p.a_start
        for li in FULL_ATTN_LAYERS:
            k_cached, v_cached = p.cached[li]
            k_shift = shift_rope(k_cached.unsqueeze(0), delta, inv_freq).squeeze(0)
            per_layer[li].append((p.b_start, p.b_start + p.length, k_shift, v_cached))

    originals = {}
    try:
        for li in FULL_ATTN_LAYERS:
            attn = model.model.layers[li].self_attn
            originals[li] = attn.forward
            attn.forward = _patched_attn_forward_factory(attn, li, per_layer[li])
        yield
    finally:
        for li, fn in originals.items():
            model.model.layers[li].self_attn.forward = fn
