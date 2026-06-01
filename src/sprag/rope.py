"""Inverse-RoPE for Qwen3.5 full-attention layers.

Qwen3.5 uses MRoPE with partial_rotary_factor=0.25 (head_dim=256 → 64 rotary dims,
192 pass-through). In text-only mode, position_ids are identical across T/H/W
axes, so MRoPE degenerates to standard 1D RoPE on the first 64 dims.

Given a key K cached at original position A (i.e. K = R_A · K_raw on the rotated
slice), we can shift it to a new position B in O(rotary_dim) by applying R_{B-A}:
    R_B · K_raw = R_{B-A} · (R_A · K_raw) = R_{B-A} · K_cached
"""
from __future__ import annotations

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    """Matches Qwen3.5's rotate_half: ([-x2, x1])."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_inv_freq(head_dim: int, partial_rotary_factor: float, rope_theta: float,
                   device=None) -> Tensor:
    rot_dim = int(head_dim * partial_rotary_factor)
    assert rot_dim % 2 == 0
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rot_dim, 2, dtype=torch.float32, device=device) / rot_dim)
    )
    return inv_freq  # shape (rot_dim // 2,)


def rope_cos_sin(positions: Tensor, inv_freq: Tensor) -> tuple[Tensor, Tensor]:
    """Compute cos/sin for given positions (1D) — duplicated to rot_dim length.

    Matches Qwen3.5's text-only path: angles = position * inv_freq, then
    emb = cat([angles, angles], dim=-1) before cos/sin.

    Args:
        positions: shape (..., L)
        inv_freq: shape (rot_dim // 2,)
    Returns:
        cos, sin: shape (..., L, rot_dim)
    """
    freqs = positions.float().unsqueeze(-1) * inv_freq  # (..., L, rot_dim//2)
    emb = torch.cat((freqs, freqs), dim=-1)             # (..., L, rot_dim)
    return emb.cos(), emb.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotation to the first rotary_dim of x; pass-through the rest.

    x: (..., L, head_dim);  cos/sin: (..., L, rotary_dim).
    """
    rot_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    out = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([out, x_pass], dim=-1)


def shift_rope(
    k_cached: Tensor,
    delta: int | Tensor,
    inv_freq: Tensor,
) -> Tensor:
    """Inverse-RoPE shift: K cached at position A -> K at position B = A + delta.

    Equivalent to applying R_delta to the rotated portion of k_cached.

    Args:
        k_cached: (..., L, head_dim) with the first rotary_dim already RoPE-rotated.
        delta: scalar shift to apply uniformly to all L tokens, OR (L,) tensor.
        inv_freq: (rot_dim // 2,) — pre-built.
    Returns:
        k_shifted: same shape as k_cached.
    """
    L = k_cached.shape[-2]
    device = k_cached.device
    if isinstance(delta, int):
        deltas = torch.full((L,), float(delta), device=device)
    else:
        deltas = delta.to(device=device, dtype=torch.float32)
    cos, sin = rope_cos_sin(deltas, inv_freq.to(device))
    # cos/sin shape (L, rot_dim); broadcast over heads.
    while cos.ndim < k_cached.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return apply_rope(k_cached, cos.to(k_cached.dtype), sin.to(k_cached.dtype))
