"""Verify (a) MRoPE on text-only == standard 1D RoPE on rotated dims,
(b) Inverse-RoPE shift property: shift_rope(RoPE(x, A), B-A) == RoPE(x, B)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from sprag.loader import DEFAULT_MODEL_PATH
from sprag.rope import build_inv_freq, rope_cos_sin, apply_rope, shift_rope


def get_cfg():
    cfg = AutoConfig.from_pretrained(DEFAULT_MODEL_PATH).text_config
    return cfg


def test_mrope_1d_equivalence():
    """Text-only MRoPE should equal 1D RoPE on the rotated dim slice."""
    cfg = get_cfg()
    head_dim = cfg.head_dim
    rot_factor = cfg.rope_parameters["partial_rotary_factor"]
    rot_dim = int(head_dim * rot_factor)
    theta = cfg.rope_parameters["rope_theta"]

    rope = Qwen3_5TextRotaryEmbedding(cfg)
    seq_len = 17
    # Match the text-only path: position_ids broadcast to (3, bs, L)
    positions = torch.arange(seq_len).view(1, 1, -1).expand(3, 1, -1)
    cos_mrope, sin_mrope = rope(torch.zeros(1, seq_len, head_dim), positions)
    # cos_mrope shape: (bs, L, rot_dim) after apply_interleaved_mrope collapse

    # Our 1D RoPE
    inv_freq = build_inv_freq(head_dim, rot_factor, theta)
    cos_1d, sin_1d = rope_cos_sin(torch.arange(seq_len), inv_freq)

    assert cos_mrope.shape[-1] == rot_dim, f"got {cos_mrope.shape}"
    err = (cos_mrope[0] - cos_1d).abs().max().item()
    print(f"  cos diff (mrope vs 1d): {err:.2e}")
    assert err < 1e-4, f"cos mismatch: {err}"
    err = (sin_mrope[0] - sin_1d).abs().max().item()
    print(f"  sin diff: {err:.2e}")
    assert err < 1e-4


def test_inverse_rope_shift():
    """shift_rope(RoPE(x, A), B-A) == RoPE(x, B)."""
    cfg = get_cfg()
    head_dim = cfg.head_dim
    rot_factor = cfg.rope_parameters["partial_rotary_factor"]
    theta = cfg.rope_parameters["rope_theta"]
    inv_freq = build_inv_freq(head_dim, rot_factor, theta)

    n_heads, L = 2, 11
    A, B = 5000, 100
    torch.manual_seed(0)
    x_raw = torch.randn(1, n_heads, L, head_dim, dtype=torch.float32)

    # Build cos/sin at A (uniform across L tokens — chunk-relative position 0..L-1
    # would be A+0 .. A+L-1)
    pos_A = torch.arange(A, A + L)
    pos_B = torch.arange(B, B + L)
    cosA, sinA = rope_cos_sin(pos_A, inv_freq)
    cosB, sinB = rope_cos_sin(pos_B, inv_freq)
    x_A = apply_rope(x_raw, cosA.unsqueeze(0).unsqueeze(0), sinA.unsqueeze(0).unsqueeze(0))
    x_B = apply_rope(x_raw, cosB.unsqueeze(0).unsqueeze(0), sinB.unsqueeze(0).unsqueeze(0))

    # Shift x_A by (B - A) to land at x_B
    delta = B - A
    x_A_shifted = shift_rope(x_A, delta, inv_freq)

    err = (x_A_shifted - x_B).abs().max().item()
    rel = err / x_B.abs().max().item()
    print(f"  shift error (A=5000 -> B=100, delta=-4900): abs={err:.2e}, rel={rel:.2e}")
    # fp32 trig at large angles (theta=1e7) — rel ~1e-4 is the precision floor.
    assert rel < 1e-3, f"shift mismatch: rel={rel}"


def test_against_hf_apply_rotary_pos_emb():
    """Ensure our apply_rope matches transformers' apply_rotary_pos_emb."""
    cfg = get_cfg()
    head_dim = cfg.head_dim
    rot_factor = cfg.rope_parameters["partial_rotary_factor"]
    theta = cfg.rope_parameters["rope_theta"]
    inv_freq = build_inv_freq(head_dim, rot_factor, theta)

    L = 13
    pos = torch.arange(L)
    cos, sin = rope_cos_sin(pos, inv_freq)
    # HF apply_rotary_pos_emb expects cos/sin shape (bs, L, rot_dim) and applies
    # unsqueeze(1) for heads.
    cos_b, sin_b = cos.unsqueeze(0), sin.unsqueeze(0)

    q = torch.randn(1, 4, L, head_dim, dtype=torch.float32)
    k = torch.randn(1, 2, L, head_dim, dtype=torch.float32)
    q_hf, k_hf = apply_rotary_pos_emb(q, k, cos_b, sin_b)

    q_ours = apply_rope(q, cos_b.unsqueeze(1), sin_b.unsqueeze(1))
    k_ours = apply_rope(k, cos_b.unsqueeze(1), sin_b.unsqueeze(1))

    eq = (q_ours - q_hf).abs().max().item()
    ek = (k_ours - k_hf).abs().max().item()
    print(f"  apply_rope vs HF: q={eq:.2e}  k={ek:.2e}")
    assert eq < 1e-5 and ek < 1e-5


if __name__ == "__main__":
    print("[test_mrope_1d_equivalence]")
    test_mrope_1d_equivalence()
    print("[test_against_hf_apply_rotary_pos_emb]")
    test_against_hf_apply_rotary_pos_emb()
    print("[test_inverse_rope_shift]")
    test_inverse_rope_shift()
    print("\nAll RoPE tests passed.")
