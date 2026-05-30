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

from .loader import FULL_ATTN_LAYERS, LINEAR_ATTN_LAYERS
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
                                  per_layer_splice: list[tuple[int, int, Tensor, Tensor]],
                                  splice_kind: str = "kv",
                                  alpha: float = 1.0,
                                  alpha_k_rot: float | None = None,
                                  alpha_k_pass: float | None = None,
                                  alpha_v: float | None = None):
    """Return a forward replacement that overwrites K and/or V at chunk positions.

    splice_kind: "kv" (default, both), "k" (only key), or "v" (only value).
    alpha: blend weight. spliced = alpha * cached + (1 - alpha) * fresh.
        alpha=1.0 is full splice (current behaviour); alpha=0.0 is no splice;
        intermediates probe the K/V drift tolerance curve.
    alpha_k_rot / alpha_k_pass: per-subspace blend for K, splitting the head
        at the rotary boundary (first rot_dim dims = RoPE-rotated, the rest =
        pass-through). Default to `alpha` when None. Lets us localize the
        α=1.0 footgun: is the sibling-misrouting in the position-coupled
        (rotary) coords or the pure-content (pass-through) coords?
    alpha_v: blend weight for V (no rotary structure). Defaults to `alpha`.
    """

    do_k = splice_kind in ("k", "kv")
    do_v = splice_kind in ("v", "kv")
    a_rot = alpha if alpha_k_rot is None else alpha_k_rot
    a_pass = alpha if alpha_k_pass is None else alpha_k_pass
    a_v = alpha if alpha_v is None else alpha_v
    blend_k = (a_rot != 1.0) or (a_pass != 1.0)
    blend_v = a_v != 1.0

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
        rot_dim = cos.shape[-1]  # rotary boundary: dims [0:rot_dim) rotated, rest pass-through

        # Splice only during prefill (multi-token forward). During decode the
        # chunk K/V is already inside past_key_values from the prefill step.
        if key_states.shape[-2] > 1:
            for b_start, b_end, k_shift, v_shift in per_layer_splice:
                if do_k:
                    k_cached = k_shift.to(key_states.dtype).to(key_states.device)
                    if blend_k:
                        k_fresh = key_states[:, :, b_start:b_end, :]
                        # per-subspace blend; both operands are already at phase
                        # b, so this is a linear interpolation of k_raw content
                        # split along the rotary boundary.
                        blended = torch.empty_like(k_fresh)
                        blended[..., :rot_dim] = (
                            a_rot * k_cached[..., :rot_dim] + (1.0 - a_rot) * k_fresh[..., :rot_dim])
                        blended[..., rot_dim:] = (
                            a_pass * k_cached[..., rot_dim:] + (1.0 - a_pass) * k_fresh[..., rot_dim:])
                        key_states[:, :, b_start:b_end, :] = blended
                    else:
                        key_states[:, :, b_start:b_end, :] = k_cached
                if do_v:
                    v_cached = v_shift.to(value_states.dtype).to(value_states.device)
                    if blend_v:
                        v_fresh = value_states[:, :, b_start:b_end, :]
                        value_states[:, :, b_start:b_end, :] = a_v * v_cached + (1.0 - a_v) * v_fresh
                    else:
                        value_states[:, :, b_start:b_end, :] = v_cached

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
def patched_full_attn(model, placements: Sequence[ChunkPlacement],
                      inv_freq: Tensor | None = None,
                      splice_layers: Sequence[int] | None = None,
                      splice_kind: str = "kv",
                      alpha: float = 1.0,
                      alpha_k_rot: float | None = None,
                      alpha_k_pass: float | None = None,
                      alpha_v: float | None = None):
    """Patch full-attn layers' forward to splice cached K/V (with Inverse-RoPE shift).

    placements: list of ChunkPlacement. Each placement contributes one splice per
        full-attn layer at (b_start, b_start + length).
    inv_freq: optional precomputed inv_freq for shift_rope. If None, builds from model config.
    splice_layers: which full-attn layers to actually splice. Defaults to all 6.
        Layers not in this set run normally — their K/V is computed fresh from
        the assembled context. This is the partial re-prefill knob.
    splice_kind: "kv" (default, both), "k" (only key), or "v" (only value).
        "v" lets attention weights be computed fresh (Q × K_fresh^T) but pulls
        in cached V — useful when K drift is severe but V content is still
        relevant.
    """
    if inv_freq is None:
        inv_freq = make_inv_freq_for(model)
    if splice_layers is None:
        splice_layers = FULL_ATTN_LAYERS
    splice_set = set(splice_layers)
    if splice_kind not in ("kv", "k", "v"):
        raise ValueError(f"splice_kind must be 'kv', 'k', or 'v'; got {splice_kind!r}")

    per_layer: dict[int, list[tuple[int, int, Tensor, Tensor]]] = {li: [] for li in splice_set}
    for p in placements:
        delta = p.b_start - p.a_start
        for li in splice_set:
            k_cached, v_cached = p.cached[li]
            k_shift = shift_rope(k_cached.unsqueeze(0), delta, inv_freq).squeeze(0)
            per_layer[li].append((p.b_start, p.b_start + p.length, k_shift, v_cached))

    originals = {}
    try:
        for li in splice_set:
            attn = model.model.layers[li].self_attn
            originals[li] = attn.forward
            attn.forward = _patched_attn_forward_factory(attn, li, per_layer[li],
                                                          splice_kind=splice_kind,
                                                          alpha=alpha,
                                                          alpha_k_rot=alpha_k_rot,
                                                          alpha_k_pass=alpha_k_pass,
                                                          alpha_v=alpha_v)
        yield
    finally:
        for li, fn in originals.items():
            model.model.layers[li].self_attn.forward = fn


@contextlib.contextmanager
def patched_linear_state(model, cached_states, alpha: float = 1.0,
                         layer_indices: Sequence[int] | None = None,
                         norm_match: bool = False):
    """Blend each linear (GatedDeltaNet) layer's end-of-prefill recurrent state
    with a cached, composed state:  S_used = alpha * S_cached + (1 - alpha) * S_fresh.

    This is the linear-attn analog of the full-attn K/V splice. Unlike K/V (a
    per-position tensor that can be RoPE-shifted into place), the GatedDeltaNet
    recurrent state is a single gated *sequential fold* over the whole prefix
    (modeling_qwen3_5: state = state*g.exp() + k⊗v). There is no position-
    independent per-chunk slice, so the only cacheable unit is a chunk's
    from-zero fold; `cached_states[li]` is the caller's composition of those
    over the retrieval set (e.g. a sum). The fresh fold is left intact and we
    only overwrite the recurrent state the *decode* will read — exactly parallel
    to how patched_full_attn leaves context hidden states fresh but overwrites
    the K/V that Q attends to.

    cached_states: dict layer_idx -> tensor broadcastable to the layer's
        recurrent state [batch, n_v_heads, head_k_dim, head_v_dim].
    alpha: 0.0 reproduces fresh exactly (no-op, the sanity baseline);
        1.0 = pure cached state for decode.
    Only the prefill forward (seq_len > 1) is blended; decode steps untouched.
    """
    if layer_indices is None:
        layer_indices = LINEAR_ATTN_LAYERS
    layer_indices = [li for li in layer_indices if cached_states.get(li) is not None]
    originals: dict[int, callable] = {}

    def make_wrapped(li, mod):
        original = mod.forward
        S_cached = cached_states[li]

        def wrapped(hidden_states, cache_params=None, **kw):
            out = original(hidden_states, cache_params=cache_params, **kw)
            # Blend only on the prefill fold; decode (seq_len==1) reads the
            # already-blended state and must not be re-injected each step.
            if (alpha != 0.0 and cache_params is not None
                    and hidden_states.shape[1] > 1):
                layer = cache_params.layers[li]
                S_fresh = layer.recurrent_states
                Sc = S_cached.to(S_fresh.dtype).to(S_fresh.device)
                if norm_match:
                    # Per-head: rescale the cached state to the fresh state's
                    # Frobenius norm, isolating DIRECTION from scale. Shapes are
                    # [batch, n_heads, k_dim, v_dim]; norm over the last two dims.
                    fn = S_fresh.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
                    cn = Sc.flatten(-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
                    Sc = Sc * (fn / cn.clamp_min(1e-8))
                # in-place copy_ preserves the static cache address
                layer.recurrent_states.copy_(alpha * Sc + (1.0 - alpha) * S_fresh)
            return out

        return wrapped

    try:
        for li in layer_indices:
            mod = model.model.layers[li].linear_attn
            originals[li] = mod.forward
            mod.forward = make_wrapped(li, mod)
        yield
    finally:
        for li, fn in originals.items():
            model.model.layers[li].linear_attn.forward = fn


def compute_chunk_linear_states(model, chunk_token_ids, layer_indices=None):
    """Run a from-zero forward over `chunk_token_ids` and return each linear
    layer's final recurrent state. This is the cacheable per-chunk unit for
    patched_linear_state (the chunk's isolated fold).

    chunk_token_ids: 1D LongTensor (or list) of one chunk's tokens.
    Returns dict layer_idx -> recurrent_state tensor on CPU.
    """
    from transformers import DynamicCache
    if layer_indices is None:
        layer_indices = LINEAR_ATTN_LAYERS
    device = next(model.parameters()).device
    ids = torch.as_tensor(chunk_token_ids, dtype=torch.long, device=device).view(1, -1)
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids=ids, past_key_values=cache, use_cache=True)
    out = {}
    for li in layer_indices:
        rs = cache.layers[li].recurrent_states
        out[li] = rs.detach().to("cpu").clone() if rs is not None else None
    return out
