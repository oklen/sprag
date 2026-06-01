"""Offline calibration: collect residual-stream activations on (T+, T-) pairs,
SVD-decompose the error subspace, set the gating threshold.

We hook the *output* of selected decoder layers (post block residual), grabbing
the activation at the LAST input token position (the query token just before
generation begins). Each forward → one vector per layer.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch import Tensor, nn


DEFAULT_MAGS_LAYERS = (11, 15, 19)


@dataclass
class MAGSParams:
    layer_indices: tuple[int, ...]
    mu_c: dict[int, Tensor]      # layer_idx -> (hidden,)
    B: dict[int, Tensor]         # layer_idx -> (k, hidden)
    tau: dict[int, float]        # layer_idx -> threshold


@contextlib.contextmanager
def grab_last_residual(model, layer_indices=DEFAULT_MAGS_LAYERS):
    """Capture the residual stream output of each chosen decoder layer at the
    last token of the prefill (assumes a single forward, not autoregressive)."""
    captured: dict[int, Tensor] = {}
    handles = []

    def make_hook(li):
        def _hook(module, inputs, output):
            # decoder_layer output is the full hidden_states tensor (bs, L, hidden);
            # the layer returns just the tensor (not tuple) on this model.
            h = output if not isinstance(output, tuple) else output[0]
            captured[li] = h[0, -1, :].detach().float().cpu().clone()
        return _hook

    for li in layer_indices:
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        handles.append(h)
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def collect_residuals(model, tokenizer, prompts, layer_indices=DEFAULT_MAGS_LAYERS,
                      forward_ctx=None):
    """Run `prompts` through model.forward (no generation), capture last-token
    residual per layer. forward_ctx: optional context manager (e.g. patched_full_attn)
    that should wrap the model call — pass a *factory* if you need per-prompt patching.
    Returns dict[layer_idx] -> tensor shape (n_prompts, hidden)."""
    bucket: dict[int, list[Tensor]] = {li: [] for li in layer_indices}
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        ctx = forward_ctx() if forward_ctx is not None else contextlib.nullcontext()
        with torch.no_grad(), ctx, grab_last_residual(model, layer_indices) as cap:
            _ = model.model(ids, use_cache=False)
        for li in layer_indices:
            bucket[li].append(cap[li])
    return {li: torch.stack(bucket[li], 0) for li in layer_indices}


def fit_mags(pos_acts: dict[int, Tensor], neg_acts: dict[int, Tensor],
              k: int = 4, tau_quantile: float = 0.95) -> MAGSParams:
    """Build mu_c, B, tau per layer.
        pos_acts[li]: (n_pos, hidden) — correct trajectories
        neg_acts[li]: (n_neg, hidden) — drifted trajectories
    """
    mu_c: dict[int, Tensor] = {}
    B: dict[int, Tensor] = {}
    tau: dict[int, float] = {}
    for li, P in pos_acts.items():
        N = neg_acts[li]
        mu = P.mean(0)
        D = N - mu                                # (n_neg, hidden) error directions
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        Bk = Vh[:k]                               # top-k right singular vectors
        # distance distribution on positive set
        proj = (P - mu) @ Bk.T                    # (n_pos, k)
        dists = proj.norm(dim=-1)
        tau_li = float(torch.quantile(dists, tau_quantile))
        mu_c[li], B[li], tau[li] = mu, Bk, tau_li
    return MAGSParams(layer_indices=tuple(pos_acts.keys()), mu_c=mu_c, B=B, tau=tau)


def save_mags(params: MAGSParams, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(
            {"layer_indices": params.layer_indices,
             "mu_c": {k: v.cpu() for k, v in params.mu_c.items()},
             "B": {k: v.cpu() for k, v in params.B.items()},
             "tau": params.tau}, f)


def load_mags(path) -> MAGSParams:
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f)
    return MAGSParams(**d)
