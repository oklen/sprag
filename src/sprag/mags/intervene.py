"""Online MAGS hook: monitor residual distance to error manifold; if past
threshold, subtract the projected error direction (orthogonal projection)."""
from __future__ import annotations

import contextlib

import torch
from torch import Tensor

from .calibrate import MAGSParams


@contextlib.contextmanager
def mags_hook(model, params: MAGSParams, alpha: float = 1.0, on_decode_only: bool = True,
              log: list | None = None):
    """Register forward_hooks on the chosen decoder layers' output (residual).

    For each new token (decode), if the last-token residual `a` satisfies
        d = || B (a - mu_c) ||_2 > tau
    then replace it with
        a_new = a - alpha * B^T B (a - mu_c)

    Args:
        on_decode_only: if True only fires when seq_len == 1 (decode pass).
                         False also fires during prefill (last token).
        log: optional list to append (layer, dist, fired) tuples for debug.
    """
    handles = []

    def make_hook(li):
        mu = params.mu_c[li]
        B = params.B[li]                       # (k, hidden)
        tau = params.tau[li]

        def _hook(module, inputs, output):
            h = output if not isinstance(output, tuple) else output[0]
            seq_len = h.shape[1]
            if on_decode_only and seq_len != 1:
                return output
            # operate on the last token
            target = h[:, -1, :]                # (bs, hidden)
            mu_dev = mu.to(target.device, target.dtype)
            Bd = B.to(target.device, target.dtype)
            delta = target - mu_dev             # (bs, hidden)
            proj = delta @ Bd.T                 # (bs, k)
            dist = proj.norm(dim=-1)            # (bs,)
            fired = (dist > tau).any().item()
            if log is not None:
                log.append((li, float(dist.max().item()), fired))
            if not fired:
                return output
            # mask per batch element
            mask = (dist > tau).float().unsqueeze(-1)         # (bs, 1)
            correction = (proj @ Bd) * mask                    # (bs, hidden)
            new_last = target - alpha * correction
            h = h.clone()
            h[:, -1, :] = new_last
            if isinstance(output, tuple):
                return (h, *output[1:])
            return h

        return _hook

    try:
        for li in params.layer_indices:
            h = model.model.layers[li].register_forward_hook(make_hook(li))
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()
