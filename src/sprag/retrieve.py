"""Top-K chunk retrieval via Jina embeddings (cosine similarity)."""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from .chunk_cache import load_meta


def load_chunk_reprs(cache_dir: Path) -> tuple[list[int], torch.Tensor]:
    """Return chunk_ids list and stacked repr matrix (N, hidden)."""
    meta = load_meta(cache_dir)
    ids = [c["id"] for c in meta["chunks"]]
    reprs = []
    for cid in ids:
        data = load_file(str(Path(cache_dir) / f"chunk_{cid:05d}.safetensors"))
        reprs.append(data["chunk_repr"].float())
    return ids, torch.stack(reprs, 0)


def topk(query_vec: torch.Tensor, chunk_reprs: torch.Tensor, k: int):
    """Cosine similarity top-K; returns (indices, scores)."""
    q = torch.nn.functional.normalize(query_vec.float().unsqueeze(0), dim=-1)
    c = torch.nn.functional.normalize(chunk_reprs.float(), dim=-1)
    sims = (q @ c.T).squeeze(0)
    scores, idx = torch.topk(sims, k=min(k, sims.shape[0]))
    return idx.tolist(), scores.tolist()
