"""Jina embeddings v5 wrapper for chunk_repr & query encoding."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def _default_jina_path() -> str:
    env = os.environ.get("SPRAG_JINA_PATH")
    if env:
        return env
    snapshot = "dd76d535f5447ca3897a9c893fb1e612ead98192"
    for root in (
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub",
    ):
        p = Path(root) / "models--jinaai--jina-embeddings-v5-text-small" / "snapshots" / snapshot
        if p.exists():
            return str(p)
    # fall through to model-id form; AutoModel will download
    return "jinaai/jina-embeddings-v5-text-small"


JINA_PATH = _default_jina_path()


class JinaEmbedder:
    def __init__(self, model_path: str | Path = None, device: str | None = None,
                 dtype: torch.dtype = torch.float32, max_length: int = 1024):
        if model_path is None:
            model_path = JINA_PATH
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def encode(self, texts: list[str], task: str = "retrieval",
               prompt_name: str = "document", batch_size: int = 8) -> torch.Tensor:
        # jina-v5 task adapters are: classification / clustering / retrieval / text-matching
        # prompt_name is "query" or "document" (only used for retrieval).
        # The .encode method on the model exists but doesn't support batch_size; chunk ourselves.
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            embs = self.model.encode(
                batch, task=task, prompt_name=prompt_name,
                max_length=self.max_length,
            )
            if isinstance(embs, list):
                embs = torch.stack([torch.as_tensor(e) for e in embs])
            else:
                embs = torch.as_tensor(embs)
            outs.append(embs.float().cpu())
        return torch.cat(outs, 0)

    def encode_query(self, texts: list[str]) -> torch.Tensor:
        return self.encode(texts, task="retrieval", prompt_name="query")

    def encode_passage(self, texts: list[str]) -> torch.Tensor:
        return self.encode(texts, task="retrieval", prompt_name="document")
