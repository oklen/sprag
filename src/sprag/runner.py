"""End-to-end runner: retrieval + assembled prefill + generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

from .assemble import ChunkPlacement, patched_full_attn, make_inv_freq_for
from .chunk_cache import load_meta
from .embed import JinaEmbedder
from .loader import FULL_ATTN_LAYERS
from .retrieve import load_chunk_reprs, topk


@dataclass
class RunnerConfig:
    cache_dir: Path
    top_k: int = 3
    max_new_tokens: int = 64
    prefix_text: str = ""           # system prompt-like preamble


@dataclass
class RunResult:
    output_text: str
    retrieved_chunk_ids: list[int]
    retrieved_scores: list[float]
    assembled_len: int


class SpragRunner:
    def __init__(self, model, tokenizer, embedder: JinaEmbedder, cfg: RunnerConfig):
        self.model = model
        self.tok = tokenizer
        self.embedder = embedder
        self.cfg = cfg
        self.inv_freq = make_inv_freq_for(model)

        self._chunk_ids, self._chunk_reprs = load_chunk_reprs(cfg.cache_dir)
        self._meta = load_meta(cfg.cache_dir)
        self._chunk_lookup = {c["id"]: c for c in self._meta["chunks"]}

    def _load_chunk_tensors(self, chunk_id: int) -> dict:
        path = Path(self.cfg.cache_dir) / f"chunk_{chunk_id:05d}.safetensors"
        return load_file(str(path))

    def _retrieve(self, query: str) -> tuple[list[int], list[float]]:
        q_vec = self.embedder.encode_query([query])[0]
        idx, scores = topk(q_vec, self._chunk_reprs, k=self.cfg.top_k)
        chunk_ids = [self._chunk_ids[i] for i in idx]
        return chunk_ids, scores

    def _build_placements(self, chunk_ids: list[int], prefix_len: int) -> tuple[list[int], list[ChunkPlacement]]:
        all_ids: list[int] = []
        placements: list[ChunkPlacement] = []
        cursor = prefix_len
        for cid in chunk_ids:
            tensors = self._load_chunk_tensors(cid)
            tok_ids = tensors["input_ids"]
            meta = self._chunk_lookup[cid]
            length = int(tok_ids.shape[0])
            cached = {li: (tensors[f"K_l{li}"], tensors[f"V_l{li}"]) for li in FULL_ATTN_LAYERS}
            placements.append(ChunkPlacement(
                a_start=int(meta["a_start"]),
                b_start=cursor,
                length=length,
                cached=cached,
            ))
            all_ids.extend(tok_ids.tolist())
            cursor += length
        return all_ids, placements

    def run(self, query: str) -> RunResult:
        chunk_ids, scores = self._retrieve(query)

        prefix_ids = (self.tok(self.cfg.prefix_text, add_special_tokens=False).input_ids
                       if self.cfg.prefix_text else [])
        query_ids = self.tok(query, add_special_tokens=False).input_ids

        chunk_ids_flat, placements = self._build_placements(chunk_ids, prefix_len=len(prefix_ids))
        assembled = prefix_ids + chunk_ids_flat + query_ids
        input_ids = torch.tensor([assembled], dtype=torch.long)

        with torch.no_grad(), patched_full_attn(self.model, placements, inv_freq=self.inv_freq):
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tok.eos_token_id,
            )
        gen_ids = out[0, input_ids.shape[1]:].tolist()
        output_text = self.tok.decode(gen_ids, skip_special_tokens=True)
        return RunResult(
            output_text=output_text,
            retrieved_chunk_ids=chunk_ids,
            retrieved_scores=scores,
            assembled_len=len(assembled),
        )


def run_baseline(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    """No retrieval, no splice: standard generation on the full prompt."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
