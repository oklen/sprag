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


_CHUNK_CACHE_RAM: dict[tuple, dict] = {}


def invalidate_chunk_ram(cache_dir) -> None:
    """Drop any in-memory copies for `cache_dir`. Call this whenever the
    on-disk cache has just been rebuilt — otherwise the next runner will
    keep serving the previous build's tensors."""
    prefix = str(Path(cache_dir).resolve())
    for k in list(_CHUNK_CACHE_RAM.keys()):
        if k[0] == prefix:
            del _CHUNK_CACHE_RAM[k]


def _load_chunks_to_device(cache_dir: Path, chunk_ids: list[int],
                            full_layers, model_dtype, device) -> dict[int, dict]:
    """Load all chunks from disk into a per-(cache_dir, device, dtype) memo.
    K/V tensors land on `device` cast to `model_dtype`; small bookkeeping
    fields stay on CPU. The memo is shared across runners that point at
    the same cache dir, so two runners (e.g. top_k=3 and top_k=6) read
    disk only once.
    """
    dev = torch.device(device)
    key = (str(Path(cache_dir).resolve()), dev.type, dev.index or 0, str(model_dtype))
    memo = _CHUNK_CACHE_RAM.setdefault(key, {})
    if memo:
        return memo
    for cid in chunk_ids:
        path = Path(cache_dir) / f"chunk_{cid:05d}.safetensors"
        t = load_file(str(path))
        gpu = {"input_ids": t["input_ids"]}  # ids stay on CPU; small
        for li in full_layers:
            gpu[f"K_l{li}"] = t[f"K_l{li}"].to(device=device, dtype=model_dtype, non_blocking=True).contiguous()
            gpu[f"V_l{li}"] = t[f"V_l{li}"].to(device=device, dtype=model_dtype, non_blocking=True).contiguous()
        memo[cid] = gpu
    return memo


class SpragRunner:
    def __init__(self, model, tokenizer, embedder: JinaEmbedder, cfg: RunnerConfig):
        self.model = model
        self.tok = tokenizer
        self.embedder = embedder
        self.cfg = cfg
        _device = next(model.parameters()).device
        self.inv_freq = make_inv_freq_for(model).to(_device)

        self._chunk_ids, self._chunk_reprs = load_chunk_reprs(cfg.cache_dir)
        self._meta = load_meta(cfg.cache_dir)
        self._chunk_lookup = {c["id"]: c for c in self._meta["chunks"]}

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self._chunk_tensors = _load_chunks_to_device(
            cfg.cache_dir, self._chunk_ids, FULL_ATTN_LAYERS, dtype, device,
        )

    def _load_chunk_tensors(self, chunk_id: int) -> dict:
        return self._chunk_tensors[chunk_id]

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
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([assembled], dtype=torch.long, device=device)

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
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
