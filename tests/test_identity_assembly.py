"""Identity-assembly sanity test.

If we extract chunk caches from doc D, then re-run forward on D with the
patched-splice mechanism using cached K/V at THEIR ORIGINAL positions (delta=0),
the assembled output should match the unpatched forward exactly (modulo bf16
non-determinism in attention reductions).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from sprag.loader import load_model, FULL_ATTN_LAYERS
from sprag.chunk_cache import capture_full_attn_kv, split_into_chunks
from sprag.assemble import patched_full_attn, ChunkPlacement


def main():
    model, tok, cfg = load_model()
    doc = "The quick brown fox jumps over the lazy dog. " * 40
    ids = tok(doc, return_tensors="pt").input_ids
    print(f"doc tokens: {ids.shape[1]}")

    # 1. Forward without any patches; capture K/V at the same time.
    with torch.no_grad(), capture_full_attn_kv(model) as kv_store:
        out_orig = model.model(ids, use_cache=False)
    h_orig = out_orig.last_hidden_state[0]

    # 2. Build chunks (32-token chunks; identity placements -> delta=0)
    chunk_size = 32
    chunks = split_into_chunks(ids[0], chunk_size=chunk_size)
    placements = []
    for ch in chunks:
        cached = {}
        for li in FULL_ATTN_LAYERS:
            sl = slice(ch.a_start, ch.a_end)
            cached[li] = (
                kv_store[li]["K"][0, :, sl, :],
                kv_store[li]["V"][0, :, sl, :],
            )
        placements.append(ChunkPlacement(
            a_start=ch.a_start, b_start=ch.a_start, length=ch.num_tokens,
            cached=cached,
        ))

    # 3. Forward with splice (delta=0 everywhere)
    with torch.no_grad(), patched_full_attn(model, placements):
        out_spliced = model.model(ids, use_cache=False)
    h_spliced = out_spliced.last_hidden_state[0]

    diff = (h_orig - h_spliced).float()
    abs_max = diff.abs().max().item()
    rel = abs_max / h_orig.float().abs().max().item()
    print(f"identity splice: abs_max={abs_max:.3e}  rel_max={rel:.3e}")
    # bf16 reductions can drift on order of 1e-2 absolute on hidden of magnitude ~10
    assert rel < 5e-2, f"identity assembly diverged: rel={rel}"
    print("Identity-assembly test passed.")


if __name__ == "__main__":
    main()
