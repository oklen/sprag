"""Build a small needle-doc chunk cache and run the SpragRunner end-to-end."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from sprag.loader import load_model
from sprag.chunk_cache import build_chunk_cache
from sprag.embed import JinaEmbedder
from sprag.runner import SpragRunner, RunnerConfig, run_baseline


HAYSTACK_PARTS = [
    "Charles Babbage proposed the Difference Engine in 1822. " * 8,
    "Ada Lovelace wrote what is considered the first computer program. " * 8,
    "Alan Turing later formalized the notion of an algorithm and built a machine. " * 8,
    # NEEDLE
    "Important fact embedded in the document: the secret keeper is Octavia "
    "and her favorite number is forty-two. Remember this for later. ",
    "John von Neumann introduced the stored-program architecture in 1945. " * 8,
    "The ENIAC was unveiled to the public in 1946 at Penn. " * 8,
    "Grace Hopper developed the first compiler in 1952. " * 8,
]
DOC = " ".join(HAYSTACK_PARTS)
QUERY = "Who is the secret keeper and what is her favorite number?"


def main():
    print("Loading Qwen3.5...")
    model, tok, cfg = load_model()
    print("Loading Jina embedder...")
    emb = JinaEmbedder()

    cache_dir = Path(__file__).resolve().parents[1] / "data" / "cache" / "smoke_needle"
    print("Building chunk cache (chunk_size=64)...")
    t0 = time.time()
    chunks, meta = build_chunk_cache(model, tok, DOC, cache_dir,
                                       chunk_size=64, embed_fn=emb.encode_passage)
    print(f"  cache built in {time.time()-t0:.1f}s; {meta['num_chunks']} chunks")

    print("\n=== Baseline (full doc inline) ===")
    full_prompt = DOC + "\n\nQ: " + QUERY + "\nA:"
    t0 = time.time()
    base_out = run_baseline(model, tok, full_prompt, max_new_tokens=40)
    print(f"  ({time.time()-t0:.1f}s) {base_out!r}")

    print("\n=== ReAttention (top-3 chunks via Jina) ===")
    runner = SpragRunner(model, tok, emb, RunnerConfig(
        cache_dir=cache_dir, top_k=3, max_new_tokens=40, prefix_text=""
    ))
    t0 = time.time()
    res = runner.run(QUERY + "\nA:")
    print(f"  retrieved chunks: {res.retrieved_chunk_ids}  scores: {[f'{s:.3f}' for s in res.retrieved_scores]}")
    print(f"  assembled_len={res.assembled_len}")
    for cid in res.retrieved_chunk_ids:
        prev = next(c for c in meta['chunks'] if c['id'] == cid)['text_preview']
        print(f"    chunk_{cid}: {prev!r}")
    print(f"  ({time.time()-t0:.1f}s) {res.output_text!r}")


if __name__ == "__main__":
    main()
