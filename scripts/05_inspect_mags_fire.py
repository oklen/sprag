"""Inspect MAGS fire rate on a single case: count how often the projection
trigger d > tau hits during decode at each layer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import torch

from sprag.loader import load_model
from sprag.embed import JinaEmbedder
from sprag.chunk_cache import build_chunk_cache
from sprag.runner import SpragRunner, RunnerConfig
from sprag.mags.calibrate import load_mags
from sprag.mags.intervene import mags_hook


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/niah/niah_16k.jsonl")
    ap.add_argument("--mags", default="data/mags/mags_16k.pkl")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--ids", nargs="+", type=int, default=[1, 6, 8],
                    help="case ids to inspect (mix of failure + success)")
    args = ap.parse_args()
    cases_all = [json.loads(l) for l in open(args.cases)]
    targets = [c for c in cases_all if c["id"] in args.ids]

    model, tok, _ = load_model()
    emb = JinaEmbedder()
    params = load_mags(args.mags)

    for case in targets:
        cache = Path("data/niah/_cache_inspect")
        if cache.exists():
            import shutil; shutil.rmtree(cache)
        build_chunk_cache(model, tok, case["haystack"], cache,
                          chunk_size=args.chunk_size, embed_fn=emb.encode_passage)

        runner = SpragRunner(model, tok, emb, RunnerConfig(
            cache_dir=cache, top_k=args.top_k, max_new_tokens=24, prefix_text=""
        ))

        log: list = []
        with mags_hook(model, params, alpha=1.0, on_decode_only=True, log=log):
            res = runner.run("\n\nQ: " + case["question"] + "\nA:")

        # Aggregate per layer
        from collections import defaultdict
        per = defaultdict(list)
        for li, dist, fired in log:
            per[li].append((dist, fired))

        print(f"\n=== case {case['id']}  ans={case['answer']!r} ===")
        print(f"  retrieved chunks: {res.retrieved_chunk_ids}")
        print(f"  output: {res.output_text!r}")
        for li in params.layer_indices:
            ds = [d for d, _ in per[li]]
            fs = [f for _, f in per[li]]
            tau = params.tau[li]
            print(f"  layer {li}: tau={tau:.3f}  "
                  f"d_min={min(ds):.3f}  d_mean={sum(ds)/len(ds):.3f}  "
                  f"d_max={max(ds):.3f}  fired={sum(fs)}/{len(fs)}")


if __name__ == "__main__":
    main()
