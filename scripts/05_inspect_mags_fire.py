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
    cases = [json.loads(l) for l in open("data/niah/niah_4k.jsonl")]
    # Pick a "wrong retrieval" case (3) and a "correct retrieval" case (4)
    targets = [cases[3], cases[4]]

    model, tok, _ = load_model()
    emb = JinaEmbedder()
    params = load_mags("data/mags/mags_4k.pkl")

    for case in targets:
        cache = Path("data/niah/_cache_inspect")
        if cache.exists():
            import shutil; shutil.rmtree(cache)
        build_chunk_cache(model, tok, case["haystack"], cache,
                          chunk_size=256, embed_fn=emb.encode_passage)

        runner = SpragRunner(model, tok, emb, RunnerConfig(
            cache_dir=cache, top_k=3, max_new_tokens=24, prefix_text=""
        ))

        log: list = []
        with mags_hook(model, params, alpha=1.0, on_decode_only=True, log=log):
            res = runner.run("Q: " + case["question"] + "\nA:")

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
