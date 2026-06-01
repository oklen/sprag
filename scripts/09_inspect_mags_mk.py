"""Fire-rate audit for MAGS on the MK suite.

For each (case, query) selected, runs the full ReAttention assembly with
mags_hook(log=...) and prints per-layer fire counts. Compares known
T+ (model gets it right) vs T- (degenerate or distractor) outcomes to
check whether the new MAGS τ is selective.
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from sprag.loader import load_model
from sprag.embed import JinaEmbedder
from sprag.chunk_cache import build_chunk_cache
from sprag.runner import SpragRunner, RunnerConfig
from sprag.mags.calibrate import load_mags
from sprag.mags.intervene import mags_hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--mags", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    args = ap.parse_args()

    model, tok, _ = load_model()
    emb = JinaEmbedder()
    params = load_mags(args.mags)
    print(f"MAGS layers={params.layer_indices}  "
          f"tau={ {li: round(t,3) for li,t in params.tau.items()} }")

    for ci in args.cases:
        cd = args.suite / f"case_{ci:02d}"
        haystack = (cd / "haystack.txt").read_text()
        queries = [json.loads(l) for l in (cd / "queries.jsonl").open()]

        cache_dir = Path("/tmp") / f"_inspect_mk_case{ci:02d}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        build_chunk_cache(model, tok, haystack, cache_dir,
                          chunk_size=args.chunk_size, embed_fn=emb.encode_passage)
        runner = SpragRunner(model, tok, emb, RunnerConfig(
            cache_dir=cache_dir, top_k=args.top_k,
            max_new_tokens=args.max_new_tokens, prefix_text=""))

        print(f"\n=== case {ci} ===")
        for q in queries:
            log: list = []
            with mags_hook(model, params, alpha=1.0, on_decode_only=True, log=log):
                res = runner.run("\n\nQ: " + q["question"] + "\nA:")

            per = defaultdict(list)
            for li, d, fired in log:
                per[li].append((d, fired))

            # crude verdict for tagging: did the output contain the gold?
            lo = res.output_text.lower()
            answer_parts = [p.strip().lower() for p in q["answer"].split("...") if p.strip()]
            ok = all(p in lo or any(s in lo for s in [p])
                     for p in answer_parts)
            tag = "OK" if ok else "X "
            print(f"  q{q['id']} t{q['template_id']} [{tag}] "
                  f"chunks={res.retrieved_chunk_ids[:4]} "
                  f"out={res.output_text[:60]!r}")
            for li in params.layer_indices:
                if not per[li]:
                    continue
                ds = [d for d, _ in per[li]]
                fs = [f for _, f in per[li]]
                tau = params.tau[li]
                print(f"    layer {li}: tau={tau:.3f}  "
                      f"d_min={min(ds):.3f} d_mean={sum(ds)/len(ds):.3f} "
                      f"d_max={max(ds):.3f}  fired={sum(fs)}/{len(fs)}")


if __name__ == "__main__":
    main()
