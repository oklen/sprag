"""Amortization sweep: one document, many queries.

Measures the per-query cost of baseline (full prompt every time) vs
ReAttention (cache built once, then top-K splice per query) on a shared
multi-needle haystack. Reports cumulative wall-clock time after k queries,
which is the curve that decides break-even.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from sprag.loader import load_model
from sprag.chunk_cache import build_chunk_cache
from sprag.embed import JinaEmbedder
from sprag.runner import SpragRunner, RunnerConfig, run_baseline


def score(out: str, answer: str) -> bool:
    lo = out.lower()
    parts = [p.strip().lower() for p in answer.split("...") if p.strip()]
    return all(p in lo for p in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", type=Path, required=True,
                    help="dir produced by scripts/data/gen_amortization.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modes", nargs="+", default=["baseline", "reattn"],
                    choices=["baseline", "reattn"])
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="restrict number of queries (for quick smoke runs)")
    args = ap.parse_args()

    haystack = (args.doc / "haystack.txt").read_text()
    queries = [json.loads(l) for l in (args.doc / "queries.jsonl").open()]
    meta = json.loads((args.doc / "meta.json").read_text())
    if args.limit:
        queries = queries[: args.limit]
    print(f"doc: {meta['actual_tokens']} tok, {len(queries)} queries, "
          f"modes={args.modes}, top_k={args.top_k}")

    print("Loading model + embedder...")
    model, tok, _ = load_model()
    embedder = JinaEmbedder() if "reattn" in args.modes else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"meta": meta, "top_k": args.top_k, "chunk_size": args.chunk_size,
              "modes": args.modes, "queries": []}

    runner = None
    cache_dir = args.out.parent / f"_cache_amort_{args.doc.name}"
    if "reattn" in args.modes:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        t0 = time.time()
        build_chunk_cache(model, tok, haystack, cache_dir,
                          chunk_size=args.chunk_size,
                          embed_fn=embedder.encode_passage)
        cache_time = time.time() - t0
        result["cache_build_time"] = cache_time
        print(f"  cache built in {cache_time:.1f}s -> {cache_dir}")
        runner = SpragRunner(model, tok, embedder, RunnerConfig(
            cache_dir=cache_dir, top_k=args.top_k,
            max_new_tokens=args.max_new_tokens, prefix_text=""
        ))

    cum = {m: 0.0 for m in args.modes}
    correct = {m: 0 for m in args.modes}
    for i, q in enumerate(queries):
        row = {"id": q["id"], "question": q["question"], "answer": q["answer"]}
        if "baseline" in args.modes:
            prompt = haystack + "\n\nQ: " + q["question"] + "\nA:"
            t0 = time.time()
            out = run_baseline(model, tok, prompt, max_new_tokens=args.max_new_tokens)
            dt = time.time() - t0
            ok = score(out, q["answer"])
            row["baseline"] = {"output": out, "correct": ok, "time": dt}
            cum["baseline"] += dt
            correct["baseline"] += int(ok)
            print(f"  q{q['id']} baseline {dt:5.1f}s "
                  f"[{'OK' if ok else 'X'}] {out!r}")
        if "reattn" in args.modes:
            t0 = time.time()
            res = runner.run("\n\nQ: " + q["question"] + "\nA:")
            dt = time.time() - t0
            ok = score(res.output_text, q["answer"])
            row["reattn"] = {"output": res.output_text, "correct": ok,
                              "time": dt, "retrieved": res.retrieved_chunk_ids}
            cum["reattn"] += dt
            correct["reattn"] += int(ok)
            print(f"  q{q['id']} reattn   {dt:5.1f}s "
                  f"chunks={res.retrieved_chunk_ids} "
                  f"[{'OK' if ok else 'X'}] {res.output_text!r}")
        result["queries"].append(row)
        with args.out.open("w") as f:
            json.dump(result, f, indent=2)

    print("\n=== Amortization summary ===")
    n = len(queries)
    for m in args.modes:
        avg = cum[m] / max(1, n)
        print(f"  {m}: total {cum[m]:.1f}s over {n} queries "
              f"(avg {avg:.2f}s/q)  acc {correct[m]}/{n}")
    if {"baseline", "reattn"}.issubset(args.modes):
        cache_t = result.get("cache_build_time", 0.0)
        # cumulative cost at k: baseline=k*avg_b, reattn=cache_t+k*avg_r
        per_q_b = cum["baseline"] / n
        per_q_r = cum["reattn"] / n
        if per_q_b - per_q_r > 1e-3:
            k_break = cache_t / (per_q_b - per_q_r)
            print(f"  break-even ~= {k_break:.1f} queries "
                  f"(cache {cache_t:.1f}s, savings {per_q_b - per_q_r:.2f}s/q)")
        else:
            print(f"  no per-query advantage (baseline {per_q_b:.2f} vs reattn {per_q_r:.2f})")


if __name__ == "__main__":
    main()
