"""Run NIAH evaluation: baseline vs ReAttention (vs Full with MAGS later)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from sprag.loader import load_model
from sprag.chunk_cache import build_chunk_cache
from sprag.embed import JinaEmbedder
from sprag.runner import SpragRunner, RunnerConfig, run_baseline
from sprag.mags.calibrate import load_mags
from sprag.mags.intervene import mags_hook


def score_output(output: str, answer: str) -> bool:
    """The answer string may contain '...' splitting required parts.
    All parts must appear in the output (case-insensitive substring)."""
    lo = output.lower()
    parts = [p.strip().lower() for p in answer.split("...") if p.strip()]
    return all(p in lo for p in parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modes", nargs="+",
                    default=["baseline", "reattn"],
                    choices=["baseline", "reattn", "full"])
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mags_path", type=Path, default=None,
                     help="Required for mode=full")
    args = ap.parse_args()

    cases = [json.loads(l) for l in args.cases.open()]
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases from {args.cases}")
    print(f"modes: {args.modes}  chunk_size={args.chunk_size}  top_k={args.top_k}")

    print("Loading model + embedder...")
    model, tok, cfg = load_model()
    embedder = JinaEmbedder() if ("reattn" in args.modes or "full" in args.modes) else None
    mags_params = load_mags(args.mags_path) if "full" in args.modes else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i, case in enumerate(cases):
        row = {"id": case["id"], "answer": case["answer"], "picks": case["answer_picks"],
               "haystack_tokens": case["haystack_tokens"], "depth": case["needle_depth"]}
        print(f"\n--- case {i}/{len(cases)-1}  tokens={case['haystack_tokens']}  depth={case['needle_depth']:.2f} ---")

        if "baseline" in args.modes:
            prompt = case["haystack"] + "\n\nQ: " + case["question"] + "\nA:"
            t0 = time.time()
            out = run_baseline(model, tok, prompt, max_new_tokens=args.max_new_tokens)
            dt = time.time() - t0
            row["baseline"] = {"output": out, "correct": score_output(out, case["answer"]), "time": dt}
            print(f"  baseline ({dt:.0f}s) [{'OK' if row['baseline']['correct'] else 'X'}] {out!r}")

        if "reattn" in args.modes or "full" in args.modes:
            tmp_cache = args.out.parent / f"_cache_case{case['id']}"
            t0 = time.time()
            build_chunk_cache(model, tok, case["haystack"], tmp_cache,
                              chunk_size=args.chunk_size,
                              embed_fn=embedder.encode_passage)
            cache_time = time.time() - t0
            runner = SpragRunner(model, tok, embedder, RunnerConfig(
                cache_dir=tmp_cache, top_k=args.top_k,
                max_new_tokens=args.max_new_tokens, prefix_text=""
            ))

            if "reattn" in args.modes:
                t0 = time.time()
                res = runner.run("\n\nQ: " + case["question"] + "\nA:")
                run_time = time.time() - t0
                row["reattn"] = {
                    "output": res.output_text,
                    "correct": score_output(res.output_text, case["answer"]),
                    "cache_time": cache_time, "run_time": run_time,
                    "retrieved": res.retrieved_chunk_ids,
                    "scores": res.retrieved_scores,
                    "assembled_len": res.assembled_len,
                }
                print(f"  reattn cache={cache_time:.0f}s run={run_time:.0f}s "
                      f"chunks={res.retrieved_chunk_ids} "
                      f"[{'OK' if row['reattn']['correct'] else 'X'}] {res.output_text!r}")

            if "full" in args.modes:
                t0 = time.time()
                with mags_hook(model, mags_params):
                    res_full = runner.run("\n\nQ: " + case["question"] + "\nA:")
                run_time = time.time() - t0
                row["full"] = {
                    "output": res_full.output_text,
                    "correct": score_output(res_full.output_text, case["answer"]),
                    "cache_time": cache_time, "run_time": run_time,
                    "retrieved": res_full.retrieved_chunk_ids,
                    "scores": res_full.retrieved_scores,
                    "assembled_len": res_full.assembled_len,
                }
                print(f"  full   run={run_time:.0f}s "
                      f"chunks={res_full.retrieved_chunk_ids} "
                      f"[{'OK' if row['full']['correct'] else 'X'}] {res_full.output_text!r}")

        results.append(row)
        with args.out.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # Summary
    print("\n=== Summary ===")
    for mode in args.modes:
        accs = [r[mode]["correct"] for r in results if mode in r]
        print(f"  {mode}: {sum(accs)}/{len(accs)} = {sum(accs)/max(1,len(accs)):.0%}")


if __name__ == "__main__":
    main()
