"""MK NIAH evaluation: multi-needle, multi-query per doc.

For each case: build chunk cache once (if reattn), run all queries
through baseline and reattn (at one or more top_k values), and
classify the output:
  - correct
  - distractor: output matches a sibling needle's answer (cross-needle
                hallucination — the failure mode that motivates MAGS)
  - other: neither gold nor any sibling answer surfaces

Numeric-word aware scorer: "101" ≡ "one hundred and one", etc.
"""
import argparse
import json
import re
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

# Map word-number gold strings to their digit form (and vice-versa).
NUM_EQUIV = {
    "forty-two": ["42"],
    "seventeen": ["17"],
    "ninety-three": ["93"],
    "one hundred and one": ["101", "one hundred one"],
}


def _expand(answer: str) -> list[str]:
    """Expand an answer into all surface forms that should count as match."""
    parts = [p.strip() for p in answer.split("...") if p.strip()]
    # For each part, expand if it has a numeric equivalent.
    expansions = []
    for p in parts:
        lo = p.lower()
        forms = [lo] + NUM_EQUIV.get(lo, [])
        expansions.append(forms)
    return expansions


def matches(output: str, answer: str) -> bool:
    lo = output.lower()
    return all(any(f in lo for f in forms) for forms in _expand(answer))


def classify(output: str, answer: str, distractors: list[str]) -> str:
    if matches(output, answer):
        return "correct"
    for d in distractors:
        if matches(output, d):
            return "distractor"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True,
                    help="dir produced by scripts/data/gen_mk_suite.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modes", nargs="+", default=["baseline", "reattn"],
                    choices=["baseline", "reattn"])
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--top_k", nargs="+", type=int, default=[3, 6])
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--limit_cases", type=int, default=None)
    args = ap.parse_args()

    suite_meta = json.loads((args.suite / "suite_meta.json").read_text())
    case_ids = [c["id"] for c in suite_meta["cases"]]
    if args.limit_cases:
        case_ids = case_ids[: args.limit_cases]
    print(f"suite: {args.suite}  cases={len(case_ids)}  "
          f"needles/case={suite_meta['n_needles']}  modes={args.modes}  "
          f"top_k={args.top_k}")

    print("Loading model + embedder...")
    model, tok, _ = load_model()
    embedder = JinaEmbedder() if "reattn" in args.modes else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    suite_results = {"suite_meta": suite_meta, "top_k": args.top_k,
                     "modes": args.modes, "cases": []}

    counts = {f"baseline": {"correct": 0, "distractor": 0, "other": 0}}
    for k in args.top_k:
        counts[f"reattn_k{k}"] = {"correct": 0, "distractor": 0, "other": 0}
    total_qs = 0
    time_acc = {kk: 0.0 for kk in counts}
    cache_acc = 0.0

    for ci in case_ids:
        cd = args.suite / f"case_{ci:02d}"
        haystack = (cd / "haystack.txt").read_text()
        queries = [json.loads(l) for l in (cd / "queries.jsonl").open()]
        meta = json.loads((cd / "meta.json").read_text())
        print(f"\n=== case {ci}  {meta['actual_tokens']} tok  "
              f"{len(queries)} queries ===")

        case_row = {"id": ci, "tokens": meta["actual_tokens"], "queries": []}
        runners: dict[int, SpragRunner] = {}
        if "reattn" in args.modes:
            cache_dir = args.out.parent / f"_cache_mk_case{ci:02d}"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            t0 = time.time()
            build_chunk_cache(model, tok, haystack, cache_dir,
                              chunk_size=args.chunk_size,
                              embed_fn=embedder.encode_passage)
            ct = time.time() - t0
            cache_acc += ct
            case_row["cache_time"] = ct
            for k in args.top_k:
                runners[k] = SpragRunner(model, tok, embedder, RunnerConfig(
                    cache_dir=cache_dir, top_k=k,
                    max_new_tokens=args.max_new_tokens, prefix_text=""
                ))
            print(f"  cache built in {ct:.1f}s")

        for q in queries:
            qrow = {"id": q["id"], "question": q["question"],
                    "answer": q["answer"], "template_id": q["template_id"],
                    "distractor_answers": q["distractor_answers"]}
            if "baseline" in args.modes:
                prompt = haystack + "\n\nQ: " + q["question"] + "\nA:"
                t0 = time.time()
                out = run_baseline(model, tok, prompt,
                                   max_new_tokens=args.max_new_tokens)
                dt = time.time() - t0
                cls = classify(out, q["answer"], q["distractor_answers"])
                counts["baseline"][cls] += 1
                time_acc["baseline"] += dt
                qrow["baseline"] = {"output": out, "class": cls, "time": dt}
                print(f"  q{q['id']} t{q['template_id']} baseline "
                      f"{dt:4.1f}s [{cls:10s}] {out[:80]!r}")
            if "reattn" in args.modes:
                for k in args.top_k:
                    kk = f"reattn_k{k}"
                    t0 = time.time()
                    res = runners[k].run("\n\nQ: " + q["question"] + "\nA:")
                    dt = time.time() - t0
                    cls = classify(res.output_text, q["answer"],
                                   q["distractor_answers"])
                    counts[kk][cls] += 1
                    time_acc[kk] += dt
                    qrow[kk] = {"output": res.output_text, "class": cls,
                                "time": dt, "retrieved": res.retrieved_chunk_ids}
                    print(f"  q{q['id']} t{q['template_id']} {kk:9s} "
                          f"{dt:4.1f}s chunks={res.retrieved_chunk_ids[:4]} "
                          f"[{cls:10s}] {res.output_text[:80]!r}")
            case_row["queries"].append(qrow)
            total_qs += 1
        suite_results["cases"].append(case_row)
        # checkpoint after every case
        with args.out.open("w") as f:
            json.dump(suite_results, f, indent=2)

    print("\n=== MK suite summary ===")
    print(f"  total queries: {total_qs}  cache total: {cache_acc:.1f}s")
    for kk, c in counts.items():
        n = sum(c.values())
        if n == 0:
            continue
        per = time_acc[kk] / n
        print(f"  {kk:11s}  correct {c['correct']:>3}/{n}  "
              f"distractor {c['distractor']:>3}  other {c['other']:>3}  "
              f"per-q {per:.2f}s")

    # Per-template breakdown
    print("\n=== Per-template (correct/n) ===")
    per_tmpl: dict[int, dict[str, dict[str, int]]] = {}
    for case in suite_results["cases"]:
        for q in case["queries"]:
            t = q["template_id"]
            per_tmpl.setdefault(t, {kk: {"correct": 0, "n": 0}
                                    for kk in counts})
            for kk in counts:
                if kk in q:
                    per_tmpl[t][kk]["n"] += 1
                    if q[kk]["class"] == "correct":
                        per_tmpl[t][kk]["correct"] += 1
    template_names = {0: "vault", 1: "secret-keeper", 2: "bookshop"}
    for t in sorted(per_tmpl):
        line = f"  template {t} ({template_names.get(t,'?')}):"
        for kk in counts:
            d = per_tmpl[t][kk]
            if d["n"]:
                line += f"  {kk}={d['correct']}/{d['n']}"
        print(line)


if __name__ == "__main__":
    main()
