"""Multi-Key NIAH suite: N cases × K needles/case, tagged for failure-mode
analysis.

Each case is a single document containing K needles drawn from distinct
(template, picks) pairs. Each query records:
  - template_id  (which family the gold needle came from)
  - distractor_answers  (the answers of *sibling* needles in the same
    doc — i.e., what a cross-needle hallucination would say)

Output layout:
  out/
    case_00/{haystack.txt, queries.jsonl, meta.json}
    case_01/...
    suite_meta.json

`queries.jsonl` rows: {id, question, answer, picks, template_id,
distractor_answers, depth_piece}
"""
import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from transformers import AutoTokenizer
from sprag.loader import DEFAULT_MODEL_PATH
from gen_niah import FILLERS, NEEDLES  # type: ignore


def sample_distinct_needles(rng, n: int):
    """Each needle entry: (text, q, a, picks, template_id).

    Constraints (so the doc is unambiguous and disambiguation is required):
      - distinct (template, picks) → distinct needle texts
      - distinct questions
      - within a template family, distinct *answer strings* (otherwise
        the model can guess by template alone and score well without
        disambiguating by key)
    """
    seen = set()
    answers_by_tid: dict[int, set[str]] = {}
    out = []
    guard = 0
    while len(out) < n and guard < 50000:
        guard += 1
        ti = rng.randrange(len(NEEDLES))
        tmpl, keys, q_tmpl, ans_tmpl = NEEDLES[ti]
        picks = {k: rng.choice(v) for k, v in keys.items()}
        key = (ti,) + tuple(sorted(picks.items()))
        if key in seen:
            continue
        q = q_tmpl.format(**picks)
        a = ans_tmpl.format(**picks)
        if any(o[1] == q for o in out):
            continue
        if a in answers_by_tid.get(ti, set()):
            continue
        seen.add(key)
        answers_by_tid.setdefault(ti, set()).add(a)
        out.append((tmpl.format(**picks), q, a, picks, ti))
    if len(out) < n:
        raise RuntimeError(f"only drew {len(out)} distinct needles; reduce n_needles")
    return out


def make_case(tok, target_tokens: int, n_needles: int, rng):
    needles = sample_distinct_needles(rng, n_needles)
    nt_total = sum(len(tok(n[0]).input_ids) for n in needles)
    budget = target_tokens - nt_total - 64
    if budget <= 0:
        raise ValueError(f"target_tokens={target_tokens} too small for {n_needles} needles")

    pieces = []
    used = 0
    while used < budget:
        s = rng.choice(FILLERS)
        n_tok = len(tok(" " + s).input_ids)
        if used + n_tok > budget + 16:
            break
        pieces.append(s)
        used += n_tok

    # Evenly-spaced insertion points (with jitter), inserted from the
    # tail so earlier indices stay valid.
    base = sorted(
        max(0, min(len(pieces),
                   int(len(pieces) * (0.10 + 0.80 * (i + 0.5) / n_needles)
                       + rng.randint(-2, 2))))
        for i in range(n_needles)
    )
    queries = []
    # iterate in reverse so insertions don't shift earlier positions
    for slot_idx in range(n_needles - 1, -1, -1):
        pos = base[slot_idx]
        text, q, a, picks, tid = needles[slot_idx]
        pieces.insert(pos, text)
        queries.append({"id": slot_idx, "question": q, "answer": a, "picks": picks,
                        "template_id": tid, "depth_piece": pos})
    queries.sort(key=lambda r: r["id"])

    # Distractor answers: for each query, list answers of *other needles
    # of the same template family* in this doc — those are the answers
    # the model might cross-pollute with.
    by_tid: dict[int, list[str]] = {}
    for q in queries:
        by_tid.setdefault(q["template_id"], []).append(q["answer"])
    for q in queries:
        same = sorted({a for a in by_tid[q["template_id"]] if a != q["answer"]})
        q["distractor_answers"] = same

    haystack = " ".join(pieces)
    total = len(tok(haystack).input_ids)
    return haystack, queries, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target_tokens", type=int, default=8192)
    ap.add_argument("--n_needles", type=int, default=6)
    ap.add_argument("--n_cases", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    suite_meta = {"target_tokens": args.target_tokens, "n_needles": args.n_needles,
                  "n_cases": args.n_cases, "seed": args.seed, "cases": []}
    for ci in range(args.n_cases):
        haystack, queries, total = make_case(tok, args.target_tokens, args.n_needles, rng)
        cd = args.out / f"case_{ci:02d}"
        cd.mkdir(exist_ok=True)
        (cd / "haystack.txt").write_text(haystack)
        with (cd / "queries.jsonl").open("w") as f:
            for q in queries:
                f.write(json.dumps(q) + "\n")
        (cd / "meta.json").write_text(json.dumps(
            {"actual_tokens": total, "n_needles": args.n_needles}, indent=2))
        suite_meta["cases"].append({"id": ci, "tokens": total})
        print(f"  case {ci:02d}: {total} tok, {args.n_needles} needles")
    (args.out / "suite_meta.json").write_text(json.dumps(suite_meta, indent=2))
    print(f"Wrote MK suite -> {args.out}")


if __name__ == "__main__":
    main()
