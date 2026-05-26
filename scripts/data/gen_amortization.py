"""Multi-needle amortization dataset: one haystack, many queries.

Plants N distinct needles (different templates + picks) into a single
filler document at random depths, and emits N (question, answer) pairs
tied to that one document. Used by scripts/06_amortization_sweep.py to
measure baseline vs ReAttention per-query cost on a shared cache.
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


def sample_distinct_needles(rng, n: int) -> list[tuple[str, str, str, dict]]:
    """Draw n needles with distinct (template_idx, picks) — so each
    question/answer is unambiguous given the document."""
    seen: set[tuple] = set()
    out = []
    guard = 0
    while len(out) < n and guard < 10000:
        guard += 1
        ti = rng.randrange(len(NEEDLES))
        tmpl, keys, q_tmpl, ans_tmpl = NEEDLES[ti]
        picks = {k: rng.choice(v) for k, v in keys.items()}
        key = (ti,) + tuple(sorted(picks.items()))
        if key in seen:
            continue
        # also need question to be unique — same vault implies same question
        q = q_tmpl.format(**picks)
        if any(o[1] == q for o in out):
            continue
        seen.add(key)
        out.append((tmpl.format(**picks), q, ans_tmpl.format(**picks), picks))
    if len(out) < n:
        raise RuntimeError(f"only drew {len(out)} distinct needles; reduce n_needles")
    return out


def make_doc(tok, target_tokens: int, n_needles: int, rng):
    needles = sample_distinct_needles(rng, n_needles)
    needle_tok_total = sum(len(tok(n[0]).input_ids) for n in needles)
    budget = target_tokens - needle_tok_total - 64  # safety
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

    # Place needles at evenly-spaced depths (with small jitter) so each lives
    # in a different region of the document. Insert from the end so earlier
    # insertion indices stay valid.
    base_positions = sorted(
        int(len(pieces) * (0.10 + 0.80 * (i + 0.5) / n_needles)
            + rng.randint(-2, 2))
        for i in range(n_needles)
    )
    base_positions = [max(0, min(len(pieces), p)) for p in base_positions]
    queries = []
    for slot, (nt, q, a, picks) in zip(reversed(list(enumerate(base_positions))), reversed(needles)):
        idx, pos = slot
        pieces.insert(pos, nt)
        queries.append({"id": idx, "question": q, "answer": a, "picks": picks,
                        "depth_piece": pos})
    queries.sort(key=lambda r: r["id"])

    haystack = " ".join(pieces)
    total = len(tok(haystack).input_ids)
    return haystack, queries, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir; writes haystack.txt + queries.jsonl + meta.json")
    ap.add_argument("--target_tokens", type=int, default=16384)
    ap.add_argument("--n_needles", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH)
    rng = random.Random(args.seed)
    haystack, queries, total = make_doc(tok, args.target_tokens, args.n_needles, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "haystack.txt").write_text(haystack)
    with (args.out / "queries.jsonl").open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    (args.out / "meta.json").write_text(json.dumps({
        "target_tokens": args.target_tokens,
        "actual_tokens": total,
        "n_needles": args.n_needles,
        "seed": args.seed,
    }, indent=2))
    print(f"Wrote {args.n_needles}-needle doc ({total} tok) -> {args.out}")


if __name__ == "__main__":
    main()
