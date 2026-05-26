"""Oracle-retrieval MK eval — does ReAttention's splice work when
retrieval is perfect?

For each query, we find the chunk that actually contains the gold
needle (by substring match on its reconstructed text) and feed only
that chunk into the ReAttention assembly. No Jina, no top_k ranking.
This isolates the splice quality from retrieval quality.

Three modes per query:
  - baseline: full prompt (haystack + Q)  — known-good reference
  - oracle_k1: ReAttention with placements = [gold_chunk]
  - oracle_k3: ReAttention with [gold_chunk, distractor_1, distractor_2]
                — gold is FIRST in retrieval order, distractors after.
                Tests whether the splice is robust to competing needles
                when the right one is already there.

Saves per-query class + per-template breakdown so we can compare to
§5e's plain-reattn numbers.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

import torch
from safetensors.torch import load_file

from sprag.loader import load_model, FULL_ATTN_LAYERS
from sprag.chunk_cache import build_chunk_cache, load_meta
from sprag.embed import JinaEmbedder
from sprag.assemble import ChunkPlacement, patched_full_attn, make_inv_freq_for
from sprag.runner import run_baseline
from gen_niah import NEEDLES  # type: ignore


NUM_EQUIV = {"forty-two": ["42"], "seventeen": ["17"], "ninety-three": ["93"],
             "one hundred and one": ["101", "one hundred one"]}


def _expand(answer: str) -> list[list[str]]:
    parts = [p.strip() for p in answer.split("...") if p.strip()]
    return [[p.lower()] + NUM_EQUIV.get(p.lower(), []) for p in parts]


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


def reconstruct_needle(template_id: int, picks: dict) -> str:
    return NEEDLES[template_id][0].format(**picks)


def find_chunk_for_needle(cache_dir: Path, meta, tok, needle_text: str) -> int:
    spine = needle_text[max(0, len(needle_text)//2 - 25): len(needle_text)//2 + 25].lower()
    for c in meta["chunks"]:
        full = tok.decode(load_file(str(cache_dir / f"chunk_{c['id']:05d}.safetensors"))["input_ids"])
        if spine in full.lower():
            return c["id"]
    head = needle_text[:30].lower()
    for c in meta["chunks"]:
        if head in c["text_preview"].lower():
            return c["id"]
    return -1


def assemble_and_generate(model, tok, cache_dir, chunk_ids, chunk_lookup,
                           question: str, inv_freq, max_new_tokens: int):
    placements = []
    flat = []
    cursor = 0
    for cid in chunk_ids:
        t = load_file(str(Path(cache_dir) / f"chunk_{cid:05d}.safetensors"))
        ids = t["input_ids"]
        L = int(ids.shape[0])
        cached = {li: (t[f"K_l{li}"], t[f"V_l{li}"]) for li in FULL_ATTN_LAYERS}
        placements.append(ChunkPlacement(
            a_start=int(chunk_lookup[cid]["a_start"]),
            b_start=cursor, length=L, cached=cached,
        ))
        flat.extend(ids.tolist())
        cursor += L
    prompt_tail_ids = tok("\n\nQ: " + question + "\nA:", add_special_tokens=False).input_ids
    device = next(model.parameters()).device
    inp = torch.tensor([flat + prompt_tail_ids], dtype=torch.long, device=device)
    with torch.no_grad(), patched_full_attn(model, placements, inv_freq=inv_freq):
        out = model.generate(
            input_ids=inp, max_new_tokens=max_new_tokens,
            do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, inp.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--modes", nargs="+",
                    default=["baseline", "oracle_k1", "oracle_k3"],
                    choices=["baseline", "oracle_k1", "oracle_k3"])
    ap.add_argument("--limit_cases", type=int, default=None)
    args = ap.parse_args()

    suite_meta = json.loads((args.suite / "suite_meta.json").read_text())
    case_ids = [c["id"] for c in suite_meta["cases"]]
    if args.limit_cases:
        case_ids = case_ids[: args.limit_cases]
    print(f"Oracle MK eval: {len(case_ids)} cases, modes={args.modes}")

    model, tok, _ = load_model()
    device = next(model.parameters()).device
    emb = JinaEmbedder() if "oracle_k1" in args.modes or "oracle_k3" in args.modes else None
    inv_freq = make_inv_freq_for(model).to(device)

    counts = {m: {"correct": 0, "distractor": 0, "other": 0} for m in args.modes}
    per_tmpl = {}
    rows = []
    skipped_no_gold = 0

    for ci in case_ids:
        cd_src = args.suite / f"case_{ci:02d}"
        haystack = (cd_src / "haystack.txt").read_text()
        queries = [json.loads(l) for l in (cd_src / "queries.jsonl").open()]
        print(f"\n=== case {ci}  {len(queries)} queries ===")

        cache_dir = args.out.parent / f"_oracle_case{ci:02d}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        if "oracle_k1" in args.modes or "oracle_k3" in args.modes:
            build_chunk_cache(model, tok, haystack, cache_dir,
                              chunk_size=args.chunk_size,
                              embed_fn=emb.encode_passage)
            meta = load_meta(cache_dir)
            chunk_lookup = {c["id"]: c for c in meta["chunks"]}

        for q in queries:
            row = {"case": ci, "id": q["id"], "template_id": q["template_id"],
                   "answer": q["answer"], "distractor_answers": q["distractor_answers"]}
            gold_chunk = -1
            other_needle_chunks: list[int] = []
            if "oracle_k1" in args.modes or "oracle_k3" in args.modes:
                gold_needle = reconstruct_needle(q["template_id"], q["picks"])
                gold_chunk = find_chunk_for_needle(cache_dir, meta, tok, gold_needle)
                if gold_chunk < 0:
                    skipped_no_gold += 1
                    print(f"  q{q['id']} SKIP: no gold chunk found")
                    continue
                for q_other in queries:
                    if q_other["id"] == q["id"]:
                        continue
                    nt_other = reconstruct_needle(q_other["template_id"], q_other["picks"])
                    cid = find_chunk_for_needle(cache_dir, meta, tok, nt_other)
                    if 0 <= cid != gold_chunk and cid not in other_needle_chunks:
                        other_needle_chunks.append(cid)

            if "baseline" in args.modes:
                t0 = time.time()
                out = run_baseline(model, tok,
                                    haystack + "\n\nQ: " + q["question"] + "\nA:",
                                    max_new_tokens=args.max_new_tokens)
                dt = time.time() - t0
                cls = classify(out, q["answer"], q["distractor_answers"])
                counts["baseline"][cls] += 1
                row["baseline"] = {"output": out, "class": cls, "time": dt}
                print(f"  q{q['id']} t{q['template_id']} baseline   "
                      f"{dt:4.1f}s [{cls:10s}] {out[:60]!r}")

            if "oracle_k1" in args.modes:
                t0 = time.time()
                out = assemble_and_generate(model, tok, cache_dir, [gold_chunk],
                                             chunk_lookup, q["question"], inv_freq,
                                             args.max_new_tokens)
                dt = time.time() - t0
                cls = classify(out, q["answer"], q["distractor_answers"])
                counts["oracle_k1"][cls] += 1
                row["oracle_k1"] = {"output": out, "class": cls, "time": dt,
                                    "gold_chunk": gold_chunk}
                print(f"  q{q['id']} t{q['template_id']} oracle_k1  "
                      f"{dt:4.1f}s gold={gold_chunk} [{cls:10s}] {out[:60]!r}")

            if "oracle_k3" in args.modes:
                ids_k3 = [gold_chunk] + other_needle_chunks[:2]
                t0 = time.time()
                out = assemble_and_generate(model, tok, cache_dir, ids_k3,
                                             chunk_lookup, q["question"], inv_freq,
                                             args.max_new_tokens)
                dt = time.time() - t0
                cls = classify(out, q["answer"], q["distractor_answers"])
                counts["oracle_k3"][cls] += 1
                row["oracle_k3"] = {"output": out, "class": cls, "time": dt,
                                    "assembled": ids_k3}
                print(f"  q{q['id']} t{q['template_id']} oracle_k3  "
                      f"{dt:4.1f}s ids={ids_k3} [{cls:10s}] {out[:60]!r}")

            for m in args.modes:
                if m in row:
                    per_tmpl.setdefault(q["template_id"], {}) \
                            .setdefault(m, {"correct": 0, "n": 0})
                    per_tmpl[q["template_id"]][m]["n"] += 1
                    if row[m]["class"] == "correct":
                        per_tmpl[q["template_id"]][m]["correct"] += 1
            rows.append(row)
        # checkpoint
        with args.out.open("w") as f:
            json.dump({"rows": rows, "counts": counts, "per_tmpl": per_tmpl,
                       "skipped_no_gold": skipped_no_gold}, f, indent=2)

    print(f"\n=== Oracle summary ===  skipped_no_gold={skipped_no_gold}")
    for m, c in counts.items():
        n = sum(c.values())
        if n:
            print(f"  {m:11s}  correct {c['correct']:>3}/{n}  "
                  f"distractor {c['distractor']:>3}  other {c['other']:>3}")
    print("\n=== Per-template ===")
    template_names = {0: "vault", 1: "secret-keeper", 2: "bookshop"}
    for t in sorted(per_tmpl):
        line = f"  t{t} ({template_names.get(t,'?')}):"
        for m in args.modes:
            if m in per_tmpl[t]:
                d = per_tmpl[t][m]
                line += f"  {m}={d['correct']}/{d['n']}"
        print(line)


if __name__ == "__main__":
    main()
