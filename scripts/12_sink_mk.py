"""StreamingLLM-style Sink + ReAttention MK eval.

Test whether keeping the first M tokens of the haystack as a *global
attention sink* — and stripping the first S tokens off every retrieved
chunk's K/V before splicing — improves the assembly. The intuition
(Xiao et al. 2024, "Attention Sink") is that the first few tokens
absorb a disproportionate share of attention mass and that the model
breaks when no sticky sink is in scope.

Layout of the assembled prefill:
  [sink (M tokens, a=[0..M))]
  [chunk_i stripped (L_i - S tokens, a=[a_i+S..a_i+L_i))]   ← retrieved chunks
  ...
  [Q tail]

ReAttention's existing per-placement RoPE rebase handles the rotation
from a_start to b_start for both sink (delta=0, no rotation) and
stripped chunks (a_start += S).

Modes:
  oracle_k3       : gold + 2 sibling-needle chunks, no sink, no strip  (= §5h control)
  sink_oracle_k3  : same chunks, M-token sink prepended, each chunk stripped of first S
  reattn_k6       : Jina top-6, no sink                                (= §5e control)
  sink_k6         : Jina top-6, M-token sink prepended, each stripped of first S
  baseline        : full prompt                                        (optional, slow)
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
from sprag.retrieve import load_chunk_reprs, topk
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


def _load_chunk(cache_dir: Path, cid: int) -> dict:
    return load_file(str(Path(cache_dir) / f"chunk_{cid:05d}.safetensors"))


def build_sink_placement(cache_dir: Path, M: int) -> tuple[ChunkPlacement, list[int]]:
    """Sink = first M tokens of chunk 0 (i.e. of the doc). a_start=0, b_start=0,
    so delta=0 and no RoPE rotation will be applied."""
    t = _load_chunk(cache_dir, 0)
    ids = t["input_ids"][:M].tolist()
    cached = {li: (t[f"K_l{li}"][:, :M, :].contiguous(),
                   t[f"V_l{li}"][:, :M, :].contiguous())
              for li in FULL_ATTN_LAYERS}
    return ChunkPlacement(a_start=0, b_start=0, length=M, cached=cached), ids


def build_chunk_placements(cache_dir: Path, chunk_ids: list[int],
                            chunk_lookup: dict, S: int, b_offset: int
                            ) -> tuple[list[ChunkPlacement], list[int]]:
    """Each placement has its first S tokens stripped: a_start += S, length -= S,
    cached tensors sliced [:, S:, :]."""
    placements, flat = [], []
    cursor = b_offset
    for cid in chunk_ids:
        t = _load_chunk(cache_dir, cid)
        ids = t["input_ids"]
        L = int(ids.shape[0])
        if L <= S:
            continue
        kept_ids = ids[S:].tolist()
        cached = {li: (t[f"K_l{li}"][:, S:, :].contiguous(),
                       t[f"V_l{li}"][:, S:, :].contiguous())
                  for li in FULL_ATTN_LAYERS}
        placements.append(ChunkPlacement(
            a_start=int(chunk_lookup[cid]["a_start"]) + S,
            b_start=cursor, length=L - S, cached=cached,
        ))
        flat.extend(kept_ids)
        cursor += L - S
    return placements, flat


def build_chunk_placements_nostrip(cache_dir: Path, chunk_ids: list[int],
                                    chunk_lookup: dict, b_offset: int
                                    ) -> tuple[list[ChunkPlacement], list[int]]:
    """Plain placement (no sink, no strip) — used for the oracle_k3 and reattn_k6
    controls so that this script reproduces §5e / §5h numbers under the same
    decoding path."""
    placements, flat = [], []
    cursor = b_offset
    for cid in chunk_ids:
        t = _load_chunk(cache_dir, cid)
        ids = t["input_ids"]
        L = int(ids.shape[0])
        cached = {li: (t[f"K_l{li}"], t[f"V_l{li}"]) for li in FULL_ATTN_LAYERS}
        placements.append(ChunkPlacement(
            a_start=int(chunk_lookup[cid]["a_start"]),
            b_start=cursor, length=L, cached=cached,
        ))
        flat.extend(ids.tolist())
        cursor += L
    return placements, flat


def run_assembled(model, tok, placements, prefix_ids, question, inv_freq,
                   max_new_tokens):
    prompt_tail_ids = tok("\n\nQ: " + question + "\nA:", add_special_tokens=False).input_ids
    device = next(model.parameters()).device
    inp = torch.tensor([prefix_ids + prompt_tail_ids], dtype=torch.long, device=device)
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
    ap.add_argument("--M", type=int, default=4, help="sink length")
    ap.add_argument("--S", type=int, default=4, help="strip length per chunk")
    ap.add_argument("--top_k", type=int, default=6, help="for sink_k6 / reattn_k6")
    ap.add_argument("--modes", nargs="+",
                    default=["oracle_k3", "sink_oracle_k3", "reattn_k6", "sink_k6"],
                    choices=["baseline", "oracle_k3", "sink_oracle_k3",
                             "reattn_k6", "sink_k6"])
    ap.add_argument("--limit_cases", type=int, default=None)
    args = ap.parse_args()

    suite_meta = json.loads((args.suite / "suite_meta.json").read_text())
    case_ids = [c["id"] for c in suite_meta["cases"]]
    if args.limit_cases:
        case_ids = case_ids[: args.limit_cases]
    print(f"Sink MK eval: {len(case_ids)} cases  M={args.M} S={args.S} "
          f"top_k={args.top_k}  modes={args.modes}")

    model, tok, _ = load_model()
    device = next(model.parameters()).device
    inv_freq = make_inv_freq_for(model).to(device)
    need_cache = any(m != "baseline" for m in args.modes)
    need_jina = ("reattn_k6" in args.modes) or ("sink_k6" in args.modes)
    emb = JinaEmbedder() if need_cache else None

    counts = {m: {"correct": 0, "distractor": 0, "other": 0} for m in args.modes}
    time_acc = {m: 0.0 for m in args.modes}
    per_tmpl: dict = {}
    rows: list = []
    skipped_no_gold = 0

    for ci in case_ids:
        cd_src = args.suite / f"case_{ci:02d}"
        haystack = (cd_src / "haystack.txt").read_text()
        queries = [json.loads(l) for l in (cd_src / "queries.jsonl").open()]
        print(f"\n=== case {ci}  {len(queries)} queries ===")

        cache_dir = args.out.parent / f"_sink_case{ci:02d}"
        if need_cache:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            build_chunk_cache(model, tok, haystack, cache_dir,
                              chunk_size=args.chunk_size,
                              embed_fn=emb.encode_passage)
            meta = load_meta(cache_dir)
            chunk_lookup = {c["id"]: c for c in meta["chunks"]}
            if need_jina:
                jina_ids, jina_reprs = load_chunk_reprs(cache_dir)

        for q in queries:
            row = {"case": ci, "id": q["id"], "template_id": q["template_id"],
                   "answer": q["answer"]}

            # oracle gold + sibling chunks
            gold_chunk = -1
            other_needle_chunks: list[int] = []
            if any(m in args.modes for m in ("oracle_k3", "sink_oracle_k3")):
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

            # Jina top_k retrieval
            jina_top = None
            if need_jina:
                q_vec = emb.encode_query(["\n\nQ: " + q["question"] + "\nA:"])[0]
                idx, _scores = topk(q_vec, jina_reprs, k=args.top_k)
                jina_top = [jina_ids[i] for i in idx]

            if "baseline" in args.modes:
                t0 = time.time()
                out = run_baseline(model, tok,
                                    haystack + "\n\nQ: " + q["question"] + "\nA:",
                                    max_new_tokens=args.max_new_tokens)
                dt = time.time() - t0
                cls = classify(out, q["answer"], q["distractor_answers"])
                counts["baseline"][cls] += 1
                time_acc["baseline"] += dt
                row["baseline"] = {"output": out, "class": cls, "time": dt}
                print(f"  q{q['id']} t{q['template_id']} baseline       "
                      f"{dt:4.1f}s [{cls:10s}] {out[:60]!r}")

            if "oracle_k3" in args.modes:
                ids_k3 = [gold_chunk] + other_needle_chunks[:2]
                placements, flat = build_chunk_placements_nostrip(
                    cache_dir, ids_k3, chunk_lookup, b_offset=0)
                t0 = time.time()
                out = run_assembled(model, tok, placements, flat,
                                     q["question"], inv_freq, args.max_new_tokens)
                dt = time.time() - t0
                cls = classify(out, q["answer"], q["distractor_answers"])
                counts["oracle_k3"][cls] += 1
                time_acc["oracle_k3"] += dt
                row["oracle_k3"] = {"output": out, "class": cls, "time": dt,
                                     "assembled": ids_k3}
                print(f"  q{q['id']} t{q['template_id']} oracle_k3      "
                      f"{dt:4.1f}s ids={ids_k3} [{cls:10s}] {out[:60]!r}")

            if "sink_oracle_k3" in args.modes:
                ids_k3 = [gold_chunk] + other_needle_chunks[:2]
                sink_pl, sink_ids = build_sink_placement(cache_dir, args.M)
                ch_pl, ch_flat = build_chunk_placements(
                    cache_dir, ids_k3, chunk_lookup, args.S, b_offset=args.M)
                placements = [sink_pl] + ch_pl
                flat = sink_ids + ch_flat
                t0 = time.time()
                out = run_assembled(model, tok, placements, flat,
                                     q["question"], inv_freq, args.max_new_tokens)
                dt = time.time() - t0
                cls = classify(out, q["answer"], q["distractor_answers"])
                counts["sink_oracle_k3"][cls] += 1
                time_acc["sink_oracle_k3"] += dt
                row["sink_oracle_k3"] = {"output": out, "class": cls, "time": dt,
                                           "assembled": ids_k3}
                print(f"  q{q['id']} t{q['template_id']} sink_oracle_k3 "
                      f"{dt:4.1f}s ids={ids_k3} [{cls:10s}] {out[:60]!r}")

            if "reattn_k6" in args.modes:
                placements, flat = build_chunk_placements_nostrip(
                    cache_dir, jina_top, chunk_lookup, b_offset=0)
                t0 = time.time()
                out = run_assembled(model, tok, placements, flat,
                                     q["question"], inv_freq, args.max_new_tokens)
                dt = time.time() - t0
                cls = classify(out, q["answer"], q["distractor_answers"])
                counts["reattn_k6"][cls] += 1
                time_acc["reattn_k6"] += dt
                row["reattn_k6"] = {"output": out, "class": cls, "time": dt,
                                     "retrieved": jina_top}
                print(f"  q{q['id']} t{q['template_id']} reattn_k6      "
                      f"{dt:4.1f}s chunks={jina_top[:4]} [{cls:10s}] {out[:60]!r}")

            if "sink_k6" in args.modes:
                sink_pl, sink_ids = build_sink_placement(cache_dir, args.M)
                ch_pl, ch_flat = build_chunk_placements(
                    cache_dir, jina_top, chunk_lookup, args.S, b_offset=args.M)
                placements = [sink_pl] + ch_pl
                flat = sink_ids + ch_flat
                t0 = time.time()
                out = run_assembled(model, tok, placements, flat,
                                     q["question"], inv_freq, args.max_new_tokens)
                dt = time.time() - t0
                cls = classify(out, q["answer"], q["distractor_answers"])
                counts["sink_k6"][cls] += 1
                time_acc["sink_k6"] += dt
                row["sink_k6"] = {"output": out, "class": cls, "time": dt,
                                   "retrieved": jina_top}
                print(f"  q{q['id']} t{q['template_id']} sink_k6        "
                      f"{dt:4.1f}s chunks={jina_top[:4]} [{cls:10s}] {out[:60]!r}")

            for m in args.modes:
                if m in row:
                    per_tmpl.setdefault(q["template_id"], {}) \
                            .setdefault(m, {"correct": 0, "n": 0})
                    per_tmpl[q["template_id"]][m]["n"] += 1
                    if row[m]["class"] == "correct":
                        per_tmpl[q["template_id"]][m]["correct"] += 1
            rows.append(row)
        with args.out.open("w") as f:
            json.dump({"M": args.M, "S": args.S, "top_k": args.top_k,
                       "rows": rows, "counts": counts,
                       "per_tmpl": per_tmpl,
                       "skipped_no_gold": skipped_no_gold}, f, indent=2)

    print(f"\n=== Sink MK summary ===  M={args.M} S={args.S}  "
          f"skipped_no_gold={skipped_no_gold}")
    for m in args.modes:
        c = counts[m]
        n = sum(c.values())
        if not n:
            continue
        per = time_acc[m] / n
        print(f"  {m:15s}  correct {c['correct']:>3}/{n}  "
              f"distractor {c['distractor']:>3}  other {c['other']:>3}  "
              f"per-q {per:.2f}s")
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
