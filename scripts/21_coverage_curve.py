"""Δ(coverage) curve for cross-context (full-doc) KV-cache reuse.

Research object (2026-05-31, after lit-search confirmed the gap): characterise
how the accuracy of REUSING a chunk's full-document-context KV cache (vs FRESH
recompute) depends on `coverage` = the fraction of that chunk's real preceding
context that is present in the assembled prompt. Find the crossover c* where
reuse becomes lossless / beneficial (the "global-context bonus" overtaking the
"context-drift" cost).

Controlled, single-target protocol (decouples coverage from chunk granularity
and from retrieval, the confounds in the earlier chunk_size sweep §5ad-RGB):
  - standard full-doc cache (each chunk's K/V built seeing the whole preceding doc)
  - ORACLE target = the one chunk that fully contains the answer (skip records
    where 0 or >1 chunks contain it → context never leaks the answer, so high
    coverage can't win for a spurious reason)
  - coverage knob = the target's IMMEDIATELY-PRECEDING c% of chunks (contiguous),
    placed right before the target; target sits last before the query
  - two arms, identical token layout [sink][ctx c%][target][Q]:
        cached: sink + ctx + target ALL spliced from cache at α=1.0 (= prior
                splice_topk_a1 / the prefill-skip regime), RoPE-shifted into place
        fresh : the same tokens, everything recomputed (plain generate)
  - Δ(c) = acc_cached(c) − acc_fresh(c); curve over c ∈ {0,25,50,75,100}%.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sprag.loader import load_model
from sprag.chunk_cache import build_chunk_cache, load_meta
from sprag.assemble import make_inv_freq_for
from sprag.rgb import load_rgb, matches, any_slot_alias_in

_spec = importlib.util.spec_from_file_location("sink_mk", ROOT / "scripts" / "12_sink_mk.py")
_sink = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sink)
build_sink_placement = _sink.build_sink_placement
build_chunk_placements_nostrip = _sink.build_chunk_placements_nostrip
run_assembled = _sink.run_assembled
_load_chunk = _sink._load_chunk


def chunk_has_answer(cache_dir, meta, tok, slots):
    """{chunk_id: any answer-slot alias present in the chunk text}."""
    return {c["id"]: any_slot_alias_in(
                tok.decode(_load_chunk(cache_dir, c["id"])["input_ids"]), slots)
            for c in meta["chunks"]}


def gen_fresh(model, tok, device, flat, query, max_new):
    tail = tok("\n\nQ: " + query + "\nA:", add_special_tokens=False).input_ids
    inp = torch.tensor([flat + tail], dtype=torch.long, device=device)
    with torch.no_grad():
        o = model.generate(input_ids=inp, max_new_tokens=max_new, do_sample=False,
                           use_cache=True, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, inp.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/benchmarks/rgb/data/en.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--coverages", type=float, nargs="+", default=[0, 25, 50, 75, 100])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min_depth", type=int, default=2,
                    help="require target chunk index ≥ this (need enough preceding "
                         "chunks for coverage resolution)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    recs = load_rgb(args.data, limit=args.limit)
    model, tok, _ = load_model()
    device = next(model.parameters()).device
    inv_freq = make_inv_freq_for(model).to(device)
    covs = args.coverages
    ckeys = [f"c{int(c)}" for c in covs]

    cached = {k: {"ok": 0, "n": 0} for k in ckeys}
    fresh = {k: {"ok": 0, "n": 0} for k in ckeys}
    rows = []
    done = set()
    if args.resume and args.out.exists():
        prev = json.loads(args.out.read_text())
        for r in prev.get("rows", []):
            rows.append(r); done.add(r["case"])
            for k in ckeys:
                if k in r:
                    cached[k]["n"] += 1; cached[k]["ok"] += r[k]["cached"]
                    fresh[k]["n"] += 1; fresh[k]["ok"] += r[k]["fresh"]
        print(f"resume: {len(done)} cases done")

    n_used = 0
    for ci, rec in enumerate(recs):
        if ci in done:
            continue
        passages, _ = rec.passages_shuffled(seed=ci)
        doc = "\n\n".join(passages)
        cache_dir = args.out.parent / f"_cov_case{ci:04d}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        build_chunk_cache(model, tok, doc, cache_dir, chunk_size=args.chunk_size, embed_fn=None)
        meta = load_meta(cache_dir)
        chunk_lookup = {c["id"]: c for c in meta["chunks"]}

        has_ans = chunk_has_answer(cache_dir, meta, tok, rec.slots)
        ans_ids = [cid for cid, h in has_ans.items() if h]
        # target = earliest chunk where the answer appears; deep enough for resolution
        t = min(ans_ids) if ans_ids else -1
        if t < args.min_depth:
            shutil.rmtree(cache_dir, ignore_errors=True)
            continue
        sink_ids = _load_chunk(cache_dir, 0)["input_ids"][:args.M].tolist()
        sink_pl, s_ids = build_sink_placement(cache_dir, args.M)

        # answerability gate: fresh with FULL preceding context (cov=100). Skip
        # records the model can't answer even with everything → pure noise.
        _, full_flat = build_chunk_placements_nostrip(
            cache_dir, list(range(0, t + 1)), chunk_lookup, b_offset=args.M)
        if not matches(gen_fresh(model, tok, device, s_ids + full_flat,
                                 rec.query, args.max_new_tokens), rec.slots):
            shutil.rmtree(cache_dir, ignore_errors=True)
            continue
        n_used += 1
        row = {"case": ci, "rid": rec.rid, "target": t, "n_prec": t}

        for c, k in zip(covs, ckeys):
            include = round(c / 100.0 * t)
            ctx_ids = list(range(t - include, t))          # immediately-preceding c%
            chunk_ids = ctx_ids + [t]
            # per-cell contamination: does any INCLUDED context chunk also carry the
            # answer? if so, both arms get it for free → exclude this cell from Δ.
            contam = any(has_ans[i] for i in ctx_ids)
            ch_pl, ch_flat = build_chunk_placements_nostrip(
                cache_dir, chunk_ids, chunk_lookup, b_offset=args.M)
            flat = s_ids + ch_flat
            cov_tok = sum(chunk_lookup[i]["num_tokens"] for i in ctx_ids) / max(1, chunk_lookup[t]["a_start"])
            t0 = time.time()
            ok_c = matches(run_assembled(model, tok, [sink_pl] + ch_pl, flat, rec.query,
                                         inv_freq, args.max_new_tokens, alpha=1.0,
                                         splice_kind="kv"), rec.slots)
            ok_f = matches(gen_fresh(model, tok, device, flat, rec.query,
                                     args.max_new_tokens), rec.slots)
            if not contam:                                  # clean cells only
                cached[k]["ok"] += ok_c; cached[k]["n"] += 1
                fresh[k]["ok"] += ok_f; fresh[k]["n"] += 1
            row[k] = {"cached": int(ok_c), "fresh": int(ok_f), "contam": int(contam),
                      "cov_tok": round(cov_tok, 3), "ntok": len(flat), "dt": time.time() - t0}
            print(f"  [{ci}] t={t:2d} cov={int(c):3d}% {'CONTAM' if contam else 'clean '} "
                  f"cached={'Y' if ok_c else '.'} fresh={'Y' if ok_f else '.'} ntok={len(flat)}")
        rows.append(row)
        shutil.rmtree(cache_dir, ignore_errors=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"chunk_size": args.chunk_size, "M": args.M, "coverages": covs,
             "cached": cached, "fresh": fresh, "n_used": n_used, "rows": rows}, indent=1))

    print(f"\n=== Δ(coverage) curve — single target, {n_used} answerable records ===")
    print(f"{'cov%':>5s} {'n_clean':>8s} {'cached':>8s} {'fresh':>8s} {'Δ=c−f':>8s}")
    for c, k in zip(covs, ckeys):
        nc = cached[k]["n"] or 1
        ca = 100 * cached[k]["ok"] / nc
        fa = 100 * fresh[k]["ok"] / nc
        print(f"{int(c):5d} {cached[k]['n']:8d} {ca:7.1f}% {fa:7.1f}% {ca - fa:+7.1f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
