"""RGB (Retrieval-Augmented Generation Benchmark) eval for the sprag methods.

Each RGB record gives a query, gold answer(s), a set of `positive` (gold) and
`negative` (distractor) passages. We concatenate the shuffled passages into one
~9K-token document (the noisy corpus), precompute a standard full-doc chunk
cache, then compare three ways of answering the query:

  baseline     full noisy doc + Q  -> generate            (stuff-everything RAG)
  raw_topk     sink + Jina top-k chunks, FRESH K/V        (short-assembly format)
  splice_topk  sink + Jina top-k chunks, CACHED K/V (α)   (ReAttention / TurboRAG)

Optional upper bounds:
  oracle_raw / oracle_splice  use answer-containing chunks instead of Jina top-k.

This is the §5-splice-decomposition story on a real RAG benchmark: how much of
the win is the short-assembly format vs. the cached-K/V splice, and does α=1.0
still carry the footgun outside the synthetic MK suite.

Scoring = RGB checkanswer (every answer slot's alias present in output).
"""
import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from sprag.loader import load_model, FULL_ATTN_LAYERS
from sprag.chunk_cache import (build_chunk_cache, build_anchor_chunk_cache,
                                  build_fixed_anchor_chunk_cache, load_meta)
from sprag.embed import JinaEmbedder
from sprag.assemble import make_inv_freq_for
from sprag.retrieve import load_chunk_reprs, topk
from sprag.runner import run_baseline
from sprag.rgb import load_rgb, matches, any_slot_alias_in

# Reuse the placement builders + run_assembled from script 12 (not importable
# as a package module — load it by path).
_spec = importlib.util.spec_from_file_location("sink_mk", ROOT / "scripts" / "12_sink_mk.py")
_sink = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sink)
build_sink_placement = _sink.build_sink_placement
build_chunk_placements_nostrip = _sink.build_chunk_placements_nostrip
run_assembled = _sink.run_assembled
_load_chunk = _sink._load_chunk


def find_oracle_chunks(cache_dir: Path, meta, tok, slots, k: int) -> list[int]:
    """Chunks whose decoded text contains an answer alias, capped to k."""
    hits = []
    for c in meta["chunks"]:
        ids = _load_chunk(cache_dir, c["id"])["input_ids"]
        if any_slot_alias_in(tok.decode(ids), slots):
            hits.append(c["id"])
        if len(hits) >= k:
            break
    return hits


def run_raw(model, tok, device, sink_ids, chunk_ids, cache_dir, question, max_new):
    """sink + chunks as fresh tokens, plain generate (no splice)."""
    flat = list(sink_ids)
    for cid in chunk_ids:
        flat.extend(_load_chunk(cache_dir, cid)["input_ids"].tolist())
    tail = tok("\n\nQ: " + question + "\nA:", add_special_tokens=False).input_ids
    inp = torch.tensor([flat + tail], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(input_ids=inp, max_new_tokens=max_new,
                             do_sample=False, use_cache=True,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inp.shape[1]:], skip_special_tokens=True), inp.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=ROOT / "data/benchmarks/rgb/data/en.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--M", type=int, default=4, help="sink length")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="splice blend: K = α·cached + (1-α)·fresh")
    ap.add_argument("--modes", nargs="+",
                    default=["baseline", "raw_topk", "splice_topk", "splice_topk_a1"],
                    choices=["baseline", "raw_topk", "splice_topk", "splice_topk_a1",
                             "k_only_topk", "v_only_topk",
                             "oracle_raw", "oracle_splice"])
    ap.add_argument("--cache_kind", type=str, default="standard",
                    choices=["standard", "anchor", "fixed"],
                    help="standard = single full-doc forward; "
                         "anchor = per-chunk [sink+chunk] forward (§5w: lower "
                         "cache->assembly drift, splice viable); "
                         "fixed = per-chunk [fixed_anchor_token x M + chunk] forward, "
                         "same fixed anchor placed once FRESH at the front of the "
                         "assembly (§5y symmetric anchor).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reuse_cache", action="store_true")
    ap.add_argument("--keep_cache", action="store_true",
                    help="keep per-case cache dirs (default: delete after each "
                         "record to bound disk at ~one case)")
    ap.add_argument("--resume", action="store_true",
                    help="if --out exists, load its rows and skip cases already "
                         "done (for long unattended runs).")
    args = ap.parse_args()

    recs = load_rgb(args.data, limit=args.limit)
    print(f"RGB eval: {len(recs)} records  chunk_size={args.chunk_size} "
          f"M={args.M} top_k={args.top_k} alpha={args.alpha} "
          f"cache_kind={args.cache_kind} modes={args.modes}")

    model, tok, _ = load_model()
    device = next(model.parameters()).device
    inv_freq = make_inv_freq_for(model).to(device)
    emb = JinaEmbedder()

    counts = {m: {"correct": 0, "n": 0} for m in args.modes}
    time_acc = {m: 0.0 for m in args.modes}
    tok_acc = {m: 0 for m in args.modes}
    rows = []
    done_cases: set[int] = set()
    if args.resume and args.out.exists():
        prev = json.loads(args.out.read_text())
        for r in prev.get("rows", []):
            rows.append(r)
            done_cases.add(r["case"])
            for m in args.modes:
                if m in r:
                    counts[m]["n"] += 1
                    counts[m]["correct"] += (r[m]["class"] == "correct")
                    time_acc[m] += r[m].get("time", 0.0)
                    tok_acc[m] += r[m].get("ntok", 0)
        print(f"  resume: {len(done_cases)} cases already done")
    use_oracle = any(m.startswith("oracle") for m in args.modes)
    use_topk = any(m.endswith("topk") for m in args.modes)

    for ci, rec in enumerate(recs):
        if ci in done_cases:
            continue
        passages, _is_gold = rec.passages_shuffled(seed=ci)
        doc = "\n\n".join(passages)
        cache_dir = args.out.parent / f"_rgb_case{ci:04d}"
        ready = (args.reuse_cache and cache_dir.exists()
                 and (cache_dir / "meta.json").exists())
        if not ready:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            if args.cache_kind == "anchor":
                build_anchor_chunk_cache(model, tok, doc, cache_dir,
                                         chunk_size=args.chunk_size,
                                         anchor_M=args.M, filler_mode="none",
                                         embed_fn=emb.encode_passage)
            elif args.cache_kind == "fixed":
                build_fixed_anchor_chunk_cache(model, tok, doc, cache_dir,
                                               chunk_size=args.chunk_size,
                                               anchor_M=args.M,
                                               embed_fn=emb.encode_passage)
            else:
                build_chunk_cache(model, tok, doc, cache_dir,
                                  chunk_size=args.chunk_size,
                                  embed_fn=emb.encode_passage)
        meta = load_meta(cache_dir)
        chunk_lookup = {c["id"]: c for c in meta["chunks"]}

        jina_top = None
        if use_topk:
            q_vec = emb.encode_query([rec.query])[0]
            jina_ids, jina_reprs = load_chunk_reprs(cache_dir)
            idx, _ = topk(q_vec, jina_reprs, k=args.top_k)
            jina_top = [jina_ids[i] for i in idx]
        oracle_top = None
        if use_oracle:
            oracle_top = find_oracle_chunks(cache_dir, meta, tok, rec.slots, args.top_k)
            if not oracle_top:           # answer not found in any chunk -> fall back
                oracle_top = jina_top if jina_top is not None else [0]

        row = {"case": ci, "rid": rec.rid, "query": rec.query,
               "slots": rec.slots, "n_chunks": meta["num_chunks"],
               "jina_top": jina_top, "oracle_top": oracle_top}

        def record(mode, out, dt, ntok):
            cls = "correct" if matches(out, rec.slots) else "wrong"
            counts[mode]["correct"] += (cls == "correct")
            counts[mode]["n"] += 1
            time_acc[mode] += dt
            tok_acc[mode] += ntok
            row[mode] = {"output": out, "class": cls, "time": dt, "ntok": ntok}
            print(f"  [{ci}] {mode:12s} {dt:5.2f}s ntok={ntok:5d} "
                  f"[{cls:7s}] {out[:55]!r}")

        if "baseline" in args.modes:
            prompt = doc + "\n\nQ: " + rec.query + "\nA:"
            ntok = len(tok(prompt, add_special_tokens=True).input_ids)
            t0 = time.time()
            out = run_baseline(model, tok, prompt, max_new_tokens=args.max_new_tokens)
            record("baseline", out, time.time() - t0, ntok)

        if args.cache_kind == "fixed":
            sink_ids = [int(meta["anchor_token_id"])] * args.M
        else:
            sink_ids = _load_chunk(cache_dir, 0)["input_ids"][:args.M].tolist()

        for mode, ids in (("raw_topk", jina_top), ("oracle_raw", oracle_top)):
            if mode not in args.modes:
                continue
            t0 = time.time()
            out, ntok = run_raw(model, tok, device, sink_ids, ids, cache_dir,
                                rec.query, args.max_new_tokens)
            record(mode, out, time.time() - t0, ntok)

        # (mode, retrieved ids, alpha, splice_kind)
        for mode, ids, a, kind in (
                ("splice_topk", jina_top, args.alpha, "kv"),
                ("splice_topk_a1", jina_top, 1.0, "kv"),
                ("k_only_topk", jina_top, 1.0, "k"),   # K cached, V fresh
                ("v_only_topk", jina_top, 1.0, "v"),   # V cached, K fresh
                ("oracle_splice", oracle_top, args.alpha, "kv")):
            if mode not in args.modes:
                continue
            ch_pl, ch_flat = build_chunk_placements_nostrip(
                cache_dir, ids, chunk_lookup, b_offset=args.M)
            if args.cache_kind == "fixed":
                # one fixed anchor placed FRESH at the front (no splice); chunks
                # spliced after it (§5y symmetric anchor).
                placements = ch_pl
                flat = sink_ids + ch_flat
            else:
                sink_pl, s_ids = build_sink_placement(cache_dir, args.M)
                placements = [sink_pl] + ch_pl
                flat = s_ids + ch_flat
            ntok = len(flat) + len(tok("\n\nQ: " + rec.query + "\nA:",
                                       add_special_tokens=False).input_ids)
            t0 = time.time()
            out = run_assembled(model, tok, placements, flat, rec.query,
                                inv_freq, args.max_new_tokens, alpha=a,
                                splice_kind=kind)
            record(mode, out, time.time() - t0, ntok)

        rows.append(row)
        if not (args.keep_cache or args.reuse_cache):
            shutil.rmtree(cache_dir, ignore_errors=True)
        with args.out.open("w") as f:
            json.dump({"data": str(args.data), "chunk_size": args.chunk_size,
                       "M": args.M, "top_k": args.top_k, "alpha": args.alpha,
                       "cache_kind": args.cache_kind,
                       "counts": counts, "rows": rows}, f, indent=2)

    print(f"\n=== RGB summary ===  {len(rows)} records  alpha={args.alpha}")
    for m in args.modes:
        n = counts[m]["n"]
        if not n:
            continue
        print(f"  {m:13s}  acc {counts[m]['correct']:>3}/{n} "
              f"({100*counts[m]['correct']/n:4.1f}%)  "
              f"per-q {time_acc[m]/n:5.2f}s  avg_tok {tok_acc[m]//n}")


if __name__ == "__main__":
    main()
