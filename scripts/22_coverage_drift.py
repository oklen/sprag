"""Representational-drift curve for cross-context KV-cache reuse: cos(K_cached,
K_fresh) and cos(V) vs coverage. The rigorous, high-power backbone for the
Δ(coverage) story (the accuracy curve in 21_coverage_curve.py is n-limited).

For a target chunk built at doc position p (full-doc context), and an assembly
[sink][contiguous preceding c%][target]:
  K_cached = chunk's stored K (post-RoPE at p), shift_rope'd to the target's
             assembly position b  (so both operands share the rotation R_b →
             cos isolates CONTENT drift, position-invariant)
  K_fresh  = the target's K recomputed in the c%-context assembly (post-RoPE at b)
  drift(c) = mean cos over heads & target tokens (per layer + overall)
V has no RoPE → cos(V_cached, V_fresh) directly. As c→100% the fresh forward sees
the same context the cache was built with → cos→1; low coverage → max drift. No
answerability gate / answer-location needed (drift is a property of any chunk) →
all records usable, high power. Also dumps first-token vs mean cos to recheck
whether drift is head-localized (it isn't, behaviorally — §5ad-RGB head test).
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sprag.loader import load_model, FULL_ATTN_LAYERS
from sprag.chunk_cache import build_chunk_cache, load_meta, capture_full_attn_kv
from sprag.assemble import make_inv_freq_for
from sprag.rope import shift_rope
from sprag.rgb import load_rgb, any_slot_alias_in

_spec = importlib.util.spec_from_file_location("sink_mk", ROOT / "scripts" / "12_sink_mk.py")
_sink = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sink)
_load_chunk = _sink._load_chunk


def target_chunk(cache_dir, meta, tok, slots, min_depth):
    """Earliest chunk whose text carries an answer alias, deep enough for
    coverage resolution. -1 if none qualifies."""
    for c in meta["chunks"]:
        if c["id"] < min_depth:
            continue
        if any_slot_alias_in(tok.decode(_load_chunk(cache_dir, c["id"])["input_ids"]), slots):
            return c["id"]
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+",
                    default=[ROOT / "data/benchmarks/rgb/data/en.json",
                             ROOT / "data/benchmarks/rgb/data/en_int.json",
                             ROOT / "data/benchmarks/rgb/data/en_fact.json"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--coverages", type=float, nargs="+", default=[0, 25, 50, 75, 100])
    ap.add_argument("--min_depth", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="records per dataset")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    model, tok, _ = load_model()
    device = next(model.parameters()).device
    inv_freq = make_inv_freq_for(model)            # CPU; cos done on CPU tensors
    covs = args.coverages
    ckeys = [f"c{int(c)}" for c in covs]

    rows = []
    done = set()
    if args.resume and args.out.exists():
        for r in json.loads(args.out.read_text()).get("rows", []):
            rows.append(r); done.add((r["dataset"], r["case"]))
        print(f"resume: {len(done)} done")

    for dpath in args.data:
        recs = load_rgb(dpath, limit=args.limit)
        dname = dpath.stem
        for ci, rec in enumerate(recs):
            if (dname, ci) in done:
                continue
            passages, _ = rec.passages_shuffled(seed=ci)
            flat_p = []                       # en_int passages are nested (multi-hop)
            for p in passages:
                flat_p.extend(p if isinstance(p, list) else [p])
            doc = "\n\n".join(flat_p)
            cache_dir = args.out.parent / f"_drift_{dname}_{ci:04d}"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            build_chunk_cache(model, tok, doc, cache_dir, chunk_size=args.chunk_size, embed_fn=None)
            meta = load_meta(cache_dir)
            chunk_lookup = {c["id"]: c for c in meta["chunks"]}
            t = target_chunk(cache_dir, meta, tok, rec.slots, args.min_depth)
            if t < 0:
                shutil.rmtree(cache_dir, ignore_errors=True); continue
            p = chunk_lookup[t]["a_start"]
            L = chunk_lookup[t]["num_tokens"]
            tc = _load_chunk(cache_dir, t)
            K_cached = {li: tc[f"K_l{li}"] for li in FULL_ATTN_LAYERS}   # post-RoPE at p
            V_cached = {li: tc[f"V_l{li}"] for li in FULL_ATTN_LAYERS}
            sink = _load_chunk(cache_dir, 0)["input_ids"][:args.M].tolist()

            row = {"dataset": dname, "case": ci, "target": t}
            t0 = time.time()
            for c, k in zip(covs, ckeys):
                include = round(c / 100.0 * t)
                ctx = list(range(t - include, t))
                flat = list(sink)
                for cid in ctx:
                    flat += _load_chunk(cache_dir, cid)["input_ids"].tolist()
                b = len(flat)                       # target assembly start position
                flat += tc["input_ids"].tolist()
                with torch.no_grad(), capture_full_attn_kv(model) as kv:
                    model.model(input_ids=torch.tensor([flat], device=device), use_cache=False)
                ck, ckf, cv = [], [], []
                for li in FULL_ATTN_LAYERS:
                    Kf = kv[li]["K"][0][:, b:b + L, :].float()                 # (n_kv,L,hd) at b
                    Kc = shift_rope(K_cached[li].unsqueeze(0).float(), b - p, inv_freq)[0]
                    cosk = F.cosine_similarity(Kf, Kc, dim=-1)                 # (n_kv,L)
                    Vf = kv[li]["V"][0][:, b:b + L, :].float()
                    cosv = F.cosine_similarity(Vf, V_cached[li].unsqueeze(0).float()[0], dim=-1)
                    ck.append(cosk.mean().item()); ckf.append(cosk[:, 0].mean().item())
                    cv.append(cosv.mean().item())
                row[k] = {"cosK": sum(ck) / len(ck), "cosK_first": sum(ckf) / len(ckf),
                          "cosV": sum(cv) / len(cv), "cosK_layers": ck, "ntok": len(flat)}
            row["dt"] = time.time() - t0
            rows.append(row)
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"  [{dname} {ci}] t={t} "
                  + " ".join(f"{k}:K={row[k]['cosK']:.3f}/V={row[k]['cosV']:.3f}" for k in ckeys))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"coverages": covs, "rows": rows}, indent=1))

    print(f"\n=== drift(coverage), n={len(rows)} ({len(args.data)} datasets) ===")
    print(f"{'cov%':>5s} {'cosK':>7s} {'cosK_1st':>9s} {'cosV':>7s}")
    for c, k in zip(covs, ckeys):
        sub = [r[k] for r in rows if k in r]
        n = len(sub) or 1
        print(f"{int(c):5d} {sum(x['cosK'] for x in sub)/n:7.4f} "
              f"{sum(x['cosK_first'] for x in sub)/n:9.4f} {sum(x['cosV'] for x in sub)/n:7.4f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
