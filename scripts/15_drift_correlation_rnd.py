"""Re-measure drift correlation under random-anchor cache.

Mirrors scripts/14_drift_correlation.py but uses the anchor_random cache
written by scripts/12_sink_mk.py (data/mk/_sink_caseXX/) and the
build_chunk_placements_random_anchor splice layout. The point is to check
whether per-chunk unique anchors DID decorrelate the sibling drift
directions (regardless of the accuracy outcome).

Run:
    python3 scripts/15_drift_correlation_rnd.py \\
        --suite data/mk/suite_8k --cache_dir_root data/mk \\
        --out data/diag/drift_corr_anchor_random.json
"""
import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

import torch

from sprag.loader import load_model, FULL_ATTN_LAYERS
from sprag.chunk_cache import load_meta
from sprag.assemble import ChunkPlacement, make_inv_freq_for
from safetensors.torch import load_file

_spec13 = importlib.util.spec_from_file_location(
    "_diag13", str(Path(__file__).resolve().parent / "13_diagnose_splice.py"))
_diag13 = importlib.util.module_from_spec(_spec13)
_spec13.loader.exec_module(_diag13)

_spec14 = importlib.util.spec_from_file_location(
    "_corr14", str(Path(__file__).resolve().parent / "14_drift_correlation.py"))
_corr14 = importlib.util.module_from_spec(_spec14)
_spec14.loader.exec_module(_corr14)


def _load_chunk(cache_dir: Path, cid: int) -> dict:
    return load_file(str(Path(cache_dir) / f"chunk_{cid:05d}.safetensors"))


def build_placements_rnd(cache_dir: Path, chunk_ids: list[int],
                          chunk_lookup: dict, M_sink: int):
    """Sink at b=0 + per-chunk [anchor_ids fresh + chunk K/V at b+M_anchor]."""
    t0 = _load_chunk(cache_dir, 0)
    sink_cached = {li: (t0[f"K_l{li}"][:, :M_sink, :].contiguous(),
                         t0[f"V_l{li}"][:, :M_sink, :].contiguous())
                    for li in FULL_ATTN_LAYERS}
    placements = [ChunkPlacement(a_start=0, b_start=0, length=M_sink, cached=sink_cached)]
    flat = t0["input_ids"][:M_sink].tolist()
    cursor = M_sink
    for cid in chunk_ids:
        meta_c = chunk_lookup[cid]
        anchor_list = meta_c.get("anchor_ids", []) or []
        if anchor_list:
            flat.extend(anchor_list)
            cursor += len(anchor_list)
        t = _load_chunk(cache_dir, cid)
        ids = t["input_ids"]
        L = int(ids.shape[0])
        cached = {li: (t[f"K_l{li}"], t[f"V_l{li}"]) for li in FULL_ATTN_LAYERS}
        placements.append(ChunkPlacement(
            a_start=int(meta_c["a_start"]),
            b_start=cursor, length=L, cached=cached,
        ))
        flat.extend(ids.tolist())
        cursor += L
    return placements, flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--cache_dir_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--limit_cases", type=int, default=None)
    args = ap.parse_args()

    suite_meta = json.loads((args.suite / "suite_meta.json").read_text())
    case_ids = [c["id"] for c in suite_meta["cases"]]
    if args.limit_cases:
        case_ids = case_ids[: args.limit_cases]

    model, tok, _ = load_model()
    device = next(model.parameters()).device
    inv_freq = make_inv_freq_for(model).to(device)

    rows = []
    for ci in case_ids:
        cd_src = args.suite / f"case_{ci:02d}"
        queries = [json.loads(l) for l in (cd_src / "queries.jsonl").open()]
        cache_dir = args.cache_dir_root / f"_sink_case{ci:02d}"
        if not cache_dir.exists():
            print(f"[skip] no cache at {cache_dir}")
            continue
        meta = load_meta(cache_dir)
        if meta.get("filler_mode") != "random_per_chunk":
            print(f"[warn] case {ci} cache filler_mode={meta.get('filler_mode')} (need random_per_chunk)")
            continue
        chunk_lookup = {c["id"]: c for c in meta["chunks"]}

        for q in queries:
            gold_needle = _diag13.reconstruct_needle(q["template_id"], q["picks"])
            gold = _diag13.find_chunk_for_needle(cache_dir, meta, tok, gold_needle)
            if gold < 0:
                continue
            sibs = []
            for q_other in queries:
                if q_other["id"] == q["id"]:
                    continue
                nt = _diag13.reconstruct_needle(q_other["template_id"], q_other["picks"])
                cid = _diag13.find_chunk_for_needle(cache_dir, meta, tok, nt)
                if 0 <= cid != gold and cid not in sibs:
                    sibs.append(cid)
                if len(sibs) >= 2:
                    break
            if len(sibs) < 2:
                continue

            chunk_ids = [gold] + sibs
            placements, flat = build_placements_rnd(cache_dir, chunk_ids, chunk_lookup, args.M)
            captures = _diag13.diagnose_one(model, tok, placements, flat,
                                              q["question"], inv_freq)

            for li, plist in captures.items():
                if len(plist) < 4:
                    continue
                dK = {}; nK = {}
                for role, cap in zip(["sink", "gold", "sib0", "sib1"], plist):
                    dK[role] = _corr14.mean_drift_per_head(cap["K_fresh"], cap["K_shifted"])
                    nK[role] = _corr14.vec_norm(dK[role])
                row = {
                    "case": ci, "qid": q["id"], "template_id": q["template_id"],
                    "layer": li,
                    "nK_gold": nK["gold"], "nK_sib0": nK["sib0"], "nK_sib1": nK["sib1"],
                    "K_corr_sib0_sib1": _corr14.per_head_cos(dK["sib0"], dK["sib1"]),
                    "K_corr_gold_sib0": _corr14.per_head_cos(dK["gold"], dK["sib0"]),
                    "K_corr_gold_sib1": _corr14.per_head_cos(dK["gold"], dK["sib1"]),
                }
                rows.append(row)
            print(f"  case {ci} q{q['id']} t{q['template_id']} captured "
                  f"{sum(len(v)>=4 for v in captures.values())} layers")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(rows, f)

    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r)

    print("\n=== Drift-correlation (anchor RANDOM per-chunk) ===")
    print(f"  layer  n   ||dK_gold|| ||dK_sib0|| ||dK_sib1||  "
          f"K(sib0,sib1)  K(gold,sib0)  K(gold,sib1)")
    for li in sorted(by_layer):
        rs = by_layer[li]
        n = len(rs)
        def avg(k): return sum(r[k] for r in rs) / n
        print(f"  L{li:>2}   {n:>3}  "
              f"{avg('nK_gold'):>10.4f} {avg('nK_sib0'):>10.4f} {avg('nK_sib1'):>10.4f}  "
              f"{avg('K_corr_sib0_sib1'):>12.4f}  "
              f"{avg('K_corr_gold_sib0'):>12.4f}  "
              f"{avg('K_corr_gold_sib1'):>12.4f}")


if __name__ == "__main__":
    main()
