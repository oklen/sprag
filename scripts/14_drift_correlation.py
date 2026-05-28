"""Step-1 probe of "are sibling K/V drifts correlated across chunks?"

For each MK query (anchor v2 cache + sink_oracle_k3 splice), capture at every
full-attn layer the per-role drift:
    d_K_role[layer] := K_fresh_role - K_shifted_role   (shape: kv_h, L, d_h)
    d_V_role[layer] := V_fresh_role - V_cached_role

For each (layer, role) collapse positions: mean over L -> (kv_h, d_h).
Then for each pair of roles (r1, r2) compute per-head cos and average:
    corr(r1, r2, layer) = mean_h cos(d_r1[h], d_r2[h])

If sibling drifts share a direction in head-space, corr(sib0, sib1) >> 0.
If they're independent, corr ≈ 0.

We also report ||d|| magnitudes so a near-zero cos isn't an artefact of
tiny drift (which we already know isn't the case from gold under standard
cache — but verify under anchor cache where gold drift IS ≈ 0).

Run (anchor cache exists in data/diag/_diag_caseXX/):
    python3 scripts/14_drift_correlation.py \\
        --suite data/mk/suite_8k --cache_dir_root data/diag \\
        --out data/diag/drift_corr_anchor_v2.json --limit_cases 5
"""
import argparse
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
from sprag.assemble import make_inv_freq_for

# Re-use the diagnostic forward and helpers from script 13.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_diag13", str(Path(__file__).resolve().parent / "13_diagnose_splice.py"))
_diag13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diag13)


def mean_drift_per_head(fresh: torch.Tensor, cached_or_shifted: torch.Tensor) -> torch.Tensor:
    """fresh, cached_or_shifted: (1, kv_h, L, d_h). Return (kv_h, d_h)."""
    diff = (fresh - cached_or_shifted).squeeze(0)  # (kv_h, L, d_h)
    return diff.mean(dim=1)                         # (kv_h, d_h)


def per_head_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """a, b: (kv_h, d_h). Return mean cos across heads."""
    cs = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return cs.mean().item()


def vec_norm(a: torch.Tensor) -> float:
    return a.flatten().norm().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--cache_dir_root", type=Path, required=True,
                    help="Dir containing the per-case cache subdirs.")
    ap.add_argument("--cache_dir_prefix", type=str, default="_diag_case",
                    help="Subdir prefix; default _diag_case (script 13 layout). "
                         "Use _sink_case for script 12 caches.")
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
        cache_dir = args.cache_dir_root / f"{args.cache_dir_prefix}{ci:02d}"
        if not cache_dir.exists():
            print(f"[skip] no cache at {cache_dir}")
            continue
        meta = load_meta(cache_dir)
        if not meta.get("anchor_conditioned"):
            print(f"[warn] case {ci} cache is not anchor-conditioned")
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
            placements, flat = _diag13.build_placements(cache_dir, chunk_ids, chunk_lookup, args.M)
            captures = _diag13.diagnose_one(model, tok, placements, flat,
                                              q["question"], inv_freq)

            # captures[layer] is a list aligned with placements:
            # 0=sink, 1=gold, 2=sib0, 3=sib1.
            for li, plist in captures.items():
                if len(plist) < 4:
                    continue
                dK = {}; dV = {}; nK = {}; nV = {}
                for role, cap in zip(["sink", "gold", "sib0", "sib1"], plist):
                    dK[role] = mean_drift_per_head(cap["K_fresh"], cap["K_shifted"])
                    dV[role] = mean_drift_per_head(cap["V_fresh"], cap["V_cached"])
                    nK[role] = vec_norm(dK[role])
                    nV[role] = vec_norm(dV[role])

                row = {
                    "case": ci, "qid": q["id"], "template_id": q["template_id"],
                    "layer": li,
                    # magnitudes (sanity)
                    "nK_gold": nK["gold"], "nK_sib0": nK["sib0"], "nK_sib1": nK["sib1"],
                    "nV_gold": nV["gold"], "nV_sib0": nV["sib0"], "nV_sib1": nV["sib1"],
                    # K cross-role drift cos
                    "K_corr_sib0_sib1": per_head_cos(dK["sib0"], dK["sib1"]),
                    "K_corr_gold_sib0": per_head_cos(dK["gold"], dK["sib0"]),
                    "K_corr_gold_sib1": per_head_cos(dK["gold"], dK["sib1"]),
                    # V cross-role drift cos
                    "V_corr_sib0_sib1": per_head_cos(dV["sib0"], dV["sib1"]),
                    "V_corr_gold_sib0": per_head_cos(dV["gold"], dV["sib0"]),
                    "V_corr_gold_sib1": per_head_cos(dV["gold"], dV["sib1"]),
                }
                rows.append(row)
            print(f"  case {ci} q{q['id']} t{q['template_id']} captured "
                  f"{sum(len(v)>=4 for v in captures.values())} layers")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(rows, f)

    # Aggregate by layer.
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r)

    print("\n=== Drift-correlation (anchor v2, per-head cos averaged) ===")
    print(f"  layer  n   ||dK_gold|| ||dK_sib0|| ||dK_sib1||  "
          f"K(sib0,sib1)  K(gold,sib0)  K(gold,sib1)  "
          f"V(sib0,sib1)  V(gold,sib0)  V(gold,sib1)")
    for li in sorted(by_layer):
        rs = by_layer[li]
        n = len(rs)
        def avg(k): return sum(r[k] for r in rs) / n
        print(f"  L{li:>2}   {n:>3}  "
              f"{avg('nK_gold'):>10.4f} {avg('nK_sib0'):>10.4f} {avg('nK_sib1'):>10.4f}  "
              f"{avg('K_corr_sib0_sib1'):>12.4f}  "
              f"{avg('K_corr_gold_sib0'):>12.4f}  "
              f"{avg('K_corr_gold_sib1'):>12.4f}  "
              f"{avg('V_corr_sib0_sib1'):>12.4f}  "
              f"{avg('V_corr_gold_sib0'):>12.4f}  "
              f"{avg('V_corr_gold_sib1'):>12.4f}")


if __name__ == "__main__":
    main()
