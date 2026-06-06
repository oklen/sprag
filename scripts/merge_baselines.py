#!/usr/bin/env python3
"""Merge baseline-comparison shards, dedup by uid, report per-cov per-arm meanNLL,
SEM, paired Wilcoxon vs 'ours', and accuracy. Usage: merge_baselines.py <prefix>"""
import sys, json, glob, numpy as np
from scipy.stats import wilcoxon

prefix = sys.argv[1]
files = sorted(glob.glob(prefix + ".s*")) or [prefix]
seen = {}
for f in files:
    try:
        for r in json.load(open(f)):
            seen[r["uid"]] = r
    except Exception as e:
        print("skip", f, e)
data = list(seen.values())
ARMS = ["fresh", "ours", "ours_compact", "rekv_origpos", "rekv", "mukv"]
covs = sorted({rw["cov"] for r in data for rw in r["rows"]})
print(f"merged n={len(data)} from {len(files)} files; covs={covs}\n")

def col(cov, arm, key):
    return [rw[arm][key] for r in data for rw in r["rows"] if rw["cov"] == cov and arm in rw]

for cov in covs:
    print(f"=== cov{cov} (n={len(col(cov,'ours','nll'))}) ===")
    base = col(cov, "ours", "nll")
    print(f"  {'arm':13s} {'meanNLL':>9s} {'SEM':>7s} {'dvs_ours':>9s} {'p_vs_ours':>10s} {'acc':>6s}")
    for a in ARMS:
        nll = col(cov, a, "nll"); acc = col(cov, a, "acc")
        if not nll: continue
        sem = np.std(nll) / np.sqrt(len(nll))
        d = np.mean(nll) - np.mean(base)
        p = ""
        if a != "ours" and len(nll) == len(base) and cov != 100:
            diff = np.array(nll) - np.array(base)
            if np.any(diff != 0):
                try: p = f"{wilcoxon(nll, base).pvalue:.2e}"
                except Exception: p = "na"
            else: p = "0"
        print(f"  {a:13s} {np.mean(nll):9.4f} {sem:7.4f} {d:+9.4f} {p:>10s} {np.mean(acc):6.3f}")
    print()
