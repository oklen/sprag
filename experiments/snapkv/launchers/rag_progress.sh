#!/bin/bash
PY=/mlx_devbox/users/caizefeng/miniconda3/envs/clamp3/bin/python
cd /home/tiger/sprag-main
$PY - <<'PYEOF'
import json,glob
rows=[]; ratios=None
for f in sorted(glob.glob("data/ragsnap_cov.s*.json")):
    try: d=json.load(open(f))
    except: continue
    ratios=ratios or d.get("ratios")
    for ds,rs in d.get("datasets",{}).items(): rows+=rs
n=len(rows); print("RAGSNAP records:",n)
if n and ratios:
    af=sum(r["acc_fresh"] for r in rows)/n
    print(f"acc_fresh(full ctx)={af:.3f}")
    print(f"{'ratio':>6} {'keepf':>7} {'acc_B':>6} {'dB':>7} {'acc_A':>6} {'dA':>7} {'gapA-B':>7}")
    for r in ratios:
        k=f"r{r}"; c=[x[k] for x in rows if k in x]
        if not c: continue
        b=sum(z['acc_B'] for z in c)/len(c); a=sum(z['acc_A'] for z in c)/len(c)
        kf=sum(z['keep_frac'] for z in c)/len(c)
        print(f"{r:6.2f} {kf:7.3f} {b:6.3f} {b-af:+7.3f} {a:6.3f} {a-af:+7.3f} {a-b:+7.3f}")
PYEOF
