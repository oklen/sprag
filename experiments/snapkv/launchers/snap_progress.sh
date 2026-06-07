#!/bin/bash
# pooled progress across all snapkv_cov.s*.json on CephFS
PY=/mlx_devbox/users/caizefeng/miniconda3/envs/clamp3/bin/python
cd /home/tiger/sprag-main
$PY - <<'PYEOF'
import json,glob
rows=[]; ratios=None
for f in sorted(glob.glob("data/snapkv_cov.s*.json")):
    try: d=json.load(open(f))
    except: continue
    ratios=ratios or d.get("ratios")
    for ds,rs in d.get("datasets",{}).items(): rows+=rs
n=len(rows)
print("SNAPKV records:",n)
if n and ratios:
    af=sum(r["acc_fresh"] for r in rows)/n
    print(f"acc_fresh={af:.3f}")
    print(f"{'ratio':>6} {'keepf':>7} {'acc':>6} {'Δfresh':>7}")
    for r in ratios:
        k=f"r{r}"; c=[x[k] for x in rows if k in x]
        if not c: continue
        acc=sum(z['acc'] for z in c)/len(c); kf=sum(z['keep_frac'] for z in c)/len(c)
        print(f"{r:6.2f} {kf:7.3f} {acc:6.3f} {acc-af:+7.3f}")
PYEOF
