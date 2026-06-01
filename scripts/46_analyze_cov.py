import json, sys
import numpy as np
from scipy import stats
data = json.load(open(sys.argv[1]))
covs = sorted({rw["cov"] for r in data for rw in r["rows"]})
print(f"n_samples={len(data)}")
print(f"{'cov':>4} {'dNLL_mean':>10} {'SEM':>7} {'wilcox_p':>9} {'%cached<fresh':>13} {'dacc':>7} {'posmatch%':>9}")
for cov in covs:
    dn, da, pm = [], [], []
    for r in data:
        for rw in r["rows"]:
            if rw["cov"]==cov:
                dn.append(rw["gold_nll_cached"]-rw["gold_nll_fresh"])
                da.append(rw["acc_cached"]-rw["acc_fresh"])
                pm.append(rw.get("pos_matched", False))
    dn=np.array(dn); da=np.array(da)
    # negative dNLL => cached better. sign rate of cached strictly better:
    better = float(np.mean(dn<0))
    nz = dn[dn!=0]
    p = stats.wilcoxon(nz).pvalue if len(nz)>5 else float("nan")
    print(f"{cov:>4} {dn.mean():>+10.4f} {dn.std(ddof=1)/np.sqrt(len(dn)):>7.4f} {p:>9.4f} {100*better:>12.1f}% {da.mean():>+7.3f} {100*np.mean(pm):>8.1f}%")
