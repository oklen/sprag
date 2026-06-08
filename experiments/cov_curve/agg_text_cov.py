import json, glob, math
from statistics import median

def mean(x): return sum(x)/len(x) if x else float('nan')
def sem(x):
    if len(x)<2: return float('nan')
    m=mean(x); v=sum((a-m)**2 for a in x)/(len(x)-1)
    return math.sqrt(v/len(x))

PAT="/home/tiger/sprag-main/data/a3b_cov_fix.s*.json"
files=sorted(glob.glob(PAT))
covs=["c0","c25","c50","c75","c100"]
# collect per-cov lists across all datasets/cases
data={c:{"nll_c":[],"nll_f":[],"ppl_c":[],"ppl_f":[],"acc_c":[],"acc_f":[],"worse":0,"n":0} for c in covs}
ndoc=0
for f in files:
    try: d=json.load(open(f))
    except Exception as e: print("ERR",f,e); continue
    for ds,cases in d.get("datasets",{}).items():
        for case in cases:
            ndoc+=1
            for c in covs:
                if c not in case: continue
                e=case[c]
                pc=e.get("ppl_cached"); pf=e.get("ppl_fresh")
                if pc is None or pf is None: continue
                # clamp to avoid log(0)
                pc=max(pc,1e-9); pf=max(pf,1e-9)
                data[c]["ppl_c"].append(pc); data[c]["ppl_f"].append(pf)
                data[c]["nll_c"].append(math.log(pc)); data[c]["nll_f"].append(math.log(pf))
                data[c]["acc_c"].append(e.get("acc_cached",0)); data[c]["acc_f"].append(e.get("acc_fresh",0))
                if math.log(pc)>math.log(pf): data[c]["worse"]+=1
                data[c]["n"]+=1

print("files:",len(files),"docs:",ndoc)
print("\n cov |  n  | NLL_cached | NLL_fresh | gap(c-f) | medPPL_c | medPPL_f | %cache-worse | acc c/f")
for c in covs:
    D=data[c]
    if D["n"]==0: continue
    gap=mean(D["nll_c"])-mean(D["nll_f"])
    print("%4s | %3d | %8.3f | %8.3f | %+7.3f | %9.1f | %9.1f | %4.0f%% | %.2f/%.2f" % (
        c, D["n"], mean(D["nll_c"]), mean(D["nll_f"]), gap,
        median(D["ppl_c"]), median(D["ppl_f"]),
        100*D["worse"]/D["n"], mean(D["acc_c"]), mean(D["acc_f"])))
