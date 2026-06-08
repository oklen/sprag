import json, glob, math
from statistics import median
def mean(x): return sum(x)/len(x) if x else float('nan')
def sem(x):
    if len(x)<2: return float('nan')
    m=mean(x); v=sum((a-m)**2 for a in x)/(len(x)-1); return math.sqrt(v/len(x))
files=sorted(glob.glob("/home/tiger/sprag-main/data/a3b_cov_3arm.s*.json"))
covs=["c0","c25","c50","c75","c100"]
D={c:{"f":[],"o":[],"c":[],"af":[],"ao":[],"ac":[],"do":[],"dc":[],"n":0} for c in covs}
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
                if e.get("contam"): continue
                pf=e.get("ppl_fresh"); po=e.get("ppl_origpos"); pc=e.get("ppl_cached")
                if None in (pf,po,pc): continue
                lf,lo,lc=math.log(max(pf,1e-9)),math.log(max(po,1e-9)),math.log(max(pc,1e-9))
                D[c]["f"].append(lf); D[c]["o"].append(lo); D[c]["c"].append(lc)
                D[c]["do"].append(lo-lf); D[c]["dc"].append(lc-lf)
                D[c]["af"].append(e.get("acc_fresh",0)); D[c]["ao"].append(e.get("acc_origpos",0)); D[c]["ac"].append(e.get("acc_cached",0))
                D[c]["n"]+=1
print("files:",len(files),"docs:",ndoc)
print("\n cov |  n  | NLL_f  NLL_o  NLL_c | origpos-fresh (SEM) | compact-fresh (SEM) | acc f/o/c")
for c in covs:
    d=D[c]
    if d["n"]==0: continue
    print("%4s | %3d | %6.3f %6.3f %6.3f | %+7.4f (%.4f) | %+7.4f (%.4f) | %.2f/%.2f/%.2f" % (
        c,d["n"],mean(d["f"]),mean(d["o"]),mean(d["c"]),
        mean(d["do"]),sem(d["do"]),mean(d["dc"]),sem(d["dc"]),
        mean(d["af"]),mean(d["ao"]),mean(d["ac"])))
print("\nCLIFF CHECK (c0): origpos cliff = %+.4f, compact cliff = %+.4f -> origpos %s" % (
    mean(D["c0"]["do"]),mean(D["c0"]["dc"]),
    "SHALLOWER (escapes via geometry)" if mean(D["c0"]["do"])<mean(D["c0"]["dc"])-0.02 else
    "SAME/also-degenerates (cliff = tiny keep-set, not convention)") if D["c0"]["n"] else "no c0 data")
