import json, glob, math

def load_all(pat):
    recs=[]
    for f in sorted(glob.glob(pat)):
        try:
            d=json.load(open(f))
            if isinstance(d,list): recs+=d
        except Exception as e:
            print("ERR",f,e)
    return recs

def mean(x): return sum(x)/len(x) if x else float('nan')
def sem(x):
    if len(x)<2: return float('nan')
    m=mean(x); v=sum((a-m)**2 for a in x)/(len(x)-1)
    return math.sqrt(v/len(x))

print("="*60)
print("#2 LAYER-WISE CROSS-MODAL (omni_lw)")
print("="*60)
r2=load_all("/home/tiger/data/omni_lw.json.s*")
print("n =",len(r2))
depths=[0,4,8,12,16,20,24,28,32,36,40,44,48]
# mean delta(d) = curve[d]-curve[0]  (negative = cached-side advantage emerging)
df=[r['delta_full'] for r in r2]
print("mean delta_full (nll_cached - nll_fresh) = %.4f  SEM %.4f" % (mean(df), sem(df)))
print("  frac records delta<0 (cached better): %.1f%%" % (100*sum(1 for x in df if x<0)/len(df)))
print()
print("depth profile: mean over records of [curve[d] - curve[0]]  (more negative = more advantage realized by layer d)")
tot=mean(df)
for d in depths:
    dd=[r['curve'][str(d)]-r['curve']['0'] for r in r2 if str(d) in r['curve']]
    frac = (mean(dd)/tot*100) if abs(tot)>1e-9 else float('nan')
    bar = int(abs(mean(dd))/ (abs(tot)+1e-9) *40) if not math.isnan(frac) else 0
    print("  d=%2d  Δ=%+.4f  (%5.1f%% of full gap)  %s" % (d, mean(dd), frac, '#'*bar))

print()
print("="*60)
print("#3 M-RoPE  ours(orig-pos) vs ours_compact vs fresh (omni_mrope_ego)")
print("="*60)
r3=load_all("/home/tiger/data/omni_mrope_ego.json.s*")
print("n videos =",len(r3))
covs=[20,40,60,80,100]
by={c:{'fresh':[], 'ours':[], 'oc':[], 'f_acc':[], 'o_acc':[], 'oc_acc':[]} for c in covs}
pen=[]  # ours_compact.nll - ours.nll pooled
for r in r3:
    for row in r['rows']:
        c=row['cov']
        if c not in by: continue
        by[c]['fresh'].append(row['fresh']['nll'])
        by[c]['ours'].append(row['ours']['nll'])
        by[c]['oc'].append(row['ours_compact']['nll'])
        by[c]['f_acc'].append(row['fresh']['acc'])
        by[c]['o_acc'].append(row['ours']['acc'])
        by[c]['oc_acc'].append(row['ours_compact']['acc'])
        pen.append(row['ours_compact']['nll']-row['ours']['nll'])
print("pooled compaction penalty  mean(ours_compact.nll - ours.nll) = %.4f  SEM %.4f  (positive = compaction worse)" % (mean(pen), sem(pen)))
print("  frac rows where compact worse than ours: %.1f%%" % (100*sum(1 for x in pen if x>0)/len(pen)))
print()
print(" cov |  fresh nll | ours nll | compact nll | ours-fresh | compact-ours | acc f/o/c")
for c in covs:
    b=by[c]
    print("  %3d | %.4f | %.4f | %.4f | %+.4f | %+.4f | %.2f/%.2f/%.2f" % (
        c, mean(b['fresh']), mean(b['ours']), mean(b['oc']),
        mean(b['ours'])-mean(b['fresh']), mean(b['oc'])-mean(b['ours']),
        mean(b['f_acc']), mean(b['o_acc']), mean(b['oc_acc'])))
