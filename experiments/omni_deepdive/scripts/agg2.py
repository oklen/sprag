import json, glob, math
import pandas as pd

def mean(x): return sum(x)/len(x) if x else float('nan')
def sem(x):
    if len(x)<2: return float('nan')
    m=mean(x); v=sum((a-m)**2 for a in x)/(len(x)-1)
    return math.sqrt(v/len(x))

# meta: question_id -> task_type, duration
df=pd.read_parquet("/home/tiger/videomme_meta/videomme/test-00000-of-00001.parquet")
qt={str(r['question_id']):(str(r['task_type']),str(r['duration'])) for _,r in df.iterrows()}
TEMP={"Temporal Perception","Temporal Reasoning"}

recs=[]
for f in sorted(glob.glob("/home/tiger/data/omni_mrope_vid.json.s*")):
    try:
        d=json.load(open(f))
        if isinstance(d,list): recs+=d
    except Exception as e: print("ERR",f,e)
print("n records:",len(recs), "t_grid set:", sorted(set(r['t_grid'] for r in recs)))

covs=[10,20,30,100]
def report(subset, name):
    print("\n===== %s  (n=%d records) =====" % (name, len(subset)))
    print(" cov | n | fresh | ours | compact | ours-fresh | compact-ours (SEM) | %compact-worse | acc f/o/c")
    for c in covs:
        F=[];O=[];C=[];pen=[];fa=[];oa=[];ca=[]
        for r in subset:
            for row in r['rows']:
                if row['cov']!=c: continue
                if 'ours' not in row or 'ours_compact' not in row: continue
                F.append(row['fresh']['nll']);O.append(row['ours']['nll']);C.append(row['ours_compact']['nll'])
                pen.append(row['ours_compact']['nll']-row['ours']['nll'])
                fa.append(row['fresh']['acc']);oa.append(row['ours']['acc']);ca.append(row['ours_compact']['acc'])
        if not O: continue
        worse=100*sum(1 for x in pen if x>0)/len(pen)
        print("  %3d | %3d | %.3f | %.3f | %.3f | %+.4f | %+.4f (%.4f) | %.0f%% | %.2f/%.2f/%.2f" % (
            c,len(O),mean(F),mean(O),mean(C),mean(O)-mean(F),mean(pen),sem(pen),worse,mean(fa),mean(oa),mean(ca)))

report(recs,"ALL")
temp=[r for r in recs if qt.get(str(r.get('uid')),('',''))[0] in TEMP]
stat=[r for r in recs if qt.get(str(r.get('uid')),('',''))[0] not in TEMP and r.get('uid')]
report(temp,"TEMPORAL (Temporal Perception+Reasoning)")
report(stat,"NON-TEMPORAL (rest)")
lng=[r for r in recs if qt.get(str(r.get('uid')),('',''))[1]=='long']
med=[r for r in recs if qt.get(str(r.get('uid')),('',''))[1]=='medium']
shr=[r for r in recs if qt.get(str(r.get('uid')),('',''))[1]=='short']
report(lng,"LONG videos")
report(med,"MEDIUM videos")
report(shr,"SHORT videos")
