"""CPU-only collection/rendering. Does not alter generation or original scorers."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,math,statistics,types,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'r2/src')]
from r2.pilots.bench_loop_motif_mini_v1 import run as X
E=X.E;OUT=X.OUT
def savecsv(name,rows):
    p=OUT/'analysis'/name;p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def render(indices,stage):
    X.configure()
    # Independent title-only adapter; original rendering source remains untouched.
    f=E._render_page
    f.__code__=f.__code__.replace(co_consts=tuple('R / G0 / G1 / G3 / L0 / L3' if c=='R / G0 / G1 / G2 / G3' else c for c in f.__code__.co_consts))
    qs=X.read(OUT/'inputs/requests.json');base=X.read(OUT/'baseline.json')
    for i in indices:
        q=qs[i];rid=q['request_id'];group=[t for t in base+X.tasks() if t['request_id']==rid]
        label=lambda t: t['arm'] if 'stage_paths' in t else t['label']
        group.sort(key=lambda t:['R','G0','G1','G3','L0','L3'].index(label(t)))
        cells=[dict(display_label=label(t),cell_id=t['cell_id'],q=t['q'],path=t['stage_paths'][stage] if 'stage_paths' in t else str(E._spec(t,stage)['mesh'])) for t in group]
        for mode,selected in [('loop_development',cells),('loop_new',cells[-2:])]:
            E._render_page(dict(page_id=stage+'__'+rid,stage=stage,mode=mode,reference_q=q,cells=selected))
        print('RENDERED '+stage+' '+rid,flush=True)
def numeric():
    X.configure();X.collect()
    rows=list(csv.DictReader((OUT/'analysis/per_output_original177.csv').open(encoding='utf-8-sig')))
    fields=E._fields()
    def num(v):
        try:return float(v)
        except:return None
    for r in rows:
        if r['method']=='G3_L0p3':r['method']='G3'
        for f in fields:r[f]=num(r[f]) if r[f] not in ('True','False') else int(r[f]=='True')
        r['I_status']='LOW_SUPPORT' if r['parent_id'].startswith('28b9') else 'PASS'
    savecsv('per_output_normalized177.csv',rows)
    parents=sorted({r['parent_id'] for r in rows});means=[]
    for stage in E.STAGES:
        for method in ('R','G0','G1','G3','L0','L3'):
            for f in fields:
                vals=[];n=0
                for parent in parents:
                    v=[r[f] for r in rows if r['stage']==stage and r['method']==method and r['parent_id']==parent and r[f] is not None and math.isfinite(r[f]) and (not f.startswith('ConflictCov_C') or r['I_status']=='PASS')]
                    if v:vals.append(statistics.mean(v));n+=len(v)
                means.append(dict(stage=stage,method=method,metric=f,parent_macro=statistics.mean(vals) if vals else None,parents=len(vals),outputs=n,status='OK' if vals else 'NA_OR_NONNUMERIC'))
    savecsv('metric_parent_macro_all177.csv',means)
    lookup={(r['method'],r['stage'],r['request_id']):r for r in rows};paired=[]
    for a,b in [('L0','G0'),('L3','G3'),('L3','L0'),('L3','G1'),('L3','R')]:
        for q in X.read(OUT/'inputs/requests.json'):
            for stage in E.STAGES:
                ar=lookup[a,stage,q['request_id']];br=lookup[b,stage,q['request_id']]
                r=dict(comparison=a+'-'+b,stage=stage,request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],I_status=ar['I_status'])
                for f in fields:r[f]=ar[f]-br[f] if ar[f] is not None and br[f] is not None else None
                paired.append(r)
    savecsv('paired_deltas_all177.csv',paired)
    interactions=[]
    for q in X.read(OUT/'inputs/requests.json'):
        for stage in E.STAGES:
            r=dict(request_id=q['request_id'],parent_id=q['parent_id'],stage=stage)
            rr=[lookup[m,stage,q['request_id']] for m in ('L3','L0','G3','G0')]
            for f in fields:r[f]=rr[0][f]-rr[1][f]-rr[2][f]+rr[3][f] if all(z[f] is not None for z in rr) else None
            interactions.append(r)
    savecsv('descriptive_interaction_all177.csv',interactions)
    print('Collected 144 rows, all177 paired and parent macro',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['render','numeric']);p.add_argument('--indices',type=int,nargs='+');p.add_argument('--stage',default='RAW');a=p.parse_args()
    if a.action=='render':render(a.indices,a.stage)
    else:numeric()
