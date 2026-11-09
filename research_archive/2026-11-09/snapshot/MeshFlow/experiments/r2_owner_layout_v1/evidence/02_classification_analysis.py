"""Summarize saved head predictions, including old cached seven-C predictions."""
import csv, json, collections
from pathlib import Path
import numpy as np
from . import head as H


def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    gate=H.read(H.OUT/'gate.json');dest=H.OUT/'classification_analysis';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():raise SystemExit('Already summarized; no overwrite')
    rr=rows(H.OUT/'evaluation_head/per_edge.csv')
    truths={(r['condition_id'],int(r['edge_id'])):r['role'] for r in rr if r['arm']=='CONTEXT'}
    cached=rows(H.OLD/'evaluation/occupancy_edges.csv')
    cases={r['condition_id'] for r in cached}
    combined=[]
    for r in rr:
        if r['condition_id'] in cases:combined.append(dict(r))
    for r in cached:
        if r['label'] not in ('LONG','B_POOL47'):continue
        role=truths[(r['condition_id'],int(r['edge_id']))]
        z=float(r['logit'])
        combined.append(dict(arm=r['label']+'_CACHED',split=r['split'],condition_id=r['condition_id'],
           parent_id=r['condition_id'].split('__')[0],edge_id=r['edge_id'],role=role,logit=z,
           probability=float(1/(1+np.exp(-np.clip(z,-60,60)))),predicted_owner=r['predicted']))
    H.csvwrite(dest/'seven_C_per_edge.csv',combined)
    summaries=[];parents=[]
    rolemap={'NULL':0,'OWNER':1,'ALIAS':2}
    for arm in sorted({r['arm'] for r in combined}):
        for split in ('TRAIN','DEV'):
            r=[r for r in combined if r['arm']==arm and r['split']==split]
            m=H.metrics(np.asarray([float(x['logit']) for x in r]),np.asarray([rolemap[x['role']] for x in r]))
            summaries.append(dict(arm=arm,split=split,**m))
            for parent in sorted({x['parent_id'] for x in r}):
                p=[x for x in r if x['parent_id']==parent]
                m=H.metrics(np.asarray([float(x['logit']) for x in p]),np.asarray([rolemap[x['role']] for x in p]))
                parents.append(dict(arm=arm,split=split,parent_id=parent,**m))
    H.csvwrite(dest/'seven_C_summary.csv',summaries);H.csvwrite(dest/'seven_C_parents.csv',parents)
    failures=[r for r in rr if (float(r['logit'])>=0)!=(r['role']=='OWNER')]
    if failures:H.csvwrite(dest/'all_failed_edges.csv',failures)
    bins=[]
    for arm in ('MLP','CONTEXT'):
        for split in ('TRAIN','DEV'):
            r=[x for x in rr if x['arm']==arm and x['split']==split]
            pp=np.array([float(x['probability']) for x in r]);yy=np.array([x['role']=='OWNER' for x in r])
            bb=np.minimum((pp*10).astype(int),9)
            for i in range(10):
                v=bb==i
                bins.append(dict(arm=arm,split=split,bin=i,lower=i/10,upper=(i+1)/10,count=int(v.sum()),
                    mean_probability=float(pp[v].mean()) if v.any() else None,
                    empirical_owner_fraction=float(yy[v].mean()) if v.any() else None))
    H.csvwrite(dest/'reliability_bins.csv',bins)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.5,4))
    for arm in ('MLP','CONTEXT'):
        r=rows(H.OUT/'heads'/arm/'learning_curve.csv')
        ax.semilogy([int(x['step']) for x in r],[float(x['loss']) for x in r],label=arm)
    ax.set(xlabel='Fixed CPU optimizer updates',ylabel='TRAIN parent-mean unweighted BCE')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(dest/'learning_curve.png',dpi=150);plt.close(fig)
    H.write(dest/'complete.json',dict(status='COMPLETE',gate=gate['status'],new_model_calls=0,
        failed_edges=len(failures),cached_old_head_forwards=0,
        files={str(p):H.sha(p) for p in dest.iterdir() if p.is_file()}))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
