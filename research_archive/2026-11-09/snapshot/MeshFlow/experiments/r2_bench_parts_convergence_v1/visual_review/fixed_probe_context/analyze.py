"""Read-only arithmetic on the existing 0/1000/2000 fixed-loss records."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import statistics

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
STEPS=(0,1000,2000)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
probe_path=ROOT/'fixed_loss_probe/manifest.json'
manifest=read(probe_path)
meta={r['probe_id']:r for r in manifest['rows']}
sources={str(probe_path):sha(probe_path)}
data={}
flat=[]
for step in STEPS:
    p=ROOT/'training/COMMON/fixed_loss'/f'step{step}.json'
    sources[str(p)]=sha(p)
    d=read(p);data[step]=d
    assert d['probe_manifest_sha256']==sha(probe_path)
    assert len(d['rows'])==96
    assert {r['probe_id'] for r in d['rows']}==set(meta)
    for r in d['rows']:
        m=meta[r['probe_id']]
        assert r['all_free_scalar_count']==9*(m['N']-m['K'])
        assert r['bulk_scalar_count']==r['all_free_scalar_count']
        assert r['bulk_velocity_sse']==r['all_free_velocity_sse']
        assert r['seam_tip_scalar_count']==0 and r['seam_tip_velocity_sse']==0
        flat.append(dict(step=step,**r,N=m['N'],K=m['K'],free_faces=m['N']-m['K']))
    for split in ('TRAIN','DEV'):
        rs=[r for r in d['rows'] if r['split']==split]
        sse=sum(r['all_free_velocity_sse'] for r in rs)
        count=sum(r['all_free_scalar_count'] for r in rs)
        assert math.isclose(sse,d[split.lower()]['ALL_free']['sse'],rel_tol=1e-12)
        assert count==d[split.lower()]['ALL_free']['count']

def agg(keys):
    groups=collections.defaultdict(list)
    for r in flat:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for kk,rs in sorted(groups.items()):
        sse=sum(r['all_free_velocity_sse'] for r in rs)
        count=sum(r['all_free_scalar_count'] for r in rs)
        out.append(dict(zip(keys,kk),sse=sse,scalar_count=count,mse=sse/count,
            n_conditions=len({r['condition_id'] for r in rs}),n_parents=len({r['parent_id'] for r in rs}),
            n_probe_rows=len(rs),min_N=min(r['N'] for r in rs),max_N=max(r['N'] for r in rs),
            min_free_faces=min(r['free_faces'] for r in rs),max_free_faces=max(r['free_faces'] for r in rs)))
    return out

condition=agg(['step','split','parent_id','condition_id','part','N','K','free_faces'])
parent=agg(['step','split','parent_id'])
part=agg(['step','split','part'])
time=agg(['step','split','t'])
overall=agg(['step','split'])
index={(r['step'],r['condition_id']):r for r in condition}
split_totals={(r['step'],r['split']):r for r in overall}
delta=[]
for end,start in [(1000,0),(2000,0),(2000,1000)]:
    for b in [r for r in condition if r['step']==end]:
        a=index[start,b['condition_id']]
        assert a['scalar_count']==b['scalar_count']
        total=split_totals[end,b['split']]
        delta.append(dict(from_step=start,to_step=end,split=b['split'],parent_id=b['parent_id'],condition_id=b['condition_id'],
            part=b['part'],N=b['N'],K=b['K'],free_faces=b['free_faces'],scalar_count=b['scalar_count'],
            start_mse=a['mse'],end_mse=b['mse'],delta_mse=b['mse']-a['mse'],delta_sse=b['sse']-a['sse'],
            contribution_to_split_mse_delta=(b['sse']-a['sse'])/total['scalar_count'],
            fraction_of_split_scalar_count=b['scalar_count']/total['scalar_count'],fraction_of_end_split_sse=b['sse']/total['sse']))

def correlation(x,y):
    mx=statistics.mean(x);my=statistics.mean(y)
    xx=sum((a-mx)**2 for a in x);yy=sum((b-my)**2 for b in y)
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(xx*yy) if xx and yy else None

descriptive=[]
for o in overall:
    cc=[r for r in condition if r['step']==o['step'] and r['split']==o['split']]
    pp=[r for r in parent if r['step']==o['step'] and r['split']==o['split']]
    ranked=sorted(cc,key=lambda r:r['sse'],reverse=True)
    entry=dict(**o,condition_equal_mean_mse=statistics.mean(r['mse'] for r in cc),parent_equal_mean_mse=statistics.mean(r['mse'] for r in pp),
        median_condition_mse=statistics.median(r['mse'] for r in cc),min_condition_mse=min(r['mse'] for r in cc),max_condition_mse=max(r['mse'] for r in cc),
        condition_N_mse_Pearson=correlation([r['N'] for r in cc],[r['mse'] for r in cc]),
        condition_free_faces_mse_Pearson=correlation([r['free_faces'] for r in cc],[r['mse'] for r in cc]),
        max_condition_scalar_weight=max(r['scalar_count']/o['scalar_count'] for r in cc),
        top1_sse_share=ranked[0]['sse']/o['sse'],top3_sse_share=sum(r['sse'] for r in ranked[:3])/o['sse'])
    descriptive.append(entry)

concentration=[]
for split in ('TRAIN','DEV'):
    dd=[r for r in delta if r['from_step']==1000 and r['to_step']==2000 and r['split']==split]
    positive=[r for r in dd if r['delta_sse']>0]
    negative=[r for r in dd if r['delta_sse']<0]
    pos=sorted(positive,key=lambda r:r['delta_sse'],reverse=True)
    net=sum(r['delta_sse'] for r in dd)
    concentration.append(dict(split=split,conditions_worse=len(positive),conditions_better=len(negative),net_delta_sse=net,
        gross_positive_delta_sse=sum(r['delta_sse'] for r in positive),gross_negative_delta_sse=sum(r['delta_sse'] for r in negative),
        top_positive_conditions=[dict(condition_id=r['condition_id'],delta_sse=r['delta_sse'],delta_mse=r['delta_mse'],
            split_mse_contribution=r['contribution_to_split_mse_delta'],share_of_positive_delta_sse=r['delta_sse']/sum(q['delta_sse'] for q in positive),
            scalar_weight=r['fraction_of_split_scalar_count'],N=r['N'],K=r['K']) for r in pos[:5]]))

def writecsv(name,rows):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name,rows in [('per_condition.csv',condition),('per_parent.csv',parent),('per_part.csv',part),('per_time.csv',time),('condition_deltas.csv',delta),('overall_and_macro_diagnostics.csv',descriptive)]:writecsv(name,rows)
result=dict(status='PASS_EXISTING_RECORD_ARITHMETIC',steps=list(STEPS),scope='Process interpretation only; no sampling, model calls, rescoring, selection or training-policy changes',
    primary_aggregation='sum(all_free_velocity_sse) / sum(all_free_scalar_count)',
    denominator='Native COMMON has 9*(N-K) scalar coordinates per t; three fixed t values per condition; identical counts at all three checkpoints',
    TRAIN_scalar_count=split_totals[2000,'TRAIN']['scalar_count'],DEV_scalar_count=split_totals[2000,'DEV']['scalar_count'],
    macro_diagnostics='Condition-equal and parent-equal means are descriptive sensitivity checks only, not replacements for the registered SSE/count metric',
    overall=descriptive,change_1000_2000=concentration,model_calls=0,new_samples=0,inputs=sources)
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
receipt=dict(status='COMPLETE',source_hashes=sources,code_sha256=sha(__file__),files={str(p):sha(p) for p in OUT.iterdir() if p.suffix in ('.csv','.json')})
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(overall=descriptive,changes=concentration,parts=part,parents=parent,time=time),indent=2))
