"""Read-only scalar-table schema correction; no geometry/scoring/model calls.

The first aggregator reused boundary_edges for input metadata. Original scores
are untouched. Restore the original mesh metric and name C metadata explicitly.
"""
from __future__ import annotations
import collections,csv,json
from pathlib import Path
import numpy as np
from . import hinge as H
C=H.C


def readcsv(p):
    with C.resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def run():
    root=H.OUT;out=root/'tables_v2';reg=C.read(root/'registration.json')
    old=C.read(root/'tables/completion.json');H.E.verify(old['files'])
    policy=C.read(C.ROOT/'r2/tools/hinge_opposite_v1/metric_policy.json')['directions']
    fields=sorted(policy);assert len(fields)==135
    qmap={(j['label'],j['q']['request_id']):j['q'] for j in reg['jobs']}
    rows=readcsv(root/'tables/per_output.csv');source={str(root/'tables/completion.json'):C.sha(root/'tables/completion.json')}
    for r in rows:
        q=qmap[r['label'],r['request_id']]
        r['C_boundary_edges']=int(q['boundary_edges'])
        sp=(H.old_score_dir(r['label'],r['request_id']) if r['stage']=='BASE_POST' else H.score_dir(r['label'],r['request_id']))/'score.json'
        s=C.read(sp);r['boundary_edges']=s['topology']['boundary_edges'];source[str(sp)]=C.sha(sp)
        for k in fields:
            x=r.get(k)
            if x in ('',None):r[k]=None
            elif x in ('True','False'):r[k]=int(x=='True')
            elif not isinstance(x,(int,float)):r[k]=float(x)
        r['closed_C']=q['closed_C']
        for k in ('N','K','noise_index'):r[k]=int(q[k])
    summary=[]
    for label in H.LABELS:
        for stage in ('BASE_POST','HINGE_OPP015'):
            rr=[r for r in rows if r['label']==label and r['stage']==stage]
            for dim,groups in (('split',('ALL','TRAIN','DEV')),('part_type',sorted({r['part_type'] for r in rr}))):
                for group in groups:
                    xx=[r for r in rr if group=='ALL' or r[dim]==group]
                    if not xx:continue
                    for field in fields:
                        byc=collections.defaultdict(list)
                        for r in xx:
                            if H.number(r.get(field)):byc[r['parent_id'],r['condition_id']].append(float(r[field]))
                        byp=collections.defaultdict(list)
                        for (pid,cid),vv in byc.items():byp[pid].append(float(np.mean(vv)))
                        vals=[r[field] for r in xx if H.number(r.get(field))]
                        summary.append(dict(label=label,stage=stage,grouping=dim,group=group,metric=field,
                            mean_parent_equal=float(np.mean([np.mean(v) for v in byp.values()])) if byp else None,
                            mean_output=float(np.mean(vals)) if vals else None,n_requests=len(xx),n_valid=len(vals),
                            n_valid_parents=len(byp),closed_C_requests=sum(r['closed_C'] for r in xx)))
    lookup={(r['label'],r['request_id'],r['stage']):r for r in rows};paired=[];cross=[]
    for j in reg['jobs']:
        label,q=j['label'],j['q'];a,b=[lookup[label,q['request_id'],s] for s in ('BASE_POST','HINGE_OPP015')]
        for field in fields:
            x,y=a[field],b[field];paired.append(dict(label=label,request_id=q['request_id'],parent_id=q['parent_id'],condition_id=q['condition_id'],split=q['split'],part_type=q['part_type'],metric=field,baseline=x,candidate=y,delta=y-x if H.number(x) and H.number(y) else None,status='PASS' if H.number(x) and H.number(y) else 'NA'))
    for rid in reg['common_request_ids']:
        for stage in ('BASE_POST','HINGE_OPP015'):
            a,b=[lookup[label,rid,stage] for label in H.LABELS]
            for field in fields:
                x,y=a[field],b[field];cross.append(dict(request_id=rid,condition_id=a['condition_id'],parent_id=a['parent_id'],split=a['split'],stage=stage,metric=field,step15000=x,step21000=y,delta=y-x if H.number(x) and H.number(y) else None))
    files={}
    for name,value in (('per_output',rows),('summary_parent_equal',summary),('paired_per_output',paired),('common_request_checkpoint_pairing',cross)):
        p=out/(name+'.csv');C.csvwrite(p,value);files[str(p)]=C.sha(p)
    result=dict(status='COMPLETE',schema='original135_with_input_C_metadata_distinct_v2',rows=len(rows),numeric_fields=fields,numeric_field_count=len(fields),files=files,
        source_score_hashes=source,correction='Only restore full-mesh boundary_edges from unchanged score.topology; C input count becomes C_boundary_edges. Previous tables retained, not authoritative for that field.',
        engineering_attempts=[dict(status='BEFORE_OUTPUT_FIELD_LOOKUP_ERROR',field='score.metrics.boundary_edges',correction='Read existing score.topology.boundary_edges',scoring_calls=0,decoder_calls=0,model_calls=0)],
        scoring_calls=0,new_pair_scans=0,decoder_calls=0,model_calls=0,new_queries=0,code_sha256=C.sha(__file__))
    C.write(out/'completion.json',result)
    note='Before interpretation, the metadata field name collision was corrected in tables_v2. The original numerical scorer, meshes, frozen definitions, pair caches and all previous tables remain unchanged. Use tables_v2 for complete 135-field tables; tables/connectivity_ledger.csv and illegal_pairs_by_role.csv are unaffected. No scoring, decoder, model or noise calls were repeated.\n'
    p=root/'SCHEMA_NOTE.md'
    if not p.exists():p.write_text(note,encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',rows=len(rows),numeric_fields=len(fields),scoring_calls=0)));return result


if __name__=='__main__':run()
