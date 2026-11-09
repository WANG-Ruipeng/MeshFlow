"""Read-only numerical aggregation of the frozen 124-field Bench results.

No rescoring, query draws or model calls. Every group includes all registered
requests belonging to its predetermined parent/split set. Signs are numerical
SEAM minus PLAIN, not an automatic declaration of improvement.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
from . import evaluate as V

C=V.C


def numeric(value):
    if value in ('', 'None', None):
        return None
    if value in ('True', 'False'):
        return float(value=='True')
    result=float(value)
    if not np.isfinite(result):
        raise ValueError('Nonfinite value in completed result CSV')
    return result


def run(root):
    root=C.resolve(root)
    reg=V.prepare(root); complete=C.read(root/'evaluation/summary_completion.json')
    V.E.verify(complete['files'])
    fields=complete['numeric_fields']
    if len(fields)!=124:
        raise ValueError('Expected complete original124 metrics')
    with (root/'evaluation/per_output.csv').open(encoding='utf-8-sig',newline='') as handle:
        outputs=list(csv.DictReader(handle))
    requests={q['request_id']:q for q in reg['requests']}
    lookup={(r['label'],r['stage'],r['request_id']):r for r in outputs}
    if len(outputs)!=96 or len(lookup)!=96:
        raise ValueError('Missing/duplicate registered output row')
    for q in reg['requests']:
        for label in V.ARMS:
            for stage in ('RAW','POST'):
                r=lookup[label,stage,q['request_id']]
                for key in ('condition_id','parent_id','split'):
                    if r[key]!=q[key]:raise ValueError('Output identity mismatch')
    parents=collections.defaultdict(list)
    for q in reg['requests']:parents[q['parent_id']].append(q)
    for qq in parents.values():
        if len(qq)!=2 or {q['noise_index'] for q in qq}!={0,1}:
            raise ValueError('Every parent requires two fixed noises')
        if len({(q['split'],q['official_split'],q['official_pretraining_exposure']) for q in qq})!=1:
            raise ValueError('Conflicting parent exposure metadata')
    group_parents={
        'TRAIN':[p for p,qq in parents.items() if qq[0]['split']=='TRAIN'],
        'DEV':[p for p,qq in parents.items() if qq[0]['split']=='DEV'],
        'DEV_OFFICIAL_TEST':[p for p,qq in parents.items() if qq[0]['split']=='DEV' and qq[0]['official_split']=='test'],
        'DEV_PRETRAIN_TRAIN':[p for p,qq in parents.items() if qq[0]['split']=='DEV' and qq[0]['official_split']=='train'],
        'ALL':list(parents),
    }
    per_parent=[]; endpoint_parent=[]
    for parent,qq in sorted(parents.items()):
        q=qq[0]
        for stage in ('RAW','POST'):
            for metric in fields:
                paired=[]; endpoints={}
                for label in V.ARMS:
                    vv=[numeric(lookup[label,stage,z['request_id']].get(metric)) for z in qq]
                    endpoints[label]=vv
                    endpoint_parent.append(dict(parent_id=parent,condition_id=q['condition_id'],split=q['split'],
                        official_split=q['official_split'],official_pretraining_exposure=q['official_pretraining_exposure'],
                        label=label,stage=stage,metric=metric,n_registered=2,n_valid=sum(v is not None for v in vv),
                        parent_mean=float(np.mean(vv)) if all(v is not None for v in vv) else None,
                        status='PASS' if all(v is not None for v in vv) else 'NA_NOT_ALL_NOISES_VALID'))
                for x,y in zip(endpoints['PLAIN'],endpoints['SEAM']):
                    if x is not None and y is not None:paired.append((x,y,y-x))
                valid=len(paired)==2
                per_parent.append(dict(parent_id=parent,condition_id=q['condition_id'],split=q['split'],
                    official_split=q['official_split'],official_pretraining_exposure=q['official_pretraining_exposure'],
                    stage=stage,metric=metric,n_registered=2,n_valid=len(paired),
                    baseline_mean=float(np.mean([x[0] for x in paired])) if valid else None,
                    candidate_mean=float(np.mean([x[1] for x in paired])) if valid else None,
                    paired_delta_mean=float(np.mean([x[2] for x in paired])) if valid else None,
                    status='PASS' if valid else 'NA_NOT_ALL_NOISES_COMMON'))
    paired_groups=[]; endpoint_groups=[]
    for group,pp in group_parents.items():
        for stage in ('RAW','POST'):
            for metric in fields:
                rr=[r for r in per_parent if r['parent_id'] in pp and r['stage']==stage and r['metric']==metric]
                good=[r for r in rr if r['status']=='PASS'];dd=[r['paired_delta_mean'] for r in good]
                paired_groups.append(dict(group=group,stage=stage,metric=metric,registered_parents=len(pp),
                    valid_parents=len(good),registered_requests=2*len(pp),valid_paired_requests=2*len(good),
                    baseline_parent_macro_mean=float(np.mean([r['baseline_mean'] for r in good])) if good else None,
                    candidate_parent_macro_mean=float(np.mean([r['candidate_mean'] for r in good])) if good else None,
                    paired_delta_parent_macro_mean=float(np.mean(dd)) if dd else None,
                    paired_delta_parent_median=float(np.median(dd)) if dd else None,
                    paired_delta_parent_min=float(min(dd)) if dd else None,
                    paired_delta_parent_max=float(max(dd)) if dd else None,
                    parent_delta_negative=sum(d<0 for d in dd),parent_delta_zero=sum(d==0 for d in dd),parent_delta_positive=sum(d>0 for d in dd),
                    status='PASS' if len(good)==len(pp) else 'NA_OR_PARTIAL'))
                for label in V.ARMS:
                    rr=[r for r in endpoint_parent if r['parent_id'] in pp and r['label']==label and r['stage']==stage and r['metric']==metric]
                    good=[r for r in rr if r['status']=='PASS'];vv=[r['parent_mean'] for r in good]
                    endpoint_groups.append(dict(group=group,label=label,stage=stage,metric=metric,
                        registered_parents=len(pp),valid_parents=len(good),registered_requests=2*len(pp),
                        parent_macro_mean=float(np.mean(vv)) if vv else None,parent_median=float(np.median(vv)) if vv else None,
                        status='PASS' if len(good)==len(pp) else 'NA_OR_PARTIAL'))
    dest=root/'evaluation/paired_complete';dest.mkdir(exist_ok=True);files={}
    for name,rows in (('paired_per_parent_stratified',per_parent),('endpoint_per_parent_stratified',endpoint_parent),
        ('paired_group_summary',paired_groups),('endpoint_group_summary',endpoint_groups)):
        p=dest/(name+'.csv');C.csvwrite(p,rows);files[str(p)]=C.sha(p)
    result=dict(status='COMPLETE',groups={k:dict(parents=len(v),requests=2*len(v)) for k,v in group_parents.items()},
        metrics=fields,metric_count=124,stage_count=2,paired_parent_rows=len(per_parent),files=files,
        parent_mean_policy='A parent value requires both registered noises. Missing/non-applicable metrics remain NA, not zero.',
        delta_policy='SEAM minus PLAIN; negative is not automatically better for every metric.',
        scope='Official test split is metadata evidence; cannot independently certify no upstream prior exposure. DEV is held out of this conditional training.',
        source_per_output_sha256=C.sha(root/'evaluation/per_output.csv'),
        score_registration_sha256=C.sha(root/'evaluation/score_registration.json'),
        aggregation_source_sha256=C.sha(__file__),model_calls=0,new_geometry_scans=0,new_queries=0)
    C.write(dest/'receipt.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(V.ROOT));a=p.parse_args();r=run(a.root)
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','metrics')}))


if __name__=='__main__':main()
