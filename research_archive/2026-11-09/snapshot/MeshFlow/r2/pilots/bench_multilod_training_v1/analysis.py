"""Lossless 177-metric collection and fixed paired comparisons, no rescoring."""
from __future__ import annotations
import argparse
import json
from . import common as C, evaluation as E

CONTRASTS=(('PLAIN1500','MULTILOD1500'),('EXPOSURE','PLAIN1500'),('EXPOSURE','MULTILOD1500'))


def collect():
    from r2.pilots.bench_adaptation_strength_v1.collect_v2 import decorate,aggregate,numeric,csvwrite
    dest=C.OUT/'results';done=dest/'completion.json'
    if done.exists():
        r=C.read(done);C.verify(r['files']);C.verify(r['source_rows']);return r
    reg=C.read(C.OUT/'evaluation/score_registration.json');C.verify(reg['source_hashes'])
    metrics=C.read(E.MANIFEST)['metric_fields'];rows=[];sources={};status_rows=[]
    for label in E.LABELS:
        for q in E.requests():
            for stage in E.STAGES:
                src,score,folder=E.engine().authenticate_score(label,q,stage)
                row=decorate(src,label,q,stage)
                row.update(eval_group=q['eval_group'],report_group=q['report_group'],
                    seam_applicability=score['seam_to_bulk']['status'],
                    result_origin='REUSED_READ_ONLY' if label=='EXPOSURE' else 'NEW_REGISTERED',
                    status_by_metric=score['status_by_metric'])
                if any(row[m]!=src[m] for m in metrics):raise ValueError('Original metric changed')
                rows.append(row)
                status_rows.append(dict(label=label,stage=stage,request_id=q['request_id'],parent_id=q['parent_id'],
                    status_by_metric=score['status_by_metric'],C_query_count=score.get('C_query_count'),
                    C_query_weight_denominator=score.get('C_query_weight_denominator'),
                    free_face_denominator=score.get('free_face_denominator'),
                    CF_tested_pair_denominator=score.get('CF_tested_pair_denominator'),
                    FF_tested_pair_denominator=score.get('FF_tested_pair_denominator'),
                    deletion_semantics=score.get('deletion_semantics')))
                for name in ('row.json','score.json','complete.json'):
                    path=folder/name;sources[str(path)]=C.sha(path)
    if len(rows)!=414:raise ValueError('Expected exactly 414 RAW/BASE/FINAL score rows')
    conditions=aggregate(rows,('label','stage','report_group','eval_group','parent_id','condition_id'),metrics)
    parents=aggregate(conditions,('label','stage','report_group','eval_group','parent_id'),metrics)
    summary=aggregate(parents,('label','stage','report_group','eval_group'),metrics)
    all_summary=aggregate(parents,('label','stage'),metrics)
    by={(r['label'],r['stage'],r['request_id']):r for r in rows};pairs=[]
    for ref,cand in CONTRASTS:
        for stage in E.STAGES:
            for q in E.requests():
                a,b=by[ref,stage,q['request_id']],by[cand,stage,q['request_id']]
                row=dict(reference=ref,candidate=cand,stage=stage,
                    **{k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','eval_group','report_group')})
                for m in metrics:
                    av,bv=a[m],b[m]
                    row[m+'__reference']=av;row[m+'__candidate']=bv
                    row[m+'__delta']=bv-av if numeric(av) and numeric(bv) else None
                pairs.append(row)
    dm=[m+'__delta' for m in metrics]
    pc=aggregate(pairs,('reference','candidate','stage','report_group','eval_group','parent_id','condition_id'),dm)
    pp=aggregate(pc,('reference','candidate','stage','report_group','eval_group','parent_id'),dm)
    ps=aggregate(pp,('reference','candidate','stage','report_group','eval_group'),dm)
    tables=dict(per_output=rows,per_condition=conditions,per_parent=parents,parent_macro_summary=summary,
        all_parent_macro_summary=all_summary,paired_per_output=pairs,paired_per_condition=pc,paired_per_parent=pp,
        paired_parent_macro_summary=ps,metric_status_and_denominators=status_rows)
    for name,table in tables.items():csvwrite(dest/(name+'.csv'),table)
    C.write(dest/'metric_rows.json',rows)
    result=dict(status='COMPLETE',rows=len(rows),paired_rows=len(pairs),metric_fields=metrics,
        reused_score_rows=138,new_score_rows=276,source_rows=sources,model_calls=0,scoring_calls=0,
        aggregation='Noise -> condition -> parent -> group; each parent equal weight. NA stays missing with valid_n.',
        I_domain='Original score status_by_metric and query denominators retained; no NA-to-zero replacement.',
        source_GT_success=False,code_sha256=C.sha(__file__),
        files={str(dest/(name+'.csv')):C.sha(dest/(name+'.csv')) for name in tables})
    result['files'][str(dest/'metric_rows.json')]=C.sha(dest/'metric_rows.json')
    C.write(done,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('collect',));p.parse_args()
    r=collect();print(json.dumps({k:v for k,v in r.items() if k not in ('files','source_rows','metric_fields')},ensure_ascii=False))


if __name__=='__main__':main()
