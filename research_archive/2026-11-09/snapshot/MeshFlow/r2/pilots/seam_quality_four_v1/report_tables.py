"""Report-only derivations from frozen saved scores; no geometry evaluation."""
import csv
import collections
import json
import numpy as np
from . import eval_common as C


def main():
    root=C.OUT
    rows=list(csv.DictReader((root/'evaluation/per_output.csv').open(encoding='utf-8-sig')))
    fields=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C',
        'ConflictCov_C_dup','free_minimum_angle_lt5_fraction','deleted_free_faces','free_total_surface_area',
        'free_only_reached_bulk_area_fraction','opposite_winding_reached_free_area_fraction',
        'C_F_valid_edge_count','C_F_valid_edge_total_length','C_F_nonmanifold_edge_count','FF_illegal_pair_count',
        'illegal_pairs_bulk-bulk','illegal_pairs_seam-bulk','illegal_pairs_seam-seam')
    result={'groups':[],'paired':[],'parents':[],'occupancy':[],'deleted_reasons':[]}
    labels=('SEAM200','FULL4_C200','LONG','ANTI_COVER','LOCAL_NOISE')
    for label in labels:
        for stage in ('RAW','POST'):
            rr=[r for r in rows if r['label']==label and r['stage']==stage]
            value=dict(label=label,stage=stage,metrics={})
            for key in fields:
                x=[float(r[key]) for r in rr if r.get(key) not in ('',None)]
                value['metrics'][key]=dict(mean=float(np.mean(x)),median=float(np.median(x)),max=max(x),sum=sum(x)) if x else None
            if label!='FULL4_C200':
                x=[float(r['free_only_reached_bulk_area_fraction']) for r in rr]
                value.update(bulk_nonzero=sum(v>0 for v in x),bulk_ge30=sum(v>=.3 for v in x))
            value['I_tail']=[dict(request_id=r['request_id'],I=float(r['ConflictCov_C'])) for r in sorted(rr,key=lambda r:float(r['ConflictCov_C']),reverse=True)[:3]]
            result['groups'].append(value)
    for baseline,candidate in (('SEAM200','LONG'),('LONG','ANTI_COVER'),('LONG','LOCAL_NOISE')):
        for stage in ('RAW','POST'):
            a={r['request_id']:r for r in rows if r['label']==baseline and r['stage']==stage}
            b={r['request_id']:r for r in rows if r['label']==candidate and r['stage']==stage}
            for field in ('FF_illegal_affected_free_fraction','ConflictCov_C','free_only_reached_bulk_area_fraction'):
                d=np.array([float(b[q][field])-float(a[q][field]) for q in a])
                result['paired'].append(dict(baseline=baseline,candidate=candidate,stage=stage,metric=field,
                    mean_delta=float(np.mean(d)),median_delta=float(np.median(d)),lower=int((d< -1e-12).sum()),
                    equal=int((abs(d)<=1e-12).sum()),higher=int((d>1e-12).sum())))
    for label in labels:
        for parent in sorted({r['parent_id'] for r in rows}):
            rr=[r for r in rows if r['label']==label and r['parent_id']==parent and r['stage']=='POST']
            result['parents'].append(dict(label=label,parent=parent[:8],**{key:float(np.mean([float(r[key]) for r in rr])) for key in ('FF_illegal_affected_free_fraction','ConflictCov_C','deleted_free_faces')}))
    occ=list(csv.DictReader((root/'evaluation/occupancy_edges.csv').open(encoding='utf-8-sig')))
    for label in ('SEAM200',*C.ARMS):
        for cond in sorted({r['condition_id'] for r in occ}):
            rr=[r for r in occ if r['label']==label and r['condition_id']==cond]
            ct=collections.Counter(('T' if r['predicted']==r['source_label_training_audit_only'] else 'F')+('P' if r['predicted']=='True' else 'N') for r in rr)
            result['occupancy'].append(dict(label=label,condition_id=cond,**dict(ct)))
    deleted=list(csv.DictReader((root/'evaluation/deleted_faces.csv').open(encoding='utf-8-sig')))
    for label in labels:
        counts=collections.Counter(r['reason'] for r in deleted if r['label']==label)
        result['deleted_reasons'].append(dict(label=label,counts=dict(counts)))
    C.write(root/'report_tables.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
