"""Compact, descriptive tables from saved per-output scores; no scoring or model calls."""
import csv
import json
import collections
from pathlib import Path
import numpy as np
from . import eval_common as C

METRICS = (
 'CF_illegal_affected_free_fraction', 'FF_illegal_affected_free_fraction',
 'ConflictCov_C', 'ConflictCov_C_dup', 'free_minimum_angle_lt5_fraction',
 'C_F_valid_edge_count', 'C_F_valid_edge_total_length',
 'free_only_reached_bulk_area_fraction', 'bulk_reach_ge_0_30_descriptive',
 'seed_connected_legal_seam_bulk_edge_count', 'illegal_pairs_seam-bulk',
 'illegal_pairs_bulk-bulk', 'nonmanifold_edges', 'C_F_nonmanifold_edge_count',
 'actual_free_faces', 'deleted_free_faces', 'free_total_surface_area',
 'EXACT_COORD_face_components', 'A_VMR__L_tau_0.001', 'PMR__L_tau_0.001',
)

def num(value):
    if value in ('True', 'False'): return float(value=='True')
    try: return float(value)
    except (ValueError, TypeError): return None

def main():
    src=C.OUT/'evaluation/per_output.csv'
    with src.open(encoding='utf-8-sig', newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==256
    out=C.OUT/'analysis';out.mkdir(exist_ok=True)
    result=[]
    for label in (*C.ARMS, 'SEAM200'):
        for split in ('TRAIN','DEV'):
            if label=='SEAM200' and split=='DEV':continue
            for stage in ('RAW','POST'):
                rr=[r for r in rows if (r['label'],r['split'],r['stage'])==(label,split,stage)]
                assert len(rr)==(16 if split=='TRAIN' else 12)
                for metric in METRICS:
                    vals=[num(r[metric]) for r in rr]
                    valid=[v for v in vals if v is not None and np.isfinite(v)]
                    result.append(dict(label=label,split=split,stage=stage,metric=metric,
                        registered=len(rr),valid=len(valid),mean=float(np.mean(valid)) if valid else None,
                        median=float(np.median(valid)) if valid else None,
                        minimum=float(np.min(valid)) if valid else None,maximum=float(np.max(valid)) if valid else None))
    C.csvwrite(out/'key_metrics.csv',result)
    lookup={(r['label'],r['stage'],r['request_id']):r for r in rows}
    transitions=[]
    for baseline,candidate in [('LONG',a) for a in C.ARMS[1:]]+[('SEAM200',a) for a in C.ARMS]:
        for split in ('TRAIN','DEV'):
            if baseline=='SEAM200' and split=='DEV':continue
            for stage in ('RAW','POST'):
                rr=[r for r in rows if (r['label'],r['split'],r['stage'])==(candidate,split,stage)]
                for r in rr:
                    b=lookup[baseline,stage,r['request_id']]
                    for metric in ('C_F_has_valid_edge','bulk_reach_ge_0_30_descriptive'):
                        old,new=num(b[metric]),num(r[metric])
                        transitions.append(dict(baseline=baseline,candidate=candidate,split=split,stage=stage,
                            request_id=r['request_id'],metric=metric,old=old,new=new,
                            transition='retained' if old==new==1 else 'gained' if old==0 and new==1 else 'lost' if old==1 and new==0 else 'absent'))
    C.csvwrite(out/'connectivity_transitions.csv',transitions)
    costs=[]
    for candidate in C.ARMS[1:]:
        for r in [r for r in rows if r['label']==candidate]:
            b=lookup['LONG',r['stage'],r['request_id']]
            d=dict(candidate=candidate,request_id=r['request_id'],split=r['split'],stage=r['stage'])
            for metric in METRICS:
                x,y=num(b[metric]),num(r[metric]);d[metric+'_delta']=y-x if x is not None and y is not None else None
            costs.append(d)
    C.csvwrite(out/'all_request_costs_vs_LONG.csv',costs)
    pointer=[]
    for q in C.read(C.OUT/'eval_manifest.json')['requests']:
        folder=C.OUT/'evaluation/C_SHARED_TIP'/q['request_id']
        audit=C.read(folder/'raw/pointer_audit.json');facts=audit['facts']
        anchors=C.read(folder/'post/anchor_identity.json')
        pointer.append(dict(request_id=q['request_id'],split=q['split'],
            source_labels_read=audit['source_labels_read'],forward_calls=audit['pointer_forward_calls'],
            argmax_nonnull=facts['pointer_argmax_nonnull'],conflict_to_null=facts['pointer_conflict_null'],
            assigned_corners=facts['aliased_corner_count'],
            retained_alias_corners=anchors['shared_tip_retained_alias_corners'],
            deleted_alias_corners=anchors['shared_tip_deleted_alias_corners'],
            retained_alias_owner_deleted=anchors['shared_tip_retained_alias_with_owner_face_deleted'],
            pointer_audit_sha256=C.sha(folder/'raw/pointer_audit.json')))
    C.csvwrite(out/'pointer_inference.csv',pointer)
    C.write(out/'tables_receipt.json',dict(status='COMPLETE_DESCRIPTIVE',source=str(src),source_sha256=C.sha(src),
        code_sha256=C.sha(__file__),new_model_calls=0,new_scans=0,
        caveats=['Four correlated noises per condition; 4 TRAIN and 3 DEV parents.',
          '30% bulk reach is a descriptive threshold, not semantic adoption or complete-body certification.',
          'No claim of statistical noninferiority or simultaneous improvement of all metrics.'],
        files={str(p):C.sha(p) for p in out.glob('*.csv')}))
    print(json.dumps({'status':'COMPLETE_DESCRIPTIVE','rows':len(result),'transition_rows':len(transitions)}))

if __name__=='__main__':main()
