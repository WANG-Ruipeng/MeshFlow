"""Source-owner reference diagnostics; never imported by training/inference.

The source completion supplies one reference layout, not the only valid open-
ended completion. Predictions are read only after saved C-only head calls exist.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
from . import data as D

OUT = D.OUT.parent / 'layout_diagnostic'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2)


def csvwrite(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def prepare_source():
    if (OUT/'source_summary.json').exists(): return D.read(OUT/'source_summary.json')
    manifest=D.read(D.OUT.parent/'eval_manifest.json')
    train=D.read(D.ROOT/'experiments/r2_user_data_scaling_32_67_v1/data/train_manifest.json')
    dev=D.read(D.ROOT/'experiments/r2_user_data_scaling_32_67_v1/data/panel_manifest.json')
    tasks=dict(train['tasks']);tasks.update({r['condition_id']:r for r in dev['conditions']})
    seats=D.read(D.ROOT/'experiments/r2_seam_abc_v1/B_DATA_REGISTRATION.json')
    seat_records={r['condition_id']:r for r in seats['records']+seats['DEV_records']}
    unique={q['condition_id']:q for q in manifest['requests']}; rows=[]; protected={}
    counts=[]
    for cid,q in sorted(unique.items()):
        if q['part_type']=='SEAT_FULL':
            rec=seat_records[cid];case=D.arrays(rec['labels_npz'])
            edges,_=D.edge_features(case['C_vertices'],case['C_faces_original'],int(case['N']))
            case['edge_records']=edges;audit=[]
            for ei,(ci,u,w) in enumerate(edges):
                role='OWNER' if case['occupied_labels'][ei] else 'ALIAS' if case['source_alias_labels'][ei] else 'NULL_UNSUPPORTED'
                audit.append(dict(edge_id=ei,C_source_face=int(case['known_ids'][ci]),
                                  source_u=int(case['C_source_vertex_ids'][u]),source_v=int(case['C_source_vertex_ids'][w]),
                                  reason='EXISTING_FROZEN_SEAT_OWNER_LABEL',role=role))
            source_paths=(rec['labels_npz'],)
        else:
            t=tasks[cid]; rec=q['source_record']; pn=rec['parent_npz']; cn=rec['condition_path']; qn=t['quality_path']
            p,c,quality=D.arrays(pn),D.arrays(cn),D.read(qn)
            assert D.sha(pn)==rec['parent_sha256'] and D.sha(cn)==rec['condition_sha256']
            assert D.sha(qn)==t['quality_sha256'] and quality['input_npz_sha256']==D.sha(pn)
            case,audit=D.source_case(p,c,quality);source_paths=(pn,cn,qn)
        bundle=D.arrays(q['inference_npz'])
        for key in ('C_vertices','C_faces_original','C_faces_model'):assert np.array_equal(bundle[key],case[key])
        for path in (*source_paths,q['inference_npz']):protected[path]=D.sha(path)
        edges=case['edge_records'];v=case['C_vertices']
        for a in audit:
            ei=a['edge_id']; u,w=edges[ei,1:];h=float((v[u,1]+v[w,1])/2)
            rows.append(dict(condition_id=cid,parent_id=q['parent_id'],part_type=q['part_type'],split=q['split'],
                             source_owner=bool(case['occupied_labels'][ei]),source_alias=bool(case['source_alias_labels'][ei]),
                             source_unsupported=bool(not case['source_supported_labels'][ei]),
                             world_y=h,**a))
        counts.append(dict(condition_id=cid,part_type=q['part_type'],split=q['split'],
                           edges=len(edges),owner=int(case['occupied_labels'].sum()),
                           alias=int(case['source_alias_labels'].sum()),null_unsupported=int((~case['source_supported_labels']).sum())))
    OUT.mkdir(parents=True,exist_ok=True);csvwrite(OUT/'source_edges.csv',rows);csvwrite(OUT/'source_conditions.csv',counts)
    summary=[]
    for part in ('BACK_FULL','SEAT_FULL'):
        for split in ('ALL',*sorted({r['split'] for r in counts if r['part_type']==part})):
            cc=[r for r in counts if r['part_type']==part and (split=='ALL' or r['split']==split)]
            summary.append(dict(part_type=part,split=split,conditions=len(cc),**{k:sum(r[k] for r in cc) for k in ('edges','owner','alias','null_unsupported')}))
    result=dict(status='COMPLETE_SOURCE_REFERENCE',summary=summary,conditions=counts,protected_inputs=protected,
                policy='Only one source completion reference; no classification inference about all valid completions; no top-edge Null heuristic',
                new_geometry_scans=0,model_forwards=0,training_updates=0,
                files={str(p):D.sha(p) for p in (OUT/'source_edges.csv',OUT/'source_conditions.csv')})
    write(OUT/'source_summary.json',result);return result


def compare_predicted():
    prepare_source();manifest=D.read(D.OUT.parent/'eval_manifest.json')
    unique={q['condition_id']:q for q in manifest['requests']}
    paths={cid:D.OUT.parent/'evaluation'/manifest['label']/'layouts'/cid for cid in unique}
    if any(not (p/'receipt.json').exists() for p in paths.values()):
        raise RuntimeError('Wait until all15 actual C-only head receipts exist; no partial comparison')
    with (OUT/'source_edges.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    cache={};protected={}
    for cid,p in paths.items():
        rec=D.read(p/'receipt.json')
        assert rec['status']=='COMPLETE' and rec['occupancy_forward_calls']==1 and rec['source_labels_read'] is False
        for file,sha in rec['files'].items():assert D.sha(file)==sha,file
        a=D.arrays(p/'layout.npz');assert np.array_equal(a['occupied'],a['occupancy_logits']>=0)
        cache[cid]=a;protected[str(p/'receipt.json')]=D.sha(p/'receipt.json')
    predictions=[]
    for r in rows:
        a=cache[r['condition_id']];ei=int(r['edge_id']);truth=r['source_owner']=='True';pred=bool(a['occupied'][ei])
        predictions.append(dict(**r,logit=float(a['occupancy_logits'][ei]),predicted_active=pred,
                                reference_class='TP' if truth and pred else 'FN' if truth else 'FP' if pred else 'TN'))
    csvwrite(OUT/'predicted_vs_source_edges.csv',predictions)
    cc=[]
    for cid,q in sorted(unique.items()):
        rr=[r for r in predictions if r['condition_id']==cid];counts=collections.Counter(r['reference_class'] for r in rr)
        cc.append(dict(condition_id=cid,parent_id=q['parent_id'],part_type=q['part_type'],split=q['split'],
                       edges=len(rr),active=sum(r['predicted_active'] for r in rr),
                       **{k:counts[k] for k in ('TP','FN','FP','TN')},
                       alias_activated=sum(r['predicted_active'] and r['source_alias']=='True' for r in rr),
                       unsupported_activated=sum(r['predicted_active'] and r['source_unsupported']=='True' for r in rr)))
    csvwrite(OUT/'predicted_vs_source_conditions.csv',cc)
    write(OUT/'prediction_completion.json',dict(status='COMPLETE',conditions=cc,source_receipts=protected,
          files={str(p):D.sha(p) for p in (OUT/'predicted_vs_source_edges.csv',OUT/'predicted_vs_source_conditions.csv')},
          threshold_logit=0,model_forwards=0,source_owner_at_inference=False,
          limitation='Reference-layout agreement, not unique-task correctness or proof of seam/bulk connection'))
    return cc


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--predicted',action='store_true');args=parser.parse_args()
    result=compare_predicted() if args.predicted else prepare_source()
    print(json.dumps(result.get('summary',result) if isinstance(result,dict) else result,ensure_ascii=False,indent=2))
