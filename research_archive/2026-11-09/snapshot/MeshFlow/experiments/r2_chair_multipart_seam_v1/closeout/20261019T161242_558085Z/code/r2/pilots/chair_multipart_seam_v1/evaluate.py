"""Reuse unchanged original Chair quality math on the fixed 44 requests."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
import collections
from pathlib import Path
import numpy as np
from .eval_prepare import OUT,LABEL
from r2.pilots.chair_back_transfer_v1 import evaluate as V
C=V.C
STAGES=('RAW','BASE_POST')


def requests(root=OUT):
    m=C.read(C.resolve(root)/'eval_manifest.json');V.E.verify(m['protected'])
    assert len(m['requests'])==44
    return m['requests']


def configure():
    V.ROOT=OUT;V.LABEL=LABEL;V.STAGES=STAGES;V.requests=requests;V.prepare=prepare


def prepare(root=OUT):
    dest=C.resolve(root)/'evaluation/score_registration.json'
    if dest.exists():
        r=C.read(dest);V.E.verify(r['protected']);V.E.verify(r['source_hashes']);return r
    qq=requests(root);old,A,_,_,_=V.E.helpers();audits=[]
    # Older SEAT inputs preserve C provenance in metadata, without the newer
    # redundant source_npz sidecar. The scorer uses the same original C/query.
    protected={str(OUT/'eval_manifest.json'):C.sha(OUT/'eval_manifest.json')}
    protected.update(C.read(OUT/'eval_manifest.json')['protected'])
    for q in qq:
        inp=C.arrays(q['inference_npz']);meta=C.arrays(q['metadata_npz'])
        cc=inp['C_vertices'][inp['C_faces_model']];original=inp['C_vertices'][inp['C_faces_original']]
        assert cc.shape==(int(q['K']),3,3) and cc.dtype==np.float32
        desc=q['query_descriptor'];reference=old.load_queries(desc)
        assert desc['identity']['C_sha256']==old.array_sha(cc) and len(reference['arrays']['C_points'])==2048
        _,_,topology=A.topology(original,meta['C_source_vertex_ids'][inp['C_faces_original']])
        assert topology['boundary_edge_count']>0
        audits.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],boundary=topology,query_sha256=desc['sha256']))
    code=V.source_hashes();code[str(Path(__file__))]=C.sha(__file__)
    result=dict(status='FROZEN',version='chair_multipart_original_quality_v1',label=LABEL,requests=qq,stages=STAGES,
        protected=protected,source_hashes=code,L=C.L,C_queries=2048,new_queries=0,
        quality_definition=old.cpu_math()['control'].frozen_definition(),input_audits=audits,
        metric_policy='Unchanged original quality scorer; bridge fields retained; no new query or GT style rank.',
        metadata_policy='SEAT has source identity metadata; BACK also has redundant source C npz. No missing source geometry is guessed.',model_calls=0)
    C.write(dest,result);return result


def aggregate():
    configure();qq=requests();rows=[];source={}
    for q in qq:
        for stage in STAGES:
            p=V.score_dir(OUT,q,stage);r=C.read(p/'audit_receipt.json');V.E.verify(r['files'])
            row=C.read(p/'row.json');row.update(label=LABEL,stage=stage,part_type=q['part_type'])
            rows.append(row);source[str(p/'audit_receipt.json')]=C.sha(p/'audit_receipt.json')
    meta=set(V.E.META)|{'label','stage','model','part_type'}
    fields=sorted(k for k in set().union(*(r.keys() for r in rows)) if k not in meta and any(V.number(r.get(k)) for r in rows))
    summary=[]
    for stage in STAGES:
        for part in ('SEAT_FULL','BACK_FULL'):
            selected=[r for r in rows if r['stage']==stage and r['part_type']==part]
            for split in ('ALL',*sorted({r['split'] for r in selected})):
                rr=[r for r in selected if split=='ALL' or r['split']==split]
                for field in fields:
                    byc=collections.defaultdict(list)
                    for r in rr:
                        if V.number(r.get(field)):byc[r['parent_id'],r['condition_id']].append(float(r[field]))
                    byp=collections.defaultdict(list)
                    for (p,c),vv in byc.items():byp[p].append(float(np.mean(vv)))
                    summary.append(dict(label=LABEL,stage=stage,part_type=part,split=split,metric=field,
                        mean_parent_equal=float(np.mean([np.mean(v) for v in byp.values()])) if byp else None,
                        n_outputs=len(rr),n_valid=sum(len(v) for v in byc.values()),n_valid_parents=len(byp)))
    dest=OUT/'evaluation/tables';files={}
    for name,value in (('per_output',rows),('summary_parent_equal',summary)):
        p=dest/(name+'.csv');C.csvwrite(p,value);files[str(p)]=C.sha(p)
    C.write(dest/'completion.json',dict(status='COMPLETE',rows=len(rows),numeric_fields=fields,files=files,source_receipts=source,
        grouping='Panels reported separately; noise->condition->parent. No unique GT-style success.',model_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','score','aggregate'));p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();configure()
    if a.action=='prepare':V.prepare(OUT)
    elif a.action=='score':V.run(OUT,STAGES,a.workers)
    else:aggregate()
