"""Frozen CPU source gate and one cached D0 anti-flatten proposal per request."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys
import time
import numpy as np
from r2.pilots.seam_slot_c200_v1 import common as C
from . import geometry as R

OUT=C.ROOT/'experiments/r2_seam_quality_four_v1/antiflatten'
OLD=C.OUT
PRIOR=C.ROOT/'experiments/r2_idea_1b_3_4a_diagnostic_v1/idea3'
ARM='SEAM_SLOT_C200'
VERSION='cached_D0_finite_C_normal_rejection_v1'
HASHES={}


def track(path):
    path=C.resolve(path);HASHES[str(path)]=C.sha(path);return path
def read(path):return C.read(track(path))
def arrays(path):return C.arrays(track(path))
def rows(path):
    with track(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def protect():
    for path,digest in HASHES.items():
        if C.sha(path)!=digest:raise ValueError('Protected input changed: '+path)


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=VERSION,scope='CPU cached C-only PREDICTED16; no D0/model/GPU call',
        resources=C.resources(),L=R.L,distance_threshold_fraction_L=.02,
        parallel_abs_cosine_min=R.COS,strict_projection_barycentric_min=R.BARY_EPS,
        shared_base_exclusion_distance=R.DIST_EPS,owner_face_excluded=False,
        probe_points=['CENTROID','TIP'],nearest='Global nearest finite C triangle, distance then original C index tie-break',
        risk='max of two (1-d/(.02L))^2 on strict-interior, off-own-base, inward r>0 and |normaldot|>cos15 gates',
        operation='Freeze each PRE_GEO nearest C face; compare RAW/PRE risk on same face. If risk rises and |tip plane height| falls, restore RAW tip normal height and retain PRE tangent. One maximum risk-rise probe per seam.',
        prohibited=['no RAW edit','no anchor/C edit','no tangential/mean-anchor compensation','no smoothing','no extra D0','no capacity guard'],
        pipeline='modified cached PRE_GEO -> unchanged original GEO_C -> original C winding export -> unchanged original scorers',
        code_hashes={str(Path(__file__)):C.sha(__file__),str(Path(R.__file__)):C.sha(R.__file__)},
        evaluation_requests=16,new_POST_budget=16,new_quality_scan_budget=16,source_edges=389,
        scientific_limits=['Two-point proxy is neither necessary nor sufficient for surface overlap or I.',
        'Source edge legality does not certify the entire source Y face is conflict-free against all C.',
        'Finite near-surface contact can be legitimate; no zero-false-positive assumption.',
        'Normal restoration has no extra magnitude cap beyond restoring that RAW height; record all movement.'])
    p=OUT/'protocol.json'
    if p.exists():
        old=C.read(p)
        for key in ('code_hashes','version','distance_threshold_fraction_L','strict_projection_barycentric_min'):
            if old[key]!=protocol[key]:raise ValueError('Protocol/code changed after registration')
        return old
    C.write(p,protocol);return protocol


def source_gate():
    reg=register()
    if (OUT/'source_gate.json').exists():
        done=C.read(OUT/'source_gate.json')
        for path,digest in done['input_hashes'].items():assert C.sha(path)==digest,path
        print(json.dumps(done['summary']));return done
    if (OUT/'source_attempt.json').exists():raise RuntimeError('Source attempt exists; inspect before explicit recovery')
    C.write(OUT/'source_attempt.json',dict(utc=C.utc(),protocol_sha256=C.sha(OUT/'protocol.json')))
    tick=time.perf_counter();test=R.selftest()
    source=read(PRIOR/'registration.json'); oldrows=rows(PRIOR/'per_source_interface.csv')
    records={r['condition_id']:r for r in source['records']}
    panel={r['condition_id'] for r in read(OLD/'eval_manifest.json')['conditions']}
    grouped={};edge_rows=[];probe_rows=[]
    for row in oldrows:grouped.setdefault(row['condition_id'],[]).append(row)
    for cid,rr in grouped.items():
        rec=records[cid];p=arrays(rec['parent_npz']);condition=arrays(rec['condition_path'])
        assert C.sha(rec['parent_npz'])==rec['parent_sha256']
        assert C.sha(rec['condition_path'])==rec['condition_sha256']
        v=p['model_vertices'].astype(float);faces=p['source_faces']
        cids=condition['region_source_face_ids'].astype(int)
        ctris=v[faces[cids]];c_lookup={int(x):i for i,x in enumerate(cids)}
        for row in rr:
            ci=int(row['C_source_face']);u,w=map(int,json.loads(row['C_directed_edge']));third=int(row['third_vertex_id'])
            result=R.probes(v[u],v[w],v[third],v[faces[ci]],ctris)
            item=dict(condition_id=cid,parent_id=row['parent_id'],group=rec['group'],
                      in_training_C200=cid in panel,edge_id=int(row['source_edge_id']),
                      C_source_face=ci,Y_source_face=int(row['Y_source_face']),owner_C_local_face=c_lookup[ci],
                      source_classification=row['classification'],source_Y_legal_C_edges=int(row['Y_legal_C_edges']),
                      rule_triggered=any(x.get('active',False) for x in result),
                      max_risk=max(x['risk'] for x in result),
                      unit_L_hinge_sum=sum(x['risk']*.02**2 for x in result),
                      valid_probes=sum(x['status']=='PASS' for x in result))
            edge_rows.append(item)
            for probe in result:probe_rows.append(dict(**item,**probe))
    assert len(edge_rows)==389 and sum(x['in_training_C200'] for x in edge_rows)==62
    summary=[]
    for label,rr in [('ALL389',edge_rows),('TRAINING62',[x for x in edge_rows if x['in_training_C200']])]:
        flagged=[x for x in rr if x['rule_triggered']]
        summary.append(dict(scope=label,edges=len(rr),triggered=len(flagged),
                            conditions_triggered=len({x['condition_id'] for x in flagged}),
                            normalized_loss=sum(x['unit_L_hinge_sum'] for x in rr)/(2*len(rr)),
                            max_risk=max(x['max_risk'] for x in rr)))
    C.csvwrite(OUT/'source_per_edge.csv',edge_rows);C.csvwrite(OUT/'source_per_probe.csv',probe_rows)
    C.csvwrite(OUT/'source_triggered.csv',[x for x in edge_rows if x['rule_triggered']])
    C.csvwrite(OUT/'source_summary.csv',summary);protect()
    value=dict(status='COMPLETE',version=VERSION,summary=summary,selftest=test,
               training_loss_gate='SOURCE_GEOMETRY_CHECK_COMPLETE_NOT_AUTOMATIC_TRAINING_AUTHORIZATION',
               scope='Known legal interface edges; other nearby source surfaces may have their own defects',
               input_hashes=HASHES,protocol_sha256=C.sha(OUT/'protocol.json'),
               outputs={str(p):C.sha(p) for p in OUT.glob('source_*.csv')},seconds=time.perf_counter()-tick,
               model_calls=0,GPU_calls=0)
    C.write(OUT/'source_gate.json',value);print(json.dumps(summary),flush=True);return value


def finish_one(q):
    from r2.pilots.seam_slot_c200_v1 import generate as G,postprocess as P
    from meshflow_control.geo_io import write_indexed_ply
    base=OLD/'evaluation'/ARM/q['request_id'];dest=OUT/'evaluation'/ARM/q['request_id']/'post'
    if (dest/'receipt.json').exists():
        rec=C.read(dest/'receipt.json')
        for p,h in rec['files'].items():assert C.sha(p)==h,p
        return C.read(dest/'movement.json')
    if dest.exists():raise RuntimeError('Incomplete geometry attempt: '+str(dest))
    raw=arrays(base/'raw/raw.npz');source=arrays(base/'raw/source_condition.npz');pre=arrays(base/'post/pre_geo.npz')['output']
    dest.mkdir(parents=True);C.write(dest/'started.json',dict(utc=C.utc(),request_id=q['request_id']))
    cverts=raw['C_vertices'].astype(float);ctris=cverts[raw['C_faces_original']];modified=pre.copy();slotrows=[]
    for slot in np.flatnonzero(raw['raw_seam_slot_mask']):
        slot=int(slot);ei=int(raw['raw_slot_edge_id'][slot]);ci,u,w=map(int,raw['edge_records'][ei])
        free=np.flatnonzero(raw['raw_anchor_ids'][slot]<0);assert len(free)==1;corner=int(free[0])
        target,info=R.restore_raw_normal(cverts[u],cverts[w],raw['raw'][slot,corner],pre[slot,corner],ctris[ci],ctris)
        modified[slot,corner]=target
        slotrows.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],
                             raw_slot_id=slot,edge_id=ei,third_corner=corner,
                             raw_tip=raw['raw'][slot,corner].tolist(),pre_tip=pre[slot,corner].tolist(),final_tip=target.tolist(),**info))
    anchors=raw['raw_anchor_ids']>=0
    assert G.same(modified[anchors],pre[anchors]) and G.same(modified[:q['K']],pre[:q['K']])
    # No baseline geometry replay: use the already audited unchanged pipeline.
    returned_pre,canonical,final,geo,orientation,anchor_receipt=P.geometry_from_proposal(raw,modified,source)
    assert G.same(returned_pre,modified)
    G.save_npz(dest/'pre_geo.npz',dict(output=modified,known_faces=raw['known_faces'],raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
    G.save_npz(dest/'post_canonical.npz',canonical);G.save_npz(dest/'post.npz',final)
    write_indexed_ply(dest/'post.ply',final['vertices'],final['faces'])
    C.write(dest/'geometry.json',dict(stats=geo['stats'],face_records=geo['face_records']))
    C.write(dest/'anchor_identity.json',anchor_receipt);C.write(dest/'source_restoration.json',orientation['restoration_summary'])
    oo=orientation['orientation_result'];C.write(dest/'orientation.json',dict(summary=oo['summary'],components=oo['component_rows'],edges=oo['edge_rows']))
    C.write(dest/'movement.json',slotrows);C.csvwrite(dest/'movement.csv',slotrows)
    C.write(dest/'receipt.json',dict(status='COMPLETE',arm=ARM,request_id=q['request_id'],version=VERSION,
        source_pre_geo_sha256=C.sha(base/'post/pre_geo.npz'),protocol_sha256=C.sha(OUT/'protocol.json'),
        files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()},model_calls=0,D0_forwards=0))
    return slotrows


def geometry():
    register();m=read(OLD/'eval_manifest.json');allrows=[];start=time.perf_counter()
    for q in m['requests']:
        allrows.extend(finish_one(q));print('ANTIFLATTEN',q['request_id'],flush=True)
    C.csvwrite(OUT/'movement.csv',allrows);protect()
    changed=[x for x in allrows if x['moved']]
    summary=dict(status='COMPLETE',requests=16,active_seam_observations=len(allrows),moved_slots=len(changed),
                 requests_moved=len({x['request_id'] for x in changed}),
                 maximum_displacement_model=max([x.get('displacement_model',0.) for x in changed] or [0.]),
                 model_calls=0,D0_forwards=0,GEO_calls=16,input_hashes=HASHES,seconds=time.perf_counter()-start)
    C.write(OUT/'geometry_completion.json',summary);return summary


def score():
    from r2.pilots.seam_slot_c200_v1 import evaluate as E
    register();assert C.read(OUT/'geometry_completion.json')['requests']==16
    m=read(OLD/'eval_manifest.json');receipts=[];tick=time.perf_counter()
    for q in m['requests']:
        receipts.append(E.score_one((str(OUT),ARM,q,'POST',C.sha(OUT/'protocol.json'))))
        print('ANTIFLATTEN_SCORE',len(receipts),q['request_id'],flush=True)
    C.write(OUT/'score_completion.json',dict(status='COMPLETE',cells=16,receipts=receipts,workers=1,threads=1,
               full_quality_scans=sum(not x['reused'] for x in receipts),model_calls=0,seconds=time.perf_counter()-tick))


def aggregate():
    register();m=read(OLD/'eval_manifest.json');allrows=[];pairs=[];keyset=None
    for q in m['requests']:
        outputs={}
        for label,root in [('BASELINE',OLD),('ANTI_FLATTEN',OUT)]:
            s=read(root/'evaluation/scores'/ARM/q['request_id']/'POST/score.json')
            numeric={**s['metrics'],**s['topology'],**s['seam_to_bulk']}
            row={k:q[k] for k in E_META};row.update(label=label,stage='POST',C_preserved_exact=s['C_preserved_exact'])
            row.update({k:v for k,v in numeric.items() if isinstance(v,(int,float,bool)) or v is None})
            outputs[label]=row;allrows.append(row)
        for metric in outputs['BASELINE']:
            if metric in (*E_META,'label','stage'):continue
            a,b=outputs['BASELINE'][metric],outputs['ANTI_FLATTEN'].get(metric)
            if isinstance(a,(float,int,bool)) and isinstance(b,(float,int,bool)):
                pairs.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],noise_index=q['noise_index'],metric=metric,
                                  baseline=a,candidate=b,delta=float(b)-float(a)))
    C.csvwrite(OUT/'per_output.csv',allrows);C.csvwrite(OUT/'paired_per_noise.csv',pairs)
    summary=[]
    for metric in sorted({x['metric'] for x in pairs}):
        rr=[x for x in pairs if x['metric']==metric]
        summary.append(dict(metric=metric,n=len(rr),baseline_mean=float(np.mean([x['baseline'] for x in rr])),
                            candidate_mean=float(np.mean([x['candidate'] for x in rr])),delta_mean=float(np.mean([x['delta'] for x in rr])),
                            improved_lower=sum(x['delta']<0 for x in rr),increased=sum(x['delta']>0 for x in rr)))
    C.csvwrite(OUT/'summary.csv',summary);protect()
    C.write(OUT/'completion.json',dict(status='COMPLETE_NUMERIC_PENDING_ACTUAL_VISUAL_REVIEW',version=VERSION,
        model_calls=0,D0_forwards=0,POST_outputs=16,quality_scans=16,input_hashes=HASHES,
        protocol_sha256=C.sha(OUT/'protocol.json'),outputs={str(p):C.sha(p) for p in OUT.glob('*.csv')}))
    return summary


E_META=('request_id','condition_id','parent_id','noise_index','N','K','split')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('source','geometry','score','aggregate','geometry_score'),required=True)
    args=parser.parse_args()
    if args.stage=='source':source_gate()
    elif args.stage=='geometry_score':geometry();score();aggregate()
    else:globals()[args.stage]()
    if 'torch' in sys.modules:raise RuntimeError('This CPU geometry/scorer process must not import torch')


if __name__=='__main__':main()
