"""CPU-only additive panel: inherited24 stay byte-identical; two noises/new C."""
from __future__ import annotations
import argparse, collections, hashlib, sys, time, types
from pathlib import Path
import numpy as np
from .common import ROOT, OUT, SOURCE_ROOT, read, resolve, sha
from r2.pilots.bench_full_seat_curriculum_v1 import panel as old
P, L, arrays, verify, protect = old.P, old.L, old.arrays, old.verify, old.protect
VERSION = 'BENCH_SEAT_COVERAGE_ADDITIVE_PANEL_V1'
LABELS = ('NATIVE_S6000', 'COVERAGE_FINAL', 'SEAM_S3000')
NEW_LABELS = LABELS[:2]

def seed(cid, ni):
    return int.from_bytes(hashlib.sha256((VERSION+'|'+cid+'|'+str(ni)).encode()).digest()[:4], 'little')

def new_bundle(row, protected):
    ns=dict(vars(old), OUT=OUT)
    fn=types.FunctionType(old.new_bundle.__code__, ns, old.new_bundle.__name__, old.new_bundle.__defaults__)
    return fn(row, protected)

def labels_for(q):
    return LABELS if q.get('panel_origin')=='INHERITED24' else NEW_LABELS

def prepare(selection_path=None):
    done=OUT/'eval_assets_complete.json'
    if done.exists():
        result=read(done);verify(result['files']);validate_assets();return result
    selection_path=resolve(selection_path or OUT/'eval_selection.json')
    selection=read(selection_path)
    if selection.get('status')!='APPROVED_INPUT_REVIEW':raise ValueError('Input review and explicit selection required')
    oldpath=resolve(selection.get('old_manifest', SOURCE_ROOT/'eval_manifest.json'))
    if oldpath!=SOURCE_ROOT/'eval_manifest.json':raise ValueError('Only completed curriculum24 may be inherited')
    previous=read(oldpath)
    if len(previous['requests'])!=24:raise ValueError('Expected exact old24')
    verify(previous['protected'])
    chosen=selection.get('new_conditions', selection.get('conditions', []))
    if not chosen or len({r['condition_id'] for r in chosen})!=len(chosen):raise ValueError('Unique new conditions required')
    data=read(OUT/'data/manifest.json');train=data.get('train',data.get('rows',[]))
    train_ids={r['condition_id'] for r in train};train_parents={r['parent_id'] for r in train}
    oldids={q['condition_id'] for q in previous['requests']}
    for q in previous['requests']:
        if q['eval_group']=='DEV' and q['parent_id'] in train_parents:raise ValueError('Old DEV parent leaked into expanded training')
        if q['eval_group']=='CURRENT_TRAIN' and q['condition_id'] not in train_ids:raise ValueError('Old qualified reference parent missing from replay')
    for r in chosen:
        if r['condition_id'] in oldids:raise ValueError('New condition repeats inherited input')
        if r.get('full_seat_review',r.get('review_status')) not in ('ACCEPT','PASS'):raise ValueError('New full-seat input not reviewed')
        group=r['eval_group']
        if group not in ('NEW_TRAIN','CURRENT_TRAIN','NEW_PARENT_HOLDOUT','NEW_DEV','DEV','SAME_PARENT_HOLDOUT'):
            raise ValueError('Unregistered new-condition evaluation membership')
        if group in ('NEW_TRAIN','CURRENT_TRAIN') and r['condition_id'] not in train_ids:raise ValueError('New training condition absent from pool')
        if group in ('NEW_PARENT_HOLDOUT','NEW_DEV','DEV') and r['parent_id'] in train_parents:raise ValueError('Heldout parent leaked')
        if group=='SAME_PARENT_HOLDOUT' and (r['parent_id'] not in train_parents or r['condition_id'] in train_ids):raise ValueError('Same-parent holdout identity differs')
    if (OUT/'evaluation_inputs').exists():raise RuntimeError('Partial panel retained; audit before recovery')
    protected={};tick=time.perf_counter()
    for p in (selection_path,oldpath,OUT/'data/manifest.json',Path(__file__)):
        protect(protected,p)
    protected.update(previous['protected'])
    # Existing score/output evidence is referenced; no copies, re-scoring or redraw of noise.
    rows=[dict(q,panel_origin='INHERITED24',baseline_reuse=True) for q in previous['requests']]
    conditions=[dict(c,panel_origin='INHERITED24') for c in previous['conditions']]
    for source in chosen:
        row=dict(source, split=source.get('split',source['eval_group']),part_type=source.get('part_type','SEAT_SURFACE'),
                 panel_origin='NEW_COVERAGE',baseline_reuse=False,
                 planned_current_training_C=source['condition_id'] in train_ids)
        base=new_bundle(row,protected)
        conditions.append({k:base[k] for k in ('condition_id','parent_id','N','K','boundary_edges','closed_C','part_type','eval_group','gallery_bounds','gallery_depth_bounds','panel_origin')})
        for ni in (0,1):
            cid=row['condition_id'];rid=cid+'__n'+str(ni);s=seed(cid,ni)
            z=np.random.RandomState(s).standard_normal((row['N']-row['K'],3,3)).astype(np.float32)
            path=OUT/'evaluation_inputs/requests'/rid/'noise.npz'
            P.array_file(path,dict(z=z,seed=np.asarray(s,np.int64)));h=protect(protected,path)
            rows.append(dict(base,request_id=rid,noise_index=ni,seed=s,noise_npz=str(path),noise_sha256=h,noise_npz_sha256=h))
    nnew=2*len(chosen);total=24+nnew;new_outputs=24+2*nnew
    manifest=dict(schema=VERSION,status='FROZEN',requests=rows,conditions=conditions,labels=list(LABELS),
        stages=['RAW','BASE_POST','CLEAN_FINAL_POST'],protected=protected,
        requests_per_label={LABELS[0]:total,LABELS[1]:total,LABELS[2]:24},
        new_requests_per_label={LABELS[0]:nnew,LABELS[1]:total,LABELS[2]:0},
        new_noise_arrays=nnew,new_query_arrays=len(chosen),new_RAW=new_outputs,new_D0=new_outputs,
        generator_forwards=50*new_outputs,new_scoring_rows=3*new_outputs,reused_scoring_rows=144,
        total_scoring_rows=3*(2*total+24),C_query_count=2048,L=L,
        group_counts=dict(collections.Counter(q['eval_group'] for q in rows)),
        inference_policy='Native baseline and candidate are all-free, no head; SEAM old24 is read-only reference only',
        maximum_head_forwards=0,all_RAW_before_D0=True,no_unique_GT_style_success=True,
        old_manifest=str(oldpath),old_manifest_sha256=sha(oldpath),
        metric_contract='Original177 numeric fields; preserve current/free denominators and original NA. Parent-equally aggregated; no unique-source reconstruction success.',
        gallery_policy='Old24 exact inherited view/section frame; new C source-fixed before generation; three columns old, two new',
        model_calls=0,head_calls=0,scoring_calls=0)
    P.immutable(OUT/'eval_manifest.json',manifest)
    checks=validate_assets();P.immutable(OUT/'eval_cpu_checks.json',checks)
    if 'torch' in sys.modules:raise RuntimeError('Panel preparation imported neural framework')
    verify(protected)
    result=dict(status='COMPLETE_CPU_PREPARATION',conditions=len(conditions),requests=total,
        new_RAW=new_outputs,reused_RAW=48,model_calls=0,GPU_calls=0,scoring_calls=0,
        seconds=time.perf_counter()-tick,files={str(p):sha(p) for p in (OUT/'eval_manifest.json',OUT/'eval_cpu_checks.json')})
    P.immutable(done,result);return result

def requests(label=None):
    if label is not None and label not in LABELS:raise ValueError('Unknown label')
    m=read(OUT/'eval_manifest.json');verify(m['protected']);rows=m['requests']
    if len({q['request_id'] for q in rows})!=len(rows):raise ValueError('Duplicate request')
    if set(collections.Counter(q['condition_id'] for q in rows).values())!={2}:raise ValueError('Two fixed noises per C required')
    return [q for q in rows if label is None or label in labels_for(q)]

def validate_assets():
    from meshflow_control.source_winding import validate_source
    checks=[];inherited={q['request_id']:q for q in read(SOURCE_ROOT/'eval_manifest.json')['requests']}
    for q in requests():
        if q['panel_origin']=='INHERITED24':
            if any(q.get(k)!=v for k,v in inherited[q['request_id']].items() if k!='baseline_reuse'):
                raise ValueError('Inherited24 request arrays, metadata or framing changed')
        for key in ('inference_npz','metadata_npz','source_npz','noise_npz'):
            if sha(q[key])!=q[key+'_sha256']:raise ValueError('Changed input '+key)
        a=arrays(q['inference_npz']);z=arrays(q['noise_npz'])['z']
        if set(a)!={'C_vertices','C_faces_original','C_faces_model','N','K'}:raise ValueError('Non-C model information')
        if z.dtype!=np.float32 or z.shape!=(q['N']-q['K'],3,3) or not np.isfinite(z).all():raise ValueError('Bad noise')
        validate_source(arrays(q['source_npz']),a['C_vertices'][a['C_faces_model']])
        d=q['query_descriptor'];bank=arrays(d['path'])
        if sha(d['path'])!=d['sha256'] or len(bank['C_points'])!=2048 or not np.isclose(bank['C_weights'].sum(),1):raise ValueError('Query differs')
        checks.append(dict(request_id=q['request_id'],origin=q['panel_origin'],source_identity=True,noise_identity=True,query_identity=True))
    return dict(status='PASS',requests=len(checks),inherited_requests=24,checks=checks,model_calls=0,geometry_scores=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','verify'));p.add_argument('--selection');a=p.parse_args()
    print(prepare(a.selection) if a.action=='prepare' else validate_assets())
