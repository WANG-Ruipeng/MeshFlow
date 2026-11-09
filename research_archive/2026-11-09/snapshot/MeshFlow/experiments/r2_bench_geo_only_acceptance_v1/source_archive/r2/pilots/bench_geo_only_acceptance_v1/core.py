"""Coverage 30-request GEO_ONLY acceptance: 26 CPU operations, four reuses.

All input states/checkpoints/noises are frozen historical assets. This module
imports no torch, runs no D0, and never calls HINGE/ALPHA/HOLE or a scorer.
Resume uses checked receipts; unresolved task directories stop without replay.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'r2/src'))
from r2.pilots.bench_backend_counterfactual_v1 import core as B
OUT=ROOT/'experiments/r2_bench_geo_only_acceptance_v1'
COVERAGE=ROOT/'experiments/r2_bench_seat_coverage_v1'
CURRICULUM=ROOT/'experiments/r2_bench_full_seat_curriculum_v1'
BACKEND=ROOT/'experiments/r2_bench_backend_counterfactual_v1'
VERSION='bench_geo_only_acceptance_v1'
LABEL='COVERAGE_FINAL'
sha,read,arrays,write,npz,verify,csv_write=B.sha,B.read,B.arrays,B.write,B.npz,B.verify,B.csv_write

def log(event,**values):
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(B.clean(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**values)),ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())

def paths(root,label,rid):
    raw=root/'evaluation'/label/rid/'raw';post=root/'evaluation'/label/rid/'post'
    cpu=root/'cpu'/label/rid
    return dict(raw_path=str(raw/'raw.npz'),proposal_path=str(post/'proposal.npz'),pre_geo_path=str(post/'pre_geo.npz'),
        base_post_path=str(post/'post.npz'),base_canonical_path=str(post/'post_canonical.npz'),
        final_post_path=str(cpu/'CLEAN_FINAL_POST/post.npz'),geometry_path=str(post/'geometry.json'),
        raw_receipt=str(raw/'receipt.json'),post_receipt=str(post/'receipt.json'),proposal_receipt=str(post/'proposal_receipt.json'),
        raw_score_path=str(root/'evaluation/scores'/label/rid/'RAW/score.json'),
        base_score_path=str(cpu/'BASE_POST/scores/score.json'),final_score_path=str(cpu/'CLEAN_FINAL_POST/scores/score.json'))

def prepare():
    destination=OUT/'manifest.json'
    if destination.exists():
        m=read(destination);verify(m['protected']);verify(m['source_hashes']);return m
    P,geo,io,code=B.dependencies();code[str(Path(__file__))]=sha(__file__)
    original=read(COVERAGE/'eval_manifest.json');qs=sorted(original['requests'],key=lambda q:(q['eval_group'],q['parent_id'],q['condition_id'],q['noise_index']))
    assert len(qs)==30 and Counter(q['eval_group'] for q in qs)=={'CURRENT_TRAIN':16,'DEV':8,'NEW_TRAIN':4,'NEW_PARENT_HOLDOUT':2}
    old=read(BACKEND/'manifest.json');oldcore=read(BACKEND/'core_complete.json');assert oldcore['status']=='COMPLETE'
    reuse={t['request_id']:t for t in old['tasks'] if t['kind']=='RAW_GEO' and t['label']==LABEL};assert len(reuse)==4
    protected={str(p):sha(p) for p in (COVERAGE/'eval_manifest.json',BACKEND/'manifest.json',BACKEND/'core_complete.json')}
    tasks=[]
    for q in qs:
        rid=q['request_id'];job=LABEL+'__'+rid+'__GEO_RAW';base=paths(COVERAGE,LABEL,rid)
        nr=CURRICULUM if q['panel_origin']=='INHERITED24' else COVERAGE
        native=paths(nr,'NATIVE_S6000',rid)
        source=COVERAGE/'evaluation'/LABEL/rid/'raw/source_condition.npz'
        for p in list(base.values())+list(native.values())+[str(source),q['case_path'],q['query_descriptor']['path'],q['inference_npz'],q['noise_npz']]:
            if not Path(p).is_file():raise FileNotFoundError(p)
            protected[p]=sha(p)
        assert sha(q['query_descriptor']['path'])==q['query_descriptor']['sha256']
        assert sha(q['case_path'])==q['case_sha256']
        reused=rid in reuse
        destination_dir=Path(reuse[rid]['output_dir']) if reused else OUT/'outputs'/job
        task=dict(job_id=job,kind='RAW_GEO',label=LABEL,request_id=rid,condition_id=q['condition_id'],
            parent_id=q['parent_id'],N=q['N'],K=q['K'],eval_group=q['eval_group'],noise_index=q['noise_index'],request=q,
            case_path=q['case_path'],source_condition_path=str(source),original=base,native=native,
            native_label='NATIVE_S6000',native_root=str(nr),reuse_geometry=reused,output_dir=str(destination_dir),
            new_raw_path=base['raw_path'],new_pre_geo_path=str(destination_dir/'pre_geo.npz'),
            new_post_path=str(destination_dir/'post.npz'),new_geometry_path=str(destination_dir/'geometry.json'),
            geometry_receipt_path=str(destination_dir/'receipt.json'))
        if reused:
            ot=reuse[rid];assert job==ot['job_id'] and ot['original']['raw_path']==base['raw_path']
            receipt=read(destination_dir/'receipt.json');verify(receipt['files'])
            oldscore=BACKEND/'scores'/job
            score_receipt=read(oldscore/'complete.json');verify(score_receipt['files'])
            task.update(old_subset_job_id=job,old_subset_score_path=str(oldscore/'score.json'),
                old_subset_score_receipt_path=str(oldscore/'complete.json'),reuse_manifest_sha256=sha(BACKEND/'manifest.json'))
            for p in list(receipt['files'])+[str(destination_dir/'receipt.json')]+list(score_receipt['files'])+[str(oldscore/'complete.json')]:protected[p]=sha(p)
        tasks.append(task)
    assert sum(t['reuse_geometry'] for t in tasks)==4
    m=dict(schema=VERSION,status='FROZEN',utc=datetime.now(timezone.utc).isoformat(),
        budget=dict(total_GEO_ONLY=30,new_GEO=26,reused_GEO=4,new_source_orientation=26,model_calls=0,training=0,D0=0,new_noise=0,OT=0),
        group_counts=dict(Counter(q['eval_group'] for q in qs)),L=B.L,radius_model=geo.RADIUS,geo_config=geo.FIXED_CONFIG,
        method='COVERAGE_FINAL fixed RAW -> unchanged GEO_C -> source winding; no D0/HINGE/ALPHA/HOLE',
        source_hashes=code,protected=protected,tasks=tasks,
        ordering='eval_group lexical, parent_id, condition_id, numeric existing noise_index',
        comparison='paired frozen request; Native references use old curriculum for inherited24 and coverage root for new6',
        conditional_source_D0='NOT_EXECUTED_BY_THIS_MODULE; root owns any separately gated two-source forward')
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/'manifest.json',m);log('REGISTERED',budget=m['budget']);return m

def one(task,manifest_sha):
    if task['reuse_geometry']:
        receipt=read(task['geometry_receipt_path']);verify(receipt['files'])
        assert receipt['GEO_calls']==1 and receipt['model_calls']==receipt['D0_forwards']==0
        return dict(status='REUSED',job_id=task['job_id'],new_GEO_calls=0,geometry_receipt_path=task['geometry_receipt_path'])
    dest=Path(task['output_dir']);complete=dest/'receipt.json'
    if complete.exists():
        r=read(complete);assert r['manifest_sha256']==manifest_sha;verify(r['files']);return r
    if dest.exists():raise RuntimeError('Unresolved CPU geometry attempt; no silent retry: '+str(dest))
    P,geo,io,_=B.dependencies();dest.mkdir(parents=True)
    write(dest/'attempt.json',dict(job_id=task['job_id'],manifest_sha256=manifest_sha,GEO_budget=1,model_calls=0))
    count=0;start=time.perf_counter()
    try:
        raw=arrays(task['new_raw_path']);source=arrays(task['source_condition_path']);n,k=task['N'],task['K']
        assert len(raw['raw'])==n and int(raw['known_faces'])==k
        assert not raw['raw_seam_slot_mask'].any(),'Native-only candidate expected'
        before=hashlib.sha256(raw['raw'].tobytes()).hexdigest();log('GEO_STARTED',job_id=task['job_id'])
        pre,canonical,post,g,orient,anchors=P.geometry_from_proposal(raw,raw['raw'].copy(),source);count=1
        assert np.array_equal(pre,raw['raw']) and hashlib.sha256(raw['raw'].tobytes()).hexdigest()==before
        assert g['stats']['radius_model']==geo.RADIUS
        log('GEO_RETURNED',job_id=task['job_id'],calls=1,deleted=g['stats']['deleted_free_faces'])
        npz(dest/'pre_geo.npz',dict(output=pre,C=raw['raw'][:k],known_faces=raw['known_faces'],raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
        npz(dest/'post_canonical.npz',canonical);npz(dest/'post.npz',post);io.write_indexed_ply(dest/'post.ply',post['vertices'],post['faces'])
        write(dest/'geometry.json',dict(stats=g['stats'],face_records=g['face_records']))
        write(dest/'anchor_identity.json',anchors);write(dest/'source_restoration.json',orient['restoration_summary'])
        od=orient['orientation_result'];write(dest/'orientation.json',dict(summary=od['summary'],components=od['component_rows'],edges=od['edge_rows']))
        oldpre=arrays(task['original']['pre_geo_path'])['output'];oldcan=arrays(task['original']['base_canonical_path'])
        proposal=arrays(task['original']['proposal_path'])['official_unconstrained_proposal']
        assert np.array_equal(oldpre[k:],proposal[k:]) and np.array_equal(oldpre[:k],raw['raw'][:k])
        ids=np.arange(n);stages=[('RAW',raw['raw'],ids,oldcan),('D0_PROPOSAL',proposal,ids,oldcan),
            ('RESTORED_C',oldpre,ids,oldcan),('D0_GEO_MERGED',B.merged(oldpre,oldcan),ids,oldcan),
            ('BASE_POST',oldcan['output'],oldcan['source_face_ids'],oldcan),
            ('RAW_GEO_MERGED',B.merged(pre,canonical),ids,canonical),('RAW_GEO_POST',canonical['output'],canonical['source_face_ids'],canonical)]
        faces=[];summaries=[]
        for name,soup,slots,can in stages:
            rr,ss=B.stage_records(task,name,soup,slots,can['source_face_kept'],can['source_face_reason'])
            for row in rr:row['eval_group']=task['eval_group']
            ss['eval_group']=task['eval_group'];faces.extend(rr);summaries.append(ss)
        csv_write(dest/'per_face_stages.csv',faces);csv_write(dest/'stage_summary.csv',summaries)
        assert 'torch' not in sys.modules
        receipt=dict(status='COMPLETE',job_id=task['job_id'],label=LABEL,request_id=task['request_id'],
            manifest_sha256=manifest_sha,new_GEO_calls=1,GEO_calls=1,source_orientation_calls=1,model_calls=0,D0_forwards=0,
            seconds=time.perf_counter()-start,post_sha256=sha(dest/'post.npz'),
            files={str(p):sha(p) for p in dest.iterdir() if p.is_file()})
        write(complete,receipt);return receipt
    except BaseException as error:
        write(dest/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),returned_GEO_calls=count,no_silent_retry=True));raise

def run():
    m=prepare();complete=OUT/'core_complete.json'
    if complete.exists():r=read(complete);verify(r['files']);return r
    if not list((OUT/'resources').glob('snapshot_*.json')):raise RuntimeError('Root resource GO is required')
    tick=time.perf_counter();results=[one(t,sha(OUT/'manifest.json')) for t in m['tasks']]
    assert len(results)==30 and sum(r['new_GEO_calls'] for r in results)==26
    verify(m['protected']);verify(m['source_hashes']);faces=[];stages=[]
    for task in m['tasks']:
        for fname,store in [('per_face_stages.csv',faces),('stage_summary.csv',stages)]:
            with (Path(task['output_dir'])/fname).open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):row['eval_group']=task['eval_group'];row['geometry_reused']=task['reuse_geometry'];store.append(row)
    csv_write(OUT/'per_face_stages.csv',faces);csv_write(OUT/'stage_summary.csv',stages)
    files={t['geometry_receipt_path']:sha(t['geometry_receipt_path']) for t in m['tasks']}
    files.update({str(OUT/p):sha(OUT/p) for p in ('manifest.json','per_face_stages.csv','stage_summary.csv')})
    value=dict(status='COMPLETE',total=30,new_GEO=26,reused_GEO=4,new_source_orientation=26,model_calls=0,D0=0,
        training=0,OT=0,new_noise=0,CPU_workers=1,CPU_threads=1,seconds=time.perf_counter()-tick,
        face_stage_rows=len(faces),stage_rows=len(stages),files=files)
    write(complete,value);log('COMPLETE',**{k:v for k,v in value.items() if k!='files'});return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run'));a=p.parse_args()
    r=prepare() if a.command=='prepare' else run();print(json.dumps({k:v for k,v in r.items() if k not in ('files','protected','tasks','source_hashes')},ensure_ascii=False))
