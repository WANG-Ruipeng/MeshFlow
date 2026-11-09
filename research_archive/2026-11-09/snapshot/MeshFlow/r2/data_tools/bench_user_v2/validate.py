"""Independent CPU acceptance of reviewed Bench v2 data and production SEAM packing.

One deterministic CPU OT sample per final condition. Completed condition receipts
are verified on resume, never silently sampled again. No model is instantiated.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'r2/src'))
import numpy as np
from r2.pilots.bench_parts_convergence_v1 import data as production
from r2.pilots.seam_abc_v1 import slot_data_extension
from r2.pilots.bench_coverage_expansion_v1.data import scaled_case

DEFAULT_OUT = ROOT/'experiments/r2_bench_user_data_v2/validation'
DEFAULT_DEV = ROOT/'experiments/r2_bench_coverage_expansion_v1/data/manifest.json'
SCHEMA = 'BENCH_USER_V2_CPU_DATA_ACCEPTANCE_V1'
SEED_DOMAIN = 'BENCH_USER_V2_VALIDATION_ONLY_NOT_TRAINING_NOISE_V1'


def resolve(p):
    p = str(p)
    if os.name == 'nt' and p.startswith('/mnt/'):
        p = p[5].upper()+':/'+p[7:]
    elif os.name != 'nt' and len(p)>2 and p[1]==':':
        p = '/mnt/'+p[0].lower()+p[2:].replace('\\','/')
    p = Path(p)
    return p if p.is_absolute() else ROOT/p


def read(p):
    return json.loads(resolve(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with resolve(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20), b''): h.update(block)
    return h.hexdigest()


def json_bytes(v):
    return (json.dumps(v, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def write_new(p, v):
    p=resolve(p); p.parent.mkdir(parents=True,exist_ok=True); data=json_bytes(v)
    if p.exists():
        if p.read_bytes()!=data: raise RuntimeError('Immutable existing output differs: '+str(p))
    else:
        with p.open('xb') as f:f.write(data)


def arrays(p):
    with np.load(resolve(p), allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def ahash(a):
    a=np.asarray(a)
    return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def equal(a,b,name):
    if not np.array_equal(a,b):raise AssertionError(name)


def scalar(a):
    return np.asarray(a).item()


def path_hash(row,key,required=True):
    if key not in row:
        if required:raise AssertionError('Missing path field: '+key)
        return None
    hk=key[:-5]+'_sha256' if key.endswith('_path') else key+'_sha256'
    expected=row.get(hk)
    if not expected:raise AssertionError('Missing expected hash: '+hk)
    actual=sha(row[key])
    if actual!=expected:raise AssertionError('Protected file hash differs: '+str(row[key]))
    return str(resolve(row[key])),actual


def resources():
    import psutil
    def command(args):
        try:
            p=subprocess.run(args,capture_output=True,text=True,timeout=15)
            return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
        except Exception as e:return dict(status='NA',reason=str(e))
    return dict(cpu_logical=psutil.cpu_count(),available_RAM_bytes=psutil.virtual_memory().available,
        numeric_threads=1,CPU_workers=1,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_read_only_inventory=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        gpu_read_only_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']))


def code_hashes():
    paths=[Path(__file__),Path(production.__file__),Path(slot_data_extension.__file__),
           ROOT/'r2/pilots/bench_coverage_expansion_v1/data.py',
           ROOT/'r2/src/meshflow_control/seam/slot_data.py',ROOT/'r2/src/meshflow_control/training/coupling.py']
    return {str(p):sha(p) for p in paths}


def input_checks(case,inp,row,parent):
    required=('full_target','pre_ot_full','source_faces','model_vertices','source_face_vertex_ids',
        'source_corner_permutations','C','Y','known_ids','free_ids','C_vertices','C_faces_original',
        'C_faces_model','C_source_vertex_ids','C_source_corner_permutations','N','K','occupied_labels',
        'edge_owner_face','edge_source_triple','source_alias_labels','source_supported_labels','edge_records','target_version_id')
    for k in required:
        if k not in case:raise AssertionError('Missing case field: '+k)
    n,k=int(scalar(case['N'])),int(scalar(case['K']))
    if not 128<=n<=256 or not 2<=k<n:raise AssertionError('N/K window')
    full=np.asarray(case['full_target']);pre=np.asarray(case['pre_ot_full'])
    v=case['model_vertices'];f=case['source_faces'];canonical=case['source_face_vertex_ids'];p=case['source_corner_permutations']
    parent_mesh=arrays(parent['target_mesh_path'])
    if set(parent_mesh)!={'vertices','faces'}:raise AssertionError('Parent target mesh NPZ schema differs')
    equal(v,parent_mesh['vertices'],'Case vs parent target vertices')
    equal(f,parent_mesh['faces'],'Case vs parent target faces')
    if parent.get('target_case_path'):
        parent_case=arrays(parent['target_case_path'])
        for name in ('source_vertices','source_faces','normalized_vertices','model_vertices','full_target',
                     'pre_ot_full','source_face_vertex_ids','source_corner_permutations','transform_json','target_version_id'):
            if name not in case or name not in parent_case:raise AssertionError('Parent canonical field missing: '+name)
            equal(case[name],parent_case[name],'Case vs parent canonical field: '+name)
    known=case['known_ids'];free=case['free_ids']
    if full.shape!=(n,3,3) or len(known)!=k or len(free)!=n-k:raise AssertionError('Target/partition dimensions')
    if full.dtype!=np.float32 or pre.dtype!=np.float32:raise AssertionError('Production target precision is not FP32')
    if f.shape!=(n,3) or canonical.shape!=(n,3) or p.shape!=(n,3):raise AssertionError('Source identity dimensions')
    for name,a in [('f',f),('canonical',canonical),('p',p),('known',known),('free',free)]:
        if a.dtype.kind not in 'iu':raise AssertionError('Integer identities required: '+name)
    if not np.isfinite(full).all() or not np.isfinite(v).all():raise AssertionError('Nonfinite target')
    if f.min()<0 or f.max()>=len(v):raise AssertionError('Source vertex index range')
    equal(np.sort(np.r_[known,free]),np.arange(n),'C/Y exact disjoint exhaustive face-ID partition')
    equal(np.sort(p,axis=1),np.tile(np.arange(3),(n,1)),'Canonical permutation')
    equal(np.take_along_axis(f,p,axis=1),canonical,'Canonical source IDs')
    equal(v[canonical],full,'Canonical model coordinates')
    equal(pre*np.float32(2),full,'PreOT half-scale')
    equal(np.take_along_axis(full,np.argsort(p,axis=1)[:,:,None],axis=1),v[f],'Source winding inverse')
    equal(case['C'],full[known],'C exact subset');equal(case['Y'],full[free],'Y exact subset')
    rebuilt=np.empty_like(full);rebuilt[known]=case['C'];rebuilt[free]=case['Y']
    equal(rebuilt,full,'C+Y exact reconstruction')
    if 'target_face_ids' not in row:raise AssertionError('Missing manifest target_face_ids')
    equal(known,np.asarray(row['target_face_ids'],np.int64),'Manifest C target-face IDs')
    equal(case['C_vertices'],v[case['C_source_vertex_ids']],'C vertex identities')
    equal(case['C_vertices'][case['C_faces_model']],case['C'],'C canonical coordinates')
    equal(case['C_vertices'][case['C_faces_original']],v[f[known]],'C original coordinates')
    equal(case['C_source_corner_permutations'],p[known],'C source corner permutations')
    equal(np.take_along_axis(case['C_faces_original'],case['C_source_corner_permutations'],axis=1),case['C_faces_model'],'C reversible indexed corners')
    if 'C_source_faces_original' in case:equal(case['C_source_faces_original'],f[known],'C source original triples')
    if 'source_face_ids' in case:equal(case['source_face_ids'],known,'C source-face domain')
    if 'free_source_face_ids' in case:equal(case['free_source_face_ids'],free,'Y source-face domain')
    version=str(scalar(case['target_version_id']))
    if row.get('target_version_id',version)!=version or parent.get('target_version_id',version)!=version:raise AssertionError('Target version identity differs')
    # Exact zero-area is an invalid effective target, distinct from tiny nonzero faces.
    tri=v[f].astype(np.float64)
    zero=np.flatnonzero(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0)
    if len(zero):raise AssertionError('Effective target has exact zero-area faces: '+str(zero.tolist()))
    edges,features=production.edge_features(case['C_vertices'],case['C_faces_original'],n)
    equal(edges,case['edge_records'],'C-only deterministic boundary order')
    if 'edge_features' in case:equal(features,case['edge_features'],'C-only features')
    b=len(edges)
    if b>n-k:raise AssertionError('Boundary exceeds reserved free-slot budget')
    own=np.asarray(case['occupied_labels']);alias=np.asarray(case['source_alias_labels']);supported=np.asarray(case['source_supported_labels'])
    if any(a.dtype!=np.bool_ or a.shape!=(b,) for a in (own,alias,supported)):raise AssertionError('Source label dimensions/types')
    if np.any(own&alias):raise AssertionError('Owner/alias overlap')
    equal(own|alias,supported,'Owner+alias source supported partition')
    owner=case['edge_owner_face'];triples=case['edge_source_triple']
    if owner.shape!=(b,) or triples.shape!=(b,3):raise AssertionError('Owner/triple dimensions')
    active=owner[own]
    if len(np.unique(active))!=len(active) or not np.isin(active,free).all():raise AssertionError('Owner faces repeat or include C')
    if not np.isin(owner[supported],free).all():raise AssertionError('Supported source faces outside Y')
    if np.any(owner[~supported]!=-1):raise AssertionError('Null source-owner sentinel')
    bulk=np.setdiff1d(free,active)
    equal(np.sort(np.r_[active,bulk]),np.sort(free),'Owner/bulk exact Y partition')
    for ei in np.flatnonzero(supported):
        fi=int(owner[ei]);_,u,w=edges[ei];vid=case['C_source_vertex_ids'];su,sw=int(vid[u]),int(vid[w])
        if not np.array_equal(triples[ei,:2],[sw,su]):raise AssertionError('Owner endpoints not opposite C')
        equal(np.sort(triples[ei]),np.sort(f[fi]),'Owner triangle source IDs')
        linked=np.flatnonzero(supported&(owner==fi))
        if not own[int(linked.min())] or int(own[linked].sum())!=1:raise AssertionError('Lowest-edge unique owner policy')
    forbidden={'Y','full_target','pre_ot_full','free_ids','free_source_face_ids','occupied_labels','edge_owner_face','edge_source_triple','source_alias_labels','source_supported_labels','source_faces','model_vertices'}
    if forbidden.intersection(inp):raise AssertionError('Source target leaked into inference-only file: '+str(forbidden.intersection(inp)))
    for name in ('C','C_vertices','C_faces_original','C_faces_model','C_source_vertex_ids','C_source_corner_permutations','N','K'):
        if name not in inp:raise AssertionError('Missing inference field: '+name)
        equal(inp[name],case[name],'Inference C identity: '+name)
    return dict(N=n,K=k,boundary_edges=b,owners=int(own.sum()),aliases=int(alias.sum()),
        null_edges=int((~supported).sum()),bulk_faces=len(bulk),closed_C=b==0,open_no_owner=b>0 and not own.any(),
        target_version_id=version,expected_free_scalars=9*(n-k)-6*int(own.sum()),effective_zero_area_faces=0)


def augmentation_checks(case):
    """One deterministic algebra-only check; no second OT or training sample."""
    scale=np.asarray([.92,1.08,1.04],np.float32)
    before={k:ahash(v) for k,v in case.items()}
    scaled=scaled_case(case,scale)
    for key in ('full_target','pre_ot_full','C_vertices'):
        equal(scaled[key],case[key]*scale,'Production joint XYZ scaling: '+key)
    for key in case:
        if key not in ('full_target','pre_ot_full','C_vertices'):
            equal(scaled[key],case[key],'Augmentation preserved identity/metadata: '+key)
    equal(scaled['pre_ot_full']*np.float32(2),scaled['full_target'],'Scaled preOT x2 contract')
    equal(scaled['C_vertices'][scaled['C_faces_model']],scaled['full_target'][scaled['known_ids']],'Scaled C/complete target same coordinates')
    # Static C/Y metadata in this training copy is not read by the production
    # sampler. Effective augmented C/Y are selected from scaled full_target.
    assembled=np.empty_like(scaled['full_target'])
    for ids in (scaled['known_ids'],scaled['free_ids']):assembled[ids]=scaled['full_target'][ids]
    equal(assembled,scaled['full_target'],'Augmented target partition')
    if before!={k:ahash(v) for k,v in case.items()}:raise AssertionError('Augmentation mutated original case')
    return dict(status='PASS',kind='STATIC_ALGEBRA_ONLY_NO_OT',scale_xyz=scale.tolist(),
        production_function='bench_coverage_expansion_v1.data.scaled_case',
        fixed_probe_not_a_training_schedule=True,original_case_unchanged=True,
        effective_target_fields=['full_target','pre_ot_full','C_vertices'],
        static_C_Y_metadata='Unscaled cached C/Y are not sampled; effective augmented subsets come from scaled full_target')


def sample_checks(case,s,meta,z,eps):
    n,k=meta['N'],meta['K'];known=np.asarray(s['known_mask']);valid=np.asarray(s['valid_mask']);anchor=s['anchor_ids']
    if valid.dtype!=np.bool_ or valid.shape!=(n,) or not valid.all():raise AssertionError('Effective case has padding or invalid targets')
    if known.dtype!=np.bool_ or known.shape!=(n,) or int(known.sum())!=k:raise AssertionError('Known mask')
    if anchor.shape!=(n,3):raise AssertionError('Anchor shape')
    equal(s['free_corner_mask'],anchor<0,'Free-corner mask')
    fixed=anchor>=0
    if not fixed[known].all():raise AssertionError('Known corners not all fixed')
    ids=s['source_face_ids'];corner=s['source_corner_permutations']
    equal(np.sort(ids),np.arange(n),'Effective sampled target every face exactly once')
    equal(np.sort(corner,axis=1),np.tile(np.arange(3),(n,1)),'Sample corner permutations')
    equal(case['full_target'][ids[:,None],corner],s['x1'],'Sample identity target reconstruction')
    equal(np.sort(ids[known]),np.sort(case['known_ids']),'Sample known source identities')
    seam=~known&fixed.any(1);bulk=~known&~fixed.any(1)
    if int(seam.sum())!=meta['owners'] or int(bulk.sum())!=meta['bulk_faces']:raise AssertionError('Sample role counts')
    if not np.all(fixed[seam]==np.array([True,True,False])):raise AssertionError('SEAM fixed endpoints/free-tip contract')
    equal(np.sort(ids[seam]),np.sort(case['edge_owner_face'][case['occupied_labels']]),'Sample owner identities')
    for key in ('x0','x1','xt','u'):
        if s[key].shape!=(n,3,3) or s[key].dtype!=np.float32 or not np.isfinite(s[key]).all():raise AssertionError('Sample float contract: '+key)
    equal(s['u'],s['x1']-s['x0'],'FM velocity')
    order=s['face_permutation'];equal(np.sort(order),np.arange(n),'Sample face permutation')
    raw_start=np.zeros_like(s['x0'])
    raw_start[k:]=np.float32(np.sqrt(.75))*z+np.float32(.5)*eps
    original_anchors=np.empty_like(anchor);original_anchors[order]=anchor
    original_fixed=original_anchors>=0
    raw_start[original_fixed]=s['C_vertices'][original_anchors[original_fixed]]
    equal(s['x0'],raw_start[order],'Exact OT-then-HYBRID0.25 noise/anchor start')
    equal(s['xt'],np.float32(.5)*s['x0']+np.float32(.5)*s['x1'],'t=.5 interpolation')
    equal(s['x0'][fixed],s['C_vertices'][anchor[fixed]],'Initial exact anchor')
    equal(s['xt'][fixed],s['x1'][fixed],'Fixed corners unchanged')
    if s['u'][fixed].any():raise AssertionError('Fixed corner velocity not zero')
    if int(scalar(s['free_coordinate_count']))!=meta['expected_free_scalars']:raise AssertionError('Free FM denominator')
    if 3*int((s['free_corner_mask']&valid[:,None]).sum())!=meta['expected_free_scalars']:raise AssertionError('Padding/free effective denominator')
    if int(scalar(s['production_OT_calls']))!=1:raise AssertionError('Expected one production bulk OT')
    return dict(effective_faces=n,padding_faces=0,supervised_free_scalars=meta['expected_free_scalars'],
        target_each_face_once=True,all_role_and_identity_checks=True,output_array_hashes={key:ahash(value) for key,value in s.items()})


def validate(manifest_path,out=DEFAULT_OUT):
    start=time.perf_counter();out=resolve(out);out.mkdir(parents=True,exist_ok=True)
    manifest_path=resolve(manifest_path);m=read(manifest_path);mh=sha(manifest_path);codes=code_hashes()
    ledger=out/'ledger.jsonl'
    def event(v):
        with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),**v),ensure_ascii=False,allow_nan=False)+'\n')
    runid=str(time.time_ns());event(dict(event='VERIFY_INVOCATION',run_id=runid,manifest=str(manifest_path),manifest_sha256=mh))
    summary=dict(schema=SCHEMA,manifest=str(manifest_path),manifest_sha256=mh,code_hashes=codes,
        new_model_imports=False,model_calls=0,GPU_compute_calls=0,training_updates=0,source_modifications=0,
        condition_results=[],new_CPU_noise_arrays=0,new_CPU_noise_scalars=0,new_CPU_OT_attempts=0,new_CPU_OT_completed=0,
        resumed_verify_only=0,workers=1,numeric_threads=1,resources_before=resources())
    try:
        train=m.get('train',[]);conditions=m.get('conditions',[]);parents=m.get('parents',[])
        if not train or len(train)>128 or len(parents)>32:raise AssertionError('Authorized nonempty <=32 parent / <=128 condition batch')
        parentmap={r['parent_id']:r for r in parents};condmap={r['condition_id']:r for r in conditions}
        if len(parentmap)!=len(parents) or len(condmap)!=len(conditions):raise AssertionError('Duplicate manifest identity')
        if len({r['condition_id'] for r in train})!=len(train):raise AssertionError('Duplicate final train condition')
        if any(r['condition_id'] not in condmap for r in train):raise AssertionError('Train condition not in conditions')
        devpath=resolve(m.get('dev_source_manifest_path',DEFAULT_DEV));dm=read(devpath)
        if m.get('dev_source_manifest_sha256') and sha(devpath)!=m['dev_source_manifest_sha256']:raise AssertionError('DEV source manifest hash')
        expected_dev=dm['dev'];dev=m.get('dev')
        if len(expected_dev)!=39 or dev!=expected_dev:raise AssertionError('Old39 DEV rows not exactly preserved')
        devparents={r['parent_id'] for r in dev};trainparents={r['parent_id'] for r in train}
        if trainparents&devparents or set(parentmap)&devparents:raise AssertionError('TRAIN/DEV source-parent leakage')
        protected=m.get('protected_source_hashes') or m.get('protected_hashes')
        if not isinstance(protected,dict) or not protected:raise AssertionError('Explicit protected_source_hashes path/hash map required')
        initial_hashes={str(manifest_path):mh,str(devpath):sha(devpath)}
        for p,h in protected.items():
            if not isinstance(h,str) or sha(p)!=h:raise AssertionError('Original protected source hash changed: '+str(p))
            initial_hashes[str(resolve(p))]=h
        for parent in parents:
            for key in ('target_mesh_path',):
                ph=path_hash(parent,key);initial_hashes[ph[0]]=ph[1]
            if parent.get('target_case_path'):
                ph=path_hash(parent,'target_case_path');initial_hashes[ph[0]]=ph[1]
        summary.update(train_conditions=len(train),train_parents=len(trainparents),dev_conditions=39,dev_parents=len(devparents),
            parent_condition_counts=dict(Counter(r['parent_id'] for r in train)))
        for row in sorted(train,key=lambda r:r['condition_id']):
            cid=row['condition_id'];receipt=out/'conditions'/(cid+'.json');seal=receipt.with_suffix('.sha256')
            attempt=out/'attempts'/(cid+'.json');ct=time.perf_counter()
            item=dict(condition_id=cid,parent_id=row['parent_id'],status='BLOCKED')
            try:
                if any(x in cid for x in ('/','\\','..')):raise AssertionError('Unsafe condition filename')
                if row['parent_id'] not in parentmap:raise AssertionError('Condition parent absent')
                parent=parentmap[row['parent_id']]
                parent_status=parent.get('status','')
                parent_review=parent.get('review_status') or parent.get('review',{}).get('decision')
                if not parent_status.startswith(('TARGET_GEOMETRY_PASS','PASS','ACCEPT')) or parent.get('reasons'):
                    raise AssertionError('Final train references a geometrically blocked parent')
                if parent_review not in ('PASS','ACCEPT','ACCEPTED'):
                    raise AssertionError('Final train parent lacks passed target visual-review status')
                if row.get('review_status') not in ('PASS','ACCEPT','ACCEPTED','ACTIVE_REVIEWED_SOURCE_CONDITION'):
                    raise AssertionError('Final condition lacks passed actual-review status')
                ih={}
                for key in ('case_path','input_path'):
                    ph=path_hash(row,key);ih[ph[0]]=ph[1];initial_hashes[ph[0]]=ph[1]
                ph=path_hash(parentmap[row['parent_id']],'target_mesh_path');ih[ph[0]]=ph[1]
                if parent.get('target_case_path'):
                    ph=path_hash(parent,'target_case_path');ih[ph[0]]=ph[1]
                if receipt.exists():
                    if not seal.exists() or seal.read_text('ascii').strip()!=sha(receipt):raise AssertionError('Receipt hash missing/differs')
                    old=read(receipt)
                    if old.get('manifest_sha256')!=mh or old.get('input_hashes')!=ih or old.get('code_hashes')!=codes:raise AssertionError('Frozen receipt inputs/code changed')
                    if old['status']!='PASS':raise AssertionError('Prior failed condition is not automatically retried')
                    item=dict(old,resume_action='VERIFY_ONLY_NO_OT');summary['resumed_verify_only']+=1
                    summary['condition_results'].append(item);continue
                if attempt.exists():raise AssertionError('Previous started attempt without completed receipt; explicit recovery needed')
                write_new(attempt,dict(schema=SCHEMA,condition_id=cid,manifest_sha256=mh,input_hashes=ih,code_hashes=codes,started_utc=datetime.now(timezone.utc).isoformat()))
                case=arrays(row['case_path']);inp=arrays(row['input_path'])
                meta=input_checks(case,inp,row,parentmap[row['parent_id']])
                aug=augmentation_checks(case)
                digest=hashlib.sha256((SEED_DOMAIN+'|'+cid).encode()).digest()
                seeds=[int.from_bytes(digest[i:i+4],'little') for i in (0,4,8)]
                z=np.random.RandomState(seeds[0]).standard_normal((meta['N']-meta['K'],3,3)).astype(np.float32)
                eps=np.random.RandomState(seeds[1]).standard_normal(z.shape).astype(np.float32)
                summary['new_CPU_noise_arrays']+=2;summary['new_CPU_noise_scalars']+=int(z.size+eps.size)
                event(dict(event='CPU_VALIDATION_NOISE',condition_id=cid,array_count=2,scalar_count=int(z.size+eps.size),seeds=seeds,not_training_noise=True))
                orig_a=production.nested_ot_map;orig_b=slot_data_extension.nested_ot_map
                def counted_ot(*args,**kwargs):
                    summary['new_CPU_OT_attempts']+=1;event(dict(event='CPU_PRODUCTION_OT_ENTER',condition_id=cid))
                    result=orig_a(*args,**kwargs)
                    summary['new_CPU_OT_completed']+=1;event(dict(event='CPU_PRODUCTION_OT_RETURN',condition_id=cid))
                    return result
                production.nested_ot_map=counted_ot;slot_data_extension.nested_ot_map=counted_ot
                try:s=production.supervised_sample(case,z,eps,.5,seeds[2],'SEAM')
                finally:production.nested_ot_map=orig_a;slot_data_extension.nested_ot_map=orig_b
                details=sample_checks(case,s,meta,z,eps)
                for p,h in ih.items():
                    if sha(p)!=h:raise AssertionError('Case/source changed during CPU acceptance')
                item=dict(schema=SCHEMA,condition_id=cid,parent_id=row['parent_id'],status='PASS',manifest_sha256=mh,
                    input_hashes=ih,code_hashes=codes,metadata=meta,validation=details,
                    augmentation_validation=aug,
                    noise_seed_domain=SEED_DOMAIN,seeds=seeds,noise_array_hashes=dict(z=ahash(z),epsilon=ahash(eps)),
                    t=.5,new_CPU_noise_arrays=2,new_CPU_OT_calls=1,model_calls=0,GPU_compute_calls=0,
                    wall_seconds=time.perf_counter()-ct,scope='Validation only; arrays not saved as inference or training noise')
                write_new(receipt,item);seal.write_text(sha(receipt)+'\n',encoding='ascii')
                event(dict(event='CONDITION_PASS',condition_id=cid,receipt=str(receipt),receipt_sha256=sha(receipt)))
            except Exception as exc:
                item.update(reason=str(exc),exception_type=type(exc).__name__,traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-ct)
                event(dict(event='CONDITION_BLOCKED',condition_id=cid,reason=str(exc)))
                # An independent failure receipt is preserved; never overwrite a prior PASS.
                failed=out/'failures'/(cid+'__'+runid+'.json');write_new(failed,item)
            summary['condition_results'].append(item)
        for p,h in initial_hashes.items():
            if sha(p)!=h:raise AssertionError('Final protected hash changed: '+p)
        summary['input_hashes_after']=initial_hashes
        if code_hashes()!=codes:raise AssertionError('Validation/production source changed during run')
        bad=[r for r in summary['condition_results'] if r['status']!='PASS']
        summary.update(status='BLOCKED' if bad else 'PASS',passed=len(train)-len(bad),blocked=len(bad),
            original_inputs_unchanged=True,all_parent_dev_boundaries=True)
    except Exception as exc:
        summary.update(status='BLOCKED',reason=str(exc),exception_type=type(exc).__name__,traceback=traceback.format_exc())
        event(dict(event='GLOBAL_BLOCKED',reason=str(exc)))
    summary['wall_seconds']=time.perf_counter()-start
    summary['torch_imported']='torch' in sys.modules
    summary['model_modules_loaded']=[name for name in sys.modules if name.startswith(('meshflow_control.models','meshflow_control.seam.model_adapter'))]
    if summary['model_modules_loaded']:
        summary['status']='BLOCKED';summary['reason']='Unexpected model module import'
    result=out/('run_'+runid+'.json');write_new(result,summary)
    pointer=dict(status=summary['status'],receipt=str(result),receipt_sha256=sha(result),manifest_sha256=mh)
    tmp=out/'latest.tmp.json';tmp.write_bytes(json_bytes(pointer));os.replace(tmp,out/'latest.json')
    event(dict(event='VERIFY_FINISHED',**pointer,new_CPU_OT_attempts=summary['new_CPU_OT_attempts'],new_CPU_OT_completed=summary['new_CPU_OT_completed']))
    print(json.dumps({k:summary[k] for k in ('status','wall_seconds','new_CPU_noise_arrays','new_CPU_OT_attempts','new_CPU_OT_completed','resumed_verify_only')},indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--out',default=str(DEFAULT_OUT))
    args=parser.parse_args();result=validate(args.manifest,args.out)
    raise SystemExit(0 if result['status']=='PASS' else 2)


if __name__=='__main__':main()
