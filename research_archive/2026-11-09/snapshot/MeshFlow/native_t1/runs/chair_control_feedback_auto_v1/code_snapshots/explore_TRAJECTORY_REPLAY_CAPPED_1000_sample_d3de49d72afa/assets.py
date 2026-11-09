"""Frozen CPU data registration and lossless model-coordinate mesh export.

Never reads this run's CONFIRM scores/images; no model, CUDA, optimizer or OT.
"""
from __future__ import annotations
from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'native_t1/runs/chair_control_feedback_auto_v1'
SOURCE = ROOT/'native_t1/runs/control_mvp48_v1'
CHAIR = ROOT/'native_t1/runs/chair_domain_control_v1'
HYBRID = ROOT/'native_t1/runs/chair_hybrid_coupling_v1'
BOUNDARY = ROOT/'native_t1/runs/chair_boundary_attachment_v1'
EXPERIMENT = 'chair_control_feedback_auto_v1'
GPU_LOCK = '/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'
START_SHA = 'd640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
BANK_KEYS = ('z', 'C', 'known_mask', 'valid_mask', 'N', 'K')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def freeze_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != data: raise ValueError('Frozen JSON differs: '+str(path))
        return
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}


def freeze_npz(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = arrays(path)
        if old.keys()!=payload.keys() or any(not bits(old[k],payload[k]) for k in old):
            raise ValueError('Frozen NPZ differs: '+str(path))
        return
    with path.open('xb') as f:
        np.savez_compressed(f, **payload); f.flush(); os.fsync(f.fileno())


def resolve(value, default=ROOT):
    s = str(value).replace('\\','/')
    for prefix in ('E:/MeshFlow/', 'e:/MeshFlow/', '/mnt/e/MeshFlow/'):
        if s.startswith(prefix): return ROOT/s[len(prefix):]
    p = Path(s)
    return p if p.is_absolute() else Path(default)/p


def noise_seed(pool, uid, task_id, candidate):
    message = json.dumps([EXPERIMENT,20261008,'evaluation_noise',pool,uid,task_id,int(candidate)], separators=(',',':'))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:4], 'little')


def make_bank(C, N, seed):
    C = np.asarray(C)
    assert C.dtype == np.float32 and C.shape[1:] == (3,3)
    k = len(C); n = int(N)
    assert 0 < k < n and np.isfinite(C).all()
    z = np.zeros((1,n,9), np.float32)
    z[0,k:] = np.random.RandomState(int(seed)).randn(n-k,3,3).astype(np.float32).reshape(n-k,9)
    return dict(z=z,C=C.copy(),known_mask=np.arange(n)<k,valid_mask=np.ones(n,bool),
                N=np.asarray(n,np.int64),K=np.asarray(k,np.int64))


def _condition(base):
    from experiments.mf_h1_local0.chair_boundary_attachment_v1.registration import _attach_topology
    from experiments.mf_h1_local0.control_mvp48.eval_scoring import prepare_reference_from_sample
    n,k = int(base['N']),int(base['K'])
    ids = np.asarray(base['source_face_ids'],np.int64)
    free = np.asarray(base['free_source_ids'],np.int64)
    full = np.asarray(base['full_target'],np.float32).reshape(n,3,3)
    payload = dict(C=full[ids].copy(),GT_full=full.copy(),GT_free=full[free].copy(),
        source_face_ids=ids.copy(),known_source_face_ids=ids.copy(),free_source_ids=free.copy(),
        original_known_source_face_ids=base['retained_to_original_face'][ids].copy(),
        original_free_source_face_ids=base['retained_to_original_face'][free].copy(),
        N=np.asarray(n,np.int64),K=np.asarray(k,np.int64),
        L0=np.asarray(base.get('bbox_L0',base['bbox_diagonal']),np.float64))
    meta = _attach_topology(payload,base,ids)
    reference = prepare_reference_from_sample(base)
    for key, target in [('coverage_points','legacy_query_points'),('pairwise_target_points','legacy_pairwise_points'),
                        ('gamma_points','legacy_gamma_points'),('gamma_weights','legacy_gamma_weights')]:
        payload[target] = reference['arrays'][key].copy()
    # Original source coordinates are archival evidence, never model inputs or
    # an approximately inverted world-space output.
    payload['C_world'] = base['source_vertices'][base['source_face_vertex_ids'][ids]].copy()
    payload['known_source_vertex_ids'] = base['source_face_vertex_ids'][ids].copy()
    payload['pre_ot_free'] = base['pre_ot_full'][free].copy()
    payload['source_corner_permutations'] = base['source_corner_permutations'].copy()
    payload['free_source_corner_permutations'] = base['source_corner_permutations'][free].copy()
    payload['transform_json'] = np.asarray(json.dumps(base['transform'],sort_keys=True))
    assert bits((payload['pre_ot_free']*np.float32(2)).reshape(-1,3,3),payload['GT_free'])
    return payload,meta


def prepare(out=OUT, feedback_parent_count=8, feedback_candidates=2):
    """Freeze 16 feedback, original 16 DEV and independent 32 CONFIRM inputs."""
    from experiments.mf_h1_local0.chair_domain_control_v1.prepare import load_task_sample, _api
    from experiments.mf_h1_local0.chair_boundary_attachment_v1.registration import _check_cache
    out = Path(out).resolve()
    if out != OUT.resolve(): raise ValueError('Only the isolated registered output is authorized')
    if (feedback_parent_count,feedback_candidates)!=(8,2): raise ValueError('Frozen 8 x 2 feedback pool')
    sources = {}
    def track(path, expected=None):
        path = resolve(path); actual = digest(path)
        if expected is not None and actual!=expected: raise ValueError('Source hash differs: '+str(path))
        sources[str(path)] = dict(sha256=actual,bytes=path.stat().st_size)
        return path
    manifest = read_json(track(CHAIR/'task_manifest.json'))
    panels = read_json(track(CHAIR/'panels.json'))
    data_manifest = read_json(track(SOURCE/'data/manifest.json'))
    track(SOURCE/'data/patches.json'); track(SOURCE/'data/load_sample.py')
    binding = read_json(track(SOURCE/'registration/train_data_binding.json'))
    parents = manifest['splits']; train = set(parents['train']); dev = set(parents['validation']); confirm = set(parents['test'])
    assert len(train)==32 and len(dev)==8 and len(confirm)==8
    assert not (train&dev or train&confirm or dev&confirm)
    assert set(binding['train_object_ids'])==train
    objects = {x['uid']:x for x in data_manifest['objects']}
    for uid in train|dev|confirm:
        obj=objects[uid]; track(SOURCE/obj['prepared_path'],obj['prepared_sha256'])
    rank = sorted(train,key=lambda u:(objects[u]['N'],u))
    feedback_parents = [min(rank[i:i+4]) for i in range(0,32,4)]
    train_tasks = [t for t in manifest['tasks'] if t['role']=='train' and t['status']=='READY']
    selected_train = [next(t for t in train_tasks if t['uid']==u and t['ratio']==.2) for u in feedback_parents]
    dev_rows = [x for x in panels['A'] if x['split']=='validation' and x['ratio']==.2 and x['alpha']==1.]
    assert len(dev_rows)==16 and {x['uid'] for x in dev_rows}==dev
    first_confirm = {}
    for x in panels['C']:
        first_confirm.setdefault(x['uid'],x)
    assert set(first_confirm)==confirm and len(first_confirm)==8
    cache_doc = read_json(track(HYBRID/'evaluation/OT_HYBRID/generated_rows.json'))
    caches = {r['case_id']:r for r in cache_doc['rows']}
    pools = {k:[] for k in ('TRAIN_FEEDBACK','DEV','CONFIRM')}; conditions = {}
    selections = [('TRAIN_FEEDBACK',t,i) for t in selected_train for i in range(2)]
    selections += [('DEV',r,int(r['noise_index'])) for r in dev_rows]
    selections += [('CONFIRM',r,i) for r in first_confirm.values() for i in range(4)]
    for pool, record, candidate in selections:
        uid = record['uid']; task_id = record['task_id']
        cid = uid+'_C1' if pool=='DEV' else uid+'_'+record['patch_id'] if pool=='CONFIRM' else task_id
        rid = pool+'__'+cid+'__n'+str(candidate)
        row = dict(id=rid,request_id=rid,split=pool,pool=pool,uid=uid,condition_id=cid,
                   candidate=candidate,noise_index=candidate,source_task_id=task_id,
                   source_case_id=record.get('case_id'),N=int(record['N']),K=int(record['K']),
                   alpha=1.,status='READY',raw_cache=None)
        try:
            if cid not in conditions:
                base = (_api(str(SOURCE)).load_sample(uid,record['patch_id'],SOURCE/'data') if pool=='CONFIRM'
                        else load_task_sample(SOURCE,manifest,task_id,1.))
                payload,meta = _condition(base)
                if pool!='TRAIN_FEEDBACK':
                    ref_path = track(resolve(record['ref_npz'],CHAIR)); ref = arrays(ref_path)
                    prefix = 'reference_' if pool=='CONFIRM' else ''
                    for key,target in [('coverage_points','legacy_query_points'),('pairwise_target_points','legacy_pairwise_points'),
                                       ('gamma_points','legacy_gamma_points'),('gamma_weights','legacy_gamma_weights')]:
                        payload[target]=ref[prefix+key].copy()
                    if 'C' in ref: assert bits(payload['C'],ref['C'])
                    if 'gt_free' in ref: assert bits(payload['GT_free'].reshape(-1,9),ref['gt_free'].reshape(-1,9))
                    meta.update(legacy_ref_npz=str(ref_path),legacy_ref_json=str(track(resolve(record['ref_json'],CHAIR))))
                cp = out/'inputs/conditions'/f'{cid}.npz'; freeze_npz(cp,payload)
                meta.update(condition_id=cid,uid=uid,pool=pool,status='READY',N=row['N'],K=row['K'],
                            condition_npz=str(cp),condition_sha256=digest(cp),C_sha256=array_hash(payload['C']),
                            source_task_id=task_id,source_patch_id=record['patch_id'],
                            world_coordinates='C_world archived directly from original source vertices; exports stay model coordinates')
                if pool=='DEV':
                    for name,relative in [('legacy_control_reference','scoring/control_references_v1r1/'+cid+'__main'),
                                          ('legacy_quality_query','scoring/queries/'+cid)]:
                        for ext in ('json','npz'):
                            p=track(BOUNDARY/(relative+'.'+ext));meta[name+'_'+ext]=str(p);meta[name+'_'+ext+'_sha256']=digest(p)
                conditions[cid]=meta
            meta=conditions[cid]; payload=arrays(meta['condition_npz'])
            if pool=='DEV':
                old=track(resolve(record['input_path'],CHAIR),record['input_sha256']); a=arrays(old)
                bank={key:a[key].copy() for key in BANK_KEYS}
                seed=int(record['gaussian_seed']); assert bits(bank['C'],payload['C'])
                assert all(bits(bank[k],make_bank(payload['C'],row['N'],seed)[k]) for k in BANK_KEYS)
                cache=caches[record['case_id']]
                cache_path=track(HYBRID/'evaluation/OT_HYBRID/cases'/(record['case_id']+'.json'))
                audit=_check_cache(record,bank,cache,cache_path)
                track(audit['raw_path'],audit['raw_sha256'])
                row['raw_cache']=dict(path=audit['raw_path'],sha256=audit['raw_sha256'],checkpoint_sha256=START_SHA,
                    case_json=str(cache_path),case_json_sha256=digest(cache_path),status='PASS',metadata=audit['legacy_metadata'])
                row['legacy_input_npz']=str(old)
            else:
                seed=noise_seed(pool,uid,task_id,candidate)
                bank=make_bank(payload['C'],row['N'],seed)
            bp=out/'inputs/inference'/f'{rid}.npz'; freeze_npz(bp,bank)
            row.update(noise_seed=seed,gaussian_seed=seed,bank_npz=str(bp),input_npz=str(bp),bank_sha256=digest(bp),
                       input_sha256=digest(bp),noise_sha256=array_hash(bank['z']),condition_npz=meta['condition_npz'],
                       condition_sha256=meta['condition_sha256'],gamma_status=meta['gamma_status'])
        except (FileNotFoundError,KeyError) as exc:
            row.update(status='BLOCKED',reason=repr(exc),bank_npz=None,condition_npz=None)
        pools[pool].append(row)
    assert [len(pools[k]) for k in pools]==[16,16,32]
    new_seeds=[r['noise_seed'] for k in ('TRAIN_FEEDBACK','CONFIRM') for r in pools[k] if r['status']=='READY']
    old_seeds={r['gaussian_seed'] for k in ('A','B','C') for r in panels[k] if 'gaussian_seed' in r}
    assert len(new_seeds)==len(set(new_seeds)) and not (set(new_seeds)&old_seeds)
    for pool,rows in pools.items(): freeze_json(out/'manifests'/f'{pool}.json',dict(schema='chair_feedback_pool_v1',pool=pool,rows=rows))
    freeze_json(out/'registration/conditions.json',list(conditions.values()))
    freeze_json(out/'manifests/TRAINING.json',dict(schema='chair_feedback_training_assets_v1',parents=parents['train'],
        tasks=train_tasks,task_manifest=str(CHAIR/'task_manifest.json'),task_manifest_sha256=digest(CHAIR/'task_manifest.json'),
        training_GT_unchanged=True,model_input_allowlist=['xt','t','N','known_mask','valid_mask']))
    freeze_json(out/'registration/source_files.json',sources)
    summary=dict(schema='chair_control_feedback_assets_v1',status='PASS' if all(r['status']=='READY' for rows in pools.values() for r in rows) else 'BLOCKED',
        gpu_lock=GPU_LOCK,train_parents=32,train_tasks=len(train_tasks),feedback_parents=feedback_parents,
        feedback_rule='rank32 by(N,uid), 8 contiguous strata of4, minimum UID each; first READY ratio.2 train task in original manifest order',
        confirm_rule='first record per UID in original panels.C order; historically observed holdout parents; new independent noise is not new shapes',
        noise_rule='SHA256 compact JSON [experiment,20261008,evaluation_noise,pool,uid,task_id,candidate], first4 bytes little-endian, MT19937 randn(F,3,3) castFP32',
        pool_counts={k:len(v) for k,v in pools.items()},conditions=len(conditions),source_files=len(sources),
        input_only_prepared=True,CONFIRM_scores_or_images_read=False,model_forwards=0,OT_calls=0,
        manifests={k:dict(path=str(out/'manifests'/f'{k}.json'),sha256=digest(out/'manifests'/f'{k}.json')) for k in pools})
    freeze_json(out/'registration/assets.json',summary)
    return summary


def _face_multiset(C):
    # Exact oriented byte tuple; stronger than unordered geometric multiset.
    return Counter(t.tobytes() for t in np.ascontiguousarray(C))


def read_binary_ply(path):
    """Read our deliberately small binary triangle-soup schema, without repair."""
    with Path(path).open('rb') as f:
        header=[]
        while True:
            line=f.readline()
            if not line: raise ValueError('Incomplete PLY header')
            header.append(line.decode('ascii').strip())
            if line==b'end_header\n':break
        assert header[0]=='ply' and header[1]=='format binary_little_endian 1.0'
        nv=int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
        nf=int(next(x.split()[-1] for x in header if x.startswith('element face ')))
        vertices=np.frombuffer(f.read(12*nv),dtype='<f4').reshape(nv,3).copy()
        dt=np.dtype([('count','u1'),('indices','<i4',(3,)),('known','u1'),('known_slot','<i4'),('source_face','<i4')])
        faces=np.frombuffer(f.read(dt.itemsize*nf),dtype=dt).copy()
        assert not f.read(1) and np.all(faces['count']==3)
        assert np.array_equal(faces['indices'],np.arange(3*nf).reshape(nf,3)) and nv==3*nf
    return vertices[faces['indices']],faces


def export_mesh(output, C, known_mask, dest_prefix, source_condition=None):
    """Lossless FP32 model-space export, exact known order/multiset and mapping.

    source_condition contributes provenance only; neither mesh coordinates nor
    generated free-face assignment is changed using GT.
    """
    output=np.asarray(output); C=np.asarray(C); known=np.asarray(known_mask)
    assert output.dtype==C.dtype==np.float32 and known.dtype==np.bool_
    output=output.reshape(-1,3,3);C=C.reshape(-1,3,3);n=len(output);k=len(C)
    assert known.shape==(n,) and known.sum()==k and 0<k<n
    assert np.isfinite(output).all() and np.isfinite(C).all() and bits(output[known],C)
    ids=np.flatnonzero(known).astype(np.int64); slots=np.full(n,-1,np.int64);slots[ids]=np.arange(k)
    source=np.full(n,-1,np.int64);payload=dict(output=output.copy(),C=C.copy(),known_mask=known.copy(),
        valid_mask=np.ones(n,bool),known_to_output_face=ids,output_to_known_slot=slots,N=np.asarray(n,np.int64),K=np.asarray(k,np.int64))
    if source_condition is not None:
        condition=arrays(source_condition) if not isinstance(source_condition,dict) else source_condition
        assert bits(condition['C'],C)
        source[ids]=np.asarray(condition['source_face_ids'],np.int64)
        for name in ('C_world','known_source_vertex_ids','original_known_source_face_ids'):
            if name in condition:payload[name]=np.asarray(condition[name]).copy()
    payload['known_source_face_id_by_output']=source
    prefix=Path(dest_prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
    npz=Path(str(prefix)+'.npz');ply=Path(str(prefix)+'.ply');receipt=Path(str(prefix)+'.export.json')
    freeze_npz(npz,payload)
    header=('ply\nformat binary_little_endian 1.0\ncomment raw FP32 model coordinates; no welding or repair\n'
        f'element vertex {3*n}\nproperty float x\nproperty float y\nproperty float z\nelement face {n}\n'
        'property list uchar int vertex_indices\nproperty uchar known\nproperty int known_slot\nproperty int source_face\nend_header\n').encode('ascii')
    dt=np.dtype([('count','u1'),('indices','<i4',(3,)),('known','u1'),('known_slot','<i4'),('source_face','<i4')])
    faces=np.zeros(n,dtype=dt);faces['count']=3;faces['indices']=np.arange(3*n).reshape(n,3)
    faces['known']=known;faces['known_slot']=slots;faces['source_face']=source
    content=header+output.astype('<f4',copy=False).tobytes()+faces.tobytes()
    if ply.exists():assert ply.read_bytes()==content,'Existing export differs'
    else:
        with ply.open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
    loaded,face_meta=read_binary_ply(ply);saved=arrays(npz)
    assert bits(loaded,output) and bits(saved['output'],output)
    assert bits(loaded[face_meta['known'].astype(bool)],C) and _face_multiset(loaded[known])==_face_multiset(C)
    assert np.array_equal(face_meta['known_slot'],slots) and np.array_equal(face_meta['source_face'],source)
    assert bits(saved['C'],C) and np.array_equal(saved['known_to_output_face'],ids)
    result=dict(schema='chair_feedback_lossless_export_v1',status='PASS',N=n,K=k,vertex_count=3*n,
        model_coordinate_frame=True,world_inverse_applied=False,all_output_FP32_bytes_preserved=True,
        C_order_winding_multiplicity_bytes_preserved=True,C_geometric_face_multiset_preserved=True,
        known_to_output_face=ids.tolist(),known_source_face_ids=source[ids].tolist(),C_sha256=array_hash(C),
        output_sha256=array_hash(output),npz=str(npz),npz_sha256=digest(npz),ply=str(ply),ply_sha256=digest(ply),
        generated_free_GT_correspondence_inferred=False,geometry_modified=False)
    freeze_json(receipt,result)
    return result


def export_cpu_check(out=OUT):
    # Duplicate known triangles, non-contiguous known slots, signed zero and
    # source associations exercise information lost by ordinary mesh writers.
    a=np.asarray([[0.,-0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float32)
    c=np.stack([a,a]); full=np.stack([a,a+np.float32(2),a]);known=np.asarray([True,False,True])
    r=export_mesh(full,c,known,Path(out)/'tests/assets/export_roundtrip',dict(C=c,source_face_ids=np.asarray([4,9])))
    assert r['known_to_output_face']==[0,2] and r['known_source_face_ids']==[4,9]
    freeze_json(Path(out)/'tests/assets/results.json',dict(status='PASS',fixtures=1,model_forwards=0,export=r))
    return r


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT));parser.add_argument('--test-export',action='store_true')
    args=parser.parse_args()
    result=export_cpu_check(args.out) if args.test_export else prepare(args.out)
    print(json.dumps(result,sort_keys=True))
