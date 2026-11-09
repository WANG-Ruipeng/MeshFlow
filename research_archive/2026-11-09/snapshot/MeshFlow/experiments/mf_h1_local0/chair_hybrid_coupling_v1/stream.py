"""Frozen 1000-update paired stream using the existing Chair C task bank.

No model construction, CUDA, optimizer, generation, or protected-file writes.
Both arms use original iid Z slots after exact inverse nested OT.  Random
domains and actual common tensor hashes are independent of lambda/arm order.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from experiments.mf_h1_local0.control_mvp48.train_binding import data_api
from experiments.mf_h1_local0.control_mvp48.train_stream import FrozenStream, sample_hash
from experiments.mf_h1_local0.chair_domain_control_v1.prepare import load_task_sample
from .coupling import (CONTRACT, array_hash, file_hash, nested_ot_map, validate_map,
                       to_noise_slots, to_target_slots, coupling_path, mapping_key)

ARMS = {'OT_BASE': 0., 'OT_HYBRID': .25}
EXPERIMENT = 'chair_hybrid_coupling_v1'
SEED = 20261004


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def domain_seed(domain, *identity, seed=SEED):
    encoded = json.dumps([EXPERIMENT, int(seed), domain, *identity], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], 'little')


def make_plan(manifest, seed=SEED):
    """Original parent/ratio/C distribution, shortened to 1000 paired batches."""
    if seed != SEED:
        raise ValueError('Only the preregistered seed=20261004 is authorized')
    parents = sorted(manifest['splits']['train'])
    if len(parents) != 32:
        raise ValueError('Require the original 32 training parents')
    tasks = defaultdict(lambda: defaultdict(list))
    for task in manifest['tasks']:
        if task['role'] == 'train' and task['status'] == 'READY':
            tasks[task['uid']][task['ratio']].append(task)
    if set(tasks) != set(parents):
        raise ValueError('Original training task availability differs')
    domains = ('parent_schedule', 'task_schedule', 'alpha', 'alpha_slot_shuffle')
    rng = {d: np.random.RandomState(domain_seed(d, seed=seed)) for d in domains}
    assignments = [parents[int(i)] for _ in range(250) for i in rng['parent_schedule'].permutation(32)]
    records = []
    for batch in range(1000):
        alphas = np.r_[np.ones(4, np.float32), rng['alpha'].uniform(.9, 1.1, 4).astype(np.float32)]
        alphas = alphas[rng['alpha_slot_shuffle'].permutation(8)]
        for slot in range(8):
            index = 8 * batch + slot
            uid = assignments[index]
            ratios = sorted(tasks[uid])
            ratio = ratios[int(rng['task_schedule'].randint(len(ratios)))]
            options = tasks[uid][ratio]
            task = options[int(rng['task_schedule'].randint(len(options)))]
            records.append(dict(sample_index=index, effective_batch_index=batch, slot_index=slot,
                pilot_step=batch+1, conditional_total_step=4001+batch, uid=uid,
                patch_id=task['task_id'], task_id=task['task_id'], N=task['N'], K=task['K'],
                ratio=ratio, alpha=float(alphas[slot]),
                gaussian_seed=domain_seed('gaussian', index, seed=seed),
                t_seed=domain_seed('time', index, seed=seed),
                permutation_seed=domain_seed('permutation', index, seed=seed),
                epsilon_seed=domain_seed('epsilon', index, seed=seed)))
    return dict(schema='chair_hybrid_train_plan_v1', experiment_id=EXPERIMENT, seed=seed,
        RNG_mode='stateless_per_sample_frozen_seeds', effective_batch_count=1000,
        batch_size=8, sample_count=8000, effective_batch_index_base=0,
        random_domains=list(domains)+['gaussian','time','permutation','epsilon'],
        parent_distribution='250 independently shuffled complete 32-parent cycles',
        context_distribution='original available ratios uniform, then original C tasks uniform within ratio',
        source_task_manifest_semantics='unchanged chair_domain_control_v1 frozen training C bank',
        object_counts=dict(Counter(r['uid'] for r in records)),
        task_counts=dict(Counter(r['task_id'] for r in records)),
        object_ratio_counts=dict(Counter(r['uid']+'/'+str(r['ratio']) for r in records)), records=records)


def freeze_plan(source_chair, out, seed=SEED):
    source_chair, out = Path(source_chair).resolve(), Path(out).resolve()
    if source_chair == out or source_chair in out.parents:
        raise ValueError('Pilot output must be isolated from the protected source')
    manifest_path = source_chair/'task_manifest.json'
    plan = make_plan(read_json(manifest_path), seed)
    plan['source_task_manifest_sha256'] = file_hash(manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    path = out/'train_plan.json'
    if path.exists():
        if read_json(path) != plan:
            raise ValueError('Existing pilot plan differs; refusing overwrite')
    else:
        with path.open('x', encoding='utf-8') as f:
            json.dump(plan, f, sort_keys=True, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    return dict(path=str(path), sha256=file_hash(path), samples=8000, updates_per_arm=1000,
                seed=seed, source_task_manifest_sha256=plan['source_task_manifest_sha256'])


def prepare_training_plan(source_mvp, source_chair, out, seed=SEED):
    binding = read_json(Path(source_mvp)/'registration/train_data_binding.json')
    manifest = read_json(Path(source_chair)/'task_manifest.json')
    if set(binding['train_object_ids']) != set(manifest['splits']['train']):
        raise ValueError('MVP binding and source Chair training split disagree')
    return freeze_plan(source_chair,out,seed)


class HybridStream(FrozenStream):
    def __init__(self, source_mvp, source_chair, out, arm, cache_dir=None):
        if arm not in ARMS:
            raise ValueError('Unregistered coupling arm')
        self.source = Path(source_mvp).resolve()
        self.source_chair, self.out = Path(source_chair).resolve(), Path(out).resolve()
        self.arm, self.lam = arm, ARMS[arm]
        binding = read_json(self.source/'registration/train_data_binding.json')
        self.api = data_api(binding['loader_path'])
        self.binding = binding
        self.manifest = read_json(self.source_chair/'task_manifest.json')
        self.task_manifest_sha256 = file_hash(self.source_chair/'task_manifest.json')
        self.plan_path = self.out/'train_plan.json'
        plan = read_json(self.plan_path)
        self.plan_sha256 = file_hash(self.plan_path)
        self.seed = plan['seed']
        self.records = plan['records']
        self.tasks = {t['task_id']:t for t in self.manifest['tasks'] if t['role']=='train' and t['status']=='READY'}
        self.offset, self.end, self.batch_index = 0, 1000, 0
        self.counts = dict(actual_OT_attempts=0, actual_OT_returns=0, OT_cache_hits=0)
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else self.out/'ot_cache'
        if self.out not in self.cache_dir.parents:
            raise ValueError('OT cache must be inside this isolated output')
        implementation_paths = [Path(__file__), Path(__file__).with_name('coupling.py'),
                                Path(__file__).resolve().parents[3]/'utils/ot_utils.py']
        self.implementation_sha256 = hashlib.sha256(''.join(file_hash(p) for p in implementation_paths).encode()).hexdigest()
        if plan != dict(make_plan(self.manifest, self.seed), source_task_manifest_sha256=self.task_manifest_sha256):
            raise ValueError('Frozen paired plan does not reproduce its registered RNG and task identity')
        if set(binding['train_object_ids']) != set(plan['object_counts']) or set(plan['object_counts'].values()) != {250}:
            raise ValueError('Frozen train parent exposure differs from 32 x 250')
        if len(self.records)!=8000 or any(sum(r['alpha']==1. for r in self.records[i:i+8])!=4 for i in range(0,8000,8)):
            raise ValueError('Require 1000 batches with 4 exact-unit and 4 Uniform alpha')

    def state_dict(self):
        return dict(seed=self.seed, batch_index=self.batch_index, sample_index=8*self.batch_index,
            RNG_mode='stateless_per_sample_frozen_seeds', plan_sha256=self.plan_sha256,
            task_manifest_sha256=self.task_manifest_sha256, offset=0, end=1000,
            coupling_contract=CONTRACT, arm=self.arm, lambda_value=self.lam)

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in expected:
            if key not in ('batch_index','sample_index') and state.get(key) != expected[key]:
                raise ValueError('Frozen stream restore identity differs: '+key)
        b = state['batch_index']
        if type(b) is not int or not 0<=b<=1000 or state['sample_index']!=8*b:
            raise ValueError('Invalid pilot stream cursor')
        self.batch_index = b

    def _mapping(self, pre, noise, ids, corners, known, record):
        key, identity = mapping_key(pre, noise, ids, corners, known, record['alpha'], self.implementation_sha256)
        identity.update(uid=record['uid'],task_id=record['task_id'],plan_sha256=self.plan_sha256,task_manifest_sha256=self.task_manifest_sha256)
        key = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir/(str(record['sample_index']).zfill(4)+'_'+key+'.npz')
        attempt = path.with_suffix('.attempt.json')
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                face, corner = validate_map(data['face'], data['corner'])
                if str(data['key'].item())!=key or json.loads(str(data['identity'].item()))!=identity:
                    raise ValueError('OT cache identity mismatch')
                expected = hashlib.sha256((array_hash(face)+array_hash(corner)).encode()).hexdigest()
                if str(data['map_sha256'].item()) != expected:
                    raise ValueError('OT cache mapping content hash mismatch')
                face, corner = face.copy(), corner.copy()
            self.counts['OT_cache_hits'] += 1
            return face, corner, key, 'HIT'
        if attempt.exists():
            raise RuntimeError('Interrupted OT attempt has no completed map; no silent retry: '+str(attempt))
        with attempt.open('x', encoding='utf-8') as f:
            json.dump(dict(key=key,identity=identity,sample_index=record['sample_index'],arm=self.arm),f,sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        self.counts['actual_OT_attempts'] += 1
        face, corner = nested_ot_map(pre, noise)
        self.counts['actual_OT_returns'] += 1
        with path.open('xb') as f:
            np.savez_compressed(f, face=face, corner=corner, key=np.asarray(key),
                identity=np.asarray(json.dumps(identity,sort_keys=True)),
                map_sha256=np.asarray(hashlib.sha256((array_hash(face)+array_hash(corner)).encode()).hexdigest()))
            f.flush(); os.fsync(f.fileno())
        return face, corner, key, 'MISS'

    def _sample(self, record):
        case = load_task_sample(self.source, self.manifest, record['task_id'], record['alpha'])
        n, k = case['N'], case['K']
        known_ids, free_ids = np.asarray(case['source_face_ids'],np.int64), np.asarray(case['free_source_ids'],np.int64)
        if set(known_ids)&set(free_ids) or sorted(np.r_[known_ids,free_ids].tolist())!=list(range(n)):
            raise ValueError('Known/free source partition differs')
        free_order, corners, order = self.api.permutations(record['permutation_seed'], n, k)
        ordered_ids = free_ids[free_order]
        pre = case['pre_ot_full'][ordered_ids][np.arange(n-k)[:,None],corners[k:]]
        noise = self.api.gaussian(record['gaussian_seed'],n-k)
        face, corner, key, cache_status = self._mapping(pre,noise,ordered_ids,corners[k:],known_ids,record)
        final_target = to_noise_slots((pre*np.float32(2)).astype(np.float32),face,corner)
        # epsilon has its own stateless seed and never advances any common RNG.
        epsilon = self.api.gaussian(record['epsilon_seed'],n-k) if self.lam else None
        t = np.asarray(self.api.training_time(record['t_seed']),np.float32)
        mixed, _, _ = coupling_path(noise,final_target,epsilon,t,self.lam)
        x0 = np.zeros((n,3,3),np.float32); x1 = np.zeros_like(x0); context = np.zeros_like(x0)
        x0[k:],x1[k:] = mixed,final_target
        context[:k] = case['full_target'].reshape(n,3,3)[known_ids][np.arange(k)[:,None],corners[:k]]
        inverse_face = np.argsort(face)
        ids = np.r_[known_ids,ordered_ids[inverse_face]]
        mapped_corners = np.concatenate((corners[:k],to_noise_slots(corners[k:],face,corner)))
        known = (np.arange(n)<k)[order]
        x0,x1,context = [a[order].reshape(n,9) for a in (x0,x1,context)]
        ids,mapped_corners = ids[order],mapped_corners[order]
        u = x1-x0
        xt = (np.float32(1)-t)*x0+t*x1
        xt[known] = context[known]
        sample = dict(x0=x0,x1=x1,u=u,xt=xt,context=context,t=t,y=np.asarray(n,np.int64),
            known_mask=known,valid_mask=np.ones(n,bool),source_face_ids=ids,
            corner_permutations=mapped_corners,face_permutation=order,
            object_id=record['uid'],patch_id=record['task_id'],task_id=record['task_id'],K=k,
            alpha=float(record['alpha']),ratio=float(self.tasks[record['task_id']]['ratio']),
            free_gaussian_before_OT=noise,free_noise_after_OT=to_target_slots(noise,face,corner),
            free_target_before_OT=pre,free_epsilon=epsilon,
            ot_face_map=face,ot_corner_map=corner,ot_cache_key=key,ot_cache_status=cache_status,
            sample_index=record['sample_index'],pilot_step=record['pilot_step'],lambda_value=self.lam,
            free_coordinate_count=(n-k)*9,
            _geometry=dict(V=case['model_vertices'],F=case['source_face_vertex_ids'],bbox_L=case['bbox_diagonal']))
        sample['lambda'] = self.lam
        shared = dict(raw_Z=array_hash(noise),Y_OT=array_hash(final_target),context=array_hash(context),
            t=array_hash(t),known_mask=array_hash(known),source_ids=array_hash(ids),
            source_corners=array_hash(mapped_corners),full_face_order=array_hash(order),
            ot_map_key=key,alpha=float(record['alpha']),task_id=record['task_id'],
            epsilon_seed=record['epsilon_seed'],N=n,K=k)
        sample['shared_input_sha256'] = hashlib.sha256(json.dumps(shared,sort_keys=True).encode()).hexdigest()
        sample['coupled_input_sha256'] = sample_hash(sample)
        self.label(sample)
        return sample

    def label(self, sample):
        previous = getattr(self,'objects',None)
        self.objects = {sample['object_id']:sample['_geometry']}
        try:
            return super().label(sample)
        finally:
            self.objects = previous

    def next_effective_batch(self):
        if not 0<=self.batch_index<1000:
            raise RuntimeError('Fixed 1000-update pilot stream exhausted')
        start = 8*self.batch_index
        samples = [self._sample(r) for r in self.records[start:start+8]]
        self.batch_index += 1
        self.last_audit = dict(batch_index=self.batch_index,
            shared_batch_sha256=hashlib.sha256(''.join(s['shared_input_sha256'] for s in samples).encode()).hexdigest(),
            coupled_batch_sha256=hashlib.sha256(''.join(s['coupled_input_sha256'] for s in samples).encode()).hexdigest(),
            effective_free_coordinates=sum(s['free_coordinate_count'] for s in samples),
            actual_OT_counts=dict(self.counts))
        return samples
