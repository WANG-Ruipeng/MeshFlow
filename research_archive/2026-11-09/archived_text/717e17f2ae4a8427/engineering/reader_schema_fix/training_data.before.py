"""Versioned USER scaling stream; historical R2 sample arithmetic is unchanged.

CPU-only. Every OT has a durable batch reservation and a reusable map receipt.
Only the registered OUT is writable; all source parents/conditions are read-only.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
from pathlib import Path
import sys, hashlib, json, time
from functools import lru_cache
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
from meshflow_control.data.dataset import TrainingDataset, training_time
from meshflow_control.data.stream import HybridStream
from meshflow_control.data.io import digest, json_hash, array_hash, bits, read_json, atomic_json, freeze_json, save_npz
from meshflow_control.training import coupling

EXPERIMENT = 'R2_USER_DATA_SCALING_32_67_V1'
NAMESPACE = EXPERIMENT + '/training/v1'
OUT = Path('/mnt/e/MeshFlow/experiments/r2_user_data_scaling_32_67_v1')
TYPES = {'SEAT_FULL', 'SEAT_HALF', 'BACK_FULL', 'BACK_HALF'}

def checked_out(path):
    p = Path(path).absolute()
    for q in (p, *p.parents):
        if q.is_symlink():
            raise ValueError('Symlink output forbidden: ' + str(q))
    if p.resolve() != OUT.resolve():
        raise ValueError('Only the registered experiment OUT is writable')
    return p.resolve()

def writable(out, path):
    out = checked_out(out); p = Path(path).absolute()
    for q in (p, *p.parents):
        if q.is_symlink():
            raise ValueError('Symlink output forbidden: ' + str(q))
    if not p.resolve().is_relative_to(out):
        raise ValueError('Output escapes experiment: ' + str(p))
    return p

def seed(domain, *identity):
    return int.from_bytes(hashlib.sha256(json.dumps([NAMESPACE, domain, *identity], separators=(',', ':')).encode()).digest()[:4], 'little')

def load_manifest(path):
    path = Path(path); m = read_json(path)
    if m.get('schema') != 'r2_user_scaling_data_v1' or m.get('status') != 'FROZEN':
        raise ValueError('A successfully frozen scaling data manifest is required')
    pools = m['pools']; names = list(pools)
    expanded = 'USER_EXPANDED_67'
    if set(names) != {'USER_SMALL', expanded}:
        raise ValueError('Exactly USER_SMALL and USER_EXPANDED_67 pools required')
    small, big = pools['USER_SMALL'], pools[expanded]
    if len(small) != 32 or len(big) != 67 or not set(small) < set(big):
        raise ValueError('Exactly nested 32 / 67 independent parents required')
    if any(len(ids) != len(set(ids)) for ids in pools.values()):
        raise ValueError('Duplicate parent IDs in a pool')
    if set(m['parents']) != set(big) or any(t['uid'] not in set(big) for t in m['tasks'].values()):
        raise ValueError('Training manifest contains parents outside the frozen 67-parent pool')
    if len(m['tasks']) != 194 or sum(t['uid'] in set(small) for t in m['tasks'].values()) != 91:
        raise ValueError('Amendment reuses exactly 194 expanded / 91 small conditions')
    for uid in big:
        p = m['parents'][uid]
        if p['uid'] != uid or not 128 <= p['N'] <= 256:
            raise ValueError('Parent N/identity differs')
        tasks = [t for t in m['tasks'].values() if t['uid'] == uid]
        if not tasks:
            raise ValueError('Parent has no qualified condition')
        seen = set()
        for t in tasks:
            n, k = int(t['N']), int(t['K'])
            ids = np.asarray(t['source_face_ids'], np.int64)
            free = np.asarray(t['free_source_ids'], np.int64)
            if n != p['N'] or not 0 < k < n or len(ids) != k or not np.array_equal(np.sort(np.r_[ids, free]), np.arange(n)):
                raise ValueError('C/free partition differs')
            face_set = tuple(sorted(ids.tolist()))
            if face_set in seen or t['condition_type'] not in TYPES:
                raise ValueError('Duplicate condition or unsupported semantic family')
            seen.add(face_set)
    return m

class Dataset(TrainingDataset):
    def __init__(self, path):
        self.root = Path(path).resolve().parent
        self.manifest = load_manifest(path)
        self.manifest_sha256 = digest(path)
        self.parents, self.tasks = self.manifest['parents'], self.manifest['tasks']

    @lru_cache(maxsize=512)
    def parent(self, uid):
        # The original reader verifies pre_OT*2 and indexed geometry identity.
        return super().parent(uid)

    @lru_cache(maxsize=2048)
    def verify_condition(self, task_id):
        row=self.tasks[task_id]
        if row['task_id']!=task_id:raise ValueError('Condition key/identity differs')
        path=self.root/row['npz']
        if digest(path)!=row['sha256']:raise ValueError('Frozen condition changed')
        full=self.parent(row['uid'])['full_target'].reshape(row['N'],3,3)
        with np.load(path,allow_pickle=False) as z:
            if not bits(z['C'],full[np.asarray(row['source_face_ids'],np.int64)]) or not bits(z['Y'],full[np.asarray(row['free_source_ids'],np.int64)]):
                raise ValueError('Condition is not the frozen exact parent partition')
        return True

    def task(self,task_id,alpha=1.0):
        self.verify_condition(task_id)
        return super().task(task_id,alpha)

def make_plan(manifest, arm, updates, *, namespace='formal'):
    if arm not in manifest['pools'] or not 1 <= updates <= 6000:
        raise ValueError('Invalid arm/update budget')
    parents = sorted(manifest['pools'][arm]); count = np.zeros(len(parents), np.int64)
    by_parent = {uid: sorted(k for k,t in manifest['tasks'].items() if t['uid'] == uid) for uid in parents}
    prng = np.random.RandomState(seed('parent_balance', namespace, arm))
    crng = np.random.RandomState(seed('uniform_condition', namespace, arm))
    records = []
    for b in range(updates):
        # Random tie order among the least-exposed parents; unique within batch.
        shuffled = prng.permutation(len(parents))
        picked = shuffled[np.argsort(count[shuffled], kind='stable')[:8]]
        count[picked] += 1
        arng = np.random.RandomState(seed('shared_augmentation', namespace, b))
        alpha = np.r_[np.ones(4, np.float32), arng.uniform(.9, 1.1, 4).astype(np.float32)]
        alpha = alpha[arng.permutation(8)]
        if np.count_nonzero(alpha == 1) != 4:
            raise ValueError('Augmented draw rounded to 1; no redraw allowed')
        for slot, pi in enumerate(picked):
            uid = parents[int(pi)]; options = by_parent[uid]
            task_id = options[int(crng.randint(len(options)))]; t = manifest['tasks'][task_id]
            i = b*8+slot; ts = seed('shared_time', namespace, i)
            records.append(dict(sample_index=i, effective_batch_index=b, step=b+1, global_step=b+1,
                slot_index=slot, arm=arm, uid=uid, task_id=task_id, N=t['N'], K=t['K'],
                ratio=t['K']/t['N'], alpha=float(alpha[slot]), clean=bool(alpha[slot] == 1),
                condition_type=t['condition_type'], region_type=t['condition_type'].split('_')[0].lower(),
                t_seed=ts, t=float(training_time(ts)),
                gaussian_seed=seed('Gaussian', namespace, arm, i, uid, task_id),
                epsilon_seed=seed('epsilon', namespace, arm, i, uid, task_id),
                permutation_seed=seed('permutation', namespace, arm, i, uid, task_id)))
    if count.max()-count.min() > 1:
        raise AssertionError('Parent exposure imbalance exceeds one')
    if namespace=='preflight':
        # The isolated timing run explicitly includes smallest K, largest N,
        # and each available semantic family; no evaluation inputs are used.
        available=[t for t in manifest['tasks'].values() if t['uid'] in set(parents)]
        ranked=[sorted(available,key=lambda t:(t['K'],-t['N'],t['task_id'])),
                sorted(available,key=lambda t:(-t['N'],t['K'],t['task_id']))]
        ranked.extend(sorted([t for t in available if t['condition_type']==family],key=lambda t:t['task_id']) for family in sorted(TYPES))
        ranked.extend([sorted(available,key=lambda t:(seed('preflight_fill',t['task_id']),t['task_id']))]*8)
        chosen=[];seen=set()
        for options in ranked:
            candidate=next((t for t in options if t['uid'] not in seen),None)
            if candidate is not None:
                chosen.append(candidate);seen.add(candidate['uid'])
            if len(chosen)==8:break
        for i,t in enumerate(chosen):
            r=records[i];uid=t['uid'];task_id=t['task_id']
            r.update(uid=uid,task_id=task_id,N=t['N'],K=t['K'],ratio=t['K']/t['N'],
                condition_type=t['condition_type'],region_type=t['condition_type'].split('_')[0].lower(),
                gaussian_seed=seed('Gaussian',namespace,arm,i,uid,task_id),
                epsilon_seed=seed('epsilon',namespace,arm,i,uid,task_id),
                permutation_seed=seed('permutation',namespace,arm,i,uid,task_id))
        count=np.asarray([sum(r['uid']==uid for r in records) for uid in parents],np.int64)
    return dict(schema='r2_scaling_training_plan_v1', namespace=namespace, arm=arm, updates=updates,
        records=records, by_parent=dict(zip(parents, count.tolist())),
        by_condition_type=dict(Counter(r['condition_type'] for r in records)),
        free_coordinate_count=sum(9*(r['N']-r['K']) for r in records),
        rng_contract='stateless shared augmentation/time; arm-and-sample Gaussian/epsilon/permutation; balanced parent then uniform unique C')

class Builder(HybridStream):
    def __init__(self, dataset, maps):
        self.data = dataset; self.maps = maps; self.new_maps = {}

    def _mapping(self, pre, noise, source_ids, corners, known_ids, record):
        ident = dict(contract=coupling.CONTRACT, pre=array_hash(pre), noise=array_hash(noise),
            source_ids=array_hash(source_ids), corners=array_hash(corners), known_ids=array_hash(known_ids),
            alpha=record['alpha'], task_id=record['task_id'], sample_index=record['sample_index'])
        key = json_hash(ident); prefix = str(record['sample_index'])
        if self.maps is not None:
            if str(self.maps[prefix+'_key'].item()) != key:
                raise ValueError('Cached OT identity differs')
            face, corner = coupling.validate_map(self.maps[prefix+'_face'], self.maps[prefix+'_corner'])
            return face, corner, key, 'HIT'
        face, corner = coupling.nested_ot_map(pre, noise)
        self.new_maps.update({prefix+'_key':np.asarray(key), prefix+'_face':face, prefix+'_corner':corner})
        return face, corner, key, 'MISS'

def validate_sample(s, row):
    n,k = int(s['y']), int(s['K']); known = s['known_mask']
    if n != row['N'] or k != row['K'] or s['free_coordinate_count'] != 9*(n-k):
        raise ValueError('N/K/denominator differs')
    if known.dtype != np.bool_ or known.sum() != k or not bits(s['xt'][known], s['context'][known]):
        raise ValueError('Known C was overwritten')
    if not np.array_equal(np.sort(s['source_face_ids']), np.arange(n)):
        raise ValueError('Face identity was lost')
    if not np.array_equal(np.sort(s['corner_permutations'], axis=1), np.tile(np.arange(3), (n,1))):
        raise ValueError('Corner inverse mapping is not bijective')
    for key in ('x0','x1','u','xt','context','free_gaussian_before_OT','free_epsilon','free_target_before_OT'):
        if not np.isfinite(s[key]).all():
            raise ValueError('Nonfinite training fixture')
    if not bits(s['u'],s['x1']-s['x0']):
        raise ValueError('Velocity target differs')
    s.update(arm=row['arm'], global_step=row['global_step'], lambda_value=.25,
        condition_type=row['condition_type'], region_type=row['region_type'])
    return s

def prepare_batch(out, dataset, records, phase, arm, batch_index, plan_hash):
    out = checked_out(out)
    folder = writable(out, out/'ot'/phase/arm); folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'batch_{batch_index:04d}.npz'; receipt = path.with_suffix('.json')
    attempt = path.with_suffix('.attempt.json')
    identity = dict(plan_sha256=plan_hash, manifest_sha256=dataset.manifest_sha256, batch_index=batch_index,
        arm=arm, records_sha256=json_hash(records), sample_count=len(records))
    if receipt.exists():
        r = read_json(receipt)
        if r['identity'] != identity or r['sha256'] != digest(path):
            raise ValueError('OT batch receipt differs')
        with np.load(path, allow_pickle=False) as z:
            maps = {k:z[k].copy() for k in z.files}
    else:
        if attempt.exists() or path.exists():
            raise RuntimeError('Incomplete OT batch; no silent retry: ' + str(attempt))
        with attempt.open('x',encoding='utf8') as f:
            json.dump(identity,f,sort_keys=True); f.flush(); os.fsync(f.fileno())
        maps = None
    builder = Builder(dataset,maps); tick=time.perf_counter(); samples=[]
    for row in records:
        samples.append(validate_sample(builder.sample(row), row))
    if maps is None:
        save_npz(path, **builder.new_maps)
        atomic_json(receipt,dict(identity=identity,sha256=digest(path),actual_OT=len(records),status='PASS',seconds=time.perf_counter()-tick))
    return samples

class Stream:
    def __init__(self,out,manifest,arm,updates,phase='formal',workers=2):
        self.out=checked_out(out); self.dataset=Dataset(manifest); self.arm=arm; self.phase=phase
        self.plan=make_plan(self.dataset.manifest,arm,updates,namespace=phase)
        self.plan.update(manifest_sha256=self.dataset.manifest_sha256)
        self.plan_path=writable(out,self.out/'plans'/phase/(arm+'.json'))
        self.plan_sha256=freeze_json(self.plan_path,self.plan)
        self.batch_index=0; self.total=updates
        if workers not in (1,2): raise ValueError('One or two CPU OT workers only')
        self.pool=ThreadPoolExecutor(max_workers=workers); self.pending={}; self.workers=workers

    def state_dict(self):
        return dict(schema='r2_scaling_stream_cursor_v1',arm=self.arm,phase=self.phase,
            batch_index=self.batch_index,plan_sha256=self.plan_sha256,manifest_sha256=self.dataset.manifest_sha256)

    def load_state_dict(self,state):
        check=self.state_dict()
        if any(state.get(k)!=v for k,v in check.items() if k!='batch_index') or not 0<=state['batch_index']<=self.total:
            raise ValueError('Stream checkpoint identity differs')
        if self.pending: raise ValueError('Cannot move a live prefetch cursor')
        self.batch_index=state['batch_index']

    def next_effective_batch(self, stop_step=None):
        stop=self.total if stop_step is None else min(stop_step,self.total)
        if self.batch_index>=stop: raise StopIteration
        for b in range(self.batch_index,min(stop,self.batch_index+self.workers)):
            if b not in self.pending:
                records=self.plan['records'][b*8:(b+1)*8]
                self.pending[b]=self.pool.submit(prepare_batch,self.out,self.dataset,records,self.phase,self.arm,b,self.plan_sha256)
        result=self.pending.pop(self.batch_index).result();self.batch_index+=1
        return result

    def close(self):
        self.pool.shutdown(wait=True)

