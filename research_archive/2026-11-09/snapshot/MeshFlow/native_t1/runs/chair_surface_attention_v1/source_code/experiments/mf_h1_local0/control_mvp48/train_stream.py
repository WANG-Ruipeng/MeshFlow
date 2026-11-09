"""Replayable CPU-only stream. Requires explicit binding to frozen Session A data."""
import copy
import hashlib
import json
import numpy as np
import torch
from native_t1.artifacts import array_hash
from native_t1.data import CYCLIC
from native_t1.losses import prepare_targets
from native_t1.objectives import GeometryObjective, objective_spec
from utils.ot_utils import optimal_sum_numpy


def equal_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def sample_hash(sample):
    keys = ('x0', 'x1', 'u', 'xt', 'context', 't', 'y', 'known_mask', 'valid_mask',
            'source_face_ids', 'corner_permutations', 'face_permutation',
            'free_gaussian_before_OT', 'free_noise_after_OT', 'free_target_before_OT')
    digest = hashlib.sha256()
    digest.update(json.dumps([sample['object_id'], sample['patch_id']]).encode())
    for key in keys:
        digest.update(key.encode()); digest.update(array_hash(np.asarray(sample[key])).encode())
    return digest.hexdigest()


class FrozenStream:
    def __init__(self, binding, plan_path, archive_path, *, seed=10):
        from .train_binding import data_api
        self.binding = binding
        self.api = data_api(binding['loader_path'])
        self.data_root = archive_path
        from native_t1.artifacts import file_sha256
        from pathlib import Path
        if Path(plan_path).resolve() != (Path(self.data_root)/'train_plan.json').resolve() or file_sha256(plan_path) != binding['plan']['sha256']:
            raise ValueError('Actual consumed plan differs from frozen binding')
        plan = self.api.load_plan('train', self.data_root)
        if plan['schema'] != 'control_mvp48_train_plan_v1':
            raise ValueError('Session A train plan schema changed')
        self.records = plan['records']
        if len(self.records) != 8000 or plan['effective_batch_count'] != 1000 or plan['batch_size'] != 8:
            raise ValueError('Require exactly 1000 batches of eight')
        train_ids = set(binding['train_object_ids'])
        used = set()
        self.objects, self.cases = {}, {}
        for index, record in enumerate(self.records):
            uid, patch = record['uid'], record['patch_id']
            if (record['sample_index'], record['effective_batch_index'], record['slot_index']) != (index, index//8, index%8):
                raise ValueError('Frozen sample order differs')
            if uid not in train_ids:
                raise ValueError('Nontraining object entered task plan')
            used.add(uid)
            cid = (uid, patch)
            if cid not in self.cases:
                case = self.api.load_sample(uid, patch, self.data_root)
                self.cases[cid] = dict(object_id=uid, patch_id=patch, known_ids=case['source_face_ids'], N=case['N'], K=case['K'])
                if uid not in self.objects:
                    self.objects[uid] = dict(V=case['model_vertices'], F=case['source_face_vertex_ids'],
                        pre=case['pre_ot_full'], Y=case['full_target'].reshape(-1,3,3), bbox_L=case['bbox_diagonal'])
                self.cases[cid]['interface_edges'] = case['interface']['edge_source_vertex_ids']
            if (record['N'], record['K']) != (self.cases[cid]['N'], self.cases[cid]['K']):
                raise ValueError('Plan and loaded patch N/K differ')
        if used != train_ids:
            raise ValueError('Task plan does not cover exactly 32 train objects')
        self.seed = seed
        self.batch_index = 0
        self.counts = dict(actual_OT_attempts=0, actual_OT_returns=0, OT_cache_hits=0)

    def state_dict(self):
        return dict(seed=self.seed, batch_index=self.batch_index, sample_index=8*self.batch_index,
            RNG_mode='stateless_per_sample_frozen_seeds', plan_sha256=self.binding['plan']['sha256'])

    def load_state_dict(self, state):
        if (state['seed'] != self.seed or not 0 <= state['batch_index'] <= 1000 or
            state['sample_index'] != 8*state['batch_index'] or
            state['plan_sha256'] != self.binding['plan']['sha256'] or
            state['RNG_mode'] != 'stateless_per_sample_frozen_seeds'):
            raise ValueError('Invalid stream restore')
        self.batch_index = state['batch_index']

    def _sample(self, record):
        cid = (record['uid'], record['patch_id'])
        case = self.cases[cid]; uid = case['object_id']; obj = self.objects[uid]
        known_ids = case['known_ids']; n = len(obj['Y']); k = len(known_ids)
        free_ids = np.asarray([i for i in range(n) if i not in set(known_ids.tolist())], np.int64)
        free_order, corners, order = self.api.permutations(record['permutation_seed'], n, k)
        pre = obj['pre'][free_ids[free_order]][np.arange(n-k)[:, None], corners[k:]]
        noise = self.api.gaussian(record['gaussian_seed'], n-k)
        self.counts['actual_OT_attempts'] += 1
        paired = optimal_sum_numpy(pre, noise, optimal=True)
        self.counts['actual_OT_returns'] += 1
        x0 = np.zeros((n, 3, 3), np.float32); x1 = np.zeros_like(x0)
        x0[k:] = paired.astype(np.float32); x1[k:] = (pre * 2).astype(np.float32)
        context = np.zeros_like(x0)
        context[:k] = obj['Y'][known_ids][np.arange(k)[:, None], corners[:k]]
        ids = np.concatenate((known_ids, free_ids[free_order]))
        known = np.arange(n) < k
        x0, x1, context = [a[order].reshape(n, 9) for a in (x0, x1, context)]
        known, ids, corners = known[order], ids[order], corners[order]
        t = np.asarray(self.api.training_time(record['t_seed']), np.float32)
        u = x1 - x0
        xt = (np.float32(1) - t) * x0 + t * x1
        xt[known] = context[known]
        sample = dict(x0=x0, x1=x1, u=u, xt=xt, context=context, t=t,
            y=np.asarray(n, np.int64), known_mask=known, valid_mask=np.ones(n, bool),
            source_face_ids=ids, corner_permutations=corners, face_permutation=order,
            object_id=uid, patch_id=case['patch_id'], K=k,
            free_gaussian_before_OT=noise, free_noise_after_OT=paired,
            free_target_before_OT=pre)
        sample['shared_input_sha256'] = sample_hash(sample)
        self.label(sample)  # Validate real source/GT correspondence before any forward.
        return sample

    def next_effective_batch(self):
        if self.batch_index >= 1000:
            raise RuntimeError('Frozen plan exhausted')
        samples = [self._sample(record) for record in self.records[8*self.batch_index:8*(self.batch_index+1)]]
        self.batch_index += 1
        return samples

    def label(self, sample):
        obj = self.objects[sample['object_id']]
        valid, known = sample['valid_mask'], sample['known_mask']
        ids, corners = sample['source_face_ids'][valid], sample['corner_permutations'][valid]
        n = len(obj['F'])
        if sorted(ids.tolist()) != list(range(n)) or not np.array_equal(np.sort(corners, 1), np.tile(np.arange(3), (n, 1))):
            raise ValueError('Invalid source face/corner permutation')
        vids = np.take_along_axis(obj['F'][ids], corners, axis=1)
        xyz = obj['V'][vids]
        km = known[valid]
        Y = sample['x1'][valid].reshape(n, 3, 3)
        C = sample['context'][valid].reshape(n, 3, 3)
        if not equal_bits(xyz[~km], Y[~km]) or not equal_bits(xyz[km], C[km]):
            raise ValueError('GT/interface mapping does not reconstruct actual coordinates')
        return dict(free_gt=Y[~km].copy(), known=C[km].copy(), free_ids=vids[~km].copy(),
                    known_ids=vids[km].copy(), bbox_L=obj['bbox_L'])


class MVPObjective(GeometryObjective):
    """Only replaces the old N112 target bank; exact original auxiliary method."""
    def __init__(self, stream):
        self.bank = stream
        self.spec = objective_spec('edge5')

    def prepare(self, samples):
        return [prepare_targets(self.bank.label(sample)) for sample in samples]


def collate(samples, device='cpu', *, pad_to=None):
    """Pad without truncation; padded coordinates/IDs never enter N or losses."""
    if not samples:
        raise ValueError('Empty batch')
    width = max(len(s['valid_mask']) for s in samples)
    if pad_to is not None:
        if pad_to < width:
            raise ValueError('Refusing to truncate real faces')
        width = pad_to
    batch = {}
    for key in ('x0', 'x1', 'u', 'context', 'valid_mask', 'known_mask'):
        rows = []
        for s in samples:
            a = np.asarray(s[key])
            out = np.zeros((width,) + a.shape[1:], dtype=a.dtype)
            out[:len(a)] = a
            rows.append(out)
        batch[key] = torch.from_numpy(np.stack(rows)).to(device)
    for key in ('t', 'y'):
        batch[key] = torch.from_numpy(np.stack([np.asarray(s[key]) for s in samples])).to(device)
    if not torch.equal(batch['valid_mask'].sum(1), batch['y']):
        raise ValueError('Total N includes padding or truncates real faces')
    batch['xt'] = (1-batch['t'][:, None, None])*batch['x0'] + batch['t'][:, None, None]*batch['x1']
    batch['xt'] = torch.where(batch['known_mask'][..., None], batch['context'], batch['xt'])
    batch['loss_mask'] = batch['valid_mask'] & ~batch['known_mask']
    return batch
