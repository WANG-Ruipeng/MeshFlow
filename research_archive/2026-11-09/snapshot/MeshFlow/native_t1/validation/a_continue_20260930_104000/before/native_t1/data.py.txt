"""Validated Native patch inputs and the original two-parent mixed-K stream.

Reading an archive never invokes OT or touches a model. Sampling conditions
contain C only; targets and source identities stay in supervision/evaluation.
The training stream preserves the mixed K4/8/12 RNG/OT/permutation semantics.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import numpy as np
import torch

FACE_COUNT = 112
SUPPORTED_K = (2, 4, 8, 12)
TRAIN_K = (4, 8, 12)
CYCLIC = np.asarray([[0,1,2],[1,2,0],[2,0,1]], dtype=np.int64)
SHARED_KEYS = ('x0','x1','u','t','known_mask','valid_mask','known_target_ids',
               'source_face_ids','corner_permutations','face_permutation')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.dtype, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _readonly(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _equal_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@dataclass(frozen=True)
class SamplingCondition:
    """Only model-visible condition coordinates; no target or source face IDs."""
    constraints: np.ndarray
    N: int = FACE_COUNT

    def __post_init__(self):
        C = np.asarray(self.constraints)
        if self.N != FACE_COUNT or C.dtype != np.float32 or C.ndim != 3 or C.shape[1:] != (3,3) or len(C) not in SUPPORTED_K:
            raise ValueError('Require N112 and finite FP32 C[K,3,3], K2/4/8/12.')
        if not np.isfinite(C).all():
            raise ValueError('Nonfinite condition.')
        object.__setattr__(self, 'constraints', _readonly(C))

    @property
    def K(self):
        return len(self.constraints)

    @property
    def known_mask(self):
        return np.arange(self.N) < self.K


@dataclass(frozen=True)
class PatchCase:
    parent_index: int
    K: int
    constraints: np.ndarray
    full_target: np.ndarray
    free_target: np.ndarray
    source_face_ids: np.ndarray
    free_source_ids: np.ndarray
    bbox_diagonal: float
    pre_ot_free: np.ndarray | None = None
    N: int = FACE_COUNT

    def __post_init__(self):
        if self.N != FACE_COUNT or self.K not in SUPPORTED_K:
            raise ValueError('Only real N112 and K2/4/8/12 are supported.')
        shapes = {'constraints':(self.K,3,3), 'full_target':(112,9), 'free_target':(112-self.K,9)}
        for name, shape in shapes.items():
            value = np.asarray(getattr(self,name))
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'{name} must be finite FP32 {shape}.')
            object.__setattr__(self,name,_readonly(value))
        ids = np.asarray(self.source_face_ids); free_ids = np.asarray(self.free_source_ids)
        if not np.issubdtype(ids.dtype,np.integer) or ids.shape != (self.K,) or len(set(ids.tolist())) != self.K or np.any((ids<0)|(ids>=112)):
            raise ValueError('Known source IDs must be distinct actual faces.')
        if not np.issubdtype(free_ids.dtype,np.integer) or free_ids.tolist() != [i for i in range(112) if i not in ids]:
            raise ValueError('Free IDs must be the unchanged ordered complement of C.')
        object.__setattr__(self,'source_face_ids',_readonly(ids,np.int64))
        object.__setattr__(self,'free_source_ids',_readonly(free_ids,np.int64))
        if not _equal_bits(self.full_target[ids].reshape(self.K,3,3),self.constraints):
            raise ValueError('C is not the exact archived source-face subset.')
        if not _equal_bits(self.full_target[free_ids],self.free_target):
            raise ValueError('Remainder is not the exact ordered source complement.')
        actual_bbox = float(np.linalg.norm(np.ptp(self.full_target.reshape(-1,3),axis=0)))
        if not np.isfinite(self.bbox_diagonal) or self.bbox_diagonal <= 0 or self.bbox_diagonal != actual_bbox:
            raise ValueError('BBox metadata differs from unchanged FP32 target coordinates.')
        if self.pre_ot_free is not None:
            pre = np.asarray(self.pre_ot_free)
            if pre.shape != (112-self.K,3,3) or not np.issubdtype(pre.dtype,np.floating) or not np.isfinite(pre).all():
                raise ValueError('Original pre-OT free coordinates are required, never inferred by dividing final targets.')
            if not _equal_bits((pre*2).astype(np.float32).reshape(-1,9),self.free_target):
                raise ValueError('Original pre-OT final x2 scaling does not match archive.')
            object.__setattr__(self,'pre_ot_free',_readonly(pre))

    def sampling_condition(self):
        return SamplingCondition(self.constraints)


@dataclass(frozen=True)
class ArchiveData:
    path: Path
    sha256: str
    cases: dict[tuple[int,int],PatchCase]


def load_archive(path, expected_sha256=None):
    """Read existing NPZ in place. No copying files, RNG use, OT, or models."""
    path = Path(path).resolve(); digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Input archive SHA256 mismatch.')
    cases = {}
    with np.load(path,allow_pickle=False) as archive:
        parents = sorted(int(match.group(1)) for key in archive.files if (match:=re.fullmatch(r'parent(\d+)_target',key)))
        for parent in parents:
            full = archive[f'parent{parent}_target'].copy()
            for K in SUPPORTED_K:
                prefix = f'parent{parent}_K{K}_'
                if prefix+'constraints' not in archive:
                    continue
                pre = archive[prefix+'pre_ot_free'].copy() if prefix+'pre_ot_free' in archive else None
                cases[parent,K] = PatchCase(parent,K,archive[prefix+'constraints'],full,
                    archive[prefix+'free_target'],archive[prefix+'source_face_ids'],
                    archive[prefix+'free_source_ids'],float(np.linalg.norm(np.ptp(full.reshape(-1,3),axis=0))),pre)
    if not cases:
        raise ValueError('No registered Native N112 patch cases found.')
    return ArchiveData(path,digest,cases)


def shared_sample_hash(sample):
    fields = {key:sample[key] for key in SHARED_KEYS}
    fields.update(xt_free=sample['xt'][~sample['known_mask']],
        parent_index=np.asarray(sample['parent_index'],dtype=np.int64), K=np.asarray(sample['K'],dtype=np.int64),
        raw_Gaussian=sample['free_gaussian_before_OT'], OT_paired_noise=sample['free_noise_after_OT'],
        pre_OT_target=sample['free_target_before_OT'])
    digest = hashlib.sha256()
    for key,value in fields.items():
        digest.update(key.encode()); digest.update(array_hash(np.asarray(value)).encode())
    return digest.hexdigest()


def mixed_k_plan(index):
    phase = index % 3
    rotation = {K:TRAIN_K[(i+phase)%3] for i,K in enumerate(TRAIN_K)}
    base = [(0,4),(0,4),(0,8),(0,12),(1,4),(1,8),(1,8),(1,12)]
    return [(parent,rotation[K]) for parent,K in base]


class NativeTrainingStream:
    """Original mixed-K two-arm stream; no optimizer or model is invoked.

    Training randomizes all face slots (including known slots); only sampling
    fixes C in first K. Known faces never participate in free-only OT or loss.
    """
    RNG_KEYS = ('noise_rng','time_rng','permutation_rng','order_rng','donor_rng')

    def __init__(self,archive,arm='Native_correct',seed=10):
        if arm not in ('Native_correct','Native_independent'):
            raise ValueError('Only correct or independently sampled context arms.')
        self.cases = archive.cases if isinstance(archive,ArchiveData) else archive
        if not {(p,k) for p in (0,1) for k in TRAIN_K} <= set(self.cases):
            raise ValueError('The original stream requires both parents and K4/8/12.')
        for p in (0,1):
            for K in TRAIN_K:
                if (p,K) not in self.cases or self.cases[p,K].pre_ot_free is None:
                    raise ValueError('Missing original pre-OT target; no replacement or inference is allowed.')
        self.arm,self.seed = arm,int(seed)
        for key,offset in zip(self.RNG_KEYS,(10000,30000,40000,60000,70000)):
            setattr(self,key,np.random.RandomState(seed+offset))
        self.batch_index=0; self.sample_index=0; self.last_audit=None

    def state_dict(self):
        packed={}
        for key in self.RNG_KEYS:
            state=getattr(self,key).get_state()
            packed[key]=[state[0],state[1].tolist(),int(state[2]),int(state[3]),float(state[4])]
        return dict(rng=packed,batch_index=self.batch_index,sample_index=self.sample_index,seed=self.seed,arm=self.arm)

    def load_state_dict(self,state):
        if int(state['seed'])!=self.seed or state['arm']!=self.arm:
            raise ValueError('Stream restore arm/seed mismatch.')
        for key in self.RNG_KEYS:
            value=state['rng'][key]
            getattr(self,key).set_state((value[0],np.asarray(value[1],dtype=np.uint32),int(value[2]),int(value[3]),float(value[4])))
        self.batch_index=int(state['batch_index']); self.sample_index=int(state['sample_index'])

    def _sample(self,parent,K):
        from utils.ot_utils import optimal_sum_numpy
        case=self.cases[parent,K]; nf=112-K
        free_order=self.permutation_rng.permutation(nf)
        corners=CYCLIC[self.permutation_rng.randint(0,3,size=112)]
        pre=case.pre_ot_free[free_order].copy()[np.arange(nf)[:,None],corners[K:]]
        raw_noise=self.noise_rng.randn(nf,3,3)
        paired=optimal_sum_numpy(pre,raw_noise,optimal=True)
        final=(pre*2).astype(np.float32)
        reference=case.free_target[free_order].reshape(nf,3,3)[np.arange(nf)[:,None],corners[K:]]
        if not _equal_bits(final,reference):
            raise RuntimeError('Original free-only pre-OT ordering/scaling changed.')
        x0=np.zeros((112,3,3),dtype=np.float32); x1=np.zeros_like(x0)
        x0[K:]=paired.astype(np.float32); x1[K:]=final
        known=np.arange(112)<K; valid=np.ones(112,dtype=bool)
        known_ids=np.full(112,-1,dtype=np.int64); known_ids[:K]=np.arange(K)
        source_ids=np.concatenate([case.source_face_ids,case.free_source_ids[free_order]])
        contexts=np.zeros((2,112,3,3),dtype=np.float32)
        for donor in range(2):
            contexts[donor,:K]=self.cases[donor,K].constraints[np.arange(K)[:,None],corners[:K]]
        order=self.permutation_rng.permutation(112)
        x0=x0[order].reshape(112,9); x1=x1[order].reshape(112,9)
        known,valid=known[order],valid[order]; known_ids=known_ids[order]; source_ids=source_ids[order]
        contexts=contexts[:,order].reshape(2,112,9); corners=corners[order]
        t=np.asarray(1/(1+np.exp(-self.time_rng.randn())),dtype=np.float32)
        donor=parent if self.arm=='Native_correct' else int(self.donor_rng.randint(2))
        u=x1-x0; xt=(np.float32(1)-t)*x0+t*x1; xt[known]=contexts[donor,known]
        sample=dict(x0=x0,x1=x1,u=u,xt=xt,t=t,y=np.asarray(112,dtype=np.int64),known_mask=known,valid_mask=valid,
            context=contexts[donor].copy(),context_by_parent=contexts,parent_index=parent,donor_index=donor,K=K,
            known_target_ids=known_ids,source_face_ids=source_ids,corner_permutations=corners,face_permutation=order,
            free_gaussian_before_OT=raw_noise,free_noise_after_OT=paired,free_target_before_OT=pre)
        sample['shared_input_sha256']=shared_sample_hash(sample)
        sample['context_sha256']=array_hash(sample['context'])
        self.sample_index+=1
        return sample

    def next_effective_batch(self,size=8):
        if size!=8:
            raise ValueError('Original effective batch is exactly 8.')
        pairs=mixed_k_plan(self.batch_index)
        samples=[self._sample(*pairs[i]) for i in self.order_rng.permutation(8)]
        self.batch_index+=1
        shared=[s['shared_input_sha256'] for s in samples]
        self.last_audit=dict(batch_index=self.batch_index,samples=8,shared_input_hashes=shared,
            shared_batch_sha256=hashlib.sha256(''.join(shared).encode()).hexdigest(),
            parent_indices=[s['parent_index'] for s in samples],K=[s['K'] for s in samples],
            donor_indices=[s['donor_index'] for s in samples],times=[float(s['t']) for s in samples],
            effective_free_coordinates=sum((112-s['K'])*9 for s in samples),
            no_full112_OT=True,official_OT_call_count=8,donor_independent_of_target_stream=True)
        return samples


def native_collate(samples,device='cpu',t_override=None):
    """Supervision batch; model arguments are exclusively returned by model_inputs."""
    if not samples:
        raise ValueError('Empty Native batch.')
    result={key:torch.from_numpy(np.stack([np.asarray(s[key]) for s in samples])).to(device)
            for key in ('x0','x1','u','t','y','known_mask','valid_mask','context')}
    if t_override is not None:
        if not 0<=float(t_override)<=1:
            raise ValueError('Time override must be in [0,1].')
        result['t']=torch.full_like(result['t'],float(t_override))
    result['xt']=(1-result['t'][:,None,None])*result['x0']+result['t'][:,None,None]*result['x1']
    result['xt']=torch.where(result['known_mask'][...,None],result['context'],result['xt'])
    result['loss_mask']=result['valid_mask']&~result['known_mask']
    if any(result[key].dtype!=torch.float32 for key in ('x0','x1','u','t','xt','context')):
        raise ValueError('Native data must remain FP32.')
    if not torch.equal(result['xt'][result['known_mask']],result['context'][result['known_mask']]):
        raise RuntimeError('Known coordinates changed.')
    result['free_coordinates']=int(result['loss_mask'].sum())*9
    return result


def model_inputs(batch):
    """Explicit allowlist prevents target/GT/source IDs from becoming conditions."""
    return tuple(batch[key] for key in ('xt','t','y','valid_mask','known_mask'))
