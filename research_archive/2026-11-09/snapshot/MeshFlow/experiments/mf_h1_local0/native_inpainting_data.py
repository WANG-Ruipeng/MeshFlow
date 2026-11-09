"""Fixed-source-face data and free-only official OT for native inpainting.

CPU only. Full targets and source IDs serve decomposition/supervision/auditing,
never additional model conditions. Sampling noise is independent Gaussian; OT
is confined to the fresh training/teacher free targets before final target x2.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from .learning_data import array_hash
from .prompt_dataset import file_sha256
from .inference_pilot_data import _preserve_cpu_rng
from .nxc112_data import _persist

PROJECT=Path(__file__).resolve().parent
HISTORY=PROJECT/'outputs/soft_matching_20260928_210321'
EXPECTED_INPUT_SHA='e9945db9ed46d695db2e655a248295525f3d5283d8aece7560797e5e6dc96252'
EXPECTED_RAW_SHA='418fdd5268f77aae13c20b1a619937a6c851e56fe2611f6b8cfc70a9bd9854a0'
BASE_SHA='b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222'
UIDS=('2af09bd8df40506c9e646678ef50aa3d','5b0e833cf2ea465e42bd82e95587d8af')
DEVELOPMENT_SEEDS=(9101,9102)
CONFIRMATION_SEEDS=(9201,9202,9203,9204)
PERMUTATIONS=np.asarray(list(itertools.permutations(range(3))),dtype=np.int64)
CYCLIC=np.asarray([[0,1,2],[1,2,0],[2,0,1]],dtype=np.int64)
SHARED_KEYS=('x0','x1','u','t','known_mask','valid_mask','known_target_ids','source_face_ids','corner_permutations','face_permutation')


def _record(path):
    path=Path(path);return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=file_sha256(path))


def _hash_fields(value,keys):
    h=hashlib.sha256()
    for key in keys:
        a=value[key]
        if torch.is_tensor(a):a=a.detach().cpu().numpy()
        h.update(key.encode());h.update(array_hash(np.asarray(a)).encode())
    return h.hexdigest()


def verify_source_faces(target,constraints,bbox):
    """Unique *whole* source face match; preserve the supplied C bytes."""
    T=np.asarray(target,dtype=np.float64).reshape(-1,3,3)
    C=np.asarray(constraints,dtype=np.float64).reshape(-1,3,3)
    tolerance=1e-6*float(bbox);rows=[];ids=[]
    for index,c in enumerate(C):
        errors=np.stack([np.linalg.norm(T[:,perm,:]-c[None],axis=-1).max(-1) for perm in PERMUTATIONS],axis=-1)
        candidates=np.flatnonzero(errors.min(-1)<=tolerance)
        if len(candidates)!=1:
            error=RuntimeError('BLOCKED_CONTEXT_NOT_SOURCE_FACE')
            error.source_face_audit=dict(target_index=index,matching_source_ids=candidates.tolist(),tolerance=tolerance)
            raise error
        source=int(candidates[0]);pi=int(errors[source].argmin());ids.append(source)
        rows.append(dict(target_index=index,source_face_id=source,source_vertex_permutation=PERMUTATIONS[pi].tolist(),
            max_vertex_error=float(errors[source,pi]),matching_source_count=1,ambiguous=False,tolerance=tolerance))
    if len(set(ids))!=len(ids):raise RuntimeError('BLOCKED_CONTEXT_NOT_SOURCE_FACE: duplicate source IDs')
    return ids,rows


def _original_pre_ot(parent,target):
    """Capture actual upstream __getitem__ coordinates immediately before OT.

    Instance-local no-noise callback avoids full-object OT. Official class code
    and global NumPy RNG are unchanged. No divide-final-by-two assumption.
    """
    from datasets.mesh_dataset import ObjaverseDataset,sort_triangle_soup
    if file_sha256(parent['path'])!=parent['sha256']:raise RuntimeError('Raw source asset changed')
    with np.load(parent['path'],allow_pickle=False) as z:
        raw={key:z[key].copy() for key in ('vertices','faces','faces_num','uid')}
    dataset=ObjaverseDataset.__new__(ObjaverseDataset)
    dataset.__dict__.update(training=True,use_custom_prior=False,do_dataset_normalize=True,
        max_face_length=800,use_rot_aug=False,use_scale_aug=False,noise_sort='ot',use_permut_aug=False,std=.3762,data=[raw])
    captured=[]
    def capture(coords):
        captured.append(coords.copy());return coords,np.zeros_like(coords)
    dataset.sample_noise=capture
    sample=dataset[0]
    if len(captured)!=1 or int(sample['num_faces'])!=112:raise RuntimeError('Official pre-OT capture failed')
    pre=sort_triangle_soup(captured[0]).reshape(112,3,3)
    final=sort_triangle_soup(np.asarray(sample['coords'])).astype(np.float32).reshape(112,9)
    if not np.array_equal(final,target) or not np.array_equal((pre.reshape(112,9)*2).astype(np.float32),target):
        raise RuntimeError('Actual pre-OT target x2 does not exactly recover archived target')
    return pre,dict(source=_record(parent['path']),pre_ot_dtype=str(pre.dtype),pre_ot_sha256=array_hash(pre),
        pre_ot_shape=list(pre.shape),official_getitem_capture=True,full_object_OT_executed=False,
        canonical_final_FP32_bitwise_equals_archive=True,target_final_multiplier=2.,noise_final_multiplier=1.,
        pre_ot_not_inferred_by_dividing_archived_target=True)


def prepare_inputs(output,persist=True):
    out=Path(output).resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('native_inpainting_'):
        raise ValueError('Use a new native_inpainting_* directory')
    if torch.cuda.is_initialized():raise RuntimeError('Prepare CPU inputs before CUDA initialization')
    with _preserve_cpu_rng():
        paths={name:HISTORY/name for name in ('SOFT_MATCHING_REPORT.md','protocol.json','inputs.npz','results.json')}
        old=json.loads(paths['results.json'].read_text(encoding='utf-8-sig'))
        protocol=json.loads(paths['protocol.json'].read_text(encoding='utf-8-sig'))
        if not paths['SOFT_MATCHING_REPORT.md'].read_text(encoding='utf-8-sig').strip():raise RuntimeError('Missing prior report')
        if (old['status']!='PASS_EXECUTION' or old['base_before']!=BASE_SHA or old['base_after']!=BASE_SHA
                or not old['protected_unchanged'] or file_sha256(paths['inputs.npz'])!=EXPECTED_INPUT_SHA
                or file_sha256(paths['protocol.json'])!=old['protocol_sha256']):raise RuntimeError('Prior fixed soft task identity differs')
        if (protocol['seeds']!=list(DEVELOPMENT_SEEDS) or protocol['K']!=[1,2] or protocol['N']!=112
                or protocol['lambda_max']!=[0,5,20] or protocol['steps']!=50):raise RuntimeError('Prior soft task protocol differs')
        with np.load(paths['inputs.npz'],allow_pickle=False) as z:old_arrays={key:z[key].copy() for key in z.files}
        recorded=old['assets']['inputs']['arrays']
        if set(old_arrays)!=set(recorded) or any(array_hash(v)!=recorded[k]['sha256'] for k,v in old_arrays.items()):raise RuntimeError('Prior input arrays changed')
        raw_path=Path(old['raw']['path'])
        if raw_path.resolve()!=(HISTORY/'raw/trajectories.npz').resolve() or file_sha256(raw_path)!=EXPECTED_RAW_SHA:raise RuntimeError('Prior raw trajectories missing/changed')
        arrays={};records=[];pre_records=[];parents=[]
        for i,uid in enumerate(UIDS):
            parent=copy.deepcopy(old['assets']['parents'][i]['source_file'])
            if parent['uid']!=uid or parent['faces']!=112:raise RuntimeError('Parent identity differs')
            parents.append(parent);target=old_arrays[f'parent{i}_target'];L=old['assets']['parents'][i]['bbox_diagonal']
            pre,pre_audit=_original_pre_ot(parent,target);pre_records.append(pre_audit)
            arrays[f'parent{i}_target']=target.copy();arrays[f'parent{i}_pre_ot']=pre.copy()
            if not np.array_equal(old_arrays[f'parent{i}_K1_constraints'],old_arrays[f'parent{i}_K2_constraints'][:1]):raise RuntimeError('Historical K relationship changed')
            for k in (1,2):
                prefix=f'parent{i}_K{k}_';C=old_arrays[prefix+'constraints'].copy()
                source_ids,matches=verify_source_faces(target,C,L)
                free_ids=np.asarray([f for f in range(112) if f not in source_ids],dtype=np.int64)
                arrays[prefix+'constraints']=C;arrays[prefix+'checks']=old_arrays[prefix+'checks'].copy()
                arrays[prefix+'source_face_ids']=np.asarray(source_ids,dtype=np.int64);arrays[prefix+'free_source_ids']=free_ids
                arrays[prefix+'free_target']=target[free_ids].copy();arrays[prefix+'pre_ot_free']=pre[free_ids].copy()
                records.append(dict(case_id=f'P{i}_K{k}',parent_index=i,parent_uid=uid,K=k,N=112,N_free=112-k,bbox_diagonal=L,
                    constraints_sha256=array_hash(C),target_sha256=array_hash(target),free_target_sha256=array_hash(target[free_ids]),
                    source_face_ids=source_ids,source_face_matches=matches,context_preserved_from_archive=True,
                    pre_ot_free_sha256=array_hash(pre[free_ids]),decomposition_source_counts=dict(known=k,free=112-k,total=112),
                    no_source_face_overlap=True,known_slots_sampling=list(range(k)),soft_tau=3*(.05*L)**2))
        for seed in DEVELOPMENT_SEEDS:
            value=old_arrays[f'noise_seed{seed}'];assert value.shape==(1,112,9) and value.dtype==np.float32
            if array_hash(value)!=old['assets']['noise_hashes'][str(seed)]:raise RuntimeError('Development Gaussian archive hash differs')
            arrays[f'noise_seed{seed}']=value.copy()
        for seed in CONFIRMATION_SEEDS:
            arrays[f'noise_seed{seed}']=torch.randn((1,112,9),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32).numpy()
        artifact=_persist(out/'inputs.npz',arrays,persist)
        if persist:
            with np.load(artifact['path'],allow_pickle=False) as z:loaded={key:z[key].copy() for key in z.files}
            if set(loaded)!=set(arrays) or any(array_hash(loaded[k])!=array_hash(v) for k,v in arrays.items()):raise RuntimeError('New input archive roundtrip differs')
            arrays=loaded
        cases=[]
        for record in records:
            i,k=record['parent_index'],record['K'];prefix=f'parent{i}_K{k}_'
            cases.append(dict(**copy.deepcopy(record),constraints=arrays[prefix+'constraints'],checks=arrays[prefix+'checks'],
                full_target=arrays[f'parent{i}_target'],free_target=arrays[prefix+'free_target'],pre_ot_free=arrays[prefix+'pre_ot_free'],
                free_source_ids=arrays[prefix+'free_source_ids']))
        bank=dict(cases=cases,cases_by_key={(c['parent_index'],c['K']):c for c in cases},parents=parents,parent_uids=list(UIDS),
            noises={seed:torch.from_numpy(arrays[f'noise_seed{seed}'].copy()) for seed in DEVELOPMENT_SEEDS+CONFIRMATION_SEEDS},
            development_seeds=list(DEVELOPMENT_SEEDS),confirmation_seeds=list(CONFIRMATION_SEEDS),N=112,
            prior_results=old,prior_raw_path=str(raw_path),prior_protocol=protocol)
        audit=dict(status='PASS',source_face_gate='PASS',source_files={k:_record(v) for k,v in paths.items()},
            prior_raw=_record(raw_path),cases=records,official_pre_OT_capture=pre_records,inputs=artifact,
            prior_input_archive_exact=True,roundtrip_reloaded=bool(persist),development_noise_source='Actual old NPZ arrays, not regenerated from seeds',
            noise_hashes={str(seed):array_hash(bank['noises'][seed].numpy()) for seed in DEVELOPMENT_SEEDS+CONFIRMATION_SEEDS},
            confirmation_noise_rule='CPU torch.randn([1,112,9], generator=local manual_seed(9201..9204), FP32, no OT, all declared before model evaluation',
            confirmation_integer_overlap='9201 previously labeled the parent1 second-face interior-point NumPy check seed; this is not the new Torch Gaussian sampling tensor. Other seeds are predeclared by the user; no external/unlogged-run novelty claim.',
            coordinate_units='Official model coordinates, no physical units',model_forwards=0,optimizer_updates=0,
            CUDA_initialized=False,caller_CPU_RNG_preserved=True,
            sampling_difference='Soft starts all112 Gaussian faces. Inpainting replaces firstK slots by unchanged C; free slots are exactly Z[:,K:]. Same free noise across contexts at fixed K.',
            full_target_use='decomposition, training supervision and evaluation only; sampler model conditions limited to C,known/valid masks,t,N',
            source_implementation={name:_record(PROJECT.parent.parent/name) for name in ('datasets/mesh_dataset.py','utils/ot_utils.py','configs/snet/base-120m-ot-v-chair.yaml')},
            training_stream_seeds=dict(seed=10,noise=10010,time=30010,permutation=40010,order=60010,donor=70010),
            teacher_stream_seed=101010,confirmation_seeds=list(CONFIRMATION_SEEDS))
    return bank,audit


class NativeTrainingStream:
    """Fresh balanced eight-example stream with donor-independent shared RNGs."""
    RNG_KEYS=('noise_rng','time_rng','permutation_rng','order_rng','donor_rng')
    def __init__(self,bank,arm='Native_correct',seed=10):
        if arm not in ('Native_correct','Native_independent'):raise ValueError('Only two authorized arms')
        self.bank,self.arm,self.seed=bank,arm,int(seed)
        self.noise_rng=np.random.RandomState(seed+10000)
        self.time_rng=np.random.RandomState(seed+30000)
        self.permutation_rng=np.random.RandomState(seed+40000)
        self.order_rng=np.random.RandomState(seed+60000)
        self.donor_rng=np.random.RandomState(seed+70000)
        self.batch_index=0;self.sample_index=0;self.last_audit=None

    def state_dict(self):
        packed={}
        for key in self.RNG_KEYS:
            state=getattr(self,key).get_state()
            packed[key]=[state[0],state[1].tolist(),int(state[2]),int(state[3]),float(state[4])]
        return dict(rng=packed,batch_index=self.batch_index,sample_index=self.sample_index,seed=self.seed,arm=self.arm)

    def load_state_dict(self,state):
        if int(state['seed'])!=self.seed or state['arm']!=self.arm:raise RuntimeError('Stream restore arm/seed mismatch')
        for key in self.RNG_KEYS:
            value=state['rng'][key]
            getattr(self,key).set_state((value[0],np.asarray(value[1],dtype=np.uint32),int(value[2]),int(value[3]),float(value[4])))
        self.batch_index=int(state['batch_index']);self.sample_index=int(state['sample_index'])

    def _sample(self,parent,k):
        from utils.ot_utils import optimal_sum_numpy
        case=self.bank['cases_by_key'][(parent,k)];nf=112-k
        # The original pipeline shuffles faces/cyclic corners before OT. Known
        # faces are absent from the pairwise matrix; final target x2 is later.
        free_order=self.permutation_rng.permutation(nf)
        corner_choice=self.permutation_rng.randint(0,3,size=112)
        corners=CYCLIC[corner_choice]
        pre=case['pre_ot_free'][free_order].copy()
        pre=pre[np.arange(nf)[:,None],corners[k:]]
        raw_noise=self.noise_rng.randn(nf,3,3)
        paired=optimal_sum_numpy(pre,raw_noise,optimal=True)
        final=(pre*2).astype(np.float32)
        if not np.array_equal(final,case['free_target'][free_order].reshape(nf,3,3)[np.arange(nf)[:,None],corners[k:]]):
            raise RuntimeError('Free target original pre-OT scaling/ordering mismatch')
        noise=paired.astype(np.float32)
        x0=np.zeros((112,3,3),dtype=np.float32);x1=np.zeros_like(x0)
        x0[k:]=noise;x1[k:]=final
        known=np.arange(112)<k;valid=np.ones(112,dtype=bool)
        known_ids=np.full(112,-1,dtype=np.int64);known_ids[:k]=np.arange(k)
        source_ids=np.concatenate([np.asarray(case['source_face_ids']),case['free_source_ids'][free_order]])
        contexts=np.zeros((2,112,3,3),dtype=np.float32)
        for donor in range(2):
            original=self.bank['cases_by_key'][(donor,k)]['constraints']
            contexts[donor,:k]=original[np.arange(k)[:,None],corners[:k]]
        order=self.permutation_rng.permutation(112)
        x0=x0[order].reshape(112,9);x1=x1[order].reshape(112,9)
        known,valid=known[order],valid[order];known_ids=known_ids[order];source_ids=source_ids[order]
        contexts=contexts[:,order].reshape(2,112,9);corners=corners[order]
        t=np.asarray(1/(1+np.exp(-self.time_rng.randn())),dtype=np.float32)
        donor=parent if self.arm=='Native_correct' else int(self.donor_rng.randint(2))
        u=x1-x0;xt=(np.float32(1)-t)*x0+t*x1;xt[known]=contexts[donor,known]
        sample=dict(x0=x0,x1=x1,u=u,xt=xt,t=t,y=np.asarray(112,dtype=np.int64),known_mask=known,valid_mask=valid,
            context=contexts[donor].copy(),context_by_parent=contexts,parent_index=parent,donor_index=donor,K=k,
            known_target_ids=known_ids,source_face_ids=source_ids,corner_permutations=corners,face_permutation=order,
            free_gaussian_before_OT=raw_noise,free_noise_after_OT=paired,free_target_before_OT=pre)
        sample['shared_input_sha256']=shared_sample_hash(sample)
        sample['context_sha256']=array_hash(sample['context'])
        self.sample_index+=1
        return sample

    def next_effective_batch(self,size=8):
        if size!=8:raise ValueError('Fixed effective batch8 required')
        pairs=[(p,k) for p in range(2) for k in (1,2) for repeat in range(2)]
        samples=[self._sample(*pairs[i]) for i in self.order_rng.permutation(8)]
        self.batch_index+=1
        shared=[s['shared_input_sha256'] for s in samples]
        donors=[s['donor_index'] for s in samples]
        self.last_audit=dict(batch_index=self.batch_index,samples=8,shared_input_hashes=shared,
            shared_batch_sha256=hashlib.sha256(''.join(shared).encode()).hexdigest(),
            parent_indices=[s['parent_index'] for s in samples],K=[s['K'] for s in samples],
            free_face_counts=[112-s['K'] for s in samples],effective_free_coordinates=sum((112-s['K'])*9 for s in samples),
            times=[float(s['t']) for s in samples],donor_indices=donors,
            context_hashes=[s['context_sha256'] for s in samples],
            accidentally_correct_count=sum(s['donor_index']==s['parent_index'] for s in samples),
            donor_rng_seed=self.seed+70000,donor_independent_of_target_stream=True,
            Gaussian_hashes=[array_hash(s['free_gaussian_before_OT']) for s in samples],
            paired_noise_hashes=[array_hash(s['free_noise_after_OT']) for s in samples],
            pre_OT_target_hashes=[array_hash(s['free_target_before_OT']) for s in samples],
            known_slots=[np.flatnonzero(s['known_mask']).tolist() for s in samples],
            no_full112_OT=True,official_OT_call_count=8)
        if self.last_audit['effective_free_coordinates']!=7956:raise RuntimeError('Effective batch free denominator differs')
        return samples


def shared_sample_hash(sample):
    fields={key:sample[key] for key in SHARED_KEYS}
    fields['xt_free']=sample['xt'][~sample['known_mask']]
    fields['parent_index']=np.asarray(sample['parent_index'],dtype=np.int64)
    fields['K']=np.asarray(sample['K'],dtype=np.int64)
    fields['raw_Gaussian']=sample['free_gaussian_before_OT']
    fields['OT_paired_noise']=sample['free_noise_after_OT']
    fields['pre_OT_target']=sample['free_target_before_OT']
    return _hash_fields(fields,tuple(fields))


def with_context(sample,donor):
    """Swap only actual clean C, using the stored role/face/corner mapping."""
    if donor not in (0,1):raise ValueError('Two authorized context parents')
    changed=copy.deepcopy(sample);changed['donor_index']=int(donor)
    changed['context']=changed['context_by_parent'][donor].copy()
    changed['xt'][changed['known_mask']]=changed['context'][changed['known_mask']]
    changed['context_sha256']=array_hash(changed['context'])
    if shared_sample_hash(changed)!=sample['shared_input_sha256']:raise RuntimeError('Context replacement changed free/target/random stream')
    return changed


def native_collate(samples,device='cpu',t_override=None):
    """Only xt/t/y/known_mask/valid_mask are passed to the native model."""
    if not samples:raise ValueError('Empty native batch')
    result={key:torch.from_numpy(np.stack([np.asarray(s[key]) for s in samples])).to(device)
            for key in ('x0','x1','u','t','y','known_mask','valid_mask','context')}
    if t_override is not None:result['t']=torch.full_like(result['t'],float(t_override))
    result['xt']=(1-result['t'][:,None,None])*result['x0']+result['t'][:,None,None]*result['x1']
    result['xt']=torch.where(result['known_mask'][...,None],result['context'],result['xt'])
    result['loss_mask']=result['valid_mask']&~result['known_mask']
    if not torch.equal(result['xt'][result['known_mask']],result['context'][result['known_mask']]):raise RuntimeError('Known coordinates changed')
    if any(result[k].dtype!=torch.float32 for k in ('x0','x1','u','t','xt','context')):raise RuntimeError('Native data must be FP32')
    result['free_coordinates']=int(result['loss_mask'].sum())*9
    return result


def run_cpu_tests(bank):
    """Bounded real-data checks, no model and no optimization."""
    if torch.cuda.is_initialized():raise RuntimeError('CPU tests must precede CUDA')
    from utils.ot_utils import optimal_sum_numpy
    correct=NativeTrainingStream(bank,'Native_correct',10)
    independent=NativeTrainingStream(bank,'Native_independent',10)
    tests=[];hashes=[];donors=[];matched=0;OT_calls=0
    for _ in range(2):
        a=correct.next_effective_batch();b=independent.next_effective_batch();OT_calls+=16
        assert correct.last_audit['shared_batch_sha256']==independent.last_audit['shared_batch_sha256']
        assert correct.last_audit['Gaussian_hashes']==independent.last_audit['Gaussian_hashes']
        assert correct.last_audit['effective_free_coordinates']==7956
        for sa,sb in zip(a,b):
            assert sa['shared_input_sha256']==sb['shared_input_sha256'];hashes.append(sa['shared_input_sha256'])
            assert np.array_equal(sa['xt'][~sa['known_mask']],sb['xt'][~sb['known_mask']])
            assert len(set(sa['source_face_ids']))==112 and sorted(sa['source_face_ids'].tolist())==list(range(112))
            assert np.count_nonzero(sa['known_mask'])==sa['K']
            assert np.count_nonzero(sa['u'][sa['known_mask']])==0
            swapped=with_context(sa,1-sa['parent_index']);assert shared_sample_hash(swapped)==sa['shared_input_sha256']
            matched+=sb['donor_index']==sb['parent_index'];donors.append(sb['donor_index'])
        batch=native_collate(a)
        assert batch['xt'].shape==(8,112,9) and batch['free_coordinates']==7956 and batch['valid_mask'].all()
        assert torch.equal(batch['xt'][batch['known_mask']],batch['context'][batch['known_mask']])
        # An independent repeated call to the unmodified upstream OT confirms
        # actual pre-x2 target and full free Gaussian, without any known faces.
        s=a[0];again=optimal_sum_numpy(s['free_target_before_OT'],s['free_gaussian_before_OT'],optimal=True);OT_calls+=1
        assert np.array_equal(again,s['free_noise_after_OT'])
    tests.extend(['two_arms_identical_target_K_Gaussian_OT_t_roles_and_permutation_stream',
        'donor_context_only_changes_known_coordinates','all8_samples_balanced_and_denominator7956',
        'no_source_overlap_or_missing_source_face','free_only_official_nested_OT_at_captured_pre_x2_scale',
        'all_known_context_coordinates_exact_FP32','known_velocity_targets_zero_and_lossmask_excludes_known'])
    checkpoint=copy.deepcopy(correct.state_dict());once=correct.next_effective_batch();OT_calls+=8
    correct.load_state_dict(checkpoint);twice=correct.next_effective_batch();OT_calls+=8
    assert all(x['shared_input_sha256']==y['shared_input_sha256'] and x['context_sha256']==y['context_sha256'] for x,y in zip(once,twice))
    assert len(set(hashes))==len(hashes)
    tests.extend(['state_restore_repeats_next_batch_bitwise','fresh_noise_for_each_access'])
    return dict(status='PASS',tests=tests,shared_samples_checked=16,donor_draws=donors,accidentally_correct_count=matched,
        sample_OT_calls=48,OT_reference_calls=2,total_OT_calls=OT_calls,model_forwards=0,optimizer_updates=0,CUDA_initialized=False,
        permutation_rule='Fresh free-face permutation, original 3 cyclic corner permutations, then full112 face permutation; all masks/time/roles share mapping')
