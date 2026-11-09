"""CPU-only real assets, N-by-C noises, and balanced two-parent 112-face data.

Only new NPZ inputs are persisted. The caller owns the single results.json and
stores the returned JSON-safe audit. No model, GPU or optimizer is used here.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .learning_data import load_bank as load_learning_bank, array_hash, base_audit
from .prompt_dataset import (RealChairStream, numpy_stream, make_prompt,
    sample_surface_primitive, file_sha256, collate_samples)
from .evaluate import guidance_metrics

PROJECT=Path(__file__).resolve().parent
OLD=PROJECT/'outputs/run_20260928_01'
LEARNING=PROJECT/'outputs/learning_diagnosis_20260928_155505'
MIXED='mixed_2edges_1face'
N_VALUES=(217,348,750,768,512)
A_SEEDS=(6101,6102)
B_SEEDS=(7101,7102,7103,7104)
A_UIDS=('90c6d1df1f83329fe1181b0e584cdf9b','6ec4d5fa96861bcd70311dea98ba710d',
        'dae8b3ac69d80cf1b94e60a139ac4b1c','d481cc9e7d6d7d64baf5b49ad7809302')
A_FACES=(768,348,750,217)
B_UIDS=('2af09bd8df40506c9e646678ef50aa3d','5b0e833cf2ea465e42bd82e95587d8af')
CANDIDATE_SEEDS=tuple(range(27100,27900,100))
EXPECTED_BANK_SHA='2ec76990046ada5105b5b70f095064c3a1dc7b3cb3497230bd18094ee8bfe625'
EXPECTED_BASE_SHA='b32197096ecf8d662bfee18bbc4abf16cc4b4690cd28e5c47d14850729d2c222'
PROMPT_KEYS=('vertices','arity','primitive_mask','points','point_primitive','point_mask')


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _state_hash(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(key.encode());digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _pipeline_inputs(parents,seed,required_faces=None):
    """Run the unchanged official __getitem__, retaining genuine fresh OT pairs."""
    from datasets.mesh_dataset import sort_triangle_soup
    stream=RealChairStream(parents,seed=seed)
    samples,targets,rows=[],{},[]
    for index,parent in enumerate(parents):
        with np.load(parent['path'],allow_pickle=False) as raw:
            vertices,faces=raw['vertices'],raw['faces'];n=int(raw['faces_num'])
            uid=raw['uid'].item()
            if isinstance(uid,bytes):uid=uid.decode()
            if str(uid)!=parent['uid']:raise ValueError('Raw UID differs from manifest')
            if required_faces is not None and n!=required_faces:raise ValueError('Required real raw face count differs')
            if n!=len(faces) or n!=int(parent['faces']) or faces.shape!=(n,3):
                raise ValueError('Raw face count/shape disagrees with original train manifest')
            if not np.issubdtype(faces.dtype,np.integer) or faces.min()<0 or faces.max()>=len(vertices):
                raise ValueError('Invalid raw triangle indices')
            if not np.isfinite(vertices).all():raise ValueError('Nonfinite raw coordinates')
            triangle=vertices[faces]
            scale=float(np.ptp(vertices,axis=0).max())
            nondegenerate=np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)>scale**2*1e-12
            unique_index_faces=len(np.unique(np.sort(faces,axis=1),axis=0))
            if required_faces is not None and (int(nondegenerate.sum())!=required_faces or unique_index_faces!=required_faces):
                raise ValueError('112 effective nondegenerate, nonduplicated raw triangles are required')
        with numpy_stream(stream.data_rng):sample=stream.dataset[index]
        sample['parent_uid']=parent['uid']
        sample['t']=float(1/(1+np.exp(-stream.time_rng.randn())))
        if (int(sample['num_faces'])!=n or int(sample['len'])!=n
                or sample['coords'].shape!=(n,9) or sample['noise'].shape!=(n,9)
                or not np.isfinite(sample['coords']).all() or not np.isfinite(sample['noise']).all()):
            raise ValueError('Official pipeline count/geometry/noise failed')
        target=sort_triangle_soup(np.asarray(sample['coords'])).astype(np.float32)
        targets[parent['uid']]=target;samples.append(sample)
        rows.append(dict(parent_uid=parent['uid'],raw_faces=n,raw_nondegenerate_faces=int(nondegenerate.sum()),
            raw_unique_index_faces=unique_index_faces,official_num_faces=int(sample['num_faces']),
            official_token_shape=list(sample['coords'].shape),official_noise_shape=list(sample['noise'].shape),
            canonical_target_sha256=array_hash(target),coordinates_finite=True,
            source_path=parent['path'],source_sha256=file_sha256(parent['path']),
            no_truncation_duplication_or_padding=True))
    return samples,targets,rows


def _stage_a_assets():
    bank=load_learning_bank(LEARNING)
    if bank['manifest_sha256']!=EXPECTED_BANK_SHA or tuple(bank['parent_uids'])!=A_UIDS:
        raise ValueError('Stage A historical learning-bank identity differs')
    report_path=LEARNING/'learning_diagnosis_results.json';report=_read(report_path)
    if (report.get('status')!='PASS_EXECUTION' or report.get('data_bank_sha256')!=EXPECTED_BANK_SHA
            or report.get('backbone_before')!=EXPECTED_BASE_SHA or report.get('backbone_after')!=EXPECTED_BASE_SHA
            or not report.get('protected_assets_unchanged')):
        raise ValueError('Historical learning run did not finish with the expected frozen assets')
    _,targets,pipeline=_pipeline_inputs(bank['parents'],seed=29110)
    for index,uid in enumerate(A_UIDS):
        if pipeline[index]['raw_faces']!=A_FACES[index] or not np.array_equal(targets[uid],bank['targets'][uid]):
            raise ValueError('Stage A bank target differs from the original official data pipeline')
    weights={}
    for arm in ('Q_correct','Q_independent'):
        record=next(c for c in report['arms'][arm]['checkpoints'] if c['step']==1000)
        path=Path(record['path'])
        if path.resolve()!= (LEARNING/arm/'checkpoints/adapter_step1000.pt').resolve():
            raise ValueError('Stage A must use the historical learning step1000 path')
        digest=file_sha256(path)
        if digest!=record['sha256']:raise ValueError('Historical step1000 file hash changed')
        saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        state_digest=_state_hash(saved['adapter'])
        if (saved.get('checkpoint_schema')!='learning_diagnosis_v1' or saved['step']!=1000
                or saved['arm']!=arm or saved['base_state_sha256']!=EXPECTED_BASE_SHA
                or state_digest!=record['adapter_sha256']):
            raise ValueError('Historical step1000 checkpoint identity/config differs')
        weights[arm]=dict(path=str(path),bytes=path.stat().st_size,sha256=digest,
                         adapter_state_sha256=state_digest,step=1000,adapter_architecture='Q_center')
        del saved
    audit=dict(status='PASS',source_learning_directory=str(LEARNING),bank_manifest_sha256=EXPECTED_BANK_SHA,
        source_results_sha256=file_sha256(report_path),parents=copy.deepcopy(bank['parents']),
        official_pipeline_checks=pipeline,adapter_checkpoints=weights,base_state_sha256=EXPECTED_BASE_SHA,
        mixed_prompt_records=[copy.deepcopy(r) for r in bank['records'] if r['kind']==MIXED],
        prompt_points_and_checks_reused_without_change=True)
    return bank,audit


def _stage_b_parents():
    split_path=OLD/'split_manifest.json';split=_read(split_path)
    if file_sha256(split['split_path'])!=split['split_sha256']:raise ValueError('Author train split hash changed')
    with np.load(split['split_path'],allow_pickle=True) as saved:
        author_train={str(row['uid']) for row in saved['npz_list'].tolist()}
    test_path=Path(split['split_path']).with_name('test.npz')
    with np.load(test_path,allow_pickle=True) as saved:
        author_test={str(row['uid']) for row in saved['npz_list'].tolist()}
    diagnostic={p['uid'] for p in split['diagnostic']}
    by_uid={p['uid']:p for p in split['train']}
    if len(set(B_UIDS))!=2 or any(uid not in by_uid or uid not in author_train or uid in author_test or uid in diagnostic for uid in B_UIDS):
        raise ValueError('Both predeclared candidates must be distinct original author-train parents, excluding old diagnostic/test')
    parents=[copy.deepcopy(by_uid[uid]) for uid in B_UIDS]
    if any(int(p['faces'])!=112 for p in parents):raise ValueError('Manifest candidate face counts are not both 112')
    return parents,dict(source_train_manifest_sha256=file_sha256(split_path),
        author_train_split_sha256=file_sha256(split['split_path']),author_test_split_sha256=file_sha256(test_path),
        original_train_membership_verified=True,old_diagnostic_and_test_excluded=True)


def _noise(n,seed):
    return torch.randn((1,n,9),generator=torch.Generator(device='cpu').manual_seed(seed),device='cpu',dtype=torch.float32)


def _persist(path,arrays,persist):
    metadata={key:dict(dtype=str(value.dtype),shape=list(value.shape),sha256=array_hash(value)) for key,value in arrays.items()}
    if not persist:return dict(path=str(path),sha256=None,arrays=metadata,persisted=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays):raise ValueError('Existing new input archive has different array keys')
            for key,value in arrays.items():
                if array_hash(saved[key])!=metadata[key]['sha256']:
                    raise ValueError('Existing input differs; never overwrite: '+str(path)+' '+key)
    else:
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
    return dict(path=str(path.resolve()),sha256=file_sha256(path),arrays=metadata,persisted=True)


def _a_bank(old,noises,record):
    records=[r for r in old['records'] if r['kind']==MIXED]
    bank=dict(old)
    bank.update(stage='A',records=records,records_by_key={(r['parent_index'],MIXED):r for r in records},
        prompts={(uid,MIXED):old['prompts'][(uid,MIXED)] for uid in A_UIDS},
        checks={(uid,MIXED):old['checks'][(uid,MIXED)] for uid in A_UIDS},
        eval_noises=noises,noises=noises,N_values=list(N_VALUES),noise_seeds=list(A_SEEDS),
        adapter_checkpoints=record['adapter_checkpoints'])
    return bank


def _b_bank(arrays,record,gate):
    selected=gate['selected_candidate_index'];parents=record['parents']
    prompts={};targets={};checks={};teacher=[]
    for index,parent in enumerate(parents):
        uid=parent['uid'];prefix=f'candidate{selected}_parent{index}_'
        prompts[(uid,MIXED)]={key:arrays[prefix+key].copy() for key in PROMPT_KEYS}
        targets[uid]=arrays[f'parent{index}_target'].copy();checks[(uid,MIXED)]=arrays[prefix+'check_points'].copy()
        sample={key:arrays[f'teacher{index}_{key}'].copy() for key in ('coords','noise','num_faces','len','t')}
        sample.update(parent_uid=uid,uid=uid,t=float(sample['t']),prompt=copy.deepcopy(prompts[(uid,MIXED)]))
        teacher.append(sample)
    noises={seed:torch.from_numpy(arrays[f'noise_seed{seed}'].copy()) for seed in B_SEEDS}
    return dict(stage='B',parents=parents,parent_uids=[p['uid'] for p in parents],uids=[p['uid'] for p in parents],
        prompts=prompts,targets=targets,checks=checks,records=record['records'],
        records_by_key={(r['parent_index'],MIXED):r for r in record['records']},
        teacher_samples=teacher,eval_noises=noises,noises=noises,gate=gate,
        N_values=[112],noise_seeds=list(B_SEEDS),manifest_sha256=record['bank_content_sha256'])


def build_banks(output_dir,persist=True):
    """Return (stage_a_bank, stage_b_bank_or_None, json_safe_audit).

    ``persist=False`` prepares/tests entirely in memory and never occupies the
    final directory. Existing NPZs may only be reused after exact array equality.
    Stage B failures are recorded and never invalidate otherwise valid Stage A.
    """
    output=Path(output_dir).resolve()
    if not output.is_relative_to(PROJECT/'outputs') or not output.name.startswith('nxC_112_sandbox_'):
        raise ValueError('Use the new authorized nxC_112_sandbox output directory')
    if torch.cuda.is_initialized():raise RuntimeError('Build CPU banks before CUDA initialization')
    old,stage_a=_stage_a_assets()
    a_noises={};a_arrays={};a_noise_audit=[]
    for n in N_VALUES:
        for seed in A_SEEDS:
            noise=old['eval_noises'][seed].clone() if n==512 else _noise(n,seed)
            a_noises[(n,seed)]=noise;a_arrays[f'N{n}_seed{seed}']=noise.numpy().copy()
            a_noise_audit.append(dict(N=n,seed=seed,shape=list(noise.shape),sha256=array_hash(noise.numpy()),
                source='Exact archived learning-bank tensor clone' if n==512 else 'CPU torch.Generator(device=cpu).manual_seed(seed); torch.randn([1,N,9],float32)',
                target_independent=True,OT_paired=False,cross_N_same_latent_claim=False))
    stage_a['noises']=a_noise_audit
    artifacts=dict(stage_a_noises=_persist(output/'stage_a_noises.npz',a_arrays,persist))
    stage_b=dict(candidate_uids=list(B_UIDS),predeclared_prompt_seed_bases=list(CANDIDATE_SEEDS),
        prompt_rule='Candidate base+parent_index; two 8-point edges and one 16-point face, original corners/endpoints included',
        independent_check_rule='Candidate base+10000+parent_index; 128 independent internal points per primitive',
        candidates=[],teacher_seed=29210,parents=[],records=[])
    gate=dict(passed=False,status='BLOCKED_ASSET',reason='Stage B not checked yet',
        selected_candidate_index=None,D_bbox_pct=None,D_common_units=None,
        thresholds=dict(diagonal_max_bbox_pct=1e-4,offdiagonal_minus_diagonal_min_bbox_percentage_points=1.0))
    arrays={};bank_b=None
    try:
        parents,identity=_stage_b_parents();stage_b.update(parents=parents,identity=identity)
        teacher,targets,pipeline=_pipeline_inputs(parents,seed=29210,required_faces=112)
        stage_b['official_pipeline_checks']=pipeline
        diagonal={uid:float(np.linalg.norm(np.ptp(targets[uid].reshape(-1,3),axis=0))) for uid in B_UIDS}
        for index,(uid,sample) in enumerate(zip(B_UIDS,teacher)):
            arrays[f'parent{index}_target']=targets[uid]
            for key in ('coords','noise','num_faces','len','t'):arrays[f'teacher{index}_{key}']=np.asarray(sample[key])
        selected=None
        for candidate,seed in enumerate(CANDIDATE_SEEDS):
            prompts=[];checks=[]
            for index,uid in enumerate(B_UIDS):
                prompt=make_prompt(targets[uid],2,np.random.RandomState(seed+index))
                check_rng=np.random.RandomState(seed+10000+index)
                check=np.stack([sample_surface_primitive(v,int(a),128,check_rng,False)
                                for v,a in zip(prompt['vertices'],prompt['arity'])])
                if (prompt['arity'].tolist()!=[2,2,3] or prompt['points'].shape!=(32,3)
                        or np.bincount(prompt['point_primitive']).tolist()!=[8,8,16]):
                    raise ValueError('The fixed mixed primitive/32-point schema changed')
                prefix=f'candidate{candidate}_parent{index}_'
                for key,value in prompt.items():arrays[prefix+key]=value
                arrays[prefix+'check_points']=check;prompts.append(prompt);checks.append(check)
            percent=np.empty((2,2),dtype=np.float64);common=np.empty_like(percent)
            for i in range(2):
                for j in range(2):
                    percent[i,j]=guidance_metrics(targets[B_UIDS[j]],checks[i],diagonal[B_UIDS[i]])['guide_rms_bbox_pct']
                    common[i,j]=percent[i,j]*diagonal[B_UIDS[i]]/100
            passed=bool(percent[0,0]<=1e-4 and percent[1,1]<=1e-4
                and percent[0,1]-percent[0,0]>=1 and percent[1,0]-percent[1,1]>=1)
            stage_b['candidates'].append(dict(candidate_index=candidate,prompt_seeds=[seed,seed+1],
                check_seeds=[seed+10000,seed+10001],D_bbox_pct=percent.tolist(),D_common_units=common.tolist(),
                normalizing_bbox_diagonals=[diagonal[uid] for uid in B_UIDS],
                row_definition='Ci source bbox is fixed across columns j; equal weight across three primitives',
                directional_gaps_bbox_percentage_points=[float(percent[0,1]-percent[0,0]),float(percent[1,0]-percent[1,1])],passed=passed))
            if passed and selected is None:selected=candidate
        for seed in B_SEEDS:arrays[f'noise_seed{seed}']=_noise(112,seed).numpy()
        stage_b['noise_rule']='Shared CPU Generator seed7101..7104 standard Gaussian [1,112,9], no target OT coupling'
        stage_b['noise_hashes']={str(seed):array_hash(arrays[f'noise_seed{seed}']) for seed in B_SEEDS}
        if selected is None:
            gate.update(status='BLOCKED_PROMPT_DISTINGUISHABILITY',reason='None of the eight predeclared geometry-only candidates passed both directional thresholds')
        else:
            chosen=stage_b['candidates'][selected]
            gate.update(passed=True,status='PASS',reason='First passing candidate in the predeclared order; all eight candidate matrices retained',
                selected_candidate_index=selected,D_bbox_pct=chosen['D_bbox_pct'],D_common_units=chosen['D_common_units'],
                directional_gaps_bbox_percentage_points=chosen['directional_gaps_bbox_percentage_points'])
            for index,uid in enumerate(B_UIDS):
                stage_b['records'].append(dict(parent_index=index,parent_uid=uid,kind=MIXED,num_faces=112,
                    path=str(output/'stage_b_inputs.npz'),array_prefix=f'candidate{selected}_parent{index}_',
                    target_array_key=f'parent{index}_target',bbox_diagonal=diagonal[uid],
                    selected_candidate_index=selected,context_seed=chosen['prompt_seeds'][index],check_seed=chosen['check_seeds'][index]))
            for i,sample in enumerate(teacher):
                prefix=f'candidate{selected}_parent{i}_'
                sample['prompt']={key:arrays[prefix+key] for key in PROMPT_KEYS}
            stage_b['teacher_audit']=base_audit(teacher)
        stage_b['bank_content_sha256']=hashlib.sha256(json.dumps({k:array_hash(v) for k,v in arrays.items()},sort_keys=True).encode()).hexdigest()
        artifacts['stage_b_inputs']=_persist(output/'stage_b_inputs.npz',arrays,persist)
        if gate['passed']:bank_b=_b_bank(arrays,stage_b,gate)
    except (ValueError,KeyError,FileNotFoundError,OSError,RuntimeError) as error:
        gate.update(passed=False,status='BLOCKED_ASSET_OR_DATA',reason=repr(error))
        stage_b['error']=repr(error);bank_b=None
    audit=dict(schema=1,stage_a=stage_a,stage_b=stage_b,stage_b_gate=gate,artifacts=artifacts,
        geometry_guide_evaluations=sum(4 for _ in stage_b['candidates']),
        candidate_selection_used_model_outputs=False,official_model_forwards=0,optimizer_updates=0,
        new_noise_is_target_independent=True,persisted=bool(persist),cuda_initialized=torch.cuda.is_initialized())
    # Fail immediately if the central result cannot serialize this entire audit.
    json.dumps(audit,allow_nan=False)
    return _a_bank(old,a_noises,stage_a),bank_b,audit


def load_banks(output_dir,audit):
    """Read saved NPZ arrays using the audit already stored in central results."""
    output=Path(output_dir).resolve();old,stage_a=_stage_a_assets()
    if audit['stage_a']['bank_manifest_sha256']!=stage_a['bank_manifest_sha256']:
        raise ValueError('Stage A historical bank identity changed')
    loaded={}
    for name,record in audit['artifacts'].items():
        path=Path(record['path']).resolve()
        if not record['persisted'] or not path.is_relative_to(output) or file_sha256(path)!=record['sha256']:
            raise ValueError('Saved new input file identity changed')
        with np.load(path,allow_pickle=False) as saved:
            values={key:saved[key].copy() for key in saved.files}
        if set(values)!=set(record['arrays']):raise ValueError('Saved new input key set changed')
        if any(array_hash(value)!=record['arrays'][key]['sha256'] for key,value in values.items()):
            raise ValueError('Saved input content identity changed')
        loaded[name]=values
    a_noises={(n,seed):torch.from_numpy(loaded['stage_a_noises'][f'N{n}_seed{seed}']) for n in N_VALUES for seed in A_SEEDS}
    for seed in A_SEEDS:
        if not torch.equal(a_noises[(512,seed)],old['eval_noises'][seed]):raise ValueError('Archived N512 noise no longer identical')
    b=None
    if audit['stage_b_gate']['passed']:
        parents,_=_stage_b_parents()
        if parents!=audit['stage_b']['parents']:raise ValueError('Stage B real parent metadata changed')
        for parent in parents:
            if file_sha256(parent['path'])!=parent['sha256']:raise ValueError('Stage B original asset changed')
        b=_b_bank(loaded['stage_b_inputs'],audit['stage_b'],audit['stage_b_gate'])
    return _a_bank(old,a_noises,stage_a),b,audit


class Balanced112Stream:
    """Each effective batch has four fresh official samples of each real parent."""
    def __init__(self,bank,seed=10):
        if seed!=10 or bank is None or not bank['gate']['passed'] or tuple(bank['parent_uids'])!=B_UIDS:
            raise ValueError('Require the passed fixed two-parent 112-face bank and seed10')
        self.bank=bank;self.parents=copy.deepcopy(bank['parents'])
        self.base=RealChairStream(self.parents,seed=seed)
        self.order_rng=np.random.RandomState(seed+60000)
        self.batch_index=0;self.last_audit=None

    def next_effective_batch(self,size=8):
        if size!=8:raise ValueError('Balanced sandbox uses exactly effective batch8')
        order=self.order_rng.permutation(np.repeat(np.arange(2),4))
        samples=[]
        with numpy_stream(self.base.data_rng):
            for parent_index in order:
                sample=self.base.dataset[int(parent_index)]
                sample['parent_uid']=self.parents[int(parent_index)]['uid']
                samples.append(sample)
        for sample in samples:
            if int(sample['num_faces'])!=112 or sample['coords'].shape!=(112,9) or sample['noise'].shape!=(112,9):
                raise RuntimeError('Actual upstream training input no longer has 112 real faces')
            sample['prompt']=copy.deepcopy(self.bank['prompts'][(sample['parent_uid'],MIXED)])
            sample['t']=float(1/(1+np.exp(-self.base.time_rng.randn())))
        batch=collate_samples(samples,'cpu')
        if (tuple(batch['tokens'].shape)!=(8,112,9) or tuple(batch['masks'].shape)!=(8,112)
                or not bool(batch['masks'].all()) or not bool((batch['num_faces']==112).all())):
            raise RuntimeError('Balanced stream N/y/mask identity channel invariant failed')
        self.batch_index+=1
        self.last_audit=dict(batch_index=self.batch_index,**base_audit(samples),parent_indices=order.tolist(),
            per_parent_counts=np.bincount(order,minlength=2).tolist(),all_y_112=True,all_masks_identical_true=True,
            prompt_type=MIXED,primitive_sample_counts=[8,8,16],source='Unchanged official __getitem__, fresh Gaussian + original OT per item',
            scheduling_rng_seed=60010,parent_metadata_is_not_a_model_input=True)
        return samples

    def state_dict(self):
        return dict(**self.base.state_dict(),order_rng=self.order_rng.get_state())

    def load_state_dict(self,state,batch_index=0):
        if set(state)!={'data_rng','prompt_rng','time_rng','order_rng'}:raise ValueError('All four stream RNG states are required')
        self.base.load_state_dict({key:state[key] for key in ('data_rng','prompt_rng','time_rng')})
        self.order_rng.set_state(state['order_rng']);self.batch_index=int(batch_index);self.last_audit=None
