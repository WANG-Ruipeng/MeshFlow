"""CPU-only fixed geometry bank and matched real-data streams for Q learning.

Targets/noise/OT/t/N remain the upstream RealChairStream path. Only prompts are
replaced. Formal sampling noises are independent CPU Gaussian draws, never OT
coupled to targets. No model, optimizer, CUDA operation or training is provided.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .prompt_dataset import (RealChairStream, PROMPT_TYPES, make_prompt,
    sample_surface_primitive, file_sha256, collate_samples)

PROJECT = Path(__file__).resolve().parent
OLD_SPLIT = PROJECT/'outputs/run_20260928_01/split_manifest.json'
PROMPT_KEYS = ('vertices','arity','primitive_mask','points','point_primitive','point_mask')
EVALUATION_SEEDS = (6101,6102)
TEACHER_SEED = 19010
MANIFEST_NAME = 'learning_bank_manifest.json'


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str((value.dtype, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _update_array(digest, name, value):
    value = np.ascontiguousarray(value)
    digest.update(name.encode())
    digest.update(str((value.dtype,value.shape)).encode())
    digest.update(value.tobytes())


def base_tensor_hash(batch):
    """Actual FP32/padded official input tensors, excluding all prompt fields."""
    digest = hashlib.sha256()
    for key in ('tokens','noise','num_faces','masks','t'):
        value = batch[key].detach().cpu().contiguous()
        _update_array(digest,key,value.numpy())
    return digest.hexdigest()


def base_sample_hash(samples):
    """Original data values, including masks implied by each real face count."""
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample['parent_uid'].encode())
        for key in ('coords','noise','num_faces','len','t'):
            _update_array(digest,key,sample[key])
        _update_array(digest,'mask',np.ones(int(sample['num_faces']),dtype=bool))
    return digest.hexdigest()


def base_audit(samples, microbatch=2):
    if len(samples) % microbatch:
        raise ValueError('The fixed microbatch must divide this batch')
    return dict(base_input_hash=base_sample_hash(samples),
        padded_batch_base_hash=base_tensor_hash(collate_samples(samples,'cpu')),
        microbatch_base_hashes=[base_tensor_hash(collate_samples(samples[i:i+microbatch],'cpu'))
                               for i in range(0,len(samples),microbatch)],
        target_parent_uids=[sample['parent_uid'] for sample in samples],
        num_faces=[int(sample['num_faces']) for sample in samples],
        times=[float(sample['t']) for sample in samples],
        valid_coordinates=sum(int(sample['num_faces'])*9 for sample in samples),
        mask_policy='Original collate_samples true N, zero padding, microbatch2; no truncation')


def _write_json_new(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)


def _save_npz_new(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,**arrays)


def _selected_parents():
    source = json.loads(OLD_SPLIT.read_text(encoding='utf-8-sig'))
    parents,seen = [],set()
    for parent in source['train']:
        uid = str(parent.get('parent_uid',parent['uid']))
        if uid not in seen:
            if str(parent['uid']) != uid:
                raise ValueError('Ambiguous author parent UID')
            parents.append(copy.deepcopy(parent));seen.add(uid)
        if len(parents)==4:
            break
    excluded = {str(p.get('parent_uid',p['uid'])) for p in source['diagnostic']}
    if len(parents)!=4 or seen & excluded:
        raise ValueError('Require first four unique old training parents; old diagnostics are excluded')
    return parents,source


def _teacher_first_per_parent(parents):
    stream = RealChairStream(parents,seed=TEACHER_SEED)
    selected,seen_at = {},{}
    consumed = 0
    for _ in range(64):
        samples = stream.next_effective_batch(8)
        for sample in samples:
            uid = sample['parent_uid']
            if uid not in selected:
                selected[uid] = copy.deepcopy(sample)
                seen_at[uid] = consumed
            consumed += 1
        if len(selected)==4:
            break
    if len(selected)!=4:
        raise RuntimeError('Fixed teacher stream did not cover the preselected four parents')
    return [selected[p['uid']] for p in parents],dict(seed=TEACHER_SEED,
        data_rng_seed=TEACHER_SEED+10000,prompt_rng_seed=TEACHER_SEED+20000,
        time_rng_seed=TEACHER_SEED+30000,stream_samples_consumed=consumed,
        selected_zero_based_stream_indices=[seen_at[p['uid']] for p in parents],
        rule='First occurrence of each preselected parent in the unchanged seed19010 RealChairStream; no model scores inspected')


def prepare_bank(output):
    """Create once; reuse only through full artifact/source hash validation.

    Returns load_bank(output). The output directory may already contain the task
    and runtime configuration. A partial bank is never overwritten or repaired.
    """
    output = Path(output).resolve()
    if not output.is_relative_to((PROJECT/'outputs').resolve()):
        raise ValueError('Bank output must be under this experiment outputs directory')
    if (output/MANIFEST_NAME).exists():
        return load_bank(output)
    output.mkdir(parents=True,exist_ok=True)
    fixed = output/'learning_bank'
    fixed.mkdir(exist_ok=False)
    parents,source = _selected_parents()
    teacher,teacher_selection = _teacher_first_per_parent(parents)
    from datasets.mesh_dataset import sort_triangle_soup
    from .evaluate import surface_distance_squared
    records = []
    for index,(parent,sample) in enumerate(zip(parents,teacher)):
        # Reorder only an already-normalized official target. The saved teacher
        # retains its original target/OT-noise pairing and ordering unchanged.
        target = sort_triangle_soup(np.asarray(sample['coords'])).astype(np.float32)
        n = int(sample['num_faces'])
        if target.shape!=(n,9) or not 20<n<800:
            raise ValueError('Source target shape/count changed; no truncation is allowed')
        with np.load(parent['path'],allow_pickle=False) as raw:
            vertices = raw['vertices']
            bounds = np.array([vertices.min(0),vertices.max(0)])
        transform = dict(source_bbox_center=((bounds[0]+bounds[1])/2).tolist(),
            source_bbox_max_extent=float((bounds[1]-bounds[0]).max()),
            final_target_multiplier=2/.3762,
            transform='Original ObjaverseDataset.__getitem__ normalization; target coords * (1/0.5)',
            canonicalization='Official sort_triangle_soup reorders target faces/corners only')
        diagonal = float(np.linalg.norm(np.ptp(target.reshape(-1,3),axis=0)))
        for kind_index,kind in enumerate(PROMPT_TYPES):
            context_seed = 17100+index*10+kind_index
            check_seed = 18100+index*10+kind_index
            prompt = make_prompt(target,kind_index,np.random.RandomState(context_seed))
            check_rng = np.random.RandomState(check_seed)
            checks = np.stack([sample_surface_primitive(v,int(a),128,check_rng,False)
                               for v,a in zip(prompt['vertices'],prompt['arity'])])
            points = np.concatenate([prompt['points'],checks.reshape(-1,3)])
            max_distance = float(np.sqrt(surface_distance_squared(points,target.reshape(-1,3,3)).max()))
            if max_distance>1e-5:
                raise ValueError('Prompt/check coordinate frame failed the true target surface check')
            path = fixed/f'parent{index}_{kind}.npz'
            _save_npz_new(path,target=target,check_points=checks,**prompt)
            records.append(dict(parent_index=index,parent_uid=parent['uid'],kind=kind,
                path=str(path.resolve()),sha256=file_sha256(path),target_array_sha256=array_hash(target),
                bbox_diagonal=diagonal,num_faces=n,transform=transform,context_seed=context_seed,
                check_seed=check_seed,prompt_points=32,checks_per_primitive=128,
                prompt_surface_max_distance=max_distance,prompt_surface_threshold=1e-5))
    teacher_path = fixed/'teacher_inputs.npz'
    arrays = {}
    for index,sample in enumerate(teacher):
        for key in ('coords','noise','num_faces','len','t'):
            arrays[f'parent{index}_{key}'] = np.asarray(sample[key])
    _save_npz_new(teacher_path,**arrays)
    noise_path = fixed/'evaluation_noises.npz'
    noises = {f'seed{seed}':torch.randn((1,512,9),generator=torch.Generator(device='cpu').manual_seed(seed),
              device='cpu',dtype=torch.float32).numpy() for seed in EVALUATION_SEEDS}
    _save_npz_new(noise_path,**noises)
    manifest = dict(schema=1,status='PASS',source_split_path=str(OLD_SPLIT.resolve()),
        source_split_sha256=file_sha256(OLD_SPLIT),author_split=source['author_split'],
        selection='First four unique parent UIDs in the old 16-parent train manifest order',
        parents=parents,parent_uids=[p['uid'] for p in parents],records=records,
        old_diagnostic_parent_uids_excluded=[p['uid'] for p in source['diagnostic']],
        prompt_selection='Fixed geometry-only seeds; no generated outputs or evaluation scores used',
        static_target_noise_pairing='NONE: canonical targets are geometry/oracle only; formal noises are independent Gaussian samples',
        teacher=dict(path=str(teacher_path.resolve()),sha256=file_sha256(teacher_path),
            selection=teacher_selection,base_audit=base_audit(teacher),default_prompt_type='mixed_2edges_1face',
            prescribed_evaluation_times=[.05,.2,.5,.8,.95],training_time_distribution_changed=False),
        evaluation_noise=dict(path=str(noise_path.resolve()),sha256=file_sha256(noise_path),
            seeds=list(EVALUATION_SEEDS),shape=[1,512,9],dtype='float32',device='cpu',
            distribution='Independent torch.Generator CPU standard Gaussian; no target OT or pairing',
            array_sha256={str(seed):array_hash(noises[f'seed{seed}']) for seed in EVALUATION_SEEDS}),
        training_stream=dict(base='Unchanged RealChairStream(parents,seed10)',
            type_rng_seed=40010,donor_rng_seed=50010,type_sampling='Independent uniform among three prompt kinds',
            independent_donor_sampling='Independent uniform four parents, with replacement every sample, including correct parent with probability 1/4',
            microbatch=2,effective_batch=8,targets_or_noise_in_prompt=False),
        generalization_claim=False,phase3=False,model_forwards=0,optimizer_updates=0)
    _write_json_new(output/MANIFEST_NAME,manifest)
    return load_bank(output)


def load_bank(output):
    output = Path(output).resolve()
    manifest_path = output/MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('status')!='PASS' or manifest.get('schema')!=1:
        raise ValueError('A completed, supported learning bank is required')
    if file_sha256(OLD_SPLIT)!=manifest['source_split_sha256']:
        raise ValueError('Old split manifest changed')
    parents = manifest['parents']
    expected,_ = _selected_parents()
    if parents!=expected or len(manifest['records'])!=12:
        raise ValueError('Fixed parent selection or bank record count changed')
    for parent in parents:
        if file_sha256(parent['path'])!=parent['sha256']:
            raise ValueError('Original real parent asset changed')
    prompts,targets,checks,by_key = {},{},{},{}
    for record in manifest['records']:
        path = Path(record['path']).resolve()
        if not path.is_relative_to(output/'learning_bank') or file_sha256(path)!=record['sha256']:
            raise ValueError('Fixed prompt path/hash changed')
        with np.load(path,allow_pickle=False) as saved:
            uid,kind = record['parent_uid'],record['kind']
            prompts[(uid,kind)] = {key:saved[key].copy() for key in PROMPT_KEYS}
            target = saved['target'].copy()
            if uid in targets and not np.array_equal(targets[uid],target):
                raise ValueError('One parent has inconsistent target coordinates')
            targets[uid]=target;checks[(uid,kind)]=saved['check_points'].copy()
            by_key[(record['parent_index'],kind)] = record
    if len(prompts)!=12 or len(by_key)!=12:
        raise ValueError('Duplicate fixed prompt records')
    teacher_record,noise_record = manifest['teacher'],manifest['evaluation_noise']
    for record in (teacher_record,noise_record):
        path = Path(record['path']).resolve()
        if not path.is_relative_to(output/'learning_bank') or file_sha256(path)!=record['sha256']:
            raise ValueError('Teacher/evaluation-noise asset changed')
    teacher=[]
    with np.load(teacher_record['path'],allow_pickle=False) as saved:
        for index,parent in enumerate(parents):
            sample={key:saved[f'parent{index}_{key}'].copy() for key in ('coords','noise','num_faces','len','t')}
            sample['t']=float(sample['t']);sample['parent_uid']=parent['uid'];sample['uid']=parent['uid']
            sample['prompt']=copy.deepcopy(prompts[(parent['uid'],'mixed_2edges_1face')])
            teacher.append(sample)
    if base_audit(teacher)!=teacher_record['base_audit']:
        raise ValueError('Saved teacher data/masks differ from the original stream')
    with np.load(noise_record['path'],allow_pickle=False) as saved:
        eval_noises={seed:torch.from_numpy(saved[f'seed{seed}'].copy()) for seed in EVALUATION_SEEDS}
    for seed,noise in eval_noises.items():
        if tuple(noise.shape)!=(1,512,9) or noise.dtype!=torch.float32 or array_hash(noise.numpy())!=noise_record['array_sha256'][str(seed)]:
            raise ValueError('Independent formal Gaussian noise identity changed')
    return dict(output=output,manifest=manifest,manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),parents=parents,parent_uids=manifest['parent_uids'],
        records=manifest['records'],records_by_key=by_key,prompts=prompts,targets=targets,
        checks=checks,eval_noises=eval_noises,teacher_samples=teacher)


class PairedLearningStream:
    """Preserve upstream target/noise/time RNG; replace only the prompt payload."""
    def __init__(self,parents,bank,arm='Q_correct',seed=10):
        if arm not in ('Q_correct','Q_independent') or seed!=10:
            raise ValueError('Only the two newly authorized Q arms with seed10 are allowed')
        self.parents=copy.deepcopy(parents)
        self.bank=bank
        if self.parents!=bank['parents'] or len(self.parents)!=4:
            raise ValueError('Training must use exactly the four predeclared bank parents')
        self.arm=arm;self.seed=seed
        self.base=RealChairStream(self.parents,seed=seed)
        self.type_rng=np.random.RandomState(seed+40000)
        self.donor_rng=np.random.RandomState(seed+50000)
        self.parent_indices={p['uid']:i for i,p in enumerate(self.parents)}
        self.batch_index=0;self.last_audit=None

    def next_effective_batch(self,size=8):
        if size!=8:
            raise ValueError('The authorized effective training batch is exactly 8')
        samples=self.base.next_effective_batch(size)
        original=base_audit(samples)
        types=self.type_rng.randint(0,len(PROMPT_TYPES),size=size)
        if self.arm=='Q_independent':
            donors=self.donor_rng.randint(0,len(self.parents),size=size)
        else:
            donors=np.asarray([self.parent_indices[s['parent_uid']] for s in samples])
        for sample,kind_index,donor in zip(samples,types,donors):
            kind=PROMPT_TYPES[int(kind_index)];uid=self.parents[int(donor)]['uid']
            sample['prompt']=copy.deepcopy(self.bank['prompts'][(uid,kind)])
        if base_sample_hash(samples)!=original['base_input_hash']:
            raise RuntimeError('Replacing a prompt changed original target/noise/t/N/mask data')
        self.batch_index+=1
        self.last_audit=dict(batch_index=self.batch_index,arm=self.arm,**original,
            prompt_types=[PROMPT_TYPES[int(k)] for k in types],prompt_type_indices=types.tolist(),
            donor_indices=donors.tolist(),donor_parent_uids=[self.parents[int(i)]['uid'] for i in donors],
            donor_matches_target=[self.parents[int(i)]['uid']==s['parent_uid'] for i,s in zip(donors,samples)])
        return samples

    def state_dict(self):
        return dict(**self.base.state_dict(),type_rng=self.type_rng.get_state(),donor_rng=self.donor_rng.get_state())

    def load_state_dict(self,state,batch_index=0):
        expected={'data_rng','prompt_rng','time_rng','type_rng','donor_rng'}
        if set(state)!=expected:
            raise ValueError('Paired stream must restore all five independent RNG states')
        self.base.load_state_dict({k:state[k] for k in ('data_rng','prompt_rng','time_rng')})
        self.type_rng.set_state(state['type_rng']);self.donor_rng.set_state(state['donor_rng'])
        self.batch_index=int(batch_index);self.last_audit=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    bank=prepare_bank(args.output)
    print(json.dumps(dict(status='PASS',manifest=str(bank['manifest_path']),
        manifest_sha256=bank['manifest_sha256'],parent_uids=bank['parent_uids'],
        prompts=len(bank['records']),teacher_samples=len(bank['teacher_samples']),
        evaluation_noise_seeds=list(bank['eval_noises']),model_forwards=0,optimizer_updates=0)),flush=True)


if __name__=='__main__':
    main()
