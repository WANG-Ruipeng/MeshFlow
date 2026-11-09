"""CPU-only fixed real triangle targets and independent noise for soft matching.

The sampler receives only cases[i]['constraints']; complete targets are held for
independent scoring. No checkpoint, model, training batch, OT or CUDA execution.
"""
from __future__ import annotations
import copy
import itertools
import json
import re
import zipfile
from pathlib import Path
import numpy as np
import torch
from .inference_pilot_data import (
    HISTORY, EXPECTED_RESULT_SHA, _load_fixed_bank, _preserve_cpu_rng, _file_record)
from .nxc112_data import PROJECT, MIXED, B_UIDS, _persist
from .learning_data import array_hash
from .prompt_dataset import file_sha256, sample_surface_primitive

NOISE_SEEDS=(9101,9102)
PERMUTATIONS=np.asarray(list(itertools.permutations(range(3))),dtype=np.int64)
PATTERN=re.compile(r'(?<![A-Za-z0-9.])(?:9101|9102)(?![A-Za-z0-9.])|seed[_-]?(?:9101|9102)(?![0-9])',re.I)


def _seed_audit(output):
    files=[];hits=[];npz_count=0
    roots=[PROJECT/'outputs',PROJECT/'tests']
    paths={p.resolve() for root in roots for p in root.rglob('*') if p.is_file()}
    paths.update(p.resolve() for p in PROJECT.glob('*.py'))
    for path in sorted(paths):
        if path.is_relative_to(output) or path.name.startswith('soft_matching'):
            continue
        if path.suffix.lower() in {'.py','.json','.jsonl','.md','.txt','.yaml','.yml'}:
            content=path.read_text(encoding='utf-8-sig')
            files.append(dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha256(path)))
            for number,line in enumerate(content.splitlines(),1):
                if PATTERN.search(line):hits.append(dict(path=str(path),line=number,text=line.strip()))
        elif path.suffix.lower()=='.npz':
            npz_count+=1
            with zipfile.ZipFile(path) as saved:
                for name in saved.namelist():
                    if PATTERN.search(name):hits.append(dict(path=str(path),array_key=name))
    if hits:raise RuntimeError('Unresolved previous 9101/9102 usage: '+json.dumps(hits))
    return dict(status='PASS_WITH_FINITE_SCOPE',seeds=list(NOISE_SEEDS),
        scope='Existing experimental outputs and tests plus experiment-root Python sources; current output and new soft_matching sources excluded; NPZ member names inspected without payload loading',
        text_files=files,text_file_count=len(files),text_bytes=sum(p['bytes'] for p in files),
        npz_archives_scanned=npz_count,previous_recorded_seed_occurrences=hits,
        no_recorded_model_or_hyperparameter_selection_use=True,
        limitations='No claim about unlogged or external runs. Numeric substrings in floats/hashes are not seed uses.')


def _matching_indices(triangles,face):
    equal=(triangles[:,None,:,:]==face[PERMUTATIONS][None,:,:,:]).all(axis=(2,3)).any(axis=1)
    return np.flatnonzero(equal)


def _cases_from_arrays(arrays,records):
    cases=[]
    for record in records:
        p=record['parent_index'];k=record['K'];prefix=f'parent{p}_K{k}_'
        cases.append(dict(**copy.deepcopy(record),
            constraints=arrays[prefix+'constraints'].copy(),
            checks=arrays[prefix+'checks'].copy(),
            target=arrays[f'parent{p}_target'].copy()))
    return cases


def prepare_inputs(output,persist=True):
    """Return (bank, JSON-safe audit); load back persisted tensors before use."""
    out=Path(output).resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('soft_matching_'):
        raise ValueError('Use a separate soft_matching_* output directory')
    if torch.cuda.is_initialized():raise RuntimeError('Prepare CPU inputs before initializing CUDA')
    with _preserve_cpu_rng():
        report_path=HISTORY/'results.json'
        if file_sha256(report_path)!=EXPECTED_RESULT_SHA:raise RuntimeError('Historical results changed')
        report=json.loads(report_path.read_text(encoding='utf-8-sig'))
        old,old_audit=_load_fixed_bank(report)
        coverage=_seed_audit(out)
        arrays={};records=[];parent_records=[]
        for index,uid in enumerate(B_UIDS):
            target=old['targets'][uid].copy();triangles=target.reshape(112,3,3)
            prompt=old['prompts'][(uid,MIXED)]
            face=np.asarray(prompt['vertices'][np.flatnonzero(prompt['arity']==3)[0]],dtype=np.float32)
            first=_matching_indices(triangles,face)
            if len(first)!=1:raise RuntimeError('Archived face is not one unique canonical target face')
            extent=np.ptp(triangles.reshape(-1,3),axis=0)
            diagonal=float(np.linalg.norm(extent));scale=float(extent.max())
            area2=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
            legal=np.flatnonzero(area2>scale**2*1e-12)
            second=next((int(i) for i in legal if i not in first),None)
            if second is None:raise RuntimeError('Missing second valid distinct target face')
            selected=np.stack([face,triangles[second]]).astype(np.float32)
            # K=1 retains the exact original independent interior checks.
            checks=np.stack([old['checks'][(uid,MIXED)][2].copy(),
                sample_surface_primitive(selected[1],3,128,np.random.RandomState(9200+index),False)])
            arrays[f'parent{index}_target']=target
            for k in (1,2):
                prefix=f'parent{index}_K{k}_'
                arrays[prefix+'constraints']=selected[:k].copy()
                arrays[prefix+'checks']=checks[:k].copy()
                records.append(dict(case_id=f'P{index}_K{k}',parent_index=index,parent_uid=uid,K=k,
                    N=112,bbox_diagonal=diagonal,bbox_extent=extent.tolist(),
                    constraint_face_indices=[int(first[0]),second][:k],
                    constraints_sha256=array_hash(selected[:k]),checks_sha256=array_hash(checks[:k]),
                    target_sha256=array_hash(target),tau=3*(0.05*diagonal)**2,
                    target_selection='K1: archived mixed prompt face; K2: append first different area-valid face in archived canonical target order',
                    constraints_selected_from_input_geometry_only=True,model_outputs_consulted=False,
                    check_seed_second=9200+index if k==2 else None,
                    exact_shared_vertices_K2=int(sum(any(np.array_equal(v,w) for w in selected[1]) for v in selected[0])) if k==2 else None))
            parent_records.append(dict(parent_index=index,parent_uid=uid,bbox_diagonal=diagonal,
                bbox_extent=extent.tolist(),coordinate_min=triangles.reshape(-1,3).min(0).tolist(),
                coordinate_max=triangles.reshape(-1,3).max(0).tolist(),faces=112,
                source_file=copy.deepcopy(old['parents'][index]),target_sha256=array_hash(target)))
        for seed in NOISE_SEEDS:
            arrays[f'noise_seed{seed}']=torch.randn((1,112,9),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32,device='cpu').numpy()
        artifact=_persist(out/'inputs.npz',arrays,persist)
        if persist:
            with np.load(artifact['path'],allow_pickle=False) as saved:
                reloaded={key:saved[key].copy() for key in saved.files}
            if set(reloaded)!=set(arrays) or any(array_hash(reloaded[key])!=array_hash(value) for key,value in arrays.items()):
                raise RuntimeError('Persisted input roundtrip differs')
            arrays=reloaded
        bank=dict(cases=_cases_from_arrays(arrays,records),parents=copy.deepcopy(old['parents']),
            parent_uids=list(B_UIDS),targets={uid:arrays[f'parent{i}_target'].copy() for i,uid in enumerate(B_UIDS)},
            noises={seed:torch.from_numpy(arrays[f'noise_seed{seed}'].copy()) for seed in NOISE_SEEDS},
            noise_seeds=list(NOISE_SEEDS),N=112)
        audit=dict(status='PASS',schema=1,model_forwards=0,optimizer_updates=0,
            source_results=_file_record(report_path),historical_bank=old_audit,
            seed_coverage=coverage,inputs=artifact,roundtrip_reloaded_from_saved_NPZ=bool(persist),
            parents=parent_records,cases=records,
            noise_rule='Independent CPU torch.Generator.manual_seed(seed), Gaussian [1,112,9] FP32, shared across all cases and settings; no OT or target coupling',
            noise_hashes={str(seed):array_hash(bank['noises'][seed].numpy()) for seed in NOISE_SEEDS},
            coordinates=dict(unit='Official normalized model coordinate units; no physical meter/micron calibration',
                transform='Raw bbox center and max-side normalization, clip [-0.5,0.5], canonical vertex/face order, divide by official std 0.3762, final coordinate multiplication by 2; prompts copied in identical coordinates',
                no_additional_alignment_or_rescaling=True,GT_endpoint_available_to_sampler=False),
            fresh_noise_seed_selection_independent_of_outputs=True,cuda_initialized=torch.cuda.is_initialized(),caller_CPU_RNG_preserved=True)
        json.dumps(audit,allow_nan=False)
    return bank,audit
