"""Fixed teacher inputs for zero-update C-relation diagnosis.

Only build_teacher_bank performs OT: exactly one original free-only assignment
for each of12 pairings. Reloading saved inputs performs no OT and no model call.
No target, source ID, or relation group is added to Native model arguments.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from native_t1.data import (NativeTrainingStream, array_hash, file_sha256,
                            load_archive, shared_sample_hash)

TIMES=(.1,.3,.5,.7,.9)
BASE_SEEDS=(19601,19602)
TRAIN_K=(4,8,12)
SCHEMA='c_relation_fixed_teacher_v1'
EXPECTED_CHECKPOINTS={
    'A':{'250':'aea34110ae1f75c108298a366e8b2668c28aa09ad8fc7d683890f5caa415e305',
         '500':'43e56e107b0c6d28766aebd8cf8a0a764d8d52ef62ba2ca6e756d881634f876b'},
    'C':{'250':'a7b558451a3e733a5173f7a7a7a180df7358340418e4fe28b11d35f05a0f1ec2',
         '500':'8a40651bcf54aebb2c48cd19a2433fb4e0679370c0da6e79e916cc727f067168'},
}
PAIR_ARRAY_FIELDS=('x0','x1','u','y','known_mask','valid_mask','context',
    'known_target_ids','source_face_ids','corner_permutations','face_permutation',
    'free_gaussian_before_OT','free_noise_after_OT','free_target_before_OT')
STATE_HASH_FIELDS=('xt','t','x0','x1','u','context','known_mask','valid_mask','y',
                   'source_face_ids','corner_permutations')
TEACHER_PROTOCOL=dict(base_CPU_seeds=list(BASE_SEEDS),
    constructor='native_t1.data.NativeTrainingStream(archive, arm=Native_correct, seed=base_seed)',
    noise_RandomState_seed='base_seed+10000',time_RandomState_seed='base_seed+30000 (draw retained but unused)',
    permutation_RandomState_seed='base_seed+40000',order_RandomState_seed='base_seed+60000 (not drawn by _sample)',
    donor_RandomState_seed='base_seed+70000 (not drawn for correct-context arm)',
    per_stream_pair_order=[[p,K] for p in (0,1) for K in TRAIN_K],
    pair_construction='For each base seed, call original _sample(parent,K) once in fixed order; never use next_effective_batch.',
    target_scale='Read genuine pre_ot_free; cyclic/face target ordering occurs first; official nested OT reorders only raw Gaussian Z against pre-scale target; multiply target by2 and cast toFP32 after OT.',
    gaussian_scale='Original float64 RandomState.randn(nfree,3,3), nested-OT reordered and then cast toFP32; no x2 multiplier on noise.',
    teacher_times=list(TIMES),random_stream_time='Each _sample draws one original logit-normal t; record it as unused, then replace with the five fixed FP32 times.',
    paired_noise_reused_for_all_five_times=True,expected_OT_calls=12,teacher_states=60,
    uses_GT_for_sampling=False,complete_generations=0,optimizer_updates=0)


def _record(path):
    path=Path(path)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=file_sha256(path))


def _same(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def _source_metadata(source_run):
    """Resolve by the already located report; no timestamp inference or PT load."""
    run=Path(source_run).resolve();report=run/'THREE_ARM_INTERFACE_REPORT.md';results=run/'results.json'
    if not report.is_file() or not results.is_file():
        raise ValueError('Source must be identified by THREE_ARM_INTERFACE_REPORT.md and results.json.')
    if 'THREE_ARM_INTERFACE_REPORT' not in report.read_text(encoding='utf-8-sig'):
        raise ValueError('Unexpected source report identity.')
    old=json.loads(results.read_text(encoding='utf-8-sig'))
    if old.get('status')!='PASS_EXECUTION':
        raise ValueError('Three-arm source experiment is not complete.')
    models={}
    for arm,steps in EXPECTED_CHECKPOINTS.items():
        models[arm]={}
        for step,expected in steps.items():
            path=run/arm/f'additional_step{step}_cumulative_step{500+int(step)}.pt'
            record=old['arms'][arm]['checkpoints'].get(step)
            if record is None or record.get('sha256')!=expected:
                if step=='250' and not path.exists():
                    models[arm][step]=dict(status='MISSING_OPTIONAL_INTERMEDIATE',path=str(path));continue
                raise RuntimeError('Source checkpoint metadata does not match authorized identity.')
            if step=='500' and not path.is_file():
                raise FileNotFoundError(path)
            models[arm][step]=dict(path=str(path),metadata_sha256=expected,
                status='PRESENT_METADATA_MATCH' if path.is_file() else 'MISSING_OPTIONAL_INTERMEDIATE',
                actual_large_file_hash_recomputed_here=False)
    source_input=old['assets']['source_inputs']
    if old['relations']['inputs']['sha256']!=source_input['sha256']:
        raise RuntimeError('Three-arm data and relation identity disagree.')
    # The source run contains no persisted teacher protocol. Never assume an
    # initial sampling Gaussian is an already-OT-matched training/teacher pair.
    roots=[run,Path(source_input['path']).resolve().parent]
    candidates=sorted({str(p) for directory in roots for p in directory.rglob('*')
                       if p.is_file() and ('teacher' in p.name.lower() or 'bank' in p.name.lower())
                       and p.suffix.lower() in ('.npz','.pt','.json')})
    if candidates:
        raise RuntimeError('Potential saved teacher pairing found; inspect before constructing a replacement: '+repr(candidates))
    return old,dict(report=_record(report),results=_record(results),checkpoint_metadata=models,
        source_inputs=copy.deepcopy(source_input),prior_teacher_pairing_search=dict(
            roots=[str(p) for p in roots],matching_teacher_or_bank_files=candidates,
            available_exact_matching_pairings=0,
            reason='No saved teacher/paired-bank artifact in identified source run or its frozen patch-input source. Training logs preserve hashes, not the12 required paired arrays; old NPZ sampling noises are unpaired.'),
        checkpoint_hash_scope='Metadata identity check only here; root asset gate checks actual large checkpoint hashes once.')


def _hash_state(sample):
    digest=hashlib.sha256()
    for name in STATE_HASH_FIELDS:
        digest.update(name.encode());digest.update(array_hash(np.asarray(sample[name])).encode())
    digest.update(array_hash(sample['labels']['corner_source_vertex_ids']).encode())
    return digest.hexdigest()


def _states_from_pairs(pairs):
    states=[]
    for pair in sorted(pairs,key=lambda s:(s['parent_index'],s['K'],s['noise_index'])):
        for time_index,value in enumerate(TIMES):
            sample=copy.deepcopy(pair)
            t=np.asarray(value,dtype=np.float32)
            sample['t']=t
            sample['xt']=(np.float32(1)-t)*sample['x0']+t*sample['x1']
            sample['xt'][sample['known_mask']]=sample['context'][sample['known_mask']]
            sample['time_index']=time_index
            sample['state_id']=f"P{sample['parent_index']}_K{sample['K']}_seed{sample['noise_seed']}_t{value:.1f}"
            sample['shared_input_sha256']=shared_sample_hash(sample)
            sample['context_sha256']=array_hash(sample['context'])
            sample['state_sha256']=_hash_state(sample)
            states.append(sample)
    if len(states)!=60 or len({s['state_id'] for s in states})!=60:
        raise RuntimeError('Expected exactly60 unique fixed teacher states.')
    return states


def _rebuild_labels(pair,corner_ids,relations=None):
    """Use persisted trusted identities; a provided source mapper rechecks them."""
    from .t1_interface_relations import build_relation_labels
    ids=np.asarray(corner_ids,dtype=np.int64)
    if ids.shape!=(112,3):
        raise ValueError('Persisted source-corner labels must be [112,3].')
    if relations is not None:
        # Mapping validation does not depend on t; create a disposable t=0.5
        # representation solely to satisfy the normal stream sample interface.
        original=copy.deepcopy(pair);original['t']=np.asarray(.5,dtype=np.float32)
        original['xt']=(np.float32(1)-original['t'])*original['x0']+original['t']*original['x1']
        original['xt'][original['known_mask']]=original['context'][original['known_mask']]
        labels=relations.prepare_sample_labels(original)
        if not _same(labels['corner_source_vertex_ids'],ids):
            raise RuntimeError('Persisted teacher source identities disagree with frozen source mapper.')
        return labels
    # No proximity matching: rebuild source-ID groups directly from saved labels.
    labels=build_relation_labels(ids,pair['known_mask'],pair['context'],pair['valid_mask'])
    labels.update(corner_source_vertex_ids=ids.copy(),parent=pair['parent_index'],K=pair['K'])
    return labels


def build_teacher_bank(source_run,out,relations):
    """Build12 pairings once, persist, then return60 states and JSON-safe audit.

    No model invocation occurs. A failure is never retried. The small progress
    record embedded in a partial NPZ preserves the number of OT attempts/returns.
    """
    started=time.perf_counter();out=Path(out).resolve()
    if not out.is_dir():
        raise ValueError('Root must create and preregister the diagnosis output directory first.')
    destination=out/'teacher_bank.npz'
    if destination.exists() or (out/'teacher_bank_partial.npz').exists():
        raise FileExistsError('Teacher bank already exists; reload it without another OT call.')
    old,source=_source_metadata(source_run)
    source_input=source['source_inputs']
    if relations.audit['inputs']['sha256']!=source_input['sha256']:
        raise RuntimeError('Relation mapper and teacher target archive differ.')
    archive=load_archive(source_input['path'],expected_sha256=source_input['sha256'])
    arrays={};pairs=[];pair_records=[]
    audit=dict(schema=SCHEMA,status='BUILDING',protocol=copy.deepcopy(TEACHER_PROTOCOL),source=source,
        source_relation_audit_sha256=hashlib.sha256(json.dumps(relations.audit,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        OT_attempts=0,OT_calls=0,model_forwards=0,model_backwards=0,optimizer_updates=0,complete_generations=0,
        paired_array_reuse=False,loaded_without_OT=False,pairs=pair_records)
    try:
        for noise_index,seed in enumerate(BASE_SEEDS):
            stream=NativeTrainingStream(archive,'Native_correct',seed=seed)
            for parent in (0,1):
                for K in TRAIN_K:
                    # _sample performs exactly one original nested OT; all other
                    # operations follow the existing official scale/order path.
                    audit['OT_attempts']+=1
                    sample=stream._sample(parent,K)
                    audit['OT_calls']+=1
                    labels=relations.prepare_sample_labels(sample)
                    pair_id=f'P{parent}_K{K}_seed{seed}'
                    pair={name:np.asarray(sample[name]).copy() for name in PAIR_ARRAY_FIELDS}
                    pair.update(parent_index=parent,donor_index=parent,K=K,noise_index=noise_index,
                        noise_seed=seed,pair_id=pair_id,unused_stream_time=float(sample['t']),labels=labels)
                    prefix=f'pair{len(pairs):02d}_'
                    hashes={}
                    for name in PAIR_ARRAY_FIELDS:
                        arrays[prefix+name]=pair[name];hashes[name]=array_hash(pair[name])
                    arrays[prefix+'corner_source_vertex_ids']=labels['corner_source_vertex_ids'].copy()
                    hashes['corner_source_vertex_ids']=array_hash(labels['corner_source_vertex_ids'])
                    pair_records.append(dict(pair_id=pair_id,array_prefix=prefix,parent_index=parent,K=K,
                        noise_index=noise_index,noise_seed=seed,actual_noise_RandomState_seed=seed+10000,
                        actual_permutation_RandomState_seed=seed+40000,unused_stream_time=float(sample['t']),
                        array_hashes=hashes,FC_count=labels['FC_count'],FF_count=labels['FF_count'],
                        source_mapping_audit=copy.deepcopy(labels['source_mapping_audit']),OT_calls=1))
                    pairs.append(pair)
        states=_states_from_pairs(pairs)
        audit.update(status='PASS',pair_count=len(pairs),state_count=len(states),
            state_records=[dict(state_id=s['state_id'],parent_index=s['parent_index'],K=s['K'],noise_index=s['noise_index'],
                noise_seed=s['noise_seed'],t=float(s['t']),state_sha256=s['state_sha256'],shared_input_sha256=s['shared_input_sha256']) for s in states],
            pair_order='Stored in base_seed,parent,K generation order; states returned in parent,K,noise,t order.',
            reconstruction='Five FP32 Xt values from the same saved x0/x1; known slots replaced with exact saved C. No new Gaussian or OT while changing t.',
            input_scope='Source IDs and GT are labels/supervision only. Native model inputs remain xt,t,y,valid_mask,known_mask.',
            seconds=time.perf_counter()-started)
        arrays['metadata_json']=np.asarray(json.dumps(audit,sort_keys=True,separators=(',',':'),allow_nan=False))
        with destination.open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        saved=_record(destination)
        # This read verifies byte preservation and reconstruction, never samples.
        reloaded,check=load_teacher_bank(destination,relations=relations,expected_sha256=saved['sha256'])
        if [s['state_sha256'] for s in states]!=[s['state_sha256'] for s in reloaded]:
            raise RuntimeError('Teacher bank round-trip changed actual state bytes.')
        audit.update(bank_file=saved,roundtrip_verified=True,reload_OT_calls=0)
        return reloaded,audit
    except BaseException as error:
        audit.update(status='FAILED_NO_AUTOMATIC_RETRY',error=repr(error),pair_count=len(pairs),seconds=time.perf_counter()-started)
        if not destination.exists():
            arrays['metadata_json']=np.asarray(json.dumps(audit,sort_keys=True,separators=(',',':'),allow_nan=False))
            with (out/'teacher_bank_partial.npz').open('xb') as stream:
                np.savez_compressed(stream,**arrays)
        raise


def load_teacher_bank(path,relations=None,expected_sha256=None):
    """Return the exact same60 states from a persisted bank, with zero OT."""
    started=time.perf_counter();path=Path(path).resolve();digest=file_sha256(path)
    if expected_sha256 is not None and digest!=expected_sha256:
        raise RuntimeError('Teacher bank file hash differs.')
    pairs=[]
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata_json'].item()))
        if metadata.get('schema')!=SCHEMA or metadata.get('status')!='PASS' or metadata.get('pair_count')!=12 or metadata.get('state_count')!=60:
            raise ValueError('Only a completed fixed12-pair/60-state bank can be reloaded.')
        if metadata['protocol']!=TEACHER_PROTOCOL:
            raise RuntimeError('Saved teacher protocol differs from frozen seed/order/time rule.')
        if relations is not None and relations.audit['inputs']['sha256']!=metadata['source']['source_inputs']['sha256']:
            raise RuntimeError('Reload mapper points to a different target archive.')
        for record in metadata['pairs']:
            prefix=record['array_prefix'];pair={name:archive[prefix+name].copy() for name in PAIR_ARRAY_FIELDS}
            ids=archive[prefix+'corner_source_vertex_ids'].copy()
            for name,value in dict(pair,corner_source_vertex_ids=ids).items():
                if array_hash(value)!=record['array_hashes'][name]:
                    raise RuntimeError('Saved pairing array changed: '+prefix+name)
            pair.update({name:record[name] for name in ('parent_index','K','noise_index','noise_seed','pair_id','unused_stream_time')})
            pair['donor_index']=pair['parent_index']
            pair['labels']=_rebuild_labels(pair,ids,relations)
            if pair['labels']['FC_count']!=record['FC_count'] or pair['labels']['FF_count']!=record['FF_count']:
                raise RuntimeError('Reloaded source relation groups differ.')
            pairs.append(pair)
    states=_states_from_pairs(pairs)
    actual={s['state_id']:(s['state_sha256'],s['shared_input_sha256']) for s in states}
    expected={s['state_id']:(s['state_sha256'],s['shared_input_sha256']) for s in metadata['state_records']}
    if actual!=expected:
        raise RuntimeError('Reconstructed fixed teacher states changed.')
    audit=copy.deepcopy(metadata)
    audit.update(bank_file=dict(path=str(path),bytes=path.stat().st_size,sha256=digest),
        loaded_without_OT=True,OT_attempts=0,OT_calls=0,original_build_OT_attempts=metadata['OT_attempts'],
        original_build_OT_calls=metadata['OT_calls'],reload_seconds=time.perf_counter()-started)
    return states,audit
