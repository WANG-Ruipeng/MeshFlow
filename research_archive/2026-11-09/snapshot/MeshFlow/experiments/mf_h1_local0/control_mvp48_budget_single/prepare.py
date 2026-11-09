"""Frozen, CPU-only inputs for the bounded MVP48 continuation.

Only the new output tree is writable. Existing source banks are hash checked and
retained verbatim. Re-entry verifies the manifest before returning completed work.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value, *, frozen=True):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists() and frozen:
        if read_json(path) != json.loads(text):
            raise ValueError('Frozen JSON identity changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def _npz(path, arrays):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(arrays):
                raise ValueError('Frozen NPZ keys changed: ' + str(path))
            for key, value in arrays.items():
                actual = stored[key]
                expected = np.asarray(value)
                if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
                    raise ValueError('Frozen NPZ array changed: ' + str(path) + '/' + key)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _seed(domain, *identity):
    message = json.dumps(['control_mvp48_budget_single_v1', domain, *identity], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:4], 'little')


def _load_api(source):
    spec = importlib.util.spec_from_file_location('budget_single_frozen_source', source / 'data/load_sample.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


def _source_audit(source):
    ready = read_json(source / 'DATA_READY.json')
    if ready.get('status') != 'READY' or ready.get('data_revision') != 'repair_v2':
        raise ValueError('Require the authoritative repair_v2 DATA_READY')
    root = Path(__file__).resolve().parents[3]
    checked = {}
    for name, spec in ready['files'].items():
        path = source / name
        if sha256(path) != spec['sha256'] or path.stat().st_size != spec['bytes']:
            raise ValueError('Source data identity changed: ' + name)
        checked[name] = spec['sha256']
    for name, expected in ready['code'].items():
        if sha256(root / name) != expected:
            raise ValueError('Source data code identity changed: ' + name)
    splits = read_json(source / 'data/splits.json')
    objects = {row['uid']: row for row in read_json(source / 'data/manifest.json')['objects']}
    if {key: len(splits[key]) for key in ('train', 'validation', 'test')} != dict(train=32, validation=8, test=8):
        raise ValueError('MVP48 split count changed')
    if len({u for key in ('train', 'validation', 'test') for u in splits[key]}) != 48:
        raise ValueError('Overlapping MVP48 parent splits')
    if any(not 128 <= item['N'] <= 256 or not 128 <= item['original_N'] <= 256 for item in objects.values()):
        raise ValueError('Natural/retained face count outside registered range')
    patches = read_json(source / 'data/patches.json')['records']
    for split, count in (('train', 4), ('validation', 2), ('test', 2)):
        for uid in splits[split]:
            if {row['patch_id'] for row in patches if row['uid'] == uid} != {f'p{i}' for i in range(count)}:
                raise ValueError('MVP48 condition identity/count changed')
    return ready, splits, objects, checked


def _make_plan(source_plan, objects, train_uids, *, probe=False):
    steps = 1000 if probe else 4000
    offset = 0 if probe else 1000
    domain = 'probe_training' if probe else 'main_training_extension'
    choices = [(uid, 'p0') for uid in train_uids[:2]] if probe else [(uid, f'p{p}') for uid in train_uids for p in range(4)]
    schedule_rng = np.random.RandomState(_seed(domain, 'schedule'))
    if probe:
        assignments = [choices[int(i)] for _ in range(steps) for i in schedule_rng.permutation([0]*4 + [1]*4)]
    else:
        # Every 128-row block covers exactly all object/patch pairs; 32000/128=250.
        assignments = [choices[int(i)] for _ in range(250) for i in schedule_rng.permutation(len(choices))]
    records = []
    for index, (uid, patch) in enumerate(assignments):
        row = dict(source_plan['records'][0])
        row.update(uid=uid, patch_id=patch, N=objects[uid]['N'], K=objects[uid]['K'],
                   sample_index=8*offset+index, effective_batch_index=offset+index//8,
                   slot_index=index%8, logical_step=offset+index//8+1,
                   gaussian_seed=_seed(domain, 'gaussian', index), t_seed=_seed(domain, 'time', index),
                   permutation_seed=_seed(domain, 'permutation', index))
        records.append(row)
    return dict(schema='control_mvp48_train_plan_v1', experiment_id='control_mvp48_budget_single_v1',
                source_plan_sha256=None, effective_batch_count=steps, batch_size=8, sample_count=len(records),
                effective_batch_index_base=0, first_effective_batch_index=offset, last_effective_batch_index=offset+steps-1,
                seed=_seed(domain, 'schedule'), rng_namespace=domain,
                RNG_mode='stateless_per_sample_frozen_seeds', model_arms_share_exact_plan=not probe,
                object_counts=dict(Counter(row['uid'] for row in records)),
                pair_counts=dict(Counter(row['uid']+'/'+row['patch_id'] for row in records)), records=records)


def _area(triangles):
    value = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    return float((np.linalg.norm(np.cross(value[:, 1]-value[:, 0], value[:, 2]-value[:, 0]), axis=1)*.5).sum())


def _donor_patch(sample, count):
    """First source-face start that reaches K, sorted-neighbor FIFO shared-edge BFS."""
    faces = sample['source_face_vertex_ids']
    incident = defaultdict(list)
    for face_id, face in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            incident[tuple(sorted((int(face[a]), int(face[b]))))].append(face_id)
    adjacency = [set() for _ in faces]
    for connected in incident.values():
        for i in connected:
            adjacency[i].update(j for j in connected if i != j)
    for start in range(len(faces)):
        queue, visited, ordered = deque([start]), {start}, []
        while queue and len(ordered) < count:
            item = queue.popleft()
            ordered.append(item)
            for neighbor in sorted(adjacency[item]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(ordered) == count:
            return np.asarray(ordered, np.int64)
    return None


def prepare(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents:
        raise ValueError('New output must be separate from protected source')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ready, splits, objects, checked = _source_audit(source)
    ready_hash = sha256(source / 'DATA_READY.json')
    panel_path = out / 'panels.json'
    if panel_path.exists():
        panels = read_json(panel_path)
        if panels['source_data_ready_sha256'] != ready_hash:
            raise ValueError('Prepared banks belong to another source')
        for path, digest in panels['frozen_files'].items():
            actual = Path(path) if Path(path).is_absolute() else out / path
            if sha256(actual) != digest:
                raise ValueError('Prepared bank identity changed: ' + path)
        if read_json(out / 'registration/data_audit.json')['status'] != 'PASS':
            raise ValueError('Prior finite CPU audit did not pass')
        return panels
    api = _load_api(source)
    source_plan = read_json(source / 'data/train_plan.json')
    train_uids = sorted(splits['train'])
    fit_uids = train_uids[:2]
    ranked = sorted(train_uids, key=lambda uid: (objects[uid]['N'], uid))
    panel_train = [min(ranked[4*i:4*i+4]) for i in range(8)]
    panels = dict(schema='control_mvp48_budget_panels_v1', source=str(source), source_data_ready_sha256=ready_hash,
                  data_revision='repair_v2', train_parents=panel_train, validation_parents=sorted(splits['validation']),
                  fit_parents=fit_uids, train_selection='Sort all train parents by (N,UID); eight consecutive four-parent strata; lexicographically first UID in each stratum',
                  fit_selection='First two training UIDs in ascending lexicographic order; p0',
                  random_domains=['native_initialization', 'main_training_extension', 'probe_training', 'generation', 'teacher', 'fit_generation', 'fit_teacher', 'plotting'],
                  generation=[], teacher=[], swap=[], fit_generation=[], fit_teacher=[], holdout=[], frozen_files={})

    def frozen(path, arrays=None, value=None):
        actual = out / path
        if arrays is not None:
            _npz(actual, arrays)
        else:
            _json(actual, value)
        panels['frozen_files'][path] = sha256(actual)
        return path

    for filename, probe in (('train_plan.json', False), ('fit_plan.json', True)):
        plan = _make_plan(source_plan, objects, train_uids, probe=probe)
        plan['source_plan_sha256'] = sha256(source / 'data/train_plan.json')
        frozen(filename, value=plan)

    from experiments.mf_h1_local0.control_mvp48.eval_scoring import prepare_reference_from_sample, score_free
    from native_t1.artifacts import array_hash
    samples, references = {}, {}

    def sample(uid, patch):
        key = (uid, patch)
        if key not in samples:
            samples[key] = api.load_sample(uid, patch, source / 'data')
        return samples[key]

    # The scope and threshold are written before any real-data distance call.
    controls = [(split, uid, 'p0') for split in ('train', 'validation') for uid in sorted(splits[split])[:2]]
    precheck_spec = dict(schema='control_mvp48_budget_finite_cpu_spec_v1', selected=controls,
                         selection='first two UIDs in each training/validation split, p0',
                         RMS_max_over_L=1e-6, GPU_workers=0, model_forwards=0, optimizer_updates=0)
    frozen('registration/cpu_positive_spec.json', value=precheck_spec)
    audit_path = out / 'registration/data_audit.json'
    if audit_path.exists():
        audit = read_json(audit_path)
        if audit.get('status') != 'PASS' or audit.get('source_data_ready_sha256') != ready_hash:
            raise ValueError('Preserved finite CPU audit is not reusable')
    else:
        audit = dict(schema='control_mvp48_budget_data_audit_v1', status='RUNNING', source_data_ready_sha256=ready_hash,
                     source_files_checked=checked, data_revision='repair_v2', split_counts={k:len(splits[k]) for k in ('train','validation','test')},
                     repair=dict(repaired_objects=40, removed_exact_zero_faces=109, removed_duplicate_faces=571,
                                 source_vertices_unchanged=True, full_object_transform_unchanged=True),
                     positive_controls=[], model_forwards=0, optimizer_updates=0, CUDA_initialized=False,
                     exploratory_read_only_self_scores_before_formal_entry=4,
                     exploratory_note='All four returned passing geometry scores; terminal inspection then raised KeyError because torch was not imported.')
        _json(audit_path, audit, frozen=False)
        try:
            for split, uid, patch in controls:
                item = sample(uid, patch)
                reference = prepare_reference_from_sample(item)
                result = score_free(item['free_target'], item['constraints'], item['free_target'], item['bbox_diagonal'], reference=reference, include_pair_audit=False)
                metrics = result['metrics']
                measured = max(metrics[key]/100. for key in ('remainder_symmetric_RMS_bbox_pct','free_to_R12_RMS_bbox_pct','R12_to_free_RMS_bbox_pct','fixed_R12_coverage_RMS_bbox_pct'))
                if measured > 1e-6 or not np.isfinite(measured):
                    raise ValueError('GT self surface positive control failed: ' + uid)
                audit['positive_controls'].append(dict(split=split, uid=uid, patch_id=patch, N=item['N'], K=item['K'],
                    RMS_max_over_L=measured, partition_and_transform='PASS_BY_STRICT_SOURCE_LOADER',
                    gamma_true_interface='PASS_BY_SOURCE_REFERENCE_BUILDER', metrics=metrics))
            audit['status'] = 'PASS'
        except BaseException as exc:
            audit.update(status='FAIL', error=repr(exc))
            _json(audit_path, audit, frozen=False)
            raise
        _json(audit_path, audit, frozen=False)
    panels['frozen_files']['registration/data_audit.json'] = sha256(audit_path)

    def reference(uid, patch):
        key = (uid, patch)
        if key not in references:
            ref = prepare_reference_from_sample(sample(uid, patch))
            stem = f'inputs/references/{uid}_{patch}'
            references[key] = dict(ref_npz=frozen(stem+'.npz', arrays=ref['arrays']),
                                   ref_json=frozen(stem+'.json', value=ref['audit']))
        return references[key]

    def generation(uid, patch, noise_index, split, domain, case_id):
        item = sample(uid, patch)
        n, k = item['N'], item['K']
        seed = _seed(domain, uid, noise_index)
        noise = api.gaussian(seed, n-k, fp32=True)
        z = np.zeros((1,n,9), np.float32)
        z[:, k:] = noise.reshape(1,n-k,9)
        arrays = dict(C=item['constraints'], gaussian=noise, z=z, N=np.asarray(n), K=np.asarray(k), L=np.asarray(item['bbox_diagonal']),
                      known_mask=np.arange(n)<k, valid_mask=np.ones(n, bool), full_target=item['full_target'], gt_free=item['free_target'])
        location = frozen('inputs/generation/'+case_id+'.npz', arrays=arrays)
        return dict(case_id=case_id, split=split, uid=uid, parent_id=uid, patch_id=patch, noise_index=noise_index,
                    N=n, K=k, gaussian_seed=seed, input_path=location, input_sha256=sha256(out/location),
                    C_area_fraction=_area(item['constraints'])/_area(item['full_target']), **reference(uid,patch))

    for split, uids in (('train', panel_train), ('validation', sorted(splits['validation']))):
        for uid in uids:
            for patch in ('p0','p1'):
                for noise_index in (0,1):
                    panels['generation'].append(generation(uid,patch,noise_index,split,'generation',f'{split}_{uid}_{patch}_n{noise_index}'))
    for uid in fit_uids:
        for noise_index in range(4):
            panels['fit_generation'].append(generation(uid,'p0',noise_index,'fit','fit_generation',f'fit_{uid}_p0_n{noise_index}'))

    prep_events = out / 'registration/prepare_events.jsonl'
    def event(value):
        prep_events.parent.mkdir(parents=True, exist_ok=True)
        with prep_events.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **value), sort_keys=True)+'\n')

    def teacher(uid, split, domain):
        from experiments.mf_h1_local0.control_mvp48.eval_teacher import prepare_teacher_sample
        item = sample(uid,'p0')
        pair_stem = f'inputs/teacher_pairs/{domain}_{uid}'
        pair_npz, pair_json = out/(pair_stem+'.npz'), out/(pair_stem+'.json')
        if pair_npz.exists() and pair_json.exists():
            identity = read_json(pair_json)
            with np.load(pair_npz, allow_pickle=False) as bank:
                arrays = {key: bank[key] for key in bank.files}
            if any(array_hash(arrays[key]) != digest for key,digest in identity['array_sha256'].items()):
                raise ValueError('Cached teacher pairing identity changed')
        else:
            # Persist attempts before the original CPU OT; interrupted attempts remain visible.
            event(dict(action='teacher_OT_attempt', uid=uid, domain=domain))
            arrays, identity = prepare_teacher_sample(item, api.gaussian(_seed(domain,uid), item['N']-item['K']))
            event(dict(action='teacher_OT_return', uid=uid, domain=domain))
            _npz(pair_npz, arrays)
            _json(pair_json, identity)
        panels['frozen_files'][pair_stem+'.npz'] = sha256(pair_npz)
        panels['frozen_files'][pair_stem+'.json'] = sha256(pair_json)
        result=[]
        for t_value in (.2,.5,.8):
            current = {key: value.copy() for key,value in arrays.items()}
            current['t'] = np.asarray(t_value,np.float32)
            current['xt'] = (np.float32(1)-current['t'])*current['x0']+current['t']*current['x1']
            current['xt'][current['known_mask']] = current['context'][current['known_mask']]
            current['free_vertex_ids'] = item['source_face_vertex_ids'][item['free_source_ids']].copy()
            current['known_vertex_ids'] = item['source_face_vertex_ids'][item['source_face_ids']].copy()
            case_id=f'{domain}_{uid}_t{int(t_value*10)}'
            location=frozen('inputs/teacher/'+case_id+'.npz', arrays=current)
            info=dict(identity, teacher_t=t_value, array_sha256={key:array_hash(value) for key,value in current.items()})
            info.update(model_input_sha256=hashlib.sha256(b''.join(current[key].tobytes() for key in ('xt','t','y','known_mask','valid_mask'))).hexdigest(), input_hash_definition='sha256 concatenated xt,t,y,known_mask,valid_mask bytes')
            result.append(dict(case_id=case_id,teacher_id=case_id, split=split,uid=uid,parent_id=uid,patch_id='p0',t=t_value,
                               N=item['N'],K=item['K'],input_path=location,input_sha256=sha256(out/location),identity=info,**reference(uid,'p0')))
        return result

    for split,uids in (('train',panel_train),('validation',sorted(splits['validation']))):
        for uid in uids:
            panels['teacher'].extend(teacher(uid,split,'teacher'))
    for uid in fit_uids:
        panels['fit_teacher'].extend(teacher(uid,'fit','fit_teacher'))

    val_uids=sorted(splits['validation'])
    for request in panels['generation']:
        if request['split'] != 'validation' or request['patch_id'] != 'p0':
            continue
        own=sample(request['uid'],'p0')
        donor_choice=None
        start=val_uids.index(request['uid'])
        for delta in range(1,len(val_uids)):
            donor_uid=val_uids[(start+delta)%len(val_uids)]
            other=sample(donor_uid,'p0')
            ids=_donor_patch(other,own['K'])
            if ids is None:
                continue
            donor_C=other['full_target'].reshape(-1,3,3)[ids].copy()
            if donor_C.tobytes() == own['constraints'].tobytes():
                continue
            donor_choice=(donor_uid,other,ids,donor_C)
            break
        case_id='swap_'+request['case_id']
        if donor_choice is None:
            panels['swap'].append(dict(request,case_id=case_id,recipient_case_id=request['case_id'],status='NOT_AVAILABLE',reason='No distinct connected donor patch of exact recipient K'))
            continue
        donor_uid,other,ids,donor_C=donor_choice
        with np.load(out/request['input_path'],allow_pickle=False) as bank:
            arrays={key:bank[key] for key in bank.files}
        arrays['recipient_C']=arrays['C'].copy()
        arrays['C']=donor_C
        location=frozen('inputs/swap/'+case_id+'.npz',arrays=arrays)
        panels['swap'].append(dict(request,case_id=case_id,recipient_case_id=request['case_id'],status='READY',
            donor_uid=donor_uid,donor_source_face_ids=ids.tolist(),donor_N=other['N'],input_path=location,input_sha256=sha256(out/location),
            recipient_C_area=_area(own['constraints']),donor_C_area=_area(donor_C),
            recipient_C_centroid=own['constraints'].mean(axis=(0,1)).tolist(),donor_C_centroid=donor_C.mean(axis=(0,1)).tolist(),
            donor_rule='UID cycle; first source-face-start reaching exact K with sorted-neighbor shared-edge FIFO BFS; original donor coordinates',
            scoring_scope='Recipient original free target and Gamma; wrong C is a one-way diagnostic, not donor completion ground truth'))

    # Original held-out generation inputs are consumed from their actual arrays.
    old=read_json(source/'registration/eval_preparation_ledger.json')
    conditions={(r['parent_id'],r['condition_id']):r for r in old['conditions']}
    for request in old['test_inputs']:
        path=source/request['path']
        if sha256(path)!=request['sha256']:
            raise ValueError('Original holdout input bytes changed')
        ref=conditions[(request['parent_id'],request['condition_id'])]
        ref_path=source/ref['path']
        if sha256(ref_path)!=ref['sha256']:
            raise ValueError('Original holdout reference bytes changed')
        panels['frozen_files'][str(path)]=request['sha256']
        panels['frozen_files'][str(ref_path)]=ref['sha256']
        meta_path=frozen('inputs/holdout_metadata/'+request['case_id']+'.json',value=ref['reference_metadata'])
        panels['holdout'].append(dict(request,uid=request['parent_id'],patch_id=request['condition_id'],noise_index=request['noise_id'],split='holdout',
            input_path=str(path),input_sha256=request['sha256'],ref_npz=str(ref_path),ref_json=meta_path,
            reference_key_prefix='reference_',reference_contains_geometry=True,target_path=str(ref_path),source_case_id=request['case_id']))
    if tuple(len(panels[key]) for key in ('generation','teacher','swap','fit_generation','fit_teacher','holdout')) != (64,48,16,8,6,32):
        raise ValueError('Frozen panel counts differ from fixed budgets')
    panels['preparation_seconds']=time.perf_counter()-started
    _json(panel_path,panels)
    return panels


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    result=prepare(args.source,args.out)
    print(json.dumps({key:len(result[key]) for key in ('generation','teacher','swap','fit_generation','fit_teacher','holdout')}))
