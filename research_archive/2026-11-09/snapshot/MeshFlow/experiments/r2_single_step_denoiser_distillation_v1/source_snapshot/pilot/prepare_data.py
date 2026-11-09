"""Freeze CPU-only distillation inputs and one parent-balanced finite stream.

Generated RAW paths are promises, never reported as completed here. Source Y is
diagnostic provenance only; every optimization target must come from D0.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json
import time
import numpy as np
try:
    from . import common as C
except ImportError:
    import common as C

DATA = C.OUT / 'manifest'
ART = Path('/mnt/e/MeshFlow-Control-artifacts')
USER = ART / 'experiments/chair_r2_user_conditions_v1'
RECIPE = ART / 'experiments/chair_condition_recipe_factorial_v1'
WHITELIST = C.ANCHOR / 'data_audit/training_whitelist.json'
DEV = C.ANCHOR / 'data_audit/dev_conditions.json'
SYNTH = C.ANCHOR / 'stream/synthetic/manifest.json'
SPLITS = C.ROOT / 'native_t1/runs/control_mvp48_v1/data/splits.json'
GENERATOR_BINDING = C.ROOT / 'experiments/r2_user_mix_from_official_v1/evaluation/checkpoints/R2_USER_MIX.json'
SCHEMA = 'r2_single_step_distillation_frozen_data_v1'


def rng(domain, *identity):
    return np.random.RandomState(C.seed(domain, *identity))


def npz_once(path, arrays):
    """A partial CPU rerun verifies existing bytes/arrays; it never overwrites."""
    path = C.checked(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(arrays):
                raise ValueError('Existing NPZ schema differs: ' + str(path))
            for k, a in arrays.items():
                a = np.asarray(a)
                if z[k].shape != a.shape or z[k].dtype != a.dtype or z[k].tobytes() != a.tobytes():
                    raise ValueError('Existing NPZ differs: ' + str(path) + ':' + k)
    else:
        C.save_npz(path, **arrays)
    return C.sha(path)


def json_once(path, obj):
    path = C.checked(path)
    if path.exists():
        if C.read(path) != obj:
            raise ValueError('Existing JSON differs: ' + str(path))
    else:
        C.write(path, obj)


def protect(path, protected, expected=None):
    path = Path(path)
    actual = C.sha(path)
    if expected and actual != expected:
        raise ValueError('Source hash mismatch: ' + str(path))
    if str(path) in protected and protected[str(path)] != actual:
        raise ValueError('Source changed during preparation: ' + str(path))
    protected[str(path)] = actual
    return actual


def guard_test():
    paths = [WHITELIST, DEV, USER / 'active_conditions.json', C.OLD / 'REPORT.md', C.ASSET,
             C.OUT / '..' / 'r2_anchor_aware_denoiser_v1' / 'forbidden_write.json']
    for p in paths:
        try:
            C.checked(p)
        except ValueError:
            pass
        else:
            raise AssertionError('Output guard did not reject before open: ' + str(p))
    json_once(DATA / 'write_guard.json', dict(status='PASS', rejected_before_open=list(map(str, paths)),
              source_files_opened_for_write=0))


def isolation(whitelist, protected):
    split = C.read(SPLITS)
    protect(SPLITS, protected)
    train = set(whitelist['parent_ids'])
    groups = split['group_ids']
    assert len(train) == 25 and train <= set(groups)
    registrations = [
        (RECIPE / 'evaluation/requests.json', 'rows'),
        (RECIPE / 'evaluation/confirmation_plan.json', 'rows'),
        (RECIPE / 'phases/confirm/evaluation/requests.json', 'rows'),
        (ART / 'experiments/r2_user_condition_finetune_pilot_v1/evaluation/registration.json', 'requests'),
        (C.OLD / 'registration.json', 'records'),
        (DEV, 'rows'),
    ]
    panels = []
    for path, key in registrations:
        protect(path, protected)
        rows = C.read(path)[key]
        ids = {r.get('parent_id', r.get('uid')) for r in rows}
        assert None not in ids and ids <= set(groups), ('Missing group ID', path, ids - set(groups))
        panels.append(dict(name=str(path), rows=len(rows), parent_ids=sorted(ids)))
    panels += [dict(name='original_MVP48_' + name, rows=len(split[name]), parent_ids=split[name])
               for name in ('validation', 'test')]
    tg = {groups[u] for u in train}
    for p in panels:
        ids = set(p['parent_ids'])
        p['group_ids'] = sorted({groups[u] for u in ids})
        p['uid_overlap'] = sorted(train & ids)
        p['group_overlap'] = sorted(tg & set(p['group_ids']))
        assert not p['uid_overlap'] and not p['group_overlap'], ('SPLIT_LEAKAGE', p)
    result = dict(status='PASS', training_parent_ids=sorted(train), training_group_ids=sorted(tg),
                  panels=panels, no_new_split_or_filter=True,
                  internal_validation='Same 25 training parents with independent unseen noise; not held-out-parent validation')
    json_once(DATA / 'parent_isolation.json', result)
    return groups


def load_source(row, protected):
    protect(row['npz'], protected, row['sha256'])
    for pk, hk in [('parent_npz', 'parent_sha256'), ('source_interface_json', 'source_interface_sha256'),
                   ('quality_condition_json', 'quality_condition_sha256')]:
        protect(row[pk], protected, row[hk])
    with np.load(row['npz'], allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    n, k = row['N'], row['K']
    assert a['C'].dtype == a['Y'].dtype == np.float32
    assert a['C'].shape == (k, 3, 3) and a['Y'].shape == (n-k, 3, 3)
    assert int(a['N']) == n and int(a['K']) == k
    assert np.isfinite(a['C']).all() and np.isfinite(a['Y']).all()
    assert sorted(np.r_[a['source_face_ids'], a['free_source_ids']].tolist()) == list(range(n))
    return a


def row_base(source, sid, split, domain, variant, groups):
    return dict(sample_id=sid, uid=source['uid'], parent_id=source['uid'], group_id=groups[source['uid']],
                condition_id=source['condition_id'], N=source['N'], K=source['K'], region_type=source['region_type'],
                tier=source['tier'], split=split, domain=domain, variant=variant,
                input_path=str(C.OUT / 'inputs' / split / (sid + '.npz')),
                source_npz=source['npz'], source_npz_sha256=source['sha256'],
                source_interface_json=source['source_interface_json'],
                teacher_cache_path=str(C.OUT / 'teacher_cache' / (sid + '.npz')),
                order='C source-array prefix then Y source-array order; no permutation',
                training_eligible=split == 'train', target='D0_EFFECTIVE_FP32_RESIDUAL_ONLY')


def generation_row(source, index, split, groups, a):
    sid = ('TR_REAL' if split == 'train' else 'INTERNAL_VAL') + '__' + source['condition_id'] + '__n' + str(index)
    r = row_base(source, sid, split, 'real', index, groups)
    n, k = r['N'], r['K']
    gs = C.seed('generator_gaussian', split, source['condition_id'], index)
    z = np.random.RandomState(gs).standard_normal((1, n, 9)).astype(np.float32)
    z[:, :k] = 0
    p = C.OUT / 'generation_inputs' / (sid + '.npz')
    h = npz_once(p, dict(C=a['C'], z=z, N=np.asarray(n, np.int64), K=np.asarray(k, np.int64),
                         known_mask=np.arange(n) < k))
    r.update(generation_input_path=str(p), generation_input_sha256=h, Gaussian_sha256=C.ah(z),
             Gaussian_seed=gs, noise_seed=gs, generated_raw_path=str(C.OUT / 'generated' / sid / 'raw.npz'),
             input_status='PLANNED_REQUIRES_AUTHORIZED_GENERATOR', source_C_sha256=C.ah(a['C']),
             generation_allowed_keys=['C', 'N', 'K', 'known_mask', 'z'],
             generation_forbidden_inputs=['Y', 'M', 'semantic labels', 'source interface IDs'])
    return r


def synthetic_row(source, variant, groups, a):
    sid = 'TR_SYNTH__' + source['condition_id'] + '__v' + str(variant)
    r = row_base(source, sid, 'train', 'synthetic', variant, groups)
    n, k = r['N'], r['K']
    width = np.float32(rng('synthetic_width', source['condition_id'], variant).uniform(.9, 1.1)) if variant >= 4 else np.float32(1)
    cm, ym = a['C'].copy(), a['Y'].copy()
    cm[..., 0] *= width
    ym[..., 0] *= width
    clean_model = np.concatenate([cm, ym], axis=0)
    clean_d = clean_model * np.float32(C.SCALE)
    eps = np.zeros((n, 3, 3), np.float32)
    if variant:
        eps[k:] = rng('synthetic_epsilon', source['condition_id'], variant).standard_normal((n-k, 3, 3)).astype(np.float32)
    xd = clean_d.copy()
    xd[k:] += np.float32(C.SIGMA) * eps[k:]
    # input_d is authoritative. The model-space display roundtrip is never used
    # to reconstruct the teacher input, and C is copied from the clean FP32 C.
    xm = xd / np.float32(C.SCALE)
    xm[:k] = cm
    arrays = dict(input_model=xm, input_d=xd, C=cm, original_C=cm.copy(), known_mask=np.arange(n) < k,
                  N=np.asarray(n, np.int64), K=np.asarray(k, np.int64), source_C=a['C'],
                  source_Y_diagnostic_d=clean_d[k:], width_factor=np.asarray(width, np.float32), epsilon=eps,
                  source_face_ids=a['source_face_ids'], free_source_ids=a['free_source_ids'])
    h = npz_once(r['input_path'], arrays)
    r.update(input_sha256=h, input_d_sha256=C.ah(xd), input_status='COMPLETE_CPU',
             C_sha256=C.ah(cm), synthetic_kind='clean' if variant == 0 else ('noise_original' if variant < 4 else 'width_noise'),
             width_factor=float(width), sigma_denoiser_units=0.0 if variant == 0 else C.SIGMA,
             epsilon_seed=None if variant == 0 else C.seed('synthetic_epsilon', source['condition_id'], variant),
             source_Y_usage='Diagnostic only; never a training target')
    return r


def fixed_monitors(rows):
    """Eight distinct parents, four real/four synthetic, seat/back and size coverage."""
    selected, used = [], set()
    for domain in ['real', 'synthetic']:
        for region, upper in [('seat', False), ('back', True), ('seat', True), ('back', False)]:
            pool = [r for r in rows if r['domain'] == domain and r['region_type'] == region and r['uid'] not in used]
            pool.sort(key=lambda r: (r['N'], r['K'], r['condition_id'], r['variant']))
            pick = pool[-1] if upper else pool[0]
            used.add(pick['uid']); selected.append(pick['sample_id'])
    assert len(used) == 8
    return selected


def build_stream(train):
    by_parent = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in train:
        by_parent[r['uid']][r['domain']][r['condition_id']].append(r)
    parents = sorted(by_parent)
    stream_rng = rng('training_stream')
    batches, exposures = [], []
    for step in range(4000):
        chosen = stream_rng.choice(parents, size=8, replace=False).tolist()
        domains = np.asarray(['real']*4 + ['synthetic']*4)
        stream_rng.shuffle(domains)
        ids, denom = [], 0
        for slot, (uid, domain) in enumerate(zip(chosen, domains.tolist())):
            conditions = sorted(by_parent[uid][domain])
            cid = conditions[int(stream_rng.randint(len(conditions)))]
            pool = by_parent[uid][domain][cid]
            r = pool[int(stream_rng.randint(len(pool)))]
            ids.append(r['sample_id']); denom += 9*(r['N']-r['K'])
            exposures.append(dict(step=step, slot=slot, sample_id=r['sample_id'], uid=uid, condition_id=cid,
                                  domain=domain, variant=r['variant'], kind=r.get('synthetic_kind', 'generated_raw'),
                                  N=r['N'], K=r['K'], free_scalar_count=9*(r['N']-r['K'])))
        batches.append(dict(step=step, sample_ids=ids, free_scalar_denominator=denom))
    stream = dict(schema=SCHEMA+'_stream', status='COMPLETE', updates=4000, batch=8, microbatch=1,
                  parent_sampling='Uniform 8 distinct parents per update; independently permuted four real/four synthetic domains; uniform condition then uniform variant',
                  objective_denominator='9 * sum valid free faces across eight examples', seed=C.seed('training_stream'), batches=batches)
    json_once(DATA / 'train_stream.json', stream)
    C.csvwrite(DATA / 'train_exposures.csv', exposures)
    summaries = []
    for keys in [('uid',), ('condition_id',), ('domain',), ('kind',), ('N',), ('K',), ('uid', 'domain')]:
        counts, scalars = Counter(), Counter()
        for r in exposures:
            key = tuple(r[k] for k in keys); counts[key] += 1; scalars[key] += r['free_scalar_count']
        for values, count in sorted(counts.items()):
            summaries.append(dict(group_by='|'.join(keys), group='|'.join(map(str, values)), exposures=count,
                                  free_scalar_exposure=scalars[values]))
    C.csvwrite(DATA / 'exposure_summary.csv', summaries)
    return stream


def prepare_dev(dev_source, groups, protected):
    by_condition = {r['condition_id']: r for r in dev_source}
    result = []
    old = C.read(C.OLD / 'registration.json')
    for r in sorted(old['records'], key=lambda x: (x['label'], x['request_id'])):
        s = by_condition[r['condition_id']]
        sid = 'DEV_REAL__' + r['label'] + '__' + r['request_id']
        q = row_base(s, sid, 'dev_real', 'real', r['candidate'], groups)
        protect(r['raw_path'], protected, r['raw_sha256'])
        protect(r['condition_source'], protected, r['condition_source_sha256'])
        with np.load(r['raw_path']) as z:
            raw = (z['raw'] if 'raw' in z.files else z['output']).astype(np.float32).reshape(-1, 3, 3)
        with np.load(r['condition_source']) as z: cm = z['C'].copy()
        assert raw.shape == (r['N'], 3, 3) and C.ah(raw[:r['K']]) == C.ah(cm)
        xd = raw * np.float32(C.SCALE)
        arrays = dict(input_model=raw, input_d=xd, C=cm, original_C=cm.copy(), known_mask=np.arange(r['N']) < r['K'],
                      N=np.asarray(r['N'], np.int64), K=np.asarray(r['K'], np.int64))
        h = npz_once(q['input_path'], arrays)
        q.update(label=r['label'], request_id=r['request_id'], candidate=r['candidate'], raw_path=r['raw_path'],
                 raw_sha256=r['raw_sha256'], input_sha256=h, input_d_sha256=C.ah(xd), input_status='COMPLETE_CPU',
                 C_sha256=C.ah(cm), teacher_reuse='NO_exact_proposal_d_missing', teacher_queries_required=1,
                 original_record=r)
        old_npz = C.OLD / 'denoised' / r['label'] / r['request_id'] / 'denoised.npz'
        protect(old_npz, protected)
        with np.load(old_npz) as z:
            assert 'proposal_d' not in z.files and 'proposal_denoiser' not in z.files
        q['historical_teacher_model_space_cache'] = str(old_npz)
        result.append(q)
    synth_manifest = C.read(SYNTH); protect(SYNTH, protected)
    synth_result = []
    for r in synth_manifest['rows']:
        s = by_condition[r['condition_id']]
        sid = 'DEV_SYNTH__' + r['synthetic_id']
        q = row_base(s, sid, 'dev_synthetic', 'synthetic', r['noise_index'], groups)
        protect(r['path'], protected, r['sha256'])
        pred = C.ANCHOR / 'synthetic_results/D0_OFFICIAL' / r['synthetic_id'] / 'prediction.npz'
        receipt = pred.with_name('receipt.json')
        rh = protect(receipt, protected); rc = C.read(receipt)
        assert rc['status'] == 'PASS' and rc['model_checkpoint_sha256'] == C.EXPECTED
        assert rc['synthetic_input_sha256'] == r['sha256']
        ph = protect(pred, protected, rc['output_sha256'])
        with np.load(r['path']) as z:
            xd, cm, clean = z['x_d'].copy(), z['C'].copy(), z['target_d'].copy()
            arrays = dict(input_model=xd/np.float32(C.SCALE), input_d=xd, C=cm, original_C=cm.copy(),
                          known_mask=z['known'].copy(), N=np.asarray(r['N'], np.int64), K=np.asarray(r['K'], np.int64),
                          source_Y_diagnostic_d=clean[r['K']:], source_C=cm.copy(),
                          supported_Y_local_corner_indices=z['supported_Y_local_corner_indices'].copy())
        arrays['input_model'][:r['K']] = cm
        with np.load(pred) as z:
            pd = z['proposal_denoiser']; assert pd.dtype == np.float32 and pd.shape == xd.shape
        h = npz_once(q['input_path'], arrays)
        q.update(input_sha256=h, input_d_sha256=C.ah(xd), C_sha256=C.ah(cm), input_status='COMPLETE_CPU',
                 clean=r['clean'], synthetic_id=r['synthetic_id'], source_input_path=r['path'],
                 source_input_sha256=r['sha256'], teacher_reuse='EXACT_FP32_DENOISER_PROPOSAL', teacher_queries_required=0,
                 teacher_reuse_path=str(pred), teacher_reuse_sha256=ph, teacher_proposal_key='proposal_denoiser',
                 teacher_reuse_receipt=str(receipt), teacher_reuse_receipt_sha256=rh,
                 original_record=r)
        synth_result.append(q)
    assert len(result) == 48 and len(synth_result) == 18
    return result, synth_result


def main():
    completion = DATA / 'preparation_completion.json'
    if completion.exists():
        d = C.read(completion)
        assert d['status'] == 'COMPLETE'
        C.verify(d['protected']); C.verify(d['artifacts'])
        print('SKIP_VERIFIED_COMPLETE', json.dumps(d['counts'])); return
    start = time.monotonic()
    guard_test()
    C.write(DATA / 'resources_before.json', C.resources())
    C.event('CPU_PREPARATION_STARTED', ledger='data_preparation_ledger.jsonl', workers=1, threads=1, GPU_calls=0)
    protected = {}
    for p in [WHITELIST, DEV, USER/'active_conditions.json', USER/'dataset_handoff_active.json', GENERATOR_BINDING]:
        protect(p, protected)
    whitelist, dev = C.read(WHITELIST), C.read(DEV)['rows']
    assert whitelist['status'] == 'PASS' and len(whitelist['rows']) == 112
    binding = C.read(GENERATOR_BINDING)
    assert binding['sha256'] == C.GENERATOR_EXPECTED and Path(binding['path']).is_file()
    groups = isolation(whitelist, protected)
    sources = sorted(whitelist['rows'], key=lambda r: r['condition_id'])
    json_once(DATA/'training_whitelist.json', whitelist)
    source_arrays = {r['condition_id']: load_source(r, protected) for r in sources + dev}
    by_parent = defaultdict(list)
    for r in sources: by_parent[r['uid']].append(r)
    internal_sources = []
    for uid in sorted(by_parent):
        rows = sorted(by_parent[uid], key=lambda r: r['condition_id'])
        internal_sources.append(rows[C.seed('internal_condition', uid) % len(rows)])
    train, generations = [], []
    for r in sources:
        a = source_arrays[r['condition_id']]
        for variant in range(2):
            q = generation_row(r, variant, 'train', groups, a); train.append(q); generations.append(q)
        for variant in range(8): train.append(synthetic_row(r, variant, groups, a))
    internal = [generation_row(r, 2, 'internal_val', groups, source_arrays[r['condition_id']]) for r in internal_sources]
    generations += internal
    assert len(generations) == 249 and len({r['Gaussian_sha256'] for r in generations}) == 249
    assert len(train) == 1120 and sum(r['domain']=='synthetic' for r in train) == 896
    dev_real, dev_synthetic = prepare_dev(dev, groups, protected)
    monitors = fixed_monitors(train)
    # One n0 USER_MIX input per all six frozen DEV conditions spans their entire
    # observed N/K and seat/back range, without using outputs for selection.
    benchmarks = [r['sample_id'] for r in sorted(dev_real, key=lambda r:(r['N'],r['K'],r['condition_id']))
                  if r['label']=='R2_USER_MIX' and r['candidate']==0]
    assert len(benchmarks) == 6
    stream = build_stream(train)
    manifest = dict(schema=SCHEMA, status='CPU_INPUTS_AND_GENERATION_PLAN_FROZEN',
                    train_inputs=train, internal_val=internal, dev_real=dev_real, dev_synthetic=dev_synthetic,
                    generation_plan=generations, monitor_ids=monitors, benchmark_ids=benchmarks,
                    train_stream_path=str(DATA/'train_stream.json'), train_stream_sha256=C.sha(DATA/'train_stream.json'),
                    generator=binding, generator_state_sha256_expected=C.GENERATOR_STATE_EXPECTED,
                    teacher_sha256=C.EXPECTED, scale=C.SCALE, sigma=C.SIGMA, parent_isolation_path=str(DATA/'parent_isolation.json'),
                    training_targets='All 1120 training examples use actual single D0 proposal_d-input_d; source Y never a label',
                    teacher_queries_required=dict(train=1120,internal_validation=25,dev_real=48,dev_synthetic=0,total=1193),
                    synthetic_dev_reuse=18, generation_count=249, new_CPU_Gaussian_count=249,
                    generated_raw_identity_audit_required='Check all 249 actual input_d hashes unique and train224/internal25 disjoint after generation',
                    model_inputs='Only input_d; known_mask used for loss/restoration, not passed as a role embedding',
                    arrays_contract=dict(input_d='Authoritative FP32 [N,3,3] denoiser coordinates',
                                         input_model='FP32 mesh coordinates; display only for noisy synthetic, no teacher inverse scaling',
                                         C='FP32 preserved C in model units',known_mask='bool[N], first K true',
                                         order='C original source-array order followed by free source-array order',
                                         no_online_permutation=True,source_Y_diagnostic_d='Never used as KD target'),
                    fixed_selection_rule='8 training monitors: distinct parents, 4 real/4 synthetic, seat/back, N/K extremes. 6 benchmarks: all six fixed USER_MIX DEV conditions at existing n0.',
                    no_GPU_model_calls=True)
    json_once(DATA/'data.json', manifest)
    C.csvwrite(DATA/'inputs_index.csv', [{k:v for k,v in r.items() if not isinstance(v,(dict,list))}
                                        for r in train+internal+dev_real+dev_synthetic])
    C.verify(protected)
    C.write(DATA/'protected_sources.json', protected)
    C.write(DATA/'resources_after.json', C.resources())
    counts = dict(train_conditions=112,train_parents=25,train_real=224,train_synthetic=896,internal_val=25,
                  generation_Gaussians=249,dev_real=48,dev_synthetic=18,training_updates=4000,exposures=32000,
                  monitor_examples=8,benchmark_examples=6,teacher_queries_required=1193,GPU_calls=0)
    C.event('CPU_PREPARATION_COMPLETE', ledger='data_preparation_ledger.jsonl', counts=counts, seconds=time.monotonic()-start)
    artifacts = {str(p): C.sha(p) for root in [DATA,C.OUT/'inputs',C.OUT/'generation_inputs']
                 for p in root.rglob('*') if p.is_file() and p != completion}
    C.write(completion, dict(status='COMPLETE',schema=SCHEMA,counts=counts,seconds=time.monotonic()-start,
                            protected=protected,artifacts=artifacts,source_sha256=C.sha(Path(__file__))))
    print(json.dumps(dict(status='COMPLETE',counts=counts,manifest=str(DATA/'data.json'))))


if __name__ == '__main__':
    main()
