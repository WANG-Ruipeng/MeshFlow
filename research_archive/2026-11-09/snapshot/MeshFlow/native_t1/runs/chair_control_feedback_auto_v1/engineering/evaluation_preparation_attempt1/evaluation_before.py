"""Frozen CPU scoring adapter; no model construction or generation.

Only input-side reference preparation may run before CONFIRM is unlocked.
The original quality and v1r1 control modules are imported read-only. A private
function namespace adapts the new NPZ key ``output`` to the original ``raw``;
no old module globals, geometry, thresholds or sources are modified.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
import traceback
import types
import numpy as np
from . import assets
from experiments.mf_h1_local0.chair_control_usability_quick_v1 import scoring as quality
from experiments.mf_h1_local0.chair_boundary_attachment_v1 import control_metrics as control

VERSION = 'chair_control_feedback_evaluation_v1'
CONTROL_SHA = 'a6036ccd2f6063a8a863591d4aa79e5358de7ab617449a4c19b133f095ecf9ab'
QUALITY_SHA = 'c287c97f0e76bf59ec99fe05f7fe494f27d69799d8ae0a938844d3a3663be01c'
POOLS = ('TRAIN_FEEDBACK', 'DEV', 'CONFIRM')
ROOT, OUT = assets.ROOT, assets.OUT
read_json, digest, resolve, arrays = assets.read_json, assets.digest, assets.resolve, assets.arrays
freeze_json, freeze_npz, bits = assets.freeze_json, assets.freeze_npz, assets.bits
write_json, write_csv = quality.write_json, quality.write_csv


def identities():
    if control.VERSION != 'chair_boundary_attachment_control_metrics_v1r1':
        raise ValueError('Wrong control version')
    if digest(control.__file__) != CONTROL_SHA or digest(quality.__file__) != QUALITY_SHA:
        raise ValueError('Original frozen scoring implementation changed')
    return dict(adapter=digest(__file__), control=control.source_identity(),
                quality=digest(quality.__file__), legacy=quality.source_identity())


def _manifest(value):
    data = read_json(resolve(value)) if isinstance(value, (str, Path)) else value
    rows = data['rows'] if isinstance(data, dict) else data
    if not rows: raise ValueError('Empty evaluation cohort')
    pool = data.get('pool', rows[0]['split']) if isinstance(data, dict) else rows[0]['split']
    if pool not in POOLS or any(r['split'] != pool for r in rows): raise ValueError('Mixed or unknown pool')
    if len({r['request_id'] for r in rows}) != len(rows): raise ValueError('Duplicate registered request')
    return pool, rows


def _condition(request, out):
    path = resolve(request['condition_npz'])
    if digest(path) != request['condition_sha256']: raise ValueError('Condition hash changed')
    metadata = next((r for r in read_json(Path(out)/'registration/conditions.json')
                     if r['condition_id'] == request['condition_id']), {})
    c = dict(metadata); c.update(arrays(path))
    c.update(condition_id=request['condition_id'], condition_sha256=request['condition_sha256'])
    c.setdefault('gamma_status', request.get('gamma_status', 'NOT_APPLICABLE' if not len(c['gamma_edges']) else 'PASS'))
    if int(c['N']) != int(request['N']) or int(c['K']) != int(request['K']): raise ValueError('N/K changed')
    for key in ('legacy_query_points', 'legacy_pairwise_points'):
        if np.asarray(c[key]).shape != (512, 3): raise ValueError('Missing original 512 queries')
    return c


def _legacy_pointer(request, condition, prefix, suffix):
    key = prefix+'_'+suffix
    value = request.get(key, condition.get(key))
    if not value: return None
    path = resolve(value)
    expected = request.get(key+'_sha256', condition.get(key+'_sha256'))
    if not expected: raise ValueError('Historical reference hash not registered: '+key)
    if digest(path) != expected: raise ValueError('Historical reference changed: '+key)
    return path


def _reference(out, request, condition):
    folder = Path(out)/'references'/request['condition_id']
    p = folder/'control.npz'; meta = folder/'control.json'
    expected = dict(condition_sha256=request['condition_sha256'], control_sources=control.source_identity(),
                    definition=control.frozen_definition(), density='main')
    if meta.exists():
        saved = read_json(meta)
        if saved['identity'] != expected or digest(p) != saved['npz_sha256']: raise ValueError('Reference changed')
        ref = dict(saved['reference']); ref['arrays'] = arrays(p)
    else:
        old_meta = _legacy_pointer(request, condition, 'legacy_control_reference', 'json')
        old_npz = _legacy_pointer(request, condition, 'legacy_control_reference', 'npz')
        if old_meta or old_npz:
            if not old_meta or not old_npz: raise ValueError('Incomplete legacy reference pair')
            old = read_json(old_meta)
            if old['cache_identity']['source_identity'] != control.source_identity(): raise ValueError('Legacy source mismatch')
            if old['cache_identity']['definition'] != control.frozen_definition(): raise ValueError('Legacy definition mismatch')
            ref = dict(old['reference']); ref['arrays'] = arrays(old_npz)
            if digest(old_npz) != old['npz_sha256']: raise ValueError('Legacy reference NPZ mismatch')
            for key, value in ref['arrays'].items():
                if assets.array_hash(value) != old['arrays_sha256'][key]: raise ValueError('Legacy outer array hash mismatch')
            origin = dict(kind='EXACT_HISTORICAL_ARRAYS', json=str(old_meta), npz=str(old_npz))
        else:
            ref = control.prepare_reference(condition, request['condition_id'], density='main')
            origin = dict(kind='ORIGINAL_FIXED_INPUT_ONLY_REFERENCE_FORMULA')
        freeze_npz(p, ref['arrays'])
        freeze_json(meta, dict(identity=expected, reference={k:v for k,v in ref.items() if k!='arrays'},
                              npz_sha256=digest(p), origin=origin))
    identity = ref['identity']
    if (identity['condition_id'] != request['condition_id'] or identity['C_sha256'] != quality.array_sha(condition['C']) or
        identity['GT_free_sha256'] != quality.array_sha(condition['GT_free'])): raise ValueError('Reference geometry mismatch')
    for key, value in ref['arrays'].items():
        if quality.array_sha(value) != ref['array_sha256'][key]: raise ValueError('Reference internal array hash changed')
    return ref


def _queries(out, request, condition):
    folder = Path(out)/'references'/request['condition_id']
    p = folder/'quality.npz'; meta = folder/'quality.json'
    expected = dict(condition_id=request['condition_id'], GT_free_sha256=quality.array_sha(condition['GT_free']),
                    common_sha256=quality.array_sha(condition['Y_common']) if len(condition.get('Y_common', [])) else None,
                    version=quality.VERSION)
    if meta.exists():
        saved = read_json(meta)
        if saved['identity'] != expected or digest(p) != saved['npz_sha256']: raise ValueError('Frozen GT queries changed')
        qb = arrays(p)
    else:
        oldmeta = _legacy_pointer(request, condition, 'legacy_quality_query', 'json')
        oldnpz = _legacy_pointer(request, condition, 'legacy_quality_query', 'npz')
        if oldmeta or oldnpz:
            if not oldmeta or not oldnpz: raise ValueError('Incomplete legacy query pair')
            saved = read_json(oldmeta)
            if saved['identity'] != expected or digest(oldnpz) != saved['npz_sha256']: raise ValueError('Old QY identity mismatch')
            qb = arrays(oldnpz)
            for key, value in qb.items():
                if quality.array_sha(value) != saved['array_hashes'][key]: raise ValueError('Old QY array hash mismatch')
            origin = str(oldnpz)
        else:
            qb = {'Q_Y2048': quality.sample_mesh_surface(condition['GT_free'], 2048,
                   np.random.RandomState(quality.seed('Q_Y', request['condition_id'], 2048)))}
            if len(condition.get('Y_common', [])):
                qb['Q_common2048'] = quality.sample_mesh_surface(condition['Y_common'], 2048,
                    np.random.RandomState(quality.seed('Q_common', request['uid'], 2048)))
            origin = 'ORIGINAL_FIXED_INPUT_ONLY_QUERY_FORMULA'
        if qb.get('Q_Y2048') is None: raise ValueError('Zero-area GT query target')
        freeze_npz(p, qb)
        freeze_json(meta, dict(identity=expected, npz_sha256=digest(p), origin=origin,
                              array_hashes={k:quality.array_sha(v) for k,v in qb.items()}))
    return qb


def prepare_references(out=OUT, manifests=None):
    """Freeze all input references and eligibility; never inspect any output."""
    out = resolve(out); manifests = manifests or [out/'manifests'/(p+'.json') for p in POOLS]
    identities(); seen = {}; started = time.perf_counter()
    for manifest in manifests:
        pool, rows = _manifest(manifest)
        for request in rows:
            cid = request['condition_id']
            if cid in seen: continue
            c = _condition(request, out); ref = _reference(out, request, c); qb = _queries(out, request, c)
            vis = ref['arrays']['GT_visibility']
            # The old function uses any VISIBLE ray as reference exposed support.
            support = int(np.any(vis == control.VISIBLE, axis=1).sum())
            seen[cid] = dict(condition_id=cid, split=pool, uid=request['uid'],
                condition_sha256=request['condition_sha256'], S_applicable=bool(ref['audits']['gamma']['status']=='PASS'),
                I_applicable=True, E_applicable=support>0, E_reference_points=support,
                C_points=2048, gamma_points_per_edge=64, directions=26,
                GT_query_sha256=quality.array_sha(qb['Q_Y2048']),
                control_reference_sha256=digest(out/'references'/cid/'control.npz'),
                quality_reference_sha256=digest(out/'references'/cid/'quality.npz'))
    frozen = dict(version=VERSION, sources=identities(), control=control.frozen_definition(),
                  quality=quality.DEFINITIONS, conditions=list(seen.values()), model_calls=0,
                  applicability='Frozen from C/GT references only; LOW_SUPPORT stays in denominator; output-induced failures never become NA')
    freeze_json(out/'references/definition_and_eligibility.json', frozen)
    return dict(status='PASS', conditions=len(seen), seconds=time.perf_counter()-started, model_calls=0)


def _adapted_arrays(path):
    z = arrays(path)
    if 'raw' not in z and 'output' in z: z['raw'] = z['output']
    return z


def _cf_scan(row):
    if row.get('legacy_score_path'):
        saved = read_json(resolve(row['legacy_score_path'])); value = saved.get('score', saved)
        scan = value.get('detail', {}).get('CF')
        if scan and scan.get('independent_verification') is True: return scan
    return None


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _contracts(request, raw, condition, result):
    n, k = int(request['N']), int(request['K']); output = raw.get('raw', raw.get('output'))
    shape = output is not None and output.shape == (n,9) and output.dtype == np.float32
    c = np.asarray(condition['C']).reshape(k,9)
    facts = dict(N_preserved=bool(shape), C_output_bitwise=bool(shape and _same(output[:k],c)),
                 finite=bool(shape and np.isfinite(output).all()), C_all_steps_bitwise=None,
                 initial_noise_bitwise=None, NPZ_C_bitwise=None, PLY_C_bitwise=None)
    path = raw.get('path')
    if path is not None:
        pathok = path.shape == (51,n,9) and path.dtype == np.float32
        facts['C_all_steps_bitwise'] = bool(pathok and all(_same(state[:k],c) for state in path))
        facts['finite'] = facts['finite'] and bool(pathok and np.isfinite(path).all())
        if pathok and not _same(path[-1],output): facts['N_preserved'] = False
    bank = arrays(resolve(request['bank_npz']))
    if request.get('bank_sha256') and digest(resolve(request['bank_npz'])) != request['bank_sha256']:
        raise ValueError('Noise bank changed')
    z = raw.get('initial_gaussian',raw.get('z'))
    if z is not None:
        facts['initial_noise_bitwise'] = _same(np.asarray(z).reshape(n,9)[k:],np.asarray(bank['z']).reshape(n,9)[k:])
    # A cached RAW cannot demonstrate its entire historical path on its own.
    # Explicit archived audits may establish those facts, but never infer them.
    for source in (result.get('contracts',{}),result.get('geometry_contracts',{}),result.get('export_audit',{})):
        for key in facts:
            if facts[key] is None and isinstance(source.get(key),bool): facts[key] = source[key]
    for key in ('export_audit_path','audit_path'):
        if result.get(key) and resolve(result[key]).exists():
            audit = read_json(resolve(result[key]))
            for name in facts:
                if facts[name] is None and isinstance(audit.get(name),bool): facts[name]=audit[name]
    return facts


def evaluate_request(request, raw_path, out_dir, label):
    """Score one fixed request; callers use at most two disjoint CPU workers."""
    out = resolve(out_dir); pool = request['split']; rid = request['request_id']
    folder = out/'evaluation'/label/pool; dest = folder/'scoring/joined'/(rid+'.json')
    rawpath = resolve(raw_path); base = {k:request.get(k) for k in
        ('request_id','uid','condition_id','split','candidate','noise_seed','N','K','source_task_id','source_case_id')}
    base.update(label=label, model_calls=0, human_usable=None, user_study='NOT_RUN')
    if not rawpath.exists(): return dict(base,status='NOT_RUN',reason='Registered raw missing')
    sources = identities(); condition = _condition(request,out)
    resultpath = rawpath.with_name('result.json'); result = read_json(resultpath) if resultpath.exists() else {}
    identity = dict(raw_sha256=digest(rawpath),condition_sha256=request['condition_sha256'],sources=sources,
                    request_id=rid,label=label,split=pool,bank_sha256=digest(resolve(request['bank_npz'])))
    if result.get('raw_sha256') and result['raw_sha256'] != identity['raw_sha256']: raise ValueError('Result raw hash differs')
    if dest.exists():
        saved = read_json(dest)
        if saved['identity'] != identity: raise ValueError('Completed score identity changed')
        for key in ('control_vectors_path','vectors_path'):
            p = saved['metrics'].get(key)
            if p and digest(resolve(p)) != saved['metrics'][key.replace('_path','_sha256')]: raise ValueError('Score vectors changed')
        return saved['metrics']
    tick = time.perf_counter(); raw = _adapted_arrays(rawpath)
    facts = _contracts(request,raw,condition,result)
    if any(facts[x] is False for x in facts):
        value = quality.clean(quality.failure_metrics(dict(base,**facts,status='INVALID_OUTPUT',reason='Hard geometry/runtime/export contract failed')))
        freeze_json(dest,dict(identity=identity,metrics=value)); return value
    reference = _reference(out,request,condition); qb = _queries(out,request,condition)
    qrequest = dict(request,arm=label,panel=pool,panels=[pool],sensitivity_check=False)
    # Derived bridge only; source NPZ is never rewritten/reordered/repaired.
    bridge = dict(result,status='CACHED' if label=='START' and request.get('raw_cache') else 'SAMPLED',
                  raw_path=str(rawpath),raw_sha256=identity['raw_sha256'])
    freeze_json(folder/'outputs'/rid/'result.json',bridge)
    namespace = dict(quality.__dict__)
    namespace.update(arrays=_adapted_arrays,query_bank=lambda *_args:qb)
    qscore = types.FunctionType(quality.score_request.__code__,namespace,quality.score_request.__name__,quality.score_request.__defaults__)
    qscore.__kwdefaults__ = quality.score_request.__kwdefaults__
    row = qscore(folder,qrequest,cf_ff=True); row.update(base); row.update(facts)
    if row['status']=='PASS':
        free = raw['raw'][int(request['K']):].reshape(-1,3,3)
        control_result = control.score_control(free,condition,reference,cf_scan=_cf_scan(row))
        cp = folder/'scoring/control'/rid
        freeze_npz(cp.with_suffix('.npz'),control_result['arrays'])
        freeze_json(cp.with_suffix('.json'),dict(identity=identity,metrics=control_result['metrics'],
            statuses=control_result['statuses'],audit=control_result['audit'],vectors_sha256=digest(cp.with_suffix('.npz'))))
        row.update(control_result['metrics'],control_statuses=control_result['statuses'],
            control_vectors_path=str(cp.with_suffix('.npz')),control_vectors_sha256=digest(cp.with_suffix('.npz')))
    row.update(adapter_seconds=time.perf_counter()-tick,scoring_sources=sources,
               raw_path=str(rawpath),raw_sha256=identity['raw_sha256'])
    freeze_json(dest,dict(identity=identity,metrics=quality.clean(row)))
    return quality.clean(row)


def _score_job(arguments):
    request, raw, out, label = arguments
    try: return evaluate_request(request,raw,out,label)
    except Exception as exc:
        return dict(request_id=request['request_id'],uid=request['uid'],candidate=request['candidate'],
            condition_id=request['condition_id'],split=request['split'],label=label,status='SCORING_BLOCKED',
            reason=repr(exc),traceback=traceback.format_exc(),model_calls=0)


def score_set(manifest,label,raw_root,out=OUT,*,workers=2,confirm_release=None):
    """Registered cohort only; failures/missing outputs retain their denominator."""
    from .aggregation import summarize
    out = resolve(out); pool, requests = _manifest(manifest)
    if pool=='CONFIRM':
        if not confirm_release: raise ValueError('CONFIRM output access requires explicit frozen release receipt')
        release = read_json(resolve(confirm_release))
        if release.get('status')!='FROZEN' or not release.get('candidate_checkpoint_sha256'):
            raise ValueError('CONFIRM candidate/formula/steps not frozen')
    if workers not in (1,2): raise ValueError('Only one or two CPU workers authorized')
    folder = out/'evaluation'/label/pool; raw_root = resolve(raw_root); jobs=[]
    for request in requests:
        raw = raw_root/request['request_id']/'raw.npz'
        if label=='START' and not raw.exists() and request.get('raw_cache'):
            cache=request['raw_cache'];raw=resolve(cache['path'])
            if digest(raw)!=cache['sha256']:raise ValueError('START RAW cache changed')
        jobs.append((request,str(raw),str(out),label))
    tick=time.perf_counter()
    if workers==1: rows=list(map(_score_job,jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor: rows=list(executor.map(_score_job,jobs))
    write_json(folder/'per_output.json',rows);write_csv(folder/'per_output_metrics.csv',rows)
    summary=summarize(rows,requests)
    summary.update(label=label,pool=pool,seconds=time.perf_counter()-tick,cpu_workers=workers,model_calls=0,
                   manifest_sha256=digest(resolve(manifest)) if isinstance(manifest,(str,Path)) else None)
    write_json(folder/'summary.json',summary);write_csv(folder/'per_parent_metrics.csv',summary['parents'])
    write_csv(folder/'per_noise_metrics.csv',summary['noise'])
    print('CPU evaluation',label,pool,summary['status'],len(rows),round(summary['seconds'],3),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));p.add_argument('--prepare',action='store_true')
    p.add_argument('--manifest');p.add_argument('--label');p.add_argument('--raw-root');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--confirm-release');a=p.parse_args()
    result=prepare_references(a.out) if a.prepare else score_set(a.manifest,a.label,a.raw_root,a.out,workers=a.workers,confirm_release=a.confirm_release)
    print(json.dumps(quality.clean(result),ensure_ascii=False))
