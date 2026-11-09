"""CPU-only, resumable free-surface scoring for the fixed usability quick pilot.

All new distances are finite closed-triangle / segment distances in original
model coordinates. Historical score fields retain their definitions and names.
No model, optimizer, OT, mesh repair, or generation is invoked here.
The legacy independent CPU pair verifier may import torch but never uses CUDA.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
import numpy as np

from experiments.mf_h1_local0.evaluate import sample_mesh_surface, surface_distance_squared
from experiments.mf_h1_local0.native_inpainting_metrics import array_sha
from experiments.mf_h1_local0.native_patch_seams import _sample_boundary
from experiments.mf_h1_local0.control_mvp48.eval_scoring import prepare_reference, score_free, source_identity
from experiments.mf_h1_local0.gpu_guidance_evaluation import shape_quality
from experiments.mf_h1_local0.native_patch_seams import nearest_segments

ROOT = Path(__file__).resolve().parents[3]
TAUS = (.005, .01, .02, .05)
ATTACH_TAUS = (.001, .003, .005, .01, .02, .05)
VERSION = 'chair_control_usability_score_v1'
DEFINITIONS = dict(
    D_sym2048='(mean d(Q_Y,F)^2 + mean d(Q_F,Y)^2)/(2*L0^2); finite closed triangles; free only',
    PR='Precision=Q_F to Y, Recall=Q_Y to F; strict distance < tau*L0; no alignment',
    sampling='area-proportional faces and sqrt barycentric; local MT19937 seed from SHA256 of namespace and condition/candidate; zero-area faces have zero probability',
    Gamma32='32 equally spaced midpoints per true GT known/free interface edge; weight=edge length/32; edge and surface versions separate',
    normalized_free_pair_distance='arithmetic mean of 6 symmetric surface distances for all four candidates, each using its own frozen Q_F2048',
    failure='Attempted model/invalid output: D=+inf, P/R/Fscore=0; incomplete assets/generation: NA; no denominator deletion',
    Best='GT-oracle minimum D of fixed candidates 0/1 or 0/1/2/3; all secondary metrics come from that same candidate',
    common='Q_common2048 from GT outside C3, same points at all levels; directed only to predicted free, copied C excluded',
    legacy='original 512 frozen GT free points, original <=1% coverage, historical %L RMS/Gamma and CF/FF names retained',
    human='human_usable=NA; user_study=NOT_RUN; no proxy threshold defines usability')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return 'INF' if value == float('inf') else '-INF' if value == -float('inf') else None
    return value


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    with temp.open('w', encoding='utf-8') as f:
        json.dump(clean(value), f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, p)


def write_csv(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows:
            w.writerow({k: 'NA' if v is None else json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in clean(row).items()})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def resolve(path, base=ROOT):
    text = str(path)
    if os.name != 'nt' and len(text) > 2 and text[1] == ':':
        text = '/mnt/' + text[0].lower() + '/' + text[3:].replace('\\', '/')
    p = Path(text)
    return p if p.is_absolute() else Path(base) / p


def arrays(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def tri(value):
    a = np.asarray(value)
    return a.reshape(-1, 3, 3)


def seed(*parts):
    return int.from_bytes(hashlib.sha256(('CHAIR_CONTROL_USABILITY_QUICK_PILOT_V1|' + '|'.join(map(str, parts))).encode()).digest()[:4], 'little')


def area(t):
    t = tri(t).astype(np.float64)
    return .5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=-1)


def tkey(value):
    return format(float(value), '.3f').rstrip('0').rstrip('.')


def pr_metrics(d2_yf, d2_fy, L0, count=2048):
    d2_yf, d2_fy = np.asarray(d2_yf), np.asarray(d2_fy)
    out = {f'GT_to_F_MSE{count}_L0': float(d2_yf.mean() / L0**2),
           f'F_to_GT_MSE{count}_L0': float(d2_fy.mean() / L0**2),
           f'D_sym{count}': float((d2_yf.mean() + d2_fy.mean()) / (2 * L0**2))}
    for tau in TAUS:
        p, r = float(np.mean(d2_fy < (tau * L0)**2)), float(np.mean(d2_yf < (tau * L0)**2))
        key = tkey(tau)
        out.update({f'Precision{count}_{key}': p, f'Recall{count}_{key}': r,
                    f'Fscore{count}_{key}': 2*p*r/(p+r) if p+r else 0.})
    return out


def weighted_stats(d2, weights, L0):
    d2, w = np.asarray(d2), np.asarray(weights)
    order = np.argsort(d2, kind='stable'); cumulative = np.cumsum(w[order])
    index = min(np.searchsorted(cumulative, .95*cumulative[-1], side='left'), len(order)-1)
    return dict(RMS_L0=float(np.sqrt(np.dot(w, d2)/w.sum())/L0),
                p95_L0=float(np.sqrt(d2[order[index]])/L0))


def gamma_points(edges, count):
    edges = np.asarray(edges, np.float64).reshape(-1, 2, 3)
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=-1)
    if not len(edges) or not np.isfinite(edges).all() or np.any(lengths <= 0): return None, None
    u = (np.arange(count, dtype=np.float64) + .5) / count
    q = (edges[:, :1] * (1-u[None, :, None]) + edges[:, 1:] * u[None, :, None]).reshape(-1, 3)
    return q, np.repeat(lengths / count, count)


def interface_metrics(free, edges, L0, count=32):
    q, w = gamma_points(edges, count)
    if q is None: return {f'Gamma{count}_status': 'NOT_APPLICABLE'}, {}
    segments = np.asarray(free)[:, ((0, 1), (1, 2), (2, 0)), :].reshape(-1, 2, 3)
    # Bounded chunks prevent all Gamma points x all free edges allocation.
    de = np.concatenate([nearest_segments(q[i:i+256], segments)[0] for i in range(0, len(q), 256)])
    ds = surface_distance_squared(q, free)
    if np.any(ds > de + 1e-10*L0**2): raise AssertionError('Surface distance exceeded finite-edge distance')
    out = {f'Gamma{count}_status': 'PASS', f'Gamma{count}_points': len(q), f'Gamma{count}_length': float(w.sum())}
    for name, distance in (('edge', de), ('surface', ds)):
        out.update({f'Gamma{count}_{name}_{k}': v for k, v in weighted_stats(distance, w, L0).items()})
        for tau in ATTACH_TAUS:
            out[f'Gamma{count}_{name}_AttachCov_{tkey(tau)}'] = float(w[distance < (tau*L0)**2].sum()/w.sum())
    return out, dict(points=q, weights=w, edge_d2=de, surface_d2=ds)


def angle_reference(target, L0):
    q = shape_quality(target, L0)
    a = tri(target).astype(np.float64)/L0; angles = []
    for i in range(3):
        u, v = a[:, (i+1)%3]-a[:, i], a[:, (i+2)%3]-a[:, i]
        den = np.linalg.norm(u, axis=-1)*np.linalg.norm(v, axis=-1)
        cosine = np.divide((u*v).sum(-1), den, out=np.ones(len(a)), where=den>0)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    values = np.stack(angles, axis=-1)
    return dict(GT_free_minimum_angle_lt5_fraction=q['min_angle_lt5_fraction'],
                GT_free_angle_p05_degrees=q['angle_p05_degrees'],
                GT_free_exact_zero_area_count=q['exact_zero_area_count'],
                GT_free_near_degenerate_count=q['near_degenerate_area_count'],
                GT_free_angle_p50_degrees=float(np.quantile(values, .5)),
                GT_free_angle_p95_degrees=float(np.quantile(values, .95)),
                GT_free_area=float(q['total_area']))


def query_bank(out, request, condition):
    folder = Path(out)/'scoring'/'queries'; folder.mkdir(parents=True, exist_ok=True)
    p = folder/(request['condition_id']+'.npz'); meta = p.with_suffix('.json')
    identity = dict(condition_id=request['condition_id'], GT_free_sha256=array_sha(condition['GT_free']),
                    common_sha256=array_sha(condition['Y_common']) if len(condition.get('Y_common', [])) else None,
                    version=VERSION)
    if p.exists():
        saved = read_json(meta)
        if saved['identity'] != identity or saved['npz_sha256'] != sha(p): raise ValueError('Frozen scoring queries changed')
        return arrays(p)
    values = {}
    sensitivity_ids = {r['condition_id'] for r in read_json(Path(out)/'requests.json') if r.get('sensitivity_check')}
    for n in ((2048, 8192) if request['condition_id'] in sensitivity_ids else (2048,)):
        values['Q_Y'+str(n)] = sample_mesh_surface(condition['GT_free'], n, np.random.RandomState(seed('Q_Y', request['condition_id'], n)))
    if len(condition.get('Y_common', [])):
        values['Q_common2048'] = sample_mesh_surface(condition['Y_common'], 2048, np.random.RandomState(seed('Q_common', request['uid'], 2048)))
    if any(v is None for v in values.values()): raise ValueError('Zero-area target query surface')
    np.savez_compressed(p, **values)
    write_json(meta, dict(identity=identity, npz_sha256=sha(p), array_hashes={k: array_sha(v) for k, v in values.items()},
                         RNG='MT19937 with SHA256 derived uint32 seeds', definitions=DEFINITIONS))
    return values


def historical_score(out, request, condition, free, cf_ff):
    historical = request.get('legacy_score_path')
    if historical and resolve(historical).exists():
        saved = read_json(resolve(historical)); value = saved.get('score', saved)
        detail = value['detail']; audit = detail['reference']
        if (detail['predicted_free_sha256'] != array_sha(free) or
            audit['known_sha256'] != array_sha(condition['C']) or audit['GT_free_sha256'] != array_sha(condition['GT_free'])):
            raise ValueError('Historical score belongs to different output/condition')
        if saved.get('identity', {}).get('score_sources') not in (None, source_identity()):
            raise ValueError('Historical scoring source identity differs')
        return value, 'CACHED_IDENTITY_VERIFIED', str(resolve(historical))
    edges = condition['gamma_edges']
    if 'legacy_gamma_points' in condition and 'legacy_gamma_weights' in condition:
        qp, weights = condition['legacy_gamma_points'], condition['legacy_gamma_weights']
    elif len(edges):
        qp, _, weights, _ = _sample_boundary(edges, 9919, 512)
    else:
        qp, weights = np.empty((0, 3)), np.empty(0)
    reference = prepare_reference(condition['C'], condition['GT_free'], float(condition['L0']), gamma_points=qp, gamma_weights=weights)
    # Frozen original 512 arrays are authoritative, including their own hashes.
    for supplied, key in (('legacy_query_points', 'coverage_points'), ('legacy_pairwise_points', 'pairwise_target_points')):
        if supplied in condition:
            reference['arrays'][key] = condition[supplied]
            reference['audit']['array_sha256'][key] = array_sha(condition[supplied])
    value = score_free(free, condition['C'], condition['GT_free'], float(condition['L0']), reference=reference, include_pair_audit=cf_ff)
    path = Path(out)/'scoring'/'legacy_scores'/(request['request_id']+'.json')
    write_json(path, value)
    return value, 'RECOMPUTED_ORIGINAL_DEFINITION', str(path)


def failure_metrics(row):
    row.update(D_sym2048=float('inf'), GT_to_F_MSE2048_L0=float('inf'), F_to_GT_MSE2048_L0=float('inf'))
    for tau in TAUS:
        for name in ('Precision', 'Recall', 'Fscore'): row[f'{name}2048_{tkey(tau)}'] = 0.
    return row


def score_request(out, request, *, cf_ff=True):
    out = Path(out); rid = request['request_id']; dest = out/'scoring'/'outputs'/(rid+'.json')
    base = {k: request.get(k) for k in ('request_id', 'uid', 'condition_id', 'context_level', 'candidate', 'arm', 'panel', 'panels')}
    result_path = out/'outputs'/rid/'result.json'
    result = read_json(result_path) if result_path.exists() else {}
    status = result.get('status', request.get('status', 'BLOCKED'))
    base.update(generation_status=status, attempted=status in ('CACHED', 'SAMPLED', 'PASS', 'FAILED', 'FAIL'),
                human_usable=None, user_study='NOT_RUN', C_preserved=None)
    if status in ('NOT_APPLICABLE', 'NA'):
        return dict(base, status='NOT_APPLICABLE', reason=result.get('reason', request.get('reason')))
    if status in ('FAILED', 'FAIL'):
        return failure_metrics(dict(base, status='GENERATION_FAILED', reason=result.get('reason')))
    if status not in ('CACHED', 'SAMPLED', 'PASS') or not result.get('raw_path'):
        return dict(base, status='BLOCKED', reason=result.get('reason', 'Output not completed'))
    rawpath = resolve(result['raw_path'], out)
    if not rawpath.exists(): return dict(base, status='BLOCKED', reason='Registered raw asset missing')
    condition_path = resolve(request['condition_npz'], out)
    condition = arrays(condition_path); rawhash = sha(rawpath)
    if result.get('raw_sha256') and result['raw_sha256'] != rawhash: raise ValueError('Raw identity changed: '+rid)
    identity = dict(raw_sha256=rawhash, condition_sha256=sha(condition_path), scoring_sha256=sha(__file__),
                    source_scorer_identity=source_identity(), cf_ff=bool(cf_ff))
    if dest.exists():
        old = read_json(dest)
        if old.get('identity') == identity: return old['metrics']
        raise ValueError('Completed scoring cache identity differs: '+rid)
    tick = time.perf_counter(); z = arrays(rawpath); raw = z['raw']
    n, k, L0 = int(condition['N']), int(condition['K']), float(condition['L0'])
    if not np.isfinite(L0) or L0 <= 0: raise ValueError('Positive original-parent L0 required')
    base.update(raw_path=str(rawpath), raw_sha256=rawhash, L0=L0, N=n, K=k, free_faces=n-k,
                known_face_ratio=k/n, known_area_ratio=float(area(condition['C']).sum()/area(condition['GT_full']).sum()),
                known_span_L0=float(np.linalg.norm(np.ptp(tri(condition['C']).reshape(-1, 3), axis=0))/L0))
    elapsed = result.get('metadata', {}).get('seconds')
    if elapsed is None: elapsed = result.get('sampling', {}).get('seconds')
    if status == 'CACHED': elapsed = result.get('source_generation_seconds')
    base.update(generation_seconds=elapsed, generation_time_source='historical_generation' if status == 'CACHED' else 'actual_new_generation')
    shapeok = raw.shape in ((n, 9), (n, 3, 3))
    base.update(C_preserved=bool(shapeok and np.array_equal(tri(raw)[:k], condition['C'])), finite=bool(np.isfinite(raw).all()), N_preserved=shapeok)
    free = tri(raw)[k:] if shapeok else np.empty((0, 3, 3))
    if not shapeok or not base['finite'] or not base['C_preserved'] or not len(free) or area(free).sum() <= 0:
        value = failure_metrics(dict(base, status='INVALID_OUTPUT', reason='N/C/finite/nonempty/positive-area contract failed'))
        write_json(dest, dict(identity=identity, metrics=value)); return value
    qb = query_bank(out, request, condition); vectors = {}
    counts = (2048, 8192) if request.get('sensitivity_check') else (2048,)
    for count in counts:
        qf = sample_mesh_surface(free, count, np.random.RandomState(seed('Q_F', request['condition_id'], request['candidate'], count)))
        dy = surface_distance_squared(qb['Q_Y'+str(count)], free)
        df = surface_distance_squared(qf, condition['GT_free'])
        base.update(pr_metrics(dy, df, L0, count))
        vectors.update({f'Q_F{count}': qf, f'GT_to_F_d2_{count}': dy, f'F_to_GT_d2_{count}': df})
        base[f'Q_F{count}_sha256'] = array_sha(qf); base[f'Q_Y{count}_sha256'] = array_sha(qb['Q_Y'+str(count)])
    for count in ((32, 128) if request.get('sensitivity_check') else (32,)):
        gm, gv = interface_metrics(free, condition['gamma_edges'], L0, count); base.update(gm)
        vectors.update({'Gamma'+str(count)+'_'+name: v for name, v in gv.items()})
    base.update(angle_reference(condition['GT_free'], L0))
    # Supplemental all-corner quantiles accompany the original small-angle score.
    fq = angle_reference(free, L0)
    for key, value in fq.items(): base[key.replace('GT_free_', 'predicted_free_')] = value
    if 'Q_common2048' in qb:
        common = surface_distance_squared(qb['Q_common2048'], free)
        base.update(common_RMS_L0=float(np.sqrt(common.mean())/L0), common_p95_L0=float(np.quantile(np.sqrt(common), .95)/L0),
                    common_query_sha256=array_sha(qb['Q_common2048']), common_GT_area=float(area(condition['Y_common']).sum()))
        for tau in TAUS: base['common_recall_'+tkey(tau)] = float(np.mean(common < (tau*L0)**2))
        vectors['common_d2'] = common
    try:
        old, reuse, opath = historical_score(out, request, condition, free, cf_ff)
        base.update(old['metrics'], legacy_score_status=old['status'], legacy_score_source=reuse, legacy_score_path=opath,
                    CF_FF_status=old['detail'].get('pair_audit_status', 'NOT_RUN'))
        if old['detail'].get('fixed_GT_coverage', {}).get('per_point_distances') is not None:
            d = np.asarray(old['detail']['fixed_GT_coverage']['per_point_distances'], np.float64)
            base.update(fixed_query_MSE=float(np.mean((d/L0)**2)), coverage_fraction=float(np.mean(d <= .01*L0)),
                        fixed_query_p95_L=float(np.quantile(d/L0, .95)), legacy_query_sha256=old['detail']['fixed_GT_coverage']['points_sha256'])
    except Exception as exc:
        # Secondary legacy / intersection failure must not discard completed new geometry.
        base.update(legacy_score_status='NOT_RUN', CF_FF_status='NOT_RUN', legacy_failure=repr(exc))
        if 'legacy_query_points' in condition:
            d = np.sqrt(surface_distance_squared(condition['legacy_query_points'], free))
            base.update(fixed_query_MSE=float(np.mean((d/L0)**2)), coverage_fraction=float(np.mean(d <= .01*L0)),
                        fixed_query_p95_L=float(np.quantile(d/L0, .95)))
    # Always recompute/check the original frozen 512 queries independently of old JSON layout.
    if 'legacy_query_points' in condition:
        dq = np.sqrt(surface_distance_squared(condition['legacy_query_points'], free))
        fixed = dict(fixed_query_MSE=float(np.mean((dq/L0)**2)), coverage_fraction=float(np.mean(dq <= .01*L0)),
                     fixed_query_p95_L=float(np.quantile(dq/L0, .95)), legacy_query_sha256=array_sha(condition['legacy_query_points']))
        for key in ('fixed_query_MSE', 'coverage_fraction', 'fixed_query_p95_L'):
            if key in base and not np.isclose(base[key], fixed[key], rtol=1e-10, atol=1e-12):
                raise ValueError('Original 512 metric disagrees with frozen query recomputation: '+key)
        base.update(fixed)
        vectors['legacy_query_d2'] = dq*dq
    else:
        base['legacy_fixed_query_status'] = 'BLOCKED_MISSING_ORIGINAL_QUERIES'
    # Legacy m.C_preserved can be a placeholder: raw contract is authoritative.
    base.update(C_preserved=True, status='PASS', scoring_seconds=time.perf_counter()-tick)
    vector_path = out/'scoring'/'vectors'/(rid+'.npz'); vector_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(vector_path, **vectors)
    base['vectors_path'] = str(vector_path); base['vectors_sha256'] = sha(vector_path)
    write_json(dest, dict(identity=identity, metrics=base))
    return clean(base)


def run(out, *, available_only=False, cf_ff=True):
    out = Path(out); reqpath = out/'requests.json'
    if not reqpath.exists(): reqpath = out/'registration'/'requests.json'
    requests = read_json(reqpath); requests = requests.get('rows', requests) if isinstance(requests, dict) else requests
    write_json(out/'scoring'/'definitions.json', dict(version=VERSION, definitions=DEFINITIONS, thresholds=TAUS, interface_thresholds=ATTACH_TAUS,
        point_counts=[2048, 8192], interface_points_per_edge=[32, 128], source_identity=source_identity(), scoring_sha256=sha(__file__)))
    rows = []
    for request in requests:
        if available_only and not (out/'outputs'/request['request_id']/'result.json').exists(): continue
        tick = time.perf_counter()
        try: row = score_request(out, request, cf_ff=cf_ff)
        except Exception as exc:
            row = {k: request.get(k) for k in ('request_id', 'uid', 'condition_id', 'context_level', 'candidate', 'arm', 'panel', 'panels')}
            row.update(status='SCORING_BLOCKED', reason=repr(exc), attempted=None)
        rows.append(row)
        ledger = out/'scoring'/'cpu_ledger.jsonl'; ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open('a', encoding='utf-8') as f:
            f.write(json.dumps(clean(dict(request_id=request['request_id'], status=row['status'], seconds=time.perf_counter()-tick,
                                         time_unix=time.time(), model_forward=0, backward=0, optimizer_update=0)))+'\n')
        print('CPU score', len(rows), request['request_id'], row['status'], flush=True)
    write_json(out/'scoring'/'per_output.json', rows); write_csv(out/'per_output_metrics.csv', rows)
    from .aggregation import finalize
    return finalize(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True)
    parser.add_argument('--available-only', action='store_true'); parser.add_argument('--skip-pairs', action='store_true')
    args = parser.parse_args(); print(json.dumps(clean(run(args.out, available_only=args.available_only, cf_ff=not args.skip_pairs)), ensure_ascii=False))
