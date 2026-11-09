"""Read saved S2000 geometry only. No model, GEO implementation, or FF/I scan."""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import traceback
import numpy as np

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'experiments/r2_seam_s2000_fragment_audit_v1'
DEST = OUT / 'numeric'
PROTOCOL = OUT / 'protocol.json'
OLD_CSV = ROOT / 'experiments/r2_seam_quality_wave1_wave2_v1/results/final/per_output.csv'
TINY = (('tiny_1e6', 1e-6), ('tiny_1e5', 1e-5), ('tiny_1e4', 1e-4))
FLAGS = ('small_angle', *(name for name, _ in TINY))
ROLES = ('free', 'seam', 'bulk', 'C')
VERSION = 'saved_s2000_fp64_fragment_attribution_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csvwrite(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except Exception as e:
        return dict(error=repr(e))


def resources():
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), cpu_logical=os.cpu_count(),
        loadavg=list(os.getloadavg()), memory=Path('/proc/meminfo').read_text(),
        gpu=command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv']),
        gpu_processes=command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv']),
        cpu_work=command(['ps', '-eo', 'pid,pcpu,pmem,args', '--sort=-pcpu']),
        analysis_workers=1, threads_per_worker=1)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def shape(triangles, L):
    t = np.asarray(triangles, np.float64) / L
    assert t.ndim == 3 and t.shape[1:] == (3, 3) and np.isfinite(t).all()
    lengths = np.stack([np.linalg.norm(t[:, (j+1) % 3] - t[:, j], axis=1) for j in range(3)], 1)
    area = .5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    angles = []
    for j in range(3):
        a, b = t[:, (j+1) % 3] - t[:, j], t[:, (j+2) % 3] - t[:, j]
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cosine = np.divide((a*b).sum(1), den, out=np.ones_like(den), where=den > 0)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    minimum = np.min(np.stack(angles, 1), 1)
    q = np.divide((lengths**2).sum(1), 4*np.sqrt(3)*area,
                  out=np.full_like(area, np.inf), where=area > 0)
    return dict(area_model2=area*L*L, area_over_L2=area, minimum_angle_degrees=minimum,
                longest_edge_model=lengths.max(1)*L, longest_edge_over_L=lengths.max(1),
                q=q, exact_zero_area=area == 0, small_angle=minimum < 5,
                **{name: area < tau for name, tau in TINY})


def dsu_components(faces, kind):
    n = len(faces)
    roots = list(range(n))
    def find(x):
        while roots[x] != x:
            roots[x] = roots[roots[x]]
            x = roots[x]
        return x
    def join(a, b):
        a, b = find(a), find(b)
        roots[max(a, b)] = min(a, b)
    buckets = {}
    for i, f in enumerate(faces):
        keys = list(map(int, f)) if kind == 'vertex' else [tuple(sorted((int(f[j]), int(f[(j+1) % 3])))) for j in range(3)]
        for key in keys:
            if key in buckets:
                join(i, buckets[key])
            else:
                buckets[key] = i
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def meta(q):
    return {k: q[k] for k in ('request_id', 'parent_id', 'condition_id', 'split', 'noise_index', 'N', 'K')}


def summarize(rows, group_fields, fields, sums=()):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in group_fields)].append(r)
    out = []
    for keys, rr in sorted(groups.items()):
        row = dict(zip(group_fields, keys), outputs=len(rr))
        for name in fields:
            vv = [float(r[name]) for r in rr if r.get(name) is not None]
            row[name+'_macro_mean'] = float(np.mean(vv)) if vv else None
            if name in sums:
                row[name+'_total'] = float(sum(vv)) if vv else None
        out.append(row)
    return out


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    completion = DEST / 'completion.json'
    if completion.exists():
        done = read(completion)
        for p, h in done['source_hashes'].items():
            assert sha(p) == h, p
        for p, h in done['output_hashes'].items():
            assert sha(p) == h, p
        return dict(status='ALREADY_COMPLETE_VERIFIED', completion=str(completion))
    tick = time.perf_counter()
    write(DEST / 'resources_before.json', resources())
    protocol = read(PROTOCOL)
    assert protocol['status'] == 'FROZEN' and len(protocol['requests']) == 28
    assert protocol['small_angle_degrees'] == 5
    assert protocol['area_over_L2_thresholds'] == [1e-6, 1e-5, 1e-4]
    L = float(protocol['L'])
    source = {**protocol['source_hashes'], str(PROTOCOL): sha(PROTOCOL), str(OLD_CSV): sha(OLD_CSV),
              str(Path(__file__)): sha(__file__)}
    before = {p: sha(p) for p in source}
    assert before == source, 'Registered input hash changed'
    write(DEST / 'source_hashes_before.json', before)
    with OLD_CSV.open(encoding='utf-8-sig', newline='') as f:
        old = {(r['request_id'], r['stage']): r for r in csv.DictReader(f) if r['label'] == 'PLAIN_S2000'}
    tables = {name: [] for name in ('per_face', 'per_output', 'retained_per_output', 'transition', 'deletion', 'components',
                                    'component_cross', 'component_output', 'original_score_checks', 'identity_checks')}
    for q in protocol['requests']:
        m = meta(q); N, K = q['N'], q['K']; rid = q['request_id']
        aa = {s: arrays(q['paths'][s]) for s in ('RAW', 'D0', 'RESTORED', 'POST', 'CANONICAL')}
        raw, post, canonical = aa['RAW'], aa['POST'], aa['CANONICAL']
        seam = np.asarray(raw['raw_seam_slot_mask'], bool)
        assert len(seam) == N and not seam[:K].any()
        roles = np.where(seam, 'seam', 'bulk'); roles[:K] = 'C'
        ids = np.asarray(post['source_face_ids'], np.int64)
        corners = np.asarray(post['source_corner_ids'], np.int64)
        keep = np.asarray(post['source_face_kept'], bool)
        assert np.array_equal(ids, np.flatnonzero(keep))
        assert len(set(ids.tolist())) == len(ids) and np.array_equal(ids[:K], np.arange(K))
        assert np.array_equal(post['raw_slot_ids'], ids)
        assert np.all(np.sort(corners, axis=1) == np.arange(3))
        assert np.array_equal(post['vertices'][post['faces']], post['output'])
        stage_tri = {'RAW': raw['raw'], 'D0': aa['D0']['official_unconstrained_proposal'],
                     'RESTORED': aa['RESTORED']['output'], 'POST': post['output']}
        bulk_restore_identical = stage_tri['D0'][roles == 'bulk'].tobytes() == stage_tri['RESTORED'][roles == 'bulk'].tobytes()
        assert bulk_restore_identical, 'Bulk coordinates changed in RESTORE'
        moved = stage_tri['RESTORED'].astype(np.float64) + canonical['corner_displacement_vector']
        stage_tri['GEO_PREDELETE'] = moved
        # source_corner_ids already carries source-C recovery/free winding permutations.
        mapped = moved[ids[:, None], corners]
        err = float(np.max(np.abs(mapped - post['output'].astype(np.float64))))
        canonical_error = float(np.max(np.abs(moved[ids] - canonical['output'].astype(np.float64))))
        assert err <= 1e-7 and canonical_error <= 1e-7, (rid, err, canonical_error)
        assert np.array_equal(post['retained_seam_mask'], seam[ids])
        geo = read(q['paths']['geometry'])
        reasons = post['source_face_reason'].astype(str)
        deleted_ids = np.flatnonzero(~keep)
        assert len(deleted_ids) == geo['stats']['deleted_free_faces']
        assert dict(collections.Counter(reasons[~keep])) == geo['stats']['deleted_by_reason']
        assert all(reasons[int(r['source_face_id'])] == r['reason'] and keep[int(r['source_face_id'])] == r['kept'] for r in geo['face_records'])
        tables['identity_checks'].append(dict(**m, status='PASS', retained_faces=len(ids), deleted_faces=len(deleted_ids),
            source_corner_mapping_max_error=err, canonical_mapping_max_error=canonical_error,
            source_slot_unique=True, corner_permutations_valid=True, retained_seam_identity_valid=True,
            restore_bulk_bitwise_unchanged=bulk_restore_identical,
            GEO_PREDELETE_definition='RESTORED plus saved canonical corner_displacement_vector; algebra only'))
        stats = {s: shape(t, L) for s, t in stage_tri.items()}
        for s in ('RAW', 'D0', 'RESTORED', 'GEO_PREDELETE'):
            assert len(stage_tri[s]) == N
        post_index = {int(s): i for i, s in enumerate(ids)}
        component_membership = {}
        post_area = stats['POST']['area_model2']
        total_free_area = float(post_area[K:].sum())
        assert total_free_area > 0
        for kind in ('edge', 'vertex'):
            groups = dsu_components(post['faces'], kind)
            face_group = np.empty(len(ids), np.int64)
            detached = np.zeros(len(ids), bool)
            n_small = 0; detached_area = 0.; detached_faces = 0
            for ci, ff in enumerate(groups):
                face_group[ff] = ci
                fids = ids[ff]
                has_C = bool(np.any(fids < K))
                area = float(post_area[ff].sum())
                small = not has_C and len(ff) <= 3 and area <= .001*total_free_area
                detached[ff] = small
                n_small += int(small)
                detached_area += area if small else 0
                detached_faces += len(ff) if small else 0
                tables['components'].append(dict(**m, connectivity=kind, component_id=ci,
                    face_count=len(ff), C_faces=int((fids < K).sum()), seam_faces=int((roles[fids] == 'seam').sum()),
                    bulk_faces=int((roles[fids] == 'bulk').sum()), has_C=has_C, area_model2=area,
                    area_fraction_total_free=area/total_free_area, small_free_only_detached=small,
                    post_face_ids=json.dumps(ff), source_raw_slot_ids=json.dumps(fids.tolist()),
                    graph_definition='All actual indexed faces including C; any shared '+kind+'; not legality filtered'))
            component_membership[kind] = (face_group, detached)
            tables['component_output'].append(dict(**m, connectivity=kind, components=len(groups),
                small_detached_components=n_small, small_detached_free_faces=detached_faces,
                small_detached_free_area=detached_area, small_detached_area_fraction=detached_area/total_free_area,
                total_free_area=total_free_area))
            for role in ('free', 'seam', 'bulk'):
                mask = ids >= K if role == 'free' else roles[ids] == role
                for tiny_name, threshold in TINY:
                    for sa in (False, True):
                        for ti in (False, True):
                            for de in (False, True):
                                selected = mask & (stats['POST']['small_angle'] == sa) & (stats['POST'][tiny_name] == ti) & (detached == de)
                                tables['component_cross'].append(dict(**m, connectivity=kind, role=role,
                                    tiny_threshold=threshold, primary_tiny=threshold == 1e-5,
                                    small_angle=sa, tiny_area=ti, small_free_only_detached=de,
                                    count=int(selected.sum()), area_model2=float(post_area[selected].sum())))
        for s, sh in stats.items():
            ss = ids if s == 'POST' else np.arange(N)
            for i, slot in enumerate(ss):
                r = dict(**m, stage=s, stage_domain='PRIMARY' if s != 'GEO_PREDELETE' else 'ALGEBRAIC_DIAGNOSTIC',
                    stage_face_index=i, source_raw_slot=int(slot), role=str(roles[slot]), final_kept=bool(keep[slot]),
                    post_face_index=post_index.get(int(slot)), GEO_reason=str(reasons[slot]))
                for key, values in sh.items():
                    val = values[i].item()
                    r[key] = val if not isinstance(val, float) or math.isfinite(val) else None
                r['q_status'] = 'FINITE' if math.isfinite(float(sh['q'][i])) else 'UNDEFINED_OR_INFINITE_DEGENERATE'
                if s == 'POST':
                    r['source_corner_ids'] = json.dumps(corners[i].tolist())
                    for kind, (component_id, detached) in component_membership.items():
                        r[kind+'_component_id'] = int(component_id[i]); r[kind+'_small_detached'] = bool(detached[i])
                tables['per_face'].append(r)
            for role in ROLES:
                select = ss >= K if role == 'free' else roles[ss] == role
                count = int(select.sum()); a = sh['area_model2'][select]
                r = dict(**m, stage=s, stage_domain='PRIMARY' if s != 'GEO_PREDELETE' else 'ALGEBRAIC_DIAGNOSTIC', role=role,
                    face_count=count, area_model2=float(a.sum()), zero_area_count=int(sh['exact_zero_area'][select].sum()))
                for flag in FLAGS:
                    n = int(sh[flag][select].sum())
                    r[flag+'_count'] = n; r[flag+'_fraction'] = n/count if count else None
                    r[flag+'_area_model2'] = float(sh['area_model2'][select & sh[flag]].sum())
                for key in ('area_over_L2', 'minimum_angle_degrees', 'longest_edge_over_L'):
                    vals = sh[key][select]
                    for stat, fn in [('mean', np.mean), ('median', np.median), ('min', np.min), ('max', np.max)]:
                        r[key+'_'+stat] = float(fn(vals)) if count else None
                if s in ('RAW', 'POST') and role == 'free':
                    oo = old[rid, s]
                    for key in ('FF_illegal_pair_count', 'FF_illegal_affected_free_fraction', 'CF_illegal_affected_free_fraction', 'ConflictCov_C'):
                        r['original_'+key] = float(oo[key])
                    diff = r['small_angle_fraction'] - float(oo['free_minimum_angle_lt5_fraction'])
                    assert abs(diff) < 1e-12 and count == int(float(oo['actual_free_faces']))
                    tables['original_score_checks'].append(dict(**m, stage=s, status='PASS', face_count=count,
                        original_count=int(float(oo['actual_free_faces'])), small_angle_fraction=r['small_angle_fraction'],
                        original_small_angle_fraction=float(oo['free_minimum_angle_lt5_fraction']), difference=diff))
                tables['per_output'].append(r)
                chosen = select if s == 'POST' else select & keep
                retained_count = int(chosen.sum())
                matched = dict(**m, stage=s, role=role, cohort='FINAL_POST_RETAINED_RAW_SLOTS',
                    face_count=retained_count, area_model2=float(sh['area_model2'][chosen].sum()),
                    zero_area_count=int(sh['exact_zero_area'][chosen].sum()))
                for flag in FLAGS:
                    n = int(sh[flag][chosen].sum())
                    matched[flag+'_count'] = n
                    matched[flag+'_fraction'] = n/retained_count if retained_count else None
                    matched[flag+'_area_model2'] = float(sh['area_model2'][chosen & sh[flag]].sum())
                tables['retained_per_output'].append(matched)
        # GEO overall is separated from retained-shape changes and deletions.
        links = [('RAW', 'D0', 'D0'), ('D0', 'RESTORED', 'RESTORE'), ('RESTORED', 'POST', 'GEO_TOTAL'),
                 ('RESTORED', 'GEO_PREDELETE', 'GEO_MOVE_DIAGNOSTIC'), ('GEO_PREDELETE', 'POST', 'GEO_DELETE_DIAGNOSTIC')]
        for a, b, transition in links:
            all_a = np.arange(N)
            kept_ids = ids if b == 'POST' else all_a
            removed = ~keep if b == 'POST' else np.zeros(N, bool)
            for role in ROLES:
                role_mask = all_a >= K if role == 'free' else roles == role
                idx = kept_ids[role_mask[kept_ids]]
                for flag in FLAGS:
                    av = stats[a][flag]
                    bv = np.zeros(N, bool)
                    bv[kept_ids] = stats[b][flag]
                    rec = dict(**m, transition=transition, from_stage=a, to_stage=b, role=role, feature=flag,
                        before_faces=int(role_mask.sum()), after_faces=len(idx),
                        before_positive=int(av[role_mask].sum()), after_positive=int(bv[idx].sum()),
                        retained_persistent=int((av[idx] & bv[idx]).sum()),
                        retained_new=int((~av[idx] & bv[idx]).sum()), retained_resolved=int((av[idx] & ~bv[idx]).sum()),
                        deleted_positive=int((role_mask & removed & av).sum()),
                        deleted_negative=int((role_mask & removed & ~av).sum()))
                    assert rec['after_positive'] == rec['before_positive']+rec['retained_new']-rec['retained_resolved']-rec['deleted_positive']
                    tables['transition'].append(rec)
        for slot in deleted_ids:
            r = dict(**m, source_raw_slot=int(slot), role=str(roles[slot]), GEO_reason=str(reasons[slot]),
                duplicate_of_source_face=int(post['duplicate_of_source_face'][slot]),
                max_GEO_corner_motion=float(np.linalg.norm(canonical['corner_displacement_vector'][slot], axis=1).max()))
            for s in ('RAW', 'D0', 'RESTORED', 'GEO_PREDELETE'):
                for field in ('area_model2', 'area_over_L2', 'minimum_angle_degrees', 'longest_edge_model', *FLAGS):
                    r[s+'_'+field] = stats[s][field][slot].item()
            tables['deletion'].append(r)
        print(json.dumps(dict(request_id=rid, status='PASS', retained=len(ids), deleted=len(deleted_ids))), flush=True)
    for name, rows in tables.items():
        csvwrite(DEST/(name+'.csv'), rows)
    fields = ('face_count', 'area_model2', 'zero_area_count', *(f+'_'+suffix for f in FLAGS for suffix in ('count', 'fraction', 'area_model2')))
    summary = summarize(tables['per_output'], ('split', 'stage', 'role'), fields,
                        sums=[f for f in fields if not f.endswith('_fraction')])
    csvwrite(DEST/'split_summary.csv', summary)
    retained_summary = summarize(tables['retained_per_output'], ('split', 'stage', 'role'), fields,
                                sums=[f for f in fields if not f.endswith('_fraction')])
    csvwrite(DEST/'retained_split_summary.csv', retained_summary)
    tfields = ('before_faces', 'after_faces', 'before_positive', 'after_positive', 'retained_persistent', 'retained_new', 'retained_resolved', 'deleted_positive', 'deleted_negative')
    csvwrite(DEST/'transition_split_summary.csv', summarize(tables['transition'], ('split', 'transition', 'role', 'feature'), tfields, sums=tfields))
    cfields = ('components', 'small_detached_components', 'small_detached_free_faces', 'small_detached_free_area', 'small_detached_area_fraction')
    component_summary = summarize(tables['component_output'], ('split', 'connectivity'), cfields, sums=cfields[:-1])
    csvwrite(DEST/'component_split_summary.csv', component_summary)
    crosses = collections.defaultdict(lambda: [0, 0.])
    cross_fields = ('split', 'connectivity', 'role', 'tiny_threshold', 'primary_tiny', 'small_angle', 'tiny_area', 'small_free_only_detached')
    for r in tables['component_cross']:
        key = tuple(r[k] for k in cross_fields)
        crosses[key][0] += r['count']; crosses[key][1] += r['area_model2']
    cross_summary = [dict(zip(cross_fields, k), count_total=v[0], area_model2_total=v[1]) for k, v in sorted(crosses.items())]
    csvwrite(DEST/'component_cross_split_summary.csv', cross_summary)
    deletion_summary = collections.defaultdict(list)
    for r in tables['deletion']:
        deletion_summary[r['split'], r['role'], r['GEO_reason']].append(r)
    ds = []
    for (split, role, reason), rr in sorted(deletion_summary.items()):
        row = dict(split=split, role=role, GEO_reason=reason, faces=len(rr))
        for s in ('RAW', 'D0', 'RESTORED', 'GEO_PREDELETE'):
            row[s+'_area_total'] = sum(r[s+'_area_model2'] for r in rr)
            for flag in FLAGS:
                row[s+'_'+flag+'_count'] = sum(r[s+'_'+flag] for r in rr)
        ds.append(row)
    csvwrite(DEST/'deletion_split_summary.csv', ds)
    write(DEST/'summary.json', dict(version=VERSION, split_stage_role=summary,
        matched_retained_split_stage_role=retained_summary, component_summary=component_summary,
        component_small_angle_tiny_cross=cross_summary, deleted_by_reason=ds,
        original_score_checks='ALL_PASS', restore_bulk_bitwise_unchanged='28/28_PASS'))
    after = {p: sha(p) for p in source}
    write(DEST/'source_hashes_after.json', after)
    assert before == after
    write(DEST/'resources_after.json', resources())
    result = dict(status='COMPLETE', version=VERSION, requests=28, primary_request_stages=112,
        additional_algebraic_GEO_PREDELETE_stages=28, seconds=time.perf_counter()-tick,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_hashes=source, source_before_after='UNCHANGED', worker_count=1, threads=1,
        model_calls=0, GPU_calls=0, GEO_calls=0, new_queries=0, FF_I_scans=0,
        row_counts={name: len(rr) for name, rr in tables.items()},
        definitions=dict(precision='FP64; normalized by frozen L before area/angles, matching old angle convention',
            tiny='Strict area/L^2 < threshold; primary1e-5 and sensitivity1e-6/1e-4',
            small_detached='POST indexed graph includes C; component contains no C, <=3 faces, <=0.001 total POST free area',
            graph='Edge adjacency and vertex adjacency separate; unrestricted by winding or illegal-face detector; descriptive, not a defect verdict',
            GEO_PREDELETE='RESTORED + saved canonical corner_displacement_vector, checked against every surviving POST source slot/corner',
            deletion='Deleted positive flags are not counted as retained geometric resolution',
            FF_I='Original RAW/POST numeric CSV reused only; D0/RESTORED/GEO_PREDELETE FF/I not computed'),
        output_hashes={str(p): sha(p) for p in [*sorted(DEST.glob('*.csv')), DEST/'summary.json']})
    write(completion, result)
    return {k: v for k, v in result.items() if k not in ('source_hashes', 'output_hashes', 'definitions')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    if not args.run:
        parser.error('Use --run for the fixed saved-geometry CPU analysis')
    try:
        print(json.dumps(run()), flush=True)
    except Exception:
        DEST.mkdir(parents=True, exist_ok=True)
        write(DEST/('failure_'+str(time.time_ns())+'.json'), dict(status='FAILED', traceback=traceback.format_exc(), model_calls=0, GEO_calls=0))
        raise
