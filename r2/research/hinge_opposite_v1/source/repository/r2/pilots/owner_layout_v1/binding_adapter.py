"""Prepared CPU adapter: new C-only owner layouts, actual GEO_C clusters.

This file does not train, sample, or invoke D0. ``run`` and ``freeze`` require
an explicit completed head/sampling gate. Importing it never dispatches work.
The historic binding module and scorers remain byte-for-byte unchanged.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import time
import types

from r2.pilots.seam_followup_v1 import binding as B

C, E, Q, PE, np = B.C, B.E, B.Q, B.PE, B.np
ROOT = C.ROOT / 'experiments/r2_owner_layout_v1'
OUT = ROOT / 'binding'
ARMS = ('LONG_OWNER', 'B_POOL47_OWNER')
DELTA = .015
CELLS = 56
VERSION = 'owner_layout_actual_cluster_binding_v2_current_root_guard'
LEGACY_SHA256 = 'c184cae7e66fe7ffa65f8f2d027a5c208b8121220fc2dc64e9a00f4fc87f1d60'
DEFAULT_GATE = ROOT / 'binding_dispatch_gate.json'


def state_merge(vertices, roots, original_members, preflat, fixed, pairs):
    """Legacy whole-cluster union with the missing CURRENT free-tip guard.

Pair order is (seam-side original cluster, bulk-side original cluster).
When the seam side is originally free, BOTH current groups must stay free.
Check again after each pair in a lateral-edge candidate, not just at entry.
"""
    v = vertices.copy(); r = roots.copy(); changed = set(); operations = []
    for aa, bb in pairs:
        a, b = int(r[int(aa)]), int(r[int(bb)])
        if int(aa) not in fixed and (a in fixed or b in fixed):
            return None, 'FREE_TIP_CURRENT_ROOT_IS_C'
        if a == b:
            continue
        aset = set(np.flatnonzero(r == a).tolist())
        bset = set(np.flatnonzero(r == b).tolist())
        af, bf = bool(aset & fixed), bool(bset & fixed)
        if af and bf:
            return None, 'DISTINCT_C_ANCHOR_CONFLICT'
        target = v[a].copy() if af else v[b].copy() if bf else ((v[a].astype(float) + v[b].astype(float)) / 2).astype(np.float32)
        root = a if af else b if bf else min(a, b)
        members = aset | bset
        original_corners = np.concatenate([original_members[x] for x in sorted(members)])
        maximum = float(np.linalg.norm(preflat[original_corners] - target.astype(float), axis=1).max())
        operations.append(dict(original_cluster_ids=sorted(members), new_root=root,
            target=target.tolist(), maximum_PRE_GEO_displacement=maximum,
            original_corner_count=len(original_corners), fixed_C=af or bf))
        for x in members:
            v[x] = target
        r[list(members)] = root
        changed.update(members)
    if not changed:
        return None, 'ALREADY_SHARED_NOOP'
    return dict(vertices=v, roots=r, changed_clusters=changed, operations=operations,
                maxdisp=max(op['maximum_PRE_GEO_displacement'] for op in operations)), None


def _private_core():
    """Rebind legacy functions into a private namespace; never monkeypatch B.

worker/aggregate/event use the new paths and two labels. candidate_inventory
and decode use the tightened state_merge. score_output is the original scorer;
its seam_to_bulk(..., applicable=True) follows raw_seam_slot_mask rather than
an old label allow-list. No function mutates the imported legacy namespace.
"""
    if C.sha(B.__file__) != LEGACY_SHA256:
        raise RuntimeError('Legacy binding source changed; adapter requires reviewed frozen version')
    ns = dict(vars(B))
    ns.update(OUT=OUT, BASE=ROOT, ARMS=ARMS, DELTAS=(DELTA,), VERSION=VERSION)
    for name, obj in vars(B).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(B):
            clone = types.FunctionType(obj.__code__, ns, obj.__name__, obj.__defaults__, obj.__closure__)
            clone.__kwdefaults__ = obj.__kwdefaults__
            ns[name] = clone
    ns['state_merge'] = state_merge
    return ns


CORE = _private_core()


def _gate(path):
    """A parent-owned gate is required, including actual complete new sampling."""
    p = C.resolve(path)
    if not p.is_file():
        raise RuntimeError('Prepared only: missing explicit binding dispatch gate: ' + str(p))
    gate = C.read(p)
    if gate.get('status') != 'PASS' or gate.get('head_gate_passed') is not True or gate.get('sampling_complete') is not True:
        raise RuntimeError('Head gate/sampling have not passed; no binding dispatch')
    if tuple(gate.get('labels', ())) != ARMS or gate.get('requests_per_label') != 28 or gate.get('binding_cells') != CELLS:
        raise RuntimeError('Gate does not match the fixed LONG/B 28-request, 56-cell matrix')
    return p, gate


def validate_arrays(base, pre, q):
    """Fail before binding if the new postprocessor lost original cluster IDs."""
    required = ('vertices', 'faces', 'output', 'known_faces', 'input_faces', 'source_face_ids',
        'source_corner_ids', 'corner_vertex_ids', 'raw_seam_slot_mask', 'output_anchor_ids',
        'raw_anchor_ids', 'source_face_kept', 'source_face_reason', 'duplicate_of_source_face')
    absent = sorted(set(required) - set(base))
    if absent:
        raise ValueError('Missing actual GEO_C identity fields: ' + ', '.join(absent))
    n, k = int(q['N']), int(q['K'])
    if pre.shape != (n, 3, 3) or base['corner_vertex_ids'].shape != (n, 3):
        raise ValueError('PRE_GEO and full raw-corner cluster map must retain N x 3 identities')
    if len(base['raw_seam_slot_mask']) != n or int(base['known_faces']) != k or int(base['input_faces']) != n:
        raise ValueError('Seam layout/N/K identity mismatch')
    if base['raw_anchor_ids'].shape != (n, 3):
        raise ValueError('raw_anchor_ids must preserve canonical raw corner identity')
    if not np.array_equal(base['vertices'][base['faces']], base['output']):
        raise ValueError('POST soup disagrees with actual indexed mesh')
    if not np.array_equal(base['source_face_ids'][:k], np.arange(k)):
        raise ValueError('C must be the first K retained source faces')
    if np.any(base['raw_seam_slot_mask'][:k]):
        raise ValueError('C faces cannot be marked as free seam slots')


def freeze(gate_path=DEFAULT_GATE):
    gate_path, gate = _gate(gate_path)
    path = OUT / 'registration.json'
    if path.exists():
        reg = C.read(path)
        CORE['verify'](reg['protected']); CORE['verify'](reg['code'])
        return reg
    manifest_path = ROOT / 'eval_manifest.json'
    manifest = C.read(manifest_path); requests = manifest['requests']
    if len(requests) != 28 or len({q['request_id'] for q in requests}) != 28:
        raise ValueError('Exactly 28 unique frozen requests required')
    protected = {str(manifest_path): C.sha(manifest_path), str(gate_path): C.sha(gate_path)}
    for q in requests:
        for key in ('inference_npz', 'metadata_npz'):
            protected[q[key]] = C.sha(q[key])
        descriptor = q['query_descriptor']
        if C.sha(descriptor['path']) != descriptor['sha256']:
            raise ValueError('Fixed query hash changed')
        protected[descriptor['path']] = descriptor['sha256']
        for arm in ARMS:
            folder = ROOT / 'evaluation' / arm / q['request_id']
            for rel in ('post/post.npz', 'post/pre_geo.npz', 'post/receipt.json'):
                p = folder / rel; protected[str(p)] = C.sha(p)
            validate_arrays(C.arrays(folder / 'post/post.npz'), C.arrays(folder / 'post/pre_geo.npz')['output'], q)
            for name in ('score.json', 'pairs.json'):
                p = ROOT / 'evaluation/scores' / arm / q['request_id'] / 'POST' / name
                protected[str(p)] = C.sha(p)
    baseline_csv = ROOT / 'evaluation/per_output.csv'
    protected[str(baseline_csv)] = C.sha(baseline_csv)
    code = {str(Path(__file__)): C.sha(__file__), str(Path(B.__file__)): C.sha(B.__file__),
            **E.source_hashes(), **Q.definition_hashes()}
    for name in ('geo_c', 'winding_solver'):
        p = Path(PE.cpu_module(name).__file__); code[str(p)] = C.sha(p)
    reg = dict(status='FROZEN', utc=C.utc(), version=VERSION, arms=list(ARMS), deltas=[DELTA], cells=CELLS,
        requests=28, requests_manifest=requests, L=C.L, protected=protected, code=code,
        parent_gate=gate, previous_definition='experiments/r2_seam_followup_v1/binding/registration.json',
        change='New C-only owner layouts; same deterministic combined queue and all geometric guards; current-root C protection tightened before every merge pair',
        unchanged='Actual original GEO_C clusters, both candidate kinds, 1.5%L maximum all-member PRE_GEO motion, source C fixed, original cleanup/orientation/124-field scorer',
        no_extra_arms='No A_COUPLE, 2%L, oracle layout, radius reclustering or new queries',
        success_definition='Only scorer-confirmed legal shared edges give surface/reachable-bulk credit; point aliases alone do not',
        model_calls=0, GPU_dispatches=0, new_noise=0, new_queries=0)
    C.write(path, reg)
    for p, h in code.items():
        dest = OUT / 'actual_code' / (h[:12] + '__' + Path(p).name)
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(C.resolve(p).read_bytes())
    CORE['event']('REGISTERED', cells=CELLS, registration_sha256=C.sha(path))
    return reg


def worker(job):
    arm, q = job
    if arm not in ARMS:
        raise ValueError('Unregistered arm')
    folder = ROOT / 'evaluation' / arm / q['request_id'] / 'post'
    validate_arrays(C.arrays(folder / 'post.npz'), C.arrays(folder / 'pre_geo.npz')['output'], q)
    return CORE['worker'](job)


def run(workers=2, gate_path=DEFAULT_GATE):
    reg = freeze(gate_path)
    CORE['verify'](reg['protected']); CORE['verify'](reg['code'])
    if (OUT / 'completion.json').exists():
        done = C.read(OUT / 'completion.json'); CORE['verify'](done['files']); return done
    C.resources(OUT / ('resources_before_' + str(time.time_ns()) + '.json'))
    jobs = [(arm, q) for q in reg['requests_manifest'] for arm in ARMS]
    tick = time.perf_counter(); CORE['event']('CPU_STARTED', authorized_cells=CELLS)
    first = worker(jobs[0])
    after = C.resources(OUT / ('resources_measured_' + str(time.time_ns()) + '.json'))
    mem = {x.split(':')[0]: int(x.split()[1]) * 1024 for x in (after.get('memory') or '').splitlines()
           if len(x.split()) > 1 and x.split()[1].isdigit()}
    peak = max(x['peak_RSS_KiB'] for x in first) * 1024
    n = 2 if workers == 2 and (os.cpu_count() or 1) >= 4 and mem.get('MemAvailable', 0) > max(4 << 30, 3 * peak) and (after['loadavg'] or [999])[0] < (os.cpu_count() or 1) - 2 else 1
    C.write(OUT / ('dispatch_' + str(time.time_ns()) + '.json'), dict(workers=n, threads_per_worker=1,
        first_actual_peak_RSS_bytes=peak, resources=after,
        reason='One actual authorized cell measured before choosing remaining independent CPU concurrency'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context('spawn')) as pool:
        todo = iter(jobs[1:]); active = {}
        for _ in range(n):
            job = next(todo, None)
            if job:
                active[pool.submit(worker, job)] = job
        while active:
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result(); active.pop(future)
                job = next(todo, None)
                if job:
                    active[pool.submit(worker, job)] = job
    result = CORE['aggregate']()
    CORE['verify'](reg['protected']); CORE['verify'](reg['code'])
    if len(result['rows']) != CELLS:
        raise AssertionError('Incomplete 56-cell matrix')
    final = dict(status='COMPLETE', cells=CELLS, model_calls=0, GPU_calls=0, new_queries=0,
        numeric_metric_count=result['numeric_metric_count'], seconds=time.perf_counter()-tick, workers=n,
        receipts=result['receipts'], protected_verified=len(reg['protected']), source_hash=C.sha(__file__),
        files={str(p): C.sha(p) for p in OUT.glob('*.csv')}, visual_review='PENDING_ACTUAL_IMAGE_REVIEW')
    C.resources(OUT / ('resources_finished_' + str(time.time_ns()) + '.json'))
    C.write(OUT / 'completion.json', final); CORE['event']('CPU_COMPLETE', cells=CELLS, seconds=final['seconds'])
    return final


def selfcheck():
    """Only small synthetic identity guards; no experiment cell or model call."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, .1, 0], [0, .3, 0]], np.float32)
    members = {i: np.asarray([i]) for i in range(4)}; r = np.arange(4, dtype=np.int64)
    t, why = state_merge(v, r, members, v.astype(float), {0, 1}, [(2, 3)])
    assert why is None and np.array_equal(t['vertices'][2], t['vertices'][3])
    assert abs(float(t['vertices'][2, 1]) - .2) < 1e-7
    assert state_merge(v, r, members, v.astype(float), {0, 1}, [(0, 1)])[1] == 'DISTINCT_C_ANCHOR_CONFLICT'
    anchored = r.copy(); anchored[2] = 0
    assert state_merge(v, anchored, members, v.astype(float), {0, 1}, [(2, 3)])[1] == 'FREE_TIP_CURRENT_ROOT_IS_C'
    assert state_merge(v, anchored, members, v.astype(float), {0, 1}, [(3, 2)])[1] == 'FREE_TIP_CURRENT_ROOT_IS_C'
    assert state_merge(v, r, members, v.astype(float), {0, 1}, [(0, 2), (2, 3)])[1] == 'FREE_TIP_CURRENT_ROOT_IS_C'
    assert B.OUT != OUT and B.ARMS == ('LONG', 'A_COUPLE') and B.state_merge is not state_merge
    assert CORE['decode'].__globals__['state_merge'] is state_merge
    return dict(status='PASS', tests=7, model_calls=0, experiment_cells=0,
                legacy_source_sha256=C.sha(B.__file__), adapter_source_sha256=C.sha(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('status', 'selfcheck', 'freeze', 'run', 'aggregate'), default='status')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--gate', default=str(DEFAULT_GATE))
    args = parser.parse_args()
    if args.stage == 'selfcheck':
        result = selfcheck()
    elif args.stage == 'freeze':
        reg = freeze(args.gate); result = dict(status=reg['status'], cells=reg['cells'])
    elif args.stage == 'run':
        result = run(args.workers, args.gate); result = {k: v for k, v in result.items() if k not in ('receipts', 'files')}
    elif args.stage == 'aggregate':
        _gate(args.gate); agg = CORE['aggregate'](); result = dict(cells=len(agg['rows']), numeric_metric_count=agg['numeric_metric_count'])
    else:
        result = dict(status='PREPARED_NO_AUTO_DISPATCH', arms=ARMS, cells=CELLS, delta_fraction=DELTA,
            gate_exists=C.resolve(args.gate).is_file(), completed=len(list((OUT / 'cells').glob('*/*/complete.json'))),
            failed=len(list((OUT / 'cells').glob('*/*/failure.json'))))
    print(json.dumps(C.clean(result), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
