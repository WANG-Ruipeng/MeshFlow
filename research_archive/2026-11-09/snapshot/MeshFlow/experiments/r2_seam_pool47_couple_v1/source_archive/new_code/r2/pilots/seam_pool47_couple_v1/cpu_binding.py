"""CPU-only fixed B_LONG_LAYOUT binding; no neural-network imports or calls."""
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
from r2.pilots.owner_layout_v1 import binding_adapter as A

B, C, E, Q, PE, np = A.B, A.C, A.E, A.Q, A.PE, A.np
BASE = C.ROOT / 'experiments/r2_seam_followup_v1/layout'
OUT = C.ROOT / 'experiments/r2_seam_pool47_couple_v1/cpu_binding'
ARMS = ('B_LONG_LAYOUT',)
DELTAS = (.015, .02)
CELLS = 56
VERSION = 'B_LONG_LAYOUT_actual_cluster_binding_v2'


def core():
    if C.sha(B.__file__) != A.LEGACY_SHA256:
        raise RuntimeError('Frozen legacy source changed')
    ns = dict(vars(B)); ns.update(BASE=BASE, OUT=OUT, ARMS=ARMS, DELTAS=DELTAS, VERSION=VERSION)
    for name, obj in vars(B).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(B):
            fn = types.FunctionType(obj.__code__, ns, obj.__name__, obj.__defaults__, obj.__closure__)
            fn.__kwdefaults__ = obj.__kwdefaults__; ns[name] = fn
    ns['state_merge'] = A.state_merge
    return ns


CORE = core()


def freeze():
    p = OUT / 'registration.json'
    if p.exists():
        reg = C.read(p); CORE['verify'](reg['protected']); CORE['verify'](reg['code']); return reg
    manifest = BASE / 'registration.json'; requests = C.read(manifest)['requests']
    assert len(requests) == 28 and len({q['request_id'] for q in requests}) == 28
    protected = {str(manifest): C.sha(manifest), str(OUT/'PROTOCOL.md'): C.sha(OUT/'PROTOCOL.md')}
    source_labels = C.ROOT / 'experiments/r2_seam_abc_v1/B_DATA_REGISTRATION.json'
    protected[str(source_labels)] = C.sha(source_labels)
    dev = C.read(source_labels)['DEV_records']
    for d in dev:
        protected[d['labels_npz']] = C.sha(d['labels_npz'])
    for q in requests:
        for key in ('inference_npz', 'metadata_npz'):
            protected[q[key]] = C.sha(q[key])
        desc = q['query_descriptor']; assert C.sha(desc['path']) == desc['sha256']
        protected[desc['path']] = desc['sha256']
        folder = BASE / 'evaluation/B_LONG_LAYOUT' / q['request_id']
        for rel in ('raw/raw.npz', 'post/post.npz', 'post/pre_geo.npz', 'post/receipt.json'):
            asset = folder / rel; protected[str(asset)] = C.sha(asset)
        A.validate_arrays(C.arrays(folder / 'post/post.npz'), C.arrays(folder / 'post/pre_geo.npz')['output'], q)
        for name in ('score.json', 'pairs.json'):
            asset = BASE / 'evaluation/scores/B_LONG_LAYOUT' / q['request_id'] / 'POST' / name
            protected[str(asset)] = C.sha(asset)
    baseline_csv = BASE / 'evaluation/per_output.csv'; protected[str(baseline_csv)] = C.sha(baseline_csv)
    code = {str(Path(__file__)): C.sha(__file__), str(Path(B.__file__)): C.sha(B.__file__),
            str(Path(A.__file__)): C.sha(A.__file__), **E.source_hashes(), **Q.definition_hashes()}
    diagnostic = Path(__file__).with_name('dev_diagnostic.py'); code[str(diagnostic)] = C.sha(diagnostic)
    for name in ('geo_c', 'winding_solver'):
        asset = Path(PE.cpu_module(name).__file__); code[str(asset)] = C.sha(asset)
    reg = dict(status='FROZEN', version=VERSION, utc=C.utc(), arms=list(ARMS), deltas=list(DELTAS),
        cells=CELLS, requests=28, requests_manifest=requests, DEV_label_records=dev, protected=protected, code=code, L=C.L,
        authorization='Parent explicitly authorized saved B_LONG_LAYOUT 28 requests x two thresholds, no new model sampling',
        algorithm='Original actual GEO_C whole clusters; TIP_CLUSTER and LATERAL_EDGE one deterministic queue; every original PRE_GEO member within delta; C immovable; current-root C guard corrected',
        preserved='All prior shape/CF/legal-CF/orientation/duplicate/nonmanifold guards, exact cleanup and original 124-field score; FF costs are measured not vetoed',
        diagnosis='All DEV retained seam tips and both lateral edges; finite segment/endpoint distances, axis and threshold breakdown; source alias labels only in separate diagnostic',
        interpretation='Feasibility/cost test, not physical upper bound; a point alias is not a shared-edge connection; collision association is not causal obstruction',
        model_calls=0, GPU_dispatches=0, new_noise=0, new_queries=0)
    C.write(p, reg)
    for path, h in code.items():
        dest = OUT / 'actual_code' / (h[:12] + '__' + Path(path).name)
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(C.resolve(path).read_bytes())
    CORE['event']('REGISTERED', cells=CELLS, registration_sha256=C.sha(p)); return reg


def worker(job):
    arm, q = job
    assert arm in ARMS
    folder = BASE / 'evaluation' / arm / q['request_id'] / 'post'
    A.validate_arrays(C.arrays(folder/'post.npz'), C.arrays(folder/'pre_geo.npz')['output'], q)
    return CORE['worker'](job)


def run(workers=2):
    reg = freeze(); done_path = OUT / 'completion.json'
    if done_path.exists():
        done = C.read(done_path); CORE['verify'](done['files']); return done
    C.resources(OUT / ('resources_before_' + str(time.time_ns()) + '.json'))
    jobs = [(ARMS[0], q) for q in reg['requests_manifest']]
    tick = time.perf_counter(); CORE['event']('CPU_STARTED', authorized_cells=CELLS)
    first = worker(jobs[0])
    snap = C.resources(OUT / ('resources_measured_' + str(time.time_ns()) + '.json'))
    mem = {x.split(':')[0]: int(x.split()[1])*1024 for x in (snap.get('memory') or '').splitlines()
           if len(x.split()) > 1 and x.split()[1].isdigit()}
    peak = max(r['peak_RSS_KiB'] for r in first) * 1024
    n = 2 if workers == 2 and (os.cpu_count() or 1) >= 4 and mem.get('MemAvailable', 0) > max(4 << 30, 3*peak) and (snap['loadavg'] or [999])[0] < (os.cpu_count() or 1)-2 else 1
    C.write(OUT/('dispatch_'+str(time.time_ns())+'.json'), dict(workers=n, threads_per_worker=1,
        first_actual_request_peak_RSS_bytes=peak, resources=snap, reason='Measure authorized first two cells, then cap independent CPU workers by load/RAM'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context('spawn')) as pool:
        todo = iter(jobs[1:]); active = {}
        for _ in range(n):
            job = next(todo, None)
            if job: active[pool.submit(worker, job)] = job
        while active:
            ready, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                future.result(); active.pop(future)
                job = next(todo, None)
                if job: active[pool.submit(worker, job)] = job
    result = CORE['aggregate'](); CORE['verify'](reg['protected']); CORE['verify'](reg['code'])
    assert len(result['rows']) == CELLS
    final = dict(status='COMPLETE', cells=CELLS, model_calls=0, GPU_calls=0, new_queries=0,
        numeric_metric_count=result['numeric_metric_count'], seconds=time.perf_counter()-tick, workers=n,
        receipts=result['receipts'], protected_verified=len(reg['protected']), source_hash=C.sha(__file__),
        files={str(p):C.sha(p) for p in OUT.glob('*.csv')}, visual_review='PENDING_PARENT')
    C.resources(OUT/('resources_finished_'+str(time.time_ns())+'.json')); C.write(done_path, final)
    CORE['event']('CPU_COMPLETE', cells=CELLS, seconds=final['seconds']); return final


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=('freeze','run','diagnose','status'), default='status')
    p.add_argument('--workers', type=int, choices=(1,2), default=2); args = p.parse_args()
    if args.stage == 'freeze':
        r = freeze(); r = dict(status=r['status'], cells=r['cells'], protected=len(r['protected']))
    elif args.stage == 'run':
        r = run(args.workers); r = {k:v for k,v in r.items() if k not in ('receipts','files')}
    elif args.stage == 'diagnose':
        from r2.pilots.seam_pool47_couple_v1.dev_diagnostic import run as diagnostic
        r = diagnostic()
    else:
        r = dict(completed=len(list((OUT/'cells').glob('*/*/complete.json'))), failed=len(list((OUT/'cells').glob('*/*/failure.json'))))
    print(json.dumps(C.clean(r)), flush=True)


if __name__ == '__main__': main()
