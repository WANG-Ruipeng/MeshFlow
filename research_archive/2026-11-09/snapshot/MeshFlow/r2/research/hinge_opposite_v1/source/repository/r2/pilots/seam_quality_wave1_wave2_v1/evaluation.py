"""Shared evaluation adapters; importing this module performs no experiment."""
from __future__ import annotations
import importlib.util
import inspect
import types
import concurrent.futures
import multiprocessing
import os
import time
from pathlib import Path
from r2.pilots.seam_abc_v1 import eval_common as C
from .protocol import ROOT, OUT, WAVE1, WAVE2, ARMS

WAVES = {'wave1': WAVE1, 'wave2': WAVE2}


def wave_for(arm):
    for wave, arms in WAVES.items():
        if arm in arms:
            return wave
    raise ValueError('Unregistered arm: ' + str(arm))


def private_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def private_functions(module, **overrides):
    """Rebind only a module's own functions; never mutate a shared module."""
    namespace = dict(vars(module))
    namespace.update(overrides)
    for name, obj in vars(module).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(module):
            fn = types.FunctionType(obj.__code__, namespace, obj.__name__,
                                    obj.__defaults__, obj.__closure__)
            fn.__kwdefaults__ = obj.__kwdefaults__
            namespace[name] = fn
    return namespace


def requests():
    qq = C.read(OUT / 'protocol.json')['evaluation']['requests']
    if len(qq) != 28 or len({q['request_id'] for q in qq}) != 28:
        raise ValueError('Exactly the original 28 distinct requests are required')
    if sum(q['split'] == 'TRAIN' for q in qq) != 16:
        raise ValueError('Original TRAIN/DEV split differs')
    return qq


def source_paths():
    return [Path(__file__), Path(__file__).with_name('protocol.py')]


def dispatch_cpu(jobs, worker, folder, workers=2):
    """Measure first real authorized cell, then bound independent CPU workers."""
    tick = time.perf_counter()
    C.resources(folder / ('resources_before_' + str(time.time_ns()) + '.json'))
    first = worker(jobs[0])
    first_receipts = first if isinstance(first, list) else [first]
    snap = C.resources()
    mem = {line.split(':')[0]: int(line.split()[1])*1024
           for line in (snap.get('memory') or '').splitlines()
           if len(line.split()) > 1 and line.split()[1].isdigit()}
    peak = max(r['peak_RSS_KiB'] for r in first_receipts)*1024
    parallel = ((os.cpu_count() or 1) >= 4 and mem.get('MemAvailable', 0) > max(4 << 30, 3*peak)
                and (snap.get('loadavg') or [999])[0] < (os.cpu_count() or 1)-2)
    n = 2 if workers == 2 and parallel else 1
    C.write(folder / ('dispatch_' + str(time.time_ns()) + '.json'), dict(
        workers=n, threads_per_worker=1, first_actual_peak_RSS_bytes=peak,
        reason='First real registered cell; no added probe, resource-based CPU dispatch', resources=snap))
    receipts = list(first_receipts)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        todo = iter(jobs[1:]); active = {}
        for _ in range(n):
            job = next(todo, None)
            if job is not None:
                active[pool.submit(worker, job)] = job
        while active:
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            results = [future.result() for future in done]
            for future, result in zip(done, results):
                active.pop(future)
                receipts.extend(result if isinstance(result, list) else [result])
            for _ in done:
                job = next(todo, None)
                if job is not None:
                    active[pool.submit(worker, job)] = job
    C.resources(folder / ('resources_finished_' + str(time.time_ns()) + '.json'))
    return receipts, n, time.perf_counter()-tick
