"""Atomic checkpoints, full branch state, resource observations and GPU lock."""
from native_t1 import runtime as _runtime
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from native_t1.artifacts import atomic_json, file_sha256, state_sha256
from native_t1.portable_checkpoint import (OFFICIAL_FILE_SHA256,
    OFFICIAL_BACKBONE_SHA256, config_sha256)
from native_t1.checkpoint import DEFAULT_CONFIG
from .model_mvp import SCHEMA, json_hash, load_mvp_checkpoint
from .train_contract import PROTOCOL


def resource_snapshot():
    """nvidia-smi only; does not initialize the calling process's CUDA context."""
    def command(args):
        value = subprocess.run(args, text=True, capture_output=True, timeout=15)
        return dict(returncode=value.returncode, stdout=value.stdout.strip(), stderr=value.stderr.strip())
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, val = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapFree'):
            memory[key + '_kib'] = int(val.split()[0])
    return dict(time_unix=time.time(), pid=os.getpid(), cpu_cores=os.cpu_count(),
        cpu_affinity_cores=len(os.sched_getaffinity(0)), load_average=list(os.getloadavg()),
        memory=memory,
        gpu=command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']),
        gpu_processes=command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits']),
        cpu_processes=command(['ps', '-eo', 'pid,pcpu,pmem,rss,nlwp,comm', '--sort=-pcpu']))


@contextlib.contextmanager
def gpu_lock(root):
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized before obtaining shared GPU lock')
    tick = time.perf_counter()
    with open(PROTOCOL['gpu_lock'], 'a+') as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                atomic_json(root / 'training/status.json', dict(status='WAITING_GPU_LOCK',
                    actual_optimizer_updates=0, lock=PROTOCOL['gpu_lock'], wait_seconds=time.perf_counter()-tick))
                time.sleep(5)
        try:
            yield time.perf_counter() - tick
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def capture_rng():
    s = np.random.get_state()
    return dict(python=random.getstate(), numpy=[s[0], s[1].tolist(), int(s[2]), int(s[3]), float(s[4])],
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    s = state['numpy']; np.random.set_state((s[0], np.asarray(s[1], np.uint32), s[2], s[3], s[4]))
    torch.random.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def tree_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: tree_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_cpu(v) for v in value)
    return copy.deepcopy(value)


def tree_hash(value):
    """Hash optimizer/RNG content, including scalar tensor state and layout."""
    digest = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            digest.update(str((str(t.dtype), tuple(t.shape))).encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=lambda k: str(k)):
                digest.update(repr(key).encode()); visit(v[key])
        elif isinstance(v, (list, tuple)):
            digest.update(type(v).__name__.encode())
            for x in v: visit(x)
        else:
            digest.update(repr(v).encode())
    visit(value)
    return digest.hexdigest()


def save_checkpoint(root, model, optimizer, stream, protocol, *, variant, step,
                    actual_updates, phase_updates, fork_sha=None):
    tick = time.perf_counter()
    name = f'{variant}_step{step}'
    path = root / 'checkpoints' / (name + '.pt')
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('Refusing to overwrite checkpoint ' + str(path))
    state = tree_cpu(model.state_dict())
    opt = tree_cpu(optimizer.state_dict())
    rng = capture_rng()
    meta = dict(variant=variant, logical_step=step, context_encoder=None if variant == 'T1' else 'geo',
        model_state_sha256=state_sha256(model), optimizer_state_sha256=tree_hash(opt),
        rng_sha256=tree_hash(rng), stream_state_sha256=json_hash(stream.state_dict()),
        protocol_sha256=json_hash(protocol), config_sha256=config_sha256(DEFAULT_CONFIG),
        official_file_sha256=OFFICIAL_FILE_SHA256, official_backbone_sha256=OFFICIAL_BACKBONE_SHA256,
        actual_updates_at_save=actual_updates, executed_phase_updates=dict(phase_updates),
        fork_checkpoint_sha256=fork_sha, official_pretraining_steps_in_logical_count=False,
        optimizer_reset_at_phase_start=variant in ('T1', 'Geo_shared'),
        loss_recipe='edge5' if variant == 'Full' else 'fm')
    payload = dict(schema=SCHEMA, metadata=meta, protocol=protocol, model=state, optimizer=opt,
                   rng=rng, stream_state=stream.state_dict(), update_in_progress=False)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    with temporary.open('rb') as handle: os.fsync(handle.fileno())
    temporary.replace(path)
    digest = file_sha256(path)
    record = dict(path=str(path), checkpoint_sha256=digest, metadata=meta)
    atomic_json(path.with_suffix('.json'), record)
    del payload, state, opt
    loaded, actual, hashes = load_mvp_checkpoint(path, 'cpu')
    if hashes['model_state_sha256'] != meta['model_state_sha256']:
        raise RuntimeError('Strict checkpoint reload mismatch')
    del loaded
    record.update(strict_reload='PASS', save_and_verify_seconds=time.perf_counter()-tick)
    return record


def restore_branch(path, stream, *, device='cuda'):
    model, meta, hashes = load_mvp_checkpoint(path, device)
    if meta['variant'] != 'Geo_shared' or meta['logical_step'] != 500:
        raise ValueError('Branch requires exact Geo_shared_step500')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    from native_t1.training import make_optimizer
    model.set_trainable(True)
    optimizer = make_optimizer(model)
    optimizer.load_state_dict(payload['optimizer'])
    if tree_hash(optimizer.state_dict()) != meta['optimizer_state_sha256']:
        raise ValueError('Restored AdamW momentum/state differs')
    stream.load_state_dict(payload['stream_state'])
    if json_hash(stream.state_dict()) != meta['stream_state_sha256']:
        raise ValueError('Restored data cursor/RNG differs')
    restore_rng(payload['rng'])
    if tree_hash(capture_rng()) != meta['rng_sha256']:
        raise ValueError('Restored model RNG differs')
    return model, optimizer, hashes['checkpoint_sha256']
