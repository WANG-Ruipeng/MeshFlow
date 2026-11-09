"""Session C IO and CPU-only dependency/resource inspection."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'native_t1/runs/control_mvp48_v1'
EVALUATION = RUN / 'evaluation'
REGISTRATION = RUN / 'registration'
LOCK = '/tmp/wrp_meshflow_jedge5_diag_gpu0.lock'
MODELS = ('T1', 'Geo', 'Full')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def owned_path(path):
    path = Path(path).resolve()
    if EVALUATION.resolve() in path.parents or (RUN/'figures').resolve() in path.parents:
        return path
    if path.parent == REGISTRATION.resolve() and path.name.startswith('eval_'):
        return path
    raise ValueError(f'Not a Session C artifact path: {path}')


def write_json(path, data, *, exclusive=False):
    path = owned_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as f:
            f.write(encoded)
    else:
        temp = owned_path(path.with_name(path.name+'.tmp'))
        temp.write_text(encoded, encoding='utf-8'); os.replace(temp, path)


def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())


def snapshot():
    return dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        gpu=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        gpu_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        gpu_process_table=command(['nvidia-smi']),
        cpu_cores=os.cpu_count(), cpu_affinity_cores=len(os.sched_getaffinity(0)),
        cpu_load=list(os.getloadavg()),
        memory={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith(('MemTotal:', 'MemAvailable:', 'SwapFree:'))},
        cpu_processes=command(['ps','-eo','pid,pcpu,pmem,nlwp,args','--sort=-pcpu'])['stdout'].splitlines()[:22])


def dependencies():
    out = {}
    for name in ('DATA_READY.json', 'TRAIN_READY.json', 'DATA_BLOCKED.json'):
        p = RUN / name
        if not p.exists():
            candidates = list(RUN.rglob(name)) if RUN.exists() else []
            if len(candidates) == 1:
                p = candidates[0]
            elif len(candidates) > 1:
                raise ValueError(f'Ambiguous readiness sentinel: {name}')
        out[name] = dict(exists=p.is_file(), path=str(p), sha256=sha256(p) if p.is_file() else None)
    return out


if __name__ == '__main__':
    observed=dependencies()
    phase='BLOCKED_DATA' if observed['DATA_BLOCKED.json']['exists'] and not observed['DATA_READY.json']['exists'] else 'CPU_PREPARATION'
    result = dict(status=phase, dependencies=observed, resource_snapshot=snapshot(),
        cuda_models_created=0, forwards=0,
        scheduling='One GPU worker only after all dependencies and endpoint identities are verified; CPU work may overlap.')
    write_json(REGISTRATION/'eval_preparation_status.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
