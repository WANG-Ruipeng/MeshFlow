"""CPU-only accounting of existing GEO cleanup receipts; no geometry recomputation."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_dedup_n_sensitivity_v1/audit'
OLD = ROOT / 'experiments/r2_target_cleanup_c200_v1'
REASONS = ('exact_repeated_vertex', 'exact_collinear', 'duplicate_C', 'duplicate_free')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()

def save(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf8')
    if p.exists():
        assert p.read_bytes() == data, f'Refuse changed audit overwrite: {p}'
    else:
        p.write_bytes(data)
    return p

def table(name, rows):
    assert rows
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    p = OUT / name
    data = stream.getvalue().encode('utf8')
    if p.exists():
        assert p.read_bytes() == data, f'Refuse changed audit overwrite: {p}'
    else:
        p.write_bytes(data)

def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {'argv': args, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except Exception as e:
        return {'argv': args, 'status': 'UNAVAILABLE', 'error': repr(e)}

def resource_snapshot():
    if (OUT / 'resource_snapshot.json').exists():
        return
    mem = {}
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            k, value = line.split(':', 1)
            if k in ('MemTotal', 'MemAvailable', 'MemFree'):
                mem[k] = value.strip()
    snapshot = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Independent CPU JSON/CSV audit only. No GPU allocation or model calls.',
        'cpu_count': os.cpu_count(), 'available_affinity_cores': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'loadavg': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
        'memory_proc': mem,
        'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        'audit_workers': 1,
        'gpu_process_memory_limit': 'nvidia-smi process memory may be unavailable under WSL; device memory is not per-process memory',
        'gpu': command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader']),
        'gpu_processes': command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader']),
        'cpu_processes': command(['ps', '-eo', 'pid,ppid,comm,pcpu,pmem,rss,args', '--sort=-pcpu']),
    }
    save('resource_snapshot.json', snapshot)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    resource_snapshot()
    inputs = {}
    regpath = OLD / 'registration.json'
    inputs[str(regpath.relative_to(ROOT))] = sha(regpath)
    reg = json.loads(regpath.read_text(encoding='utf8'))
    requests = {r['request_id']: r for r in reg['requests']}
    assert len(requests) == 24
    rows, totals = [], []
    for arm in ('ORIGINAL_C200', 'DEDUP_C200'):
        paths = sorted((OLD / 'evaluation/post' / arm).glob('*/geometry.json'))
        assert len(paths) == 24
        assert {p.parent.name for p in paths} == set(requests)
        counter, nsum, ksum = Counter(), 0, 0
        for p in paths:
            inputs[str(p.relative_to(ROOT))] = sha(p)
            g = json.loads(p.read_text(encoding='utf8'))
            stats, faces = g['stats'], g['face_records']
            req = requests[p.parent.name]
            assert len(faces) == int(stats['input_faces']) == int(req['N'])
            assert int(stats['known_faces']) == int(req['K'])
            assert len({f['source_face_id'] for f in faces}) == len(faces)
            assert all(f['kept'] for f in faces if f['known'])
            deleted = [f for f in faces if not f['kept']]
            counts = Counter(f['reason'] for f in deleted)
            assert not (set(counts) - set(REASONS)), counts
            assert counts == Counter(stats['deleted_by_reason'])
            assert sum(counts.values()) == int(stats['deleted_free_faces'])
            assert int(stats['input_faces']) - int(stats['output_faces']) == sum(counts.values())
            row = {'model': arm, 'request_id': req['request_id'], 'parent_id': req['parent_id'],
                   'split': req['split'], 'candidate': req['candidate'], 'N': req['N'], 'K': req['K'],
                   'input_free': req['N'] - req['K'], 'retained_free': stats['output_faces'] - req['K'],
                   'deleted_free': sum(counts.values()), **{k: counts[k] for k in REASONS},
                   'geometry_path': str(p.relative_to(ROOT)), 'geometry_sha256': sha(p)}
            rows.append(row)
            counter.update(counts)
            nsum += row['input_free']
            ksum += row['retained_free']
        totals.append({'model': arm, 'requests': 24, 'input_free': nsum, 'retained_free': ksum,
                       'deleted_free': sum(counter.values()), **{k: counter[k] for k in REASONS}})
    table('old_geo_deletions_per_output.csv', rows)
    table('old_geo_deletions_summary.csv', totals)
    src = ROOT / 'native_t1/postprocessing.py'
    inputs[str(src.relative_to(ROOT))] = sha(src)
    code = src.read_text(encoding='utf8')
    assert code.index('reason = _degenerate(corrected[i])') < code.index('if reason is None and key in seen:')
    assert 'return tuple(sorted(_numeric(p) for p in tri))' in code
    for rel, h in inputs.items():
        assert sha(ROOT / rel) == h, f'Input modified: {rel}'
    result = {
        'status': 'PASS', 'stage': 'OLD_GEO_RECEIPT_ACCOUNTING', 'requests_per_model': 24,
        'rows': len(rows), 'summary': totals, 'protected_inputs': inputs,
        'audit_source_sha256': sha(__file__),
        'classification': {
            'priority': ['exact_repeated_vertex', 'exact_collinear', 'duplicate_C_or_free_if_not_degenerate'],
            'duplicate_identity': 'Exact numerical triangle coordinate set, ignoring winding; C has priority, then first retained free face.',
            'limitation': 'Causes are mutually exclusive deletion-mechanism classes. Degenerate faces may also be duplicate geometrically but are counted only as degenerate. Does not certify that removed faces were visually useless or that output is complete.',
            'source': 'native_t1/postprocessing.py:_degenerate/process',
        },
        'new_training_updates': 0, 'new_model_calls': 0, 'new_geometry_processing': 0, 'new_quality_scoring': 0,
    }
    save('legacy_deletion_audit.json', result)
    print(json.dumps({'status': 'PASS', 'summary': totals, 'output': str(OUT)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
