"""Serial CPU benchmark driver; no training or official-queue interaction."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse, datetime, json, platform, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
from ..exporter import RadialSupport
from ..io_utils import file_hash, write_json, save_npz
from ..layout_student.deployment import make_fixed, named_quality, predict_raw
from ..layout_value.optimizer import optimize_layout
from ..legacy.geometry_task import fibonacci, hull_faces
from ..legacy.networks import DomainVelocity
from .prepare import verify, require
from .timed_eval import evaluate_candidate

HOST_COMMAND = r"""[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); $o=Get-CimInstance Win32_OperatingSystem; $p=@(Get-Process cl,clang-cl,link,lld-link -ErrorAction SilentlyContinue); $c=Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter \"Name='_Total'\"; [ordered]@{utc=[DateTime]::UtcNow.ToString('o');compiler_process_count=$p.Count;compiler_pids=@($p|ForEach-Object{$_.Id});free_physical_GiB=[double]$o.FreePhysicalMemory/1048576;free_commit_GiB=[double]$o.FreeVirtualMemory/1048576;cpu_percent=[double]$c.PercentProcessorTime}|ConvertTo-Json -Compress"""


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def host_status():
    command = HOST_COMMAND.replace('\\"', '"')
    result = subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
                             '-NoProfile', '-NonInteractive', '-Command', command],
                            text=True, encoding='utf-8', errors='replace', capture_output=True,
                            timeout=30, check=True)
    return json.loads(result.stdout.lstrip('\ufeff').strip())


def host_ready(state, p):
    rule = p['host_gate']
    return (state['compiler_process_count'] == 0
            and state['free_physical_GiB'] >= rule['minimum_free_physical_GiB']
            and state['free_commit_GiB'] >= rule['minimum_free_commit_GiB']
            and state['cpu_percent'] < rule['maximum_host_cpu_percent'])


def wait_ready(run, p, quiet_samples=1):
    count = 0
    while count < quiet_samples:
        state = host_status()
        append_jsonl(run / 'HOST_SAMPLES.jsonl', state)
        count = count + 1 if host_ready(state, p) else 0
        if count < quiet_samples:
            write_json(run / 'PROGRESS.json', {'status': 'WAITING_FOR_QUIET_HOST',
                       'utc': utc(), 'host': state, 'quiet_samples': count,
                       'required_samples': quiet_samples}, replace=(run / 'PROGRESS.json').exists())
            print(json.dumps({'event': 'WAITING_FOR_QUIET_HOST', 'host': state}), flush=True)
            time.sleep(p['host_gate']['poll_seconds'])
    return state


def append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + '\n')


def warmup(run, p):
    before = time.perf_counter()
    support = RadialSupport()
    u = fibonacci(258)
    f = hull_faces(u).astype(np.int32)
    y = np.log(1 / np.sqrt((support.u**2 / np.array([1.5, .8, 1.])**2).sum(1))).astype(np.float32)
    rv, rf = support.mesh(y)
    fixed, base_report = evaluate_candidate(u, f, y, support, rv, rf,
                                           seeds=(1593499, 1693499), points=8192)
    initializers = {}
    for arm, binding in p['model_bindings'].items():
        start = time.perf_counter()
        net = DomainVelocity().cpu().eval().requires_grad_(False)
        net.load_state_dict(torch.load(binding['path'], map_location='cpu', weights_only=True)['model'])
        try:
            raw, meta = predict_raw(net, y, support, u, f)
            initializers[arm] = {'status': 'WARMED', 'elapsed_seconds': time.perf_counter()-start,
                                 'raw_metadata': meta}
        except Exception as exc:
            initializers[arm] = {'status': 'SYNTHETIC_INITIALIZER_FAILED_NO_RETRY',
                                 'error': repr(exc), 'elapsed_seconds': time.perf_counter()-start}
    q = named_quality(fixed['vertices'], f)
    result = optimize_layout(y, u, f, {'min_angle_deg': q['min_triangle_angle_deg'],
                             'angle_q05_deg': q['triangle_min_angle_p05_deg']},
                             device='cpu', max_steps=4, max_seconds=3600,
                             seed=1493499, snapshot_steps=(0, 1, 4))
    require(result['stats']['iterations'] == 4, 'Synthetic warmup optimizer incomplete')
    info = {'scope': 'ONE_TIME_SYNTHETIC_WARMUP_NOT_A_PARENT_RESULT',
            'elapsed_seconds': time.perf_counter()-before, 'actual_optimizer_updates': 4,
            'initializers': initializers, 'baseline_audit': base_report,
            'cuda_initialized': torch.cuda.is_initialized(),
            'does_not_select_any_configuration': True}
    write_json(run / 'WARMUP.json', info)
    return info


def run(protocol_path, gate_path):
    import_start = time.perf_counter()
    from .runner import run_case, case_folder
    from .report import build_report
    from .observer_adapter import source_identity
    protocol_path = Path(protocol_path).resolve()
    root = protocol_path.parent
    p = verify(protocol_path)
    require(not (root / 'STARTED.json').exists(), 'An attempted benchmark is never overwritten or automatically restarted')
    gate = json.loads(Path(gate_path).read_text())
    require(gate['status'] == 'PASS' and gate['protocol_sha256'] == file_hash(protocol_path), 'Actual test gate missing/mismatched')
    for path, digest in gate['source_sha256'].items():
        require(file_hash(path) == digest, 'Tested source changed: ' + path)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'Benchmark must remain CPU-only')
    write_json(root / 'STARTED.json', {'utc': utc(), 'pid': os.getpid(),
               'protocol_sha256': file_hash(protocol_path), 'gate_sha256': file_hash(gate_path),
               'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                           'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads()},
               'adapter_identity': source_identity(), 'startup_import_seconds': (time.perf_counter() - float(os.environ['TTQ_PROCESS_START_MONOTONIC'])) if 'TTQ_PROCESS_START_MONOTONIC' in os.environ else None,
               'late_import_seconds': time.perf_counter()-import_start})
    before = time.perf_counter()
    completed = []
    try:
        wait_ready(root, p, quiet_samples=3)
        warm = warmup(root, p)
        wait_ready(root, p, quiet_samples=3)
        for index, spec in enumerate(p['case_order']):
            host_before = wait_ready(root, p)
            folder = case_folder(root, spec['uid'], spec['arm'], spec['repeat'])
            case = run_case(protocol_path, spec['uid'], spec['arm'], spec['repeat'], folder)
            host_after = host_status()
            append_jsonl(root / 'HOST_SAMPLES.jsonl', host_after)
            resource_record = {'before': host_before, 'after': host_after,
                               'boundary_resource_gate_pass': host_ready(host_before, p) and host_ready(host_after, p),
                               'continuous_host_isolation_not_certified': True,
                               'host_probes_outside_case_wall': True}
            write_json(folder / 'RESOURCES.json', resource_record)
            completed.append({**spec, 'task_path': str(folder / 'TASK.json'),
                              'task_sha256': file_hash(folder / 'TASK.json'),
                              'status': case.get('status'),
                              'boundary_resource_gate_pass': resource_record['boundary_resource_gate_pass']})
            progress = {'status': 'BENCHMARK_RUNNING', 'completed_cases': len(completed),
                        'planned_cases': p['expected_cases'], 'last_case': completed[-1],
                        'elapsed_driver_seconds': time.perf_counter()-before, 'utc': utc()}
            write_json(root / 'PROGRESS.json', progress, replace=True)
            print(json.dumps(progress), flush=True)
            build_report(root, partial=True)
        verify(protocol_path)
        summary = build_report(root, partial=False)
        final = {'status': 'ALL_PLANNED_ATTEMPTS_COMPLETE', 'cases': completed,
                 'protocol_sha256': file_hash(protocol_path), 'total_driver_wall_seconds': time.perf_counter()-before,
                 'one_time_warmup_seconds': warm['elapsed_seconds'],
                 'all_boundary_resource_gates_pass': all(r['boundary_resource_gate_pass'] for r in completed),
                 'cuda_initialized': torch.cuda.is_initialized(), 'source_inputs_unchanged': True,
                 'scientific_speedup_claim': False, 'finished_utc': utc()}
        write_json(root / 'COMPLETE.json', final)
        write_json(root / 'PROGRESS.json', {k: v for k, v in final.items() if k != 'cases'}, replace=True)
        return final
    except BaseException as exc:
        fail = {'status': 'DRIVER_INTERRUPTED', 'error': repr(exc), 'traceback': traceback.format_exc(),
                'completed_cases': completed, 'all_original_attempts_retained': True,
                'no_automatic_retry': True, 'utc': utc()}
        write_json(root / 'FAILED.json', fail)
        write_json(root / 'PROGRESS.json', fail, replace=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--gate', required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.gate)
    print(json.dumps({'status': result['status'], 'cases': len(result['cases'])}), flush=True)


if __name__ == '__main__':
    main()