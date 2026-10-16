"""Read-only bounded resource telemetry for the authorized frozen sampling wave."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
import psutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--seconds', type=int, default=1200)
    a = p.parse_args()
    if not a.name.replace('_', '').isalnum() or not 1 <= a.seconds <= 3600:
        raise ValueError('Invalid bounded telemetry request')
    root = Path(__file__).resolve().parents[3] / 'experiments/r2_seam_occupancy_rollout_v1/resources'
    root.mkdir(parents=True, exist_ok=True)
    path = root / (a.name + '.jsonl')
    stop = root / (a.name + '.stop')
    start = time.monotonic()
    with path.open('x', encoding='utf-8') as f:
        while time.monotonic() - start < a.seconds and not stop.exists():
            row = dict(unix=time.time(), elapsed=time.monotonic()-start,
                       cpu_cores=os.cpu_count(), cpu_percent=psutil.cpu_percent(),
                       ram_available=psutil.virtual_memory().available,
                       loadavg=list(os.getloadavg()))
            for key, query in [('gpu', ['--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu']),
                               ('processes', ['--query-compute-apps=pid,process_name,used_gpu_memory'])]:
                try:
                    r = subprocess.run(['nvidia-smi', *query, '--format=csv,noheader,nounits'],
                                       capture_output=True, text=True, timeout=10)
                    row[key] = dict(returncode=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip())
                except Exception as e:
                    row[key] = dict(error=repr(e))
            f.write(json.dumps(row)+'\n'); f.flush()
            time.sleep(1)
        f.write(json.dumps(dict(event='MONITOR_COMPLETE', unix=time.time(),
                               elapsed=time.monotonic()-start, stopped=stop.exists()))+'\n')
    print(json.dumps(dict(status='COMPLETE', path=str(path))))


if __name__ == '__main__':
    main()
