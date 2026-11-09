from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_seam_quality_four_v1'
BASE = ROOT / 'experiments/r2_seam_slot_c200_v1'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def resources():
    import psutil
    def command(args):
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                gpu=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
                processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
                CPU_cores=os.cpu_count(), CPU_load=list(os.getloadavg()),
                RAM_available_bytes=psutil.virtual_memory().available,
                CPU_processes=command(['ps','-eo','pid,pcpu,pmem,args','--sort=-pcpu']))

def event(kind, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'ledger.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               event=kind, **fields), ensure_ascii=False, allow_nan=False)+'\n')

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    r = resources()
    write(OUT / 'resources_initial.json', r)
    event('RESOURCE_INSPECTION', GPU=r['gpu'], CPU_cores=r['CPU_cores'],
          RAM_available_bytes=r['RAM_available_bytes'])
    print(json.dumps(r, indent=2))
