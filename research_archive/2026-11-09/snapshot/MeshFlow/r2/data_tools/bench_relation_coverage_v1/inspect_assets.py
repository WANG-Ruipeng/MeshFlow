"""Read-only schema and resource inspection; no model imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import json
from pathlib import Path
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[3]
def read(p): return json.loads((ROOT / p).read_text('utf-8-sig'))
def brief(x):
    return [(k, len(v) if isinstance(v, (list, dict)) else str(v)[:120]) for k,v in x.items()]
if __name__ == '__main__':
    m = read('experiments/r2_bench_user_data_v2_expansion_v1/manifest.json')
    print('MANIFEST', brief(m))
    r = m['train'][0]
    print('ROW', json.dumps(r)[:9000])
    for name in ('case_path', 'input_path'):
        a = np.load(r[name], allow_pickle=False)
        print(name, [(k, a[k].shape, str(a[k].dtype)) for k in a.files])
    print('DEV', json.dumps(m['dev'][0])[:3500])
    p = m['parents'][0]
    print('PARENT', json.dumps(p)[:2500])
    for name in ('target_mesh_path', 'target_case_path'):
        if name in p:
            a = np.load(p[name], allow_pickle=False)
            print(name, [(k, a[k].shape, str(a[k].dtype)) for k in a.files])
    for p in ('experiments/r2_bench_trajectory_crossover_v1/registration.json',
              'experiments/r2_bench_user_v2_training_v1/registration.json',
              'experiments/r2_bench_user_data_v2_expansion_v1/all_admission_decisions.json'):
        x = read(p); print(p, brief(x))
        if 'requests' in x: print('REQUEST', json.dumps(x['requests'][0])[:6000])
    print('RESOURCE', json.dumps(dict(logical_cores=psutil.cpu_count(), physical_cores=psutil.cpu_count(logical=False),
          memory=psutil.virtual_memory()._asdict(), load=psutil.getloadavg(),
          processes=[dict(pid=p.pid, name=p.info['name'], cmd=p.info['cmdline']) for p in psutil.process_iter(['name','cmdline'])
                     if (p.info['name'] or '').startswith('python')]), default=str))
