"""Resource-only end-of-experiment observation; no CUDA/model initialization."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.bench_three_mechanisms_v1 import common as C
r=C.register_resource('final_report_only')
print(json.dumps({k:r[k] for k in ('gpu_name','total_mib','used_mib','free_mib',
    'gpu_util_percent','cpu_cores','available_ram_bytes','compute_processes')}))
