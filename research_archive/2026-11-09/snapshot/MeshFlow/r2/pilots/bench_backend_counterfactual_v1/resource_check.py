"""Read-only runtime measurements for the registered CPU-only comparison."""
import json, os, subprocess, time
from pathlib import Path
import psutil

def run():
    root=Path(__file__).resolve().parents[3]/'experiments/r2_bench_backend_counterfactual_v1'
    root.mkdir(parents=True,exist_ok=True)
    def cmd(args):
        p=subprocess.run(args,capture_output=True,text=True)
        return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    info={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'gpu':cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
          'gpu_processes':cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
          'cpu_logical':psutil.cpu_count(),'cpu_physical':psutil.cpu_count(logical=False),
          'ram':psutil.virtual_memory()._asdict(),'cpu_percent_1sec':psutil.cpu_percent(1),
          'processes':[],'plan':{'GPU_workers':0,'core_workers':1,'score_workers_max':2,
          'render_workers_max':1,'threads_per_process':1,'max_overlapping_CPU_workers':3,
          'reason':'Only 10 new geometry jobs; independent scores/render may overlap; no neural calls.'}}
    for p in psutil.process_iter(['pid','name','cmdline','memory_info','cpu_times']):
        try:
            if any(s in (p.info['name'] or '').lower() for s in ('python','blender')):
                i=p.info;i['memory_info']=i['memory_info']._asdict();i['cpu_times']=i['cpu_times']._asdict()
                info['processes'].append(i)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    path=root/'resources'/('snapshot_'+str(time.time_ns())+'.json');path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'path':str(path),'cpu_logical':info['cpu_logical'],'cpu_pct':info['cpu_percent_1sec'],
        'available_GiB':info['ram']['available']/2**30,'gpu':info['gpu'],'active_python_count':len(info['processes'])}))

if __name__=='__main__':run()
