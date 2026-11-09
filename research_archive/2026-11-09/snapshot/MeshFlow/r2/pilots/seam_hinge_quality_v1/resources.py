"""Measured resource record; no model dispatch."""
import argparse,time,json,shutil
from .protocol import OUT
from r2.pilots.seam_abc_v1 import generate as G
def snapshot(label):
    import psutil
    OUT.mkdir(parents=True,exist_ok=True)
    r=G.resources();r.update(label=label,cpu_count=psutil.cpu_count(),ram_total=psutil.virtual_memory().total,
        ram_available=psutil.virtual_memory().available,disk_free=shutil.disk_usage(OUT).free,
        host_python_processes=[dict(pid=p.pid,name=p.info['name'],rss=p.info['memory_info'].rss if p.info['memory_info'] else None)
            for p in psutil.process_iter(['name','memory_info']) if 'python' in (p.info['name'] or '').lower()])
    G.write(OUT/'resources'/(label+'_'+str(time.time_ns())+'.json'),r);return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');print(json.dumps(snapshot(p.parse_args().label)))
