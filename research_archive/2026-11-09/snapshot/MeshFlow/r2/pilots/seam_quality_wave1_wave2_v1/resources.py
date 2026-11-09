"""Read-only resource snapshots, including GPU process occupancy and CPU load."""
import os,time,json,argparse,subprocess
from .protocol import OUT
from r2.pilots.seam_abc_v1 import generate as G

def snapshot(label):
    import psutil
    r=G.resources();r['label']=label
    r['host_ram_total_bytes']=psutil.virtual_memory().total
    r['processes_cpu_ram']=[dict(pid=p.pid,name=p.info.get('name'),rss_bytes=p.info.get('memory_info').rss if p.info.get('memory_info') else None)
        for p in psutil.process_iter(['name','memory_info']) if any(k in (p.info.get('name') or '').lower() for k in ('python','wsl','nvidia'))]
    r['disk']=dict(zip(('total','used','free'),__import__('shutil').disk_usage(OUT)))
    G.write(OUT/'resources'/(label+'_'+str(time.time_ns())+'.json'),r)
    print(json.dumps(r),flush=True);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');snapshot(p.parse_args().label)
