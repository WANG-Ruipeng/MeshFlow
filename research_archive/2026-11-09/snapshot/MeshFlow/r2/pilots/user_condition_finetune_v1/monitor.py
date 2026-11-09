"""Read-only device monitor; never stops an observed process."""
from pathlib import Path
import argparse,json,os,subprocess,time
p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--phase",required=True);a=p.parse_args()
out=Path(a.out)/"resources";out.mkdir(parents=True,exist_ok=True)
path=out/(a.phase+"_monitor.jsonl");stop=out/(a.phase+"_monitor.stop")
if path.exists(): raise RuntimeError("Existing monitor receipt: use a distinct phase")
with path.open("a") as f:
 while not stop.exists():
    r=subprocess.run(["nvidia-smi","--query-gpu=memory.total,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True,capture_output=True)
    row=dict(time_unix=time.time(),pid=os.getpid(),gpu=r.stdout.strip(),returncode=r.returncode)
    c=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True,capture_output=True)
    row["compute_processes"]=c.stdout.strip()
    mem=dict(l.split(":",1) for l in Path("/proc/meminfo").read_text().splitlines())
    row["mem_available_kib"]=int(mem["MemAvailable"].split()[0])
    f.write(json.dumps(row)+"\n");f.flush()
    time.sleep(2)
 print(json.dumps(dict(phase=a.phase,status="MONITOR_STOPPED",path=str(path))))
