"""Measured-resource coordinator; one locked GPU worker, independent CPU scorers."""
from pathlib import Path
import argparse, json, math, os, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[3]
PREFIX='experiments.mf_h1_local0.equal_nfe_confirmation.'

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def resources():
    mem={s.split(':')[0]:int(s.split(':')[1].strip().split()[0])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemTotal:','MemAvailable:'))}
    def cmd(args):
        p=subprocess.run(args,capture_output=True,text=True);return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
    return dict(timestamp=time.time(),cpu_affinity=len(os.sched_getaffinity(0)),load=list(os.getloadavg()),memory=mem,
        gpu=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
        processes=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']))

def run(out):
    out=Path(out).resolve();protocol=read(out/'registration/protocol.json')
    dest=out/'dispatch.json'
    if dest.exists():raise FileExistsError('No automatic dispatch replay')
    logs=out/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    state=dict(status='RUNNING',started_timestamp=time.time(),GPU_max_workers=1,CPU_max_observed=0,events=[],CPU_workers=[],GPU_groups=[],failure=None,
        timing_note='CPU process seconds overlap GPU and other CPU tasks; never sum as user wait. Dispatch wall is end-to-end orchestration only.')
    running=[];pending=[];cpu_peak=0;cpu_completed=0;tick=time.perf_counter()
    def cpu_poll():
        nonlocal cpu_peak,cpu_completed
        for item in list(running):
            p=item['process'];rec=item['record']
            try:
                status=Path(f'/proc/{p.pid}/status').read_text()
                for ln in status.splitlines():
                    if ln.startswith(('VmHWM:','VmRSS:')):rec['peak_process_RSS_bytes']=max(rec['peak_process_RSS_bytes'],int(ln.split()[1])*1024)
            except FileNotFoundError:pass
            code=p.poll()
            if code is not None:
                rec.update(exit_code=code,wall_seconds=time.perf_counter()-item['tick'],completed_timestamp=time.time())
                item['log'].close();running.remove(item);cpu_peak=max(cpu_peak,rec['peak_process_RSS_bytes']);cpu_completed+=1
                state['events'].append(dict(kind='CPU_complete',group_id=rec['group_id'],resources=resources()))
                print('CPU_COMPLETE',rec['group_id'],'exit',code,flush=True)
        if not pending:return
        snap=resources();cores=snap['cpu_affinity'];available_cores=max(0,math.floor(cores-snap['load'][0]-2))
        memory_slots=max(0,math.floor((snap['memory']['MemAvailable']-4*1024**3)/(1.2*max(cpu_peak,256*1024**2))))
        slots=min(1-len(running),available_cores,memory_slots) if not cpu_completed else min(available_cores,memory_slots)
        # Unknown process peak: observe one already-authorized scoring group.
        if not cpu_completed and running:slots=0
        if pending and slots>0:
            group=pending.pop(0);log=(logs/(group+'_score.log')).open('x')
            rec=dict(group_id=group,started_timestamp=time.time(),peak_process_RSS_bytes=0,resource_before=snap,threads=1,observed_prior_peak_RSS_bytes=cpu_peak)
            process=subprocess.Popen([sys.executable,'-B','-m',PREFIX+'evaluation','--out',str(out),'--group',group],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            rec['pid']=process.pid;running.append(dict(process=process,record=rec,log=log,tick=time.perf_counter()));state['CPU_workers'].append(rec)
            state['CPU_max_observed']=max(state['CPU_max_observed'],len(running));slots-=1
            print('CPU_START',group,'concurrency',len(running),flush=True)
            # Remeasure before each added CPU process.
            snap=resources()
        write(dest,state)
    try:
        for group in protocol['groups']:
            gid=group['group_id'];snap=resources()
            while True:
                values=snap['gpu']['stdout'].splitlines()[0].split(',')
                total,free=float(values[1]),float(values[3])
                if free>=max(2048,total*.1)+1536:break
                state['events'].append(dict(kind='GPU_resource_wait',group_id=gid,resources=snap))
                print('GPU_RESOURCE_WAIT',gid,flush=True)
                cpu_poll();write(dest,state);time.sleep(5);snap=resources()
            rec=dict(group_id=gid,resource_before=snap,started_timestamp=time.time())
            log=(logs/(gid+'_gpu.log')).open('x')
            process=subprocess.Popen([sys.executable,'-B','-m',PREFIX+'worker','--out',str(out),'--group',gid],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            rec['pid']=process.pid;state['GPU_groups'].append(rec);write(dest,state);start=time.perf_counter()
            print('GPU_START',gid,flush=True)
            while process.poll() is None:
                cpu_poll();write(dest,state);time.sleep(.5)
            log.close();rec.update(exit_code=process.returncode,wall_seconds=time.perf_counter()-start,resource_after=resources())
            print('GPU_COMPLETE',gid,'exit',process.returncode,flush=True)
            if any(t['kind']=='formal' for t in group['tasks']):pending.append(gid)
            cpu_poll();write(dest,state)
            if process.returncode!=0:
                state['failure']=dict(group_id=gid,reason='GPU worker failed; no further GPU dispatch or retries')
                break
        while pending or running:
            cpu_poll();write(dest,state);time.sleep(.5)
        score_failed=[r['group_id'] for r in state['CPU_workers'] if r.get('exit_code')!=0]
        state['status']='PASS' if state['failure'] is None and not score_failed else 'STOPPED'
        state['CPU_failed_groups']=score_failed
    except BaseException as e:
        state['status']='STOPPED';state['failure']=repr(e);raise
    finally:
        state.update(wall_seconds=time.perf_counter()-tick,finished_timestamp=time.time(),CPU_peak_process_RSS_bytes=cpu_peak,resource_final=resources())
        write(dest,state)
    print(json.dumps(dict(status=state['status'],GPU_groups=len(state['GPU_groups']),CPU_groups=len(state['CPU_workers']),wall_seconds=state['wall_seconds'])),flush=True)
    if state['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
