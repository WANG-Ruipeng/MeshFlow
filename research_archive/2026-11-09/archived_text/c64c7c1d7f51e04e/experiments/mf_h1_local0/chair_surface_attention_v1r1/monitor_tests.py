"""CPU-only telemetry regressions and optional read-only nvidia-smi smoke."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from unittest.mock import patch
from . import monitor


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _fake_run(args,**kwargs):
    if '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu' in args:
        return subprocess.CompletedProcess(args,0,'16303, 4000, 12303, 7\n','')
    return subprocess.CompletedProcess(args,0,f'{os.getpid()}, python, 2048\n999999, other, 512\n','')


def run(out,*,real_smoke=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Preserve prior monitor regression evidence; do not overwrite')
    source=Path(__file__).parent
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('monitor.py','monitor_tests.py')}
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    result=dict(status='RUNNING',checks={},real_model_calls=0,CUDA_calls=0,optimizer_updates=0,source_hashes=hashes)
    original_append=monitor.append_event;seen=[]
    def guarded_append(path,payload):
        assert 'time_unix' not in payload and 'pid' not in payload
        seen.append(dict(payload));return original_append(path,payload)
    with patch.object(monitor.subprocess,'run',side_effect=_fake_run),patch.object(monitor,'append_event',side_effect=guarded_append):
        m=monitor._MemoryMonitor(out/'mock_success.jsonl');before=time.time();a=m.poll();b=m.poll();s=m.summary(before)
        durable=_read(m.path)
        assert len(durable)==2 and len(seen)==2 and s['samples']==2
        assert s['peak_process_used_mib']==2048 and s['minimum_global_free_mib']==12303
        assert all(r['pid']==os.getpid() and r['sample_started_unix']<=r['time_unix'] for r in durable)
        assert a['time_unix']==durable[0]['sample_started_unix'] and b['time_unix']==durable[1]['sample_started_unix']
        assert m.summary(time.time()+1)['samples']==0
        m.stop()  # safe even if a caller only uses synchronous polling
        result['checks']['reserved_keys_and_sync_poll']=dict(status='PASS',samples=len(durable),summary=s)
    def na_run(args,**kwargs):
        value=_fake_run(args,**kwargs)
        if any('query-compute-apps' in a for a in args):value.stdout=f'{os.getpid()}, [Not Found], [N/A]\n'
        return value
    with patch.object(monitor.subprocess,'run',side_effect=na_run):
        m=monitor._MemoryMonitor(out/'mock_na.jsonl');m.poll();s=m.summary(0)
        assert s['peak_process_used_mib'] is None and s['peak_global_used_mib']==4000
        assert s['process_peak_status']=='NA_NVIDIA_SMI_PROCESS_MEMORY_UNAVAILABLE'
        result['checks']['process_NA_preserved']=dict(status='PASS',summary=s)
    with patch.object(monitor.subprocess,'run',side_effect=subprocess.TimeoutExpired('nvidia-smi',10)):
        m=monitor._MemoryMonitor(out/'mock_timeout.jsonl');r=m.poll();s=m.summary(0)
        assert r['status']=='ERROR' and 'TimeoutExpired' in r['error'] and s['samples']==1 and s['collection_errors']==1
        assert s['peak_global_used_mib'] is None
        result['checks']['subprocess_failure_durable']=dict(status='PASS',error=r['error'])
    with patch.object(monitor.subprocess,'run',side_effect=_fake_run),patch.object(monitor,'append_event',side_effect=OSError('injected telemetry write error')):
        m=monitor._MemoryMonitor(out/'mock_background_failure.jsonl');m.start()
        assert m.stop_event.wait(2),'Background exception was not observed';m.stop()
        assert not m.thread.is_alive() and len(m.background_errors)==1 and m.summary(0)['samples']==0
        emergency=_read(m.error_path);assert len(emergency)==1 and 'injected telemetry write error' in emergency[0]['error']
        try:m.poll()
        except RuntimeError as exc:assert 'Background GPU monitor failed' in str(exc)
        else:raise AssertionError('Synchronous poll silently ignored prior background failure')
        result['checks']['background_failure_evidence']=dict(status='PASS',error_file=str(m.error_path),automatic_restart=False)
    active=[0];maximum=[0];counter_lock=threading.Lock()
    def slow_run(args,**kwargs):
        with counter_lock:active[0]+=1;maximum[0]=max(maximum[0],active[0])
        time.sleep(.002)
        try:return _fake_run(args,**kwargs)
        finally:
            with counter_lock:active[0]-=1
    with patch.object(monitor.subprocess,'run',side_effect=slow_run):
        m=monitor._MemoryMonitor(out/'mock_concurrent.jsonl',interval=.005);m.start()
        for _ in range(4):m.poll()
        m.stop();rows=_read(m.path)
        assert maximum[0]==1 and len(rows)==len(m.rows) and len(rows)>=4 and not m.background_errors
        result['checks']['background_and_direct_serialized']=dict(status='PASS',durable_samples=len(rows),maximum_simultaneous_subprocesses=maximum[0])
    if real_smoke:
        m=monitor._MemoryMonitor(out/'real_nvidia_smi.jsonl',interval=.1);m.start()
        try:
            for _ in range(2):
                row=m.poll();assert row['gpu_returncode']==0 and row['status']=='OK'
                assert 'global_free_mib' in row
        finally:m.stop()
        s=m.summary(0);assert s['samples']>=2 and not s['background_errors']
        assert len(_read(m.path))==s['samples']
        result['checks']['real_nvidia_smi_read_only']=dict(status='PASS',summary=s,
            scope='Actual read-only nvidia-smi subprocess queries; no CUDA library, model, training, or GPU workload created')
    result['status']='PASS'
    (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(run(args.out),sort_keys=True))
