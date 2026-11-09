"""Experiment-local GPU telemetry; importing/polling never initializes CUDA.

The protected append_event owns its time_unix/pid keys. Sampling timestamps are
separate payload fields. Memory summaries use sampling start, not log-write time.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.common import append_event


class _MemoryMonitor:
    def __init__(self,path,*,interval=.2,command_timeout=10.):
        if not math.isfinite(interval) or interval<=0:raise ValueError('Positive finite interval required')
        if not math.isfinite(command_timeout) or command_timeout<=0:raise ValueError('Positive finite timeout required')
        self.path=Path(path);self.error_path=self.path.with_name(self.path.name+'.errors.jsonl')
        self.interval=float(interval);self.command_timeout=float(command_timeout)
        self.stop_event=threading.Event();self.lock=threading.Lock();self.poll_lock=threading.Lock()
        self.rows=[];self.background_errors=[];self.started=False
        self.thread=threading.Thread(target=self._loop,name='surface-r1-memory-monitor',daemon=True)

    def _command(self,args):
        return subprocess.run(args,capture_output=True,text=True,timeout=self.command_timeout)

    def _sample(self):
        start=time.time();row=dict(time_unix=start,pid=os.getpid(),sample_started_unix=start,
            process_used_mib=None,status='OK')
        try:
            gpu=self._command(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'])
            row.update(gpu=gpu.stdout.strip(),gpu_stderr=gpu.stderr.strip(),gpu_returncode=gpu.returncode)
            if gpu.returncode==0:
                values=list(csv.reader(io.StringIO(gpu.stdout)))[0]
                total,used,free,util=map(float,values)
                if not all(math.isfinite(v) and v>=0 for v in (total,used,free,util)) or total<=0:
                    raise ValueError('Nonfinite/negative nvidia-smi GPU memory or utilization')
                row.update(total_mib=total,global_used_mib=used,global_free_mib=free,utilization=util)
            else:row.update(status='ERROR',error='nvidia-smi GPU query failed')
            processes=self._command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                '--format=csv,noheader,nounits'])
            row.update(processes=processes.stdout.strip(),process_stderr=processes.stderr.strip(),
                process_returncode=processes.returncode)
            if processes.returncode!=0:
                row.update(status='ERROR',process_error='nvidia-smi process query failed')
            else:
                values=[]
                for fields in csv.reader(io.StringIO(processes.stdout)):
                    if len(fields)<3 or fields[0].strip()!=str(os.getpid()):continue
                    try:value=float(fields[-1])
                    except ValueError:continue  # WSL commonly reports [N/A].
                    if math.isfinite(value) and value>=0:values.append(value)
                if values:row['process_used_mib']=sum(values)
        except Exception as exc:
            row.update(status='ERROR',error=repr(exc),collection_traceback=traceback.format_exc())
        row.update(sample_finished_unix=time.time(),sample_duration_seconds=time.time()-start)
        return row

    def poll(self):
        """One synchronous, serialized sample; persistence errors are not hidden."""
        with self.poll_lock:
            with self.lock:errors=list(self.background_errors)
            if errors:raise RuntimeError('Background GPU monitor failed; inspect '+str(self.error_path))
            row=self._sample()
            payload={key:value for key,value in row.items() if key not in ('time_unix','pid')}
            assert 'time_unix' not in payload and 'pid' not in payload
            # Only durable samples enter summaries. The helper adds its reserved
            # event metadata once; sample_started_unix preserves collection time.
            append_event(self.path,payload)
            with self.lock:self.rows.append(row)
            return row

    def _record_background_failure(self,exc):
        failure=dict(time_unix=time.time(),pid=os.getpid(),event='MONITOR_BACKGROUND_FAILURE',
            error=repr(exc),traceback=traceback.format_exc(),monitor_path=str(self.path),
            thread_stopped=True,automatic_restart=False)
        with self.lock:
            self.background_errors.append(failure)
            try:
                self.error_path.parent.mkdir(parents=True,exist_ok=True)
                # An independent emergency writer avoids re-entering a failing
                # append_event implementation and preserves the original error.
                with self.error_path.open('a',encoding='utf8') as handle:
                    handle.write(json.dumps(failure,sort_keys=True,allow_nan=False)+'\n')
                    handle.flush();os.fsync(handle.fileno())
            except Exception as persistence_error:
                failure['emergency_persistence_error']=repr(persistence_error)

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                self.poll()
                self.stop_event.wait(self.interval)
        except Exception as exc:
            self._record_background_failure(exc)
            self.stop_event.set()

    def start(self):
        if self.started:raise RuntimeError('A monitor may only be started once')
        self.started=True;self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.started:self.thread.join(timeout=2*self.command_timeout+5)
        if self.thread.is_alive():raise RuntimeError('GPU monitor did not stop within its command timeouts')

    def summary(self,since):
        with self.lock:
            rows=[row for row in self.rows if row['time_unix']>=since]
            failures=list(self.background_errors)
        global_rows=[r for r in rows if 'global_used_mib' in r]
        process=[r['process_used_mib'] for r in rows if r.get('process_used_mib') is not None]
        return dict(samples=len(rows),peak_global_used_mib=max((r['global_used_mib'] for r in global_rows),default=None),
            minimum_global_free_mib=min((r['global_free_mib'] for r in global_rows),default=None),
            peak_process_used_mib=max(process,default=None),
            process_peak_status='MEASURED' if process else 'NA_NVIDIA_SMI_PROCESS_MEMORY_UNAVAILABLE',
            sampling_interval_seconds=self.interval,collection_errors=sum(r['status']!='OK' for r in rows),
            background_errors=failures,background_healthy=not failures,
            scope='nvidia-smi samples selected by sample-start time; process/global values are sampled observations and brief unsampled peaks remain possible')


MemoryMonitor=_MemoryMonitor
