"""CPU driver integration: real production methods, fake OS resources/children.

No model imports or forward; no real subprocess. Each fake child materializes
the documented outputs of its CLI, so production barriers/cache/summary execute.
This tests dispatch plumbing, not the scientific implementations inside workers.
"""
from __future__ import annotations
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
from r2.pilots.bench_parts_convergence_v1 import run_all as R

REAL_ROOT=Path('/mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1')
OUT=REAL_ROOT/('dispatch_production_preflight_20261019_'+str(time.time_ns()))
OUT.mkdir(exist_ok=False)
REG=R.read(REAL_ROOT/'registration.json')
SNAP=dict(gpu_name='SYNTHETIC_CPU_FIXTURE',gpu_total_mib=16384.,gpu_free_mib=15000.,gpu_used_mib=1384.,
          gpu_utilization=10.,cpu_logical=24,cpu_utilization=5.,memory_available_mib=64000.,gpu_processes=[])
CHECKS=[]

def record(name,**fields):CHECKS.append(dict(check=name,status='PASS',**fields))
def jl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row)+'\n')

class Children:
    def __init__(self):self.started=[];self.completed=[];self.serial=700000;self.first_train=None;self.fail=False
    def popen(self,command,**kwargs):
        assert kwargs['env']['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        assert command[1:3]==['-B','-m']
        self.serial+=1
        process=Child(self,command,self.serial,kwargs)
        self.started.append(process)
        return process

class Child:
    def __init__(self,owner,command,pid,kwargs):
        self.owner=owner;self.command=list(command);self.pid=pid;self.count=0;self.code=None;self.kwargs=kwargs
        self.module=command[3].split('.')[-1];self.argv=command[4:]
        self.start=time.time();self.realized=False
        if '--root' in self.argv:self.root=Path(self.value('--root'))
        else:self.root=None
        if self.module in ('train','sample','postprocess'):assert 'CUDA_VISIBLE_DEVICES' not in kwargs['env']
        else:assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        if self.module=='train':
            self.arm=self.value('--arm');self.target=int(self.value('--target'))
            status=self.root/'training'/self.arm/'status.json'
            self.start_step=R.read(status)['step'] if status.exists() else 12000
            if self.arm=='COMMON' and not status.exists():self.start_step=0
            R.write(status,dict(arm=self.arm,status='TRAINING',step=self.start_step+25,pid=self.pid,
                last=dict(peak_reserved_bytes=2000*1024**2)))
        if self.module in ('sample','postprocess'):
            stage='raw' if self.module=='sample' else 'post';label=self.value('--label')
            for index in (0,1):
                folder=self.root/'evaluation'/label/f'fixture_{index}'/stage
                R.write(folder/'receipt.json',dict(peak_reserved_bytes=1800*1024**2))
                if stage=='post':R.write(folder/'proposal_receipt.json',dict(peak_reserved_bytes=1800*1024**2))
    def value(self,key):return self.argv[self.argv.index(key)+1]
    def values(self,key):
        values=[]
        for x in self.argv[self.argv.index(key)+1:]:
            if x.startswith('--'):break
            values.append(x)
        return values
    def poll(self):
        self.count+=1
        if self.code is None and self.count>=7:
            if self.owner.fail:self.code=9
            else:self.outputs();self.code=0
            self.owner.completed.append(self)
        return self.code
    def send_signal(self,sig):raise AssertionError('Unexpected signal in passing fixture')
    def outputs(self):
        if self.realized:return
        self.realized=True
        if self.module=='train':
            folder=self.root/'training'/self.arm;binary=b'fixture checkpoint '+str(self.target).encode()
            rolling=folder/'rolling.pt';rolling.write_bytes(binary)
            receipt=dict(step=self.target,sha256=R.sha(rolling),generator_state_sha256='fixture_state')
            R.write(rolling.with_suffix('.json'),receipt)
            (folder/f'generator_step{self.target}.pt').write_bytes(binary)
            if self.target in (12000,15000,21000):
                named=folder/f'step{self.target}.pt';named.write_bytes(binary)
                R.write(named.with_suffix('.json'),receipt)
            R.write(folder/'status.json',dict(arm=self.arm,status='READY',step=self.target,pid=self.pid,checkpoint=str(rolling)))
            for step in (self.start_step,self.target):
                stat=lambda n:dict(sse=float(n),count=n)
                train=dict(ALL_free=stat(30),bulk=stat(30),seam_tip=stat(3 if self.arm=='SEAM' else 0),
                    parts={k:stat(3) for k in REG['major_parts']})
                R.write(folder/'fixed_loss'/f'step{step}.json',dict(arm='PLAIN' if self.arm=='COMMON' else self.arm,
                    step=step,probe_manifest_sha256='frozenfixture',train=train,dev=train))
            rows=[dict(step=k,FM=.1,input_seconds=.01,update_seconds=.1,unix=time.time()) for k in range(self.start_step+1,self.target+1)]
            jl(folder/'training_log.jsonl',rows)
            jl(folder/'ledger.jsonl',[dict(event='FORWARD_COMPLETE'),dict(event='BACKWARD_COMPLETE')])
            jl(folder/'exposure.jsonl',[dict(parent_id='fixture_parent',part='SEAT_SURFACE')])
        elif self.module=='sample':
            label=self.value('--label');subset=self.value('--panel')
            R.write(self.root/'evaluation'/label/('raw_complete_'+subset+'.json'),
                dict(status='COMPLETE',outputs=48 if subset=='full' else 8))
        elif self.module=='postprocess':
            label=self.value('--label');subset=self.value('--panel')
            for other in self.values('--barrier-labels'):
                assert (self.root/'evaluation'/other/('raw_complete_'+subset+'.json')).exists()
            R.write(self.root/'evaluation'/label/('post_complete_'+subset+'.json'),
                dict(status='COMPLETE',outputs=48 if subset=='full' else 8))
        elif self.module=='evaluation':
            stage=self.argv[0];labels=self.values('--labels');subset=self.value('--subset')
            key='__'.join(labels)+'_'+subset
            if stage=='score':
                assert int(self.value('--workers'))==2
                for label in labels:
                    assert (self.root/'evaluation'/label/('post_complete_'+subset+'.json')).exists()
                    assert ('--plain' in self.argv)==label.startswith('NATIVE')
                    R.write(self.root/'evaluation/completions'/(label+'_'+subset+'.json'),dict(status='COMPLETE'))
            elif stage=='aggregate':
                for label in labels:assert (self.root/'evaluation/completions'/(label+'_'+subset+'.json')).exists()
                folder=self.root/'evaluation/tables'/key;R.write(folder/'completion.json',dict(status='COMPLETE'))
                with (folder/'summary_parent_equal.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=['part_type','label','metric','mean_parent_equal']);writer.writeheader()
                    for label in labels:writer.writerow(dict(part_type='ALL',label=label,metric='FF',mean_parent_equal=.1))
            elif stage=='gallery':
                folder=self.root/'gallery'/key;folder.mkdir(parents=True)
                (folder/'index.html').write_text('CPU fixture, not actual mesh visualization')
                R.write(folder/'completion.json',dict(status='RENDER_COMPLETE_PENDING_REVIEW'))
            else:raise AssertionError(stage)
        elif self.module=='curves':R.write(Path(self.value('--out'))/'progress.json',dict(status='fixture'))
        elif self.module=='run_all':
            assert self.argv[0]=='--checkpoint-content-audit'
            R.write(Path(self.argv[4]),dict(status='PASS',scope='Mock worker; saved tensor content NOT tested here'))
        else:raise AssertionError(self.module)

def fixture(name):
    root=OUT/name;root.mkdir();R.write(root/'registration.json',REG)
    return R.Driver(root,poll_seconds=1)

started=time.perf_counter();children=Children()
original_progress_peak=R.Driver.progress_peak
def isolated_cpu_peak_guard(self,job,snapshot):
    # Test-only reproduction of the already reported minimal fix. Production
    # source is not edited by this audit; later unshimmed acceptance is separate.
    if not job['gpu']:return None
    return original_progress_peak(self,job,snapshot)
shim=os.environ.get('TEST_KNOWN_CPU_PEAK_GUARD')=='1'
with patch.object(R.C,'resource_snapshot',return_value=copy.deepcopy(SNAP)), \
     patch.object(R.subprocess,'Popen',side_effect=children.popen), \
     patch.object(R.time,'sleep',return_value=None), \
     patch.object(R.Driver,'progress_peak',isolated_cpu_peak_guard if shim else original_progress_peak):
    driver=fixture('normal_production')
    initial=driver.gate('sampling',1,0)
    assert initial['additional_workers']==1
    record('actual_gate_logging_kind_keyword',additional_workers=initial['additional_workers'])
    for kind in ('training','sampling','denoising'):
        with patch.object(R.C,'resource_snapshot',return_value=dict(SNAP,gpu_free_mib=3000.)):
            assert driver.gate(kind,2,0)['additional_workers']==0
    with patch.object(R.C,'resource_snapshot',return_value=dict(SNAP,gpu_utilization=99.)):
        assert driver.gate('training',1,1,dict(process_peak_estimate_mib=3000,scope='CPU synthetic fixture'))['additional_workers']==0
    record('low_memory_and_saturated_active_worker_block')
    result=driver.run()
    assert result['status']=='READY_FOR_VISUAL_REVIEW' and result['endpoint']==15000
    for arm in ('COMMON','PLAIN','SEAM'):
        expected=12000 if arm=='COMMON' else 15000
        assert R.read(driver.root/'training'/arm/'status.json')['step']==expected
        assert (driver.root/'training'/arm/f'step{expected}.pt').exists()
    assert (driver.root/'summary.json').exists() and (driver.root/'report_ready.json').exists()
    real_commands=[p.command for p in children.started]
    record('actual_driver_run_all_production_methods',child_substitutions=len(real_commands),endpoint=result['endpoint'])
    stages=[p.module for p in children.started]
    assert {'train','sample','postprocess','evaluation','curves','run_all'}<=set(stages)
    record('actual_launch_reap_and_final_summary',modules=sorted(set(stages)))
    n=len(children.started)
    driver.evaluation(15000,('PLAIN','SEAM'),'full')
    assert len(children.started)==n
    record('completed_all_stage_caches_no_new_child')
    job=driver.jobs[0]
    assert driver.launch({k:job[k] for k in ('key','kind','gpu','arm','start_step','target','command')}) is None
    record('successful_job_receipt_reuse')
    bad=fixture('failure_production');children.fail=True
    spec=dict(key='expected_nonzero',kind='CPU_dummy',gpu=False,command=bad.cmd('curves','--root',bad.root,'--out',bad.root/'curves'))
    try:bad.wave([spec])
    except R.DispatchBlocked as exc:assert 'exit=9' in str(exc)
    else:raise AssertionError('Child failure not blocked')
    count=len(children.started)
    try:bad.launch(spec)
    except R.DispatchBlocked as exc:assert 'no silent retry' in str(exc)
    else:raise AssertionError('Failed child retried')
    assert len(children.started)==count
    record('actual_nonzero_child_stops_and_refuses_retry')
    children.fail=False

assert 'torch' not in sys.modules
receipt=dict(status='PASS_PRODUCTION_DRIVER_CPU_SUBSTITUTES',run_all_sha256=R.sha(R.__file__),checks=CHECKS,
    elapsed_seconds=time.perf_counter()-started,real_subprocess_calls=0,GPU_model_calls=0,torch_imported=False,
    test_only_known_fix_shim=shim,
    limitations=['OS resources and child worker implementations are replaced; no GPU/model scientific execution verified',
                'Child output fixtures validate dispatcher field/sequence contracts, not actual scoring/gallery internals',
                'Actual checkpoint tensor identity audit remains deferred to real final checkpoints'],
    commands=real_commands)
R.write(OUT/'receipt.json',receipt)
print(json.dumps(dict(status=receipt['status'],checks=len(CHECKS),children=len(real_commands),receipt=str(OUT/'receipt.json'))))
