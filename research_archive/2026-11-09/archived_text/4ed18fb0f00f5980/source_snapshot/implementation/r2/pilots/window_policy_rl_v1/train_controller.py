"""Finite on-policy batches; each fresh rollout is consumed at most once."""
from .bootstrap import bind,ROOT,OUT
D,E=bind()
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import atexit
from .run import snapshot,stamp,PREFIX,verify_completed


class ResidentD0Client:
    """One separate owned D0; lazy load begins only on the first real batch."""
    def __init__(self):
        self.process=None;self.log=None;self.closed=False
        self.directory=OUT/'training/resident_d0_sessions'/('service_'+stamp())
        atexit.register(self.close)
    def process_job(self,path):
        if self.process is None:
            logs=OUT/'dispatch/training';logs.mkdir(parents=True,exist_ok=True)
            self.log=(logs/(self.directory.name+'.log')).open('w',encoding='utf8')
            self.process=subprocess.Popen([sys.executable,'-u','-B','-m',PREFIX+'resident_d0',
                '--service-dir',str(self.directory)],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=self.log,text=True,bufsize=1)
        if self.process.poll() is not None:raise RuntimeError('Resident D0 stopped; no automatic restart')
        self.process.stdin.write(json.dumps(dict(job=str(path)))+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Resident D0 closed without a completion; preserve logs')
        response=json.loads(line)
        if response.get('status')!='COMPLETE' or response.get('job')!=str(path):raise RuntimeError('Resident D0 failure: '+line)
        return response
    def close(self):
        if self.closed:return
        self.closed=True
        if self.process is not None:
            if self.process.stdin and not self.process.stdin.closed:self.process.stdin.close()
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:print('Owned D0 still exiting; preserved PID '+str(self.process.pid),flush=True)
            if self.log:self.log.close()

def admission(tag,required_mib):
    while True:
        path,s=snapshot(tag)
        headroom=max(2048.,.1*s['_memory_total_mib'])
        if s['_memory_free_mib']-headroom>=required_mib:return path
        print(json.dumps(dict(event='WAIT_RESOURCES',phase=tag,free_mib=s['_memory_free_mib'],headroom_mib=headroom,required_mib=required_mib)),flush=True)
        time.sleep(15)

def invoke(module,job,phase=None):
    log=job.with_suffix('.log')
    args=[sys.executable,'-B','-m',PREFIX+module,'--job',str(job)]
    if phase:args+=['--phase',phase]
    with log.open('a',encoding='utf8') as f:
        p=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('Stopped without automatic retry: '+str(job)+'; log '+str(log))

def run():
    gate_path=OUT/'analysis/decode_gate.json';gate=D.read(gate_path)
    if gate['status']!='PASS':raise ValueError('Training forbidden: calibration did not PASS')
    math_path=OUT/'engineering/math_gate.json';math_gate=D.read(math_path);assert math_gate['status']=='PASS'
    for source_path,digest in math_gate['source_files'].items():
        if D.sha(source_path)!=digest:raise ValueError('Mathematically tested source changed: '+source_path)
    config_path=OUT/'training_config.json';config=D.read(config_path)
    assert config['batches']==64 and config['sigma']==gate['selected_sigma']
    from .prepare import training
    t=training(config['batches']);mp=OUT/'inputs/manifest.json';m=D.read(mp)
    tm=OUT/'inputs/training_manifest.json';rp=OUT/'registration.json';rh=D.sha(rp)
    source=m['source_binding'];qmap={q['request_id']:q for q in t['training_requests']}
    prior=None;completed=[];d0_service=ResidentD0Client()
    for group in t['groups']:
        b=group['batch_index'];stage=OUT/'training'/f'batch_{b:03d}';jobs=OUT/'jobs/training'/f'batch_{b:03d}'
        qs=[qmap[rid] for rid in group['requests']];sample_job=jobs/'sample.json'
        sample=dict(stage_dir=str(stage),label='RL_TRAIN',requests=qs,sigma=config['sigma'],model_mode='current',
                    source_binding=source,registration_path=str(rp),registration_sha256=rh,manifest_path=str(mp),
                    training_manifest_path=str(tm),ledger_dir=str(stage/'sample_ledger'),cpu_threads=1,adapter_init_seed=2026101501)
        if prior:sample.update(adapter_checkpoint=prior['path'],adapter_checkpoint_sha256=prior['sha256'])
        D.write(sample_job,sample)
        if (stage/'sample_ledger/completion.json').exists():verify_completed(sample_job,stage/'sample_ledger')
        else:
            if (stage/'sample_ledger').exists():raise RuntimeError('Unresolved sample attempt')
            resource=admission(f'train_{b:03d}_sample',2048.)
            D.write(jobs/'sample_admission.json',dict(resources=str(resource),worker_count=1,
                rationale='Only eight dependent on-policy rollouts in this batch. One resident model avoids multiple startup/hash costs and staged worker admission for tiny shards; no next batch allowed before update. Calibration parallelism measured separately.'))
            invoke('sample_worker',sample_job)
        for phase in ('d0','cpu'):
            p=jobs/(phase+'.json')
            existed=p.exists()
            if not p.exists():
                resource=admission(f'train_{b:03d}_{phase}',3072.) if phase=='d0' else snapshot(f'train_{b:03d}_cpu')[0]
                D.write(p,dict(jobs=[dict(stage_dir=str(stage),label='RL_TRAIN',requests=qs)],
                    manifest_path=str(mp),registration_path=str(rp),ledger_dir=str(stage/(phase+'_ledger')),
                    resource_snapshot_path=str(resource),workers=4 if phase=='cpu' else 2,stages=['post','score']))
            if phase=='d0' and existed and not (stage/'d0_ledger/completion.json').exists():
                fresh=admission(f'train_{b:03d}_d0_resume',3072.)
                D.write(jobs/'admissions'/(stamp()+'.json'),dict(phase='d0',resources=str(fresh),reason='Fresh admission for previously written unlaunched job'))
            if phase=='cpu' and not (stage/'cpu_ledger/completion.json').exists():
                fresh,current=snapshot(f'train_{b:03d}_cpu_actual')
                D.write(jobs/'admissions'/(stamp()+'.json'),dict(phase='cpu',resources=str(fresh),worker_count=4,threads_each=1,measured_prior_RSS_MiB_per_worker=120))
                if current['cpu_count']<4:raise RuntimeError('Insufficient CPU cores for registered scoring dispatch')
            if phase=='d0':d0_service.process_job(p)
            else:invoke('training_cpu',p,phase)
        rewards=[];samples=[]
        for q in qs:
            rid=q['request_id'];receipt_path=stage/'raw'/rid/'receipt.json';receipt=D.read(receipt_path)
            score_path=stage/'scores'/rid/'POST'/'score.json'
            # The original CPU worker receipt is the authoritative path registry.
            cpu=D.read(stage/'cpu_ledger/completion.json')
            row=next(r for r in cpu['rows'] if r['request_id']==rid)
            score_path=Path(row['score_paths']['POST']);score=D.read(score_path)
            rewards.append(dict(request_id=rid,reward=int(score['topology']['C_F_valid_edge_count']>0),
                post_score_path=str(score_path),post_score_sha256=D.sha(score_path)))
            samples.append(dict(q=q,window_buffer_path=receipt['window_buffer_path'],window_buffer_sha256=receipt['window_buffer_sha256'],
                receipt_path=str(receipt_path),receipt_sha256=D.sha(receipt_path)))
        reward_path=stage/'rewards.json';D.write(reward_path,dict(records=rewards,definition='int(unchanged POST topology.C_F_valid_edge_count > 0)',
                sigma=config['sigma'],batch_index=b,all_eight_retained=True))
        update_job=jobs/'update.json'
        if not update_job.exists():
            resource=admission(f'train_{b:03d}_update',4096.)
            D.write(update_job,dict(samples=samples,reward_path=str(reward_path),reward_sha256=D.sha(reward_path),
                prior_checkpoint=prior,batch_index=b,source_binding=source,sigma=config['sigma'],output_dir=str(stage/'update'),
                resource_snapshot_path=str(resource),registration_path=str(rp),registration_sha256=rh,
                math_gate_path=str(math_path),math_gate_sha256=D.sha(math_path),decode_gate_path=str(gate_path),decode_gate_sha256=D.sha(gate_path),
                training_manifest_path=str(tm),training_manifest_sha256=D.sha(tm),training_config_path=str(config_path),training_config_sha256=D.sha(config_path)))
        cp=stage/'update/checkpoint.pt'
        if (stage/'update/completion.json').exists():
            done=D.read(stage/'update/completion.json');cpr=D.read(stage/'update/checkpoint.json')
            if done['status']!='COMPLETE' or done['identity']['job_sha256']!=D.sha(update_job) or cpr['status']!='COMPLETE' or cpr['batch_index']!=b or cpr['opportunities_completed']!=b+1 or cpr['sha256']!=D.sha(cp):raise ValueError('Completed checkpoint identity mismatch')
            metrics=D.read(stage/'update/training_metrics.json')
            if metrics['status']!='COMPLETE':raise ValueError('Missing completed update metrics')
        else:
            fresh=admission(f'train_{b:03d}_actual_update',4096.)
            D.write(jobs/'admissions'/(stamp()+'.json'),dict(phase='update',resources=str(fresh)))
            for source_path,digest in math_gate['source_files'].items():
                if D.sha(source_path)!=digest:raise ValueError('Tested source changed during run')
            invoke('train_worker',update_job)
        assert cp.exists()
        prior=dict(path=str(cp),sha256=D.sha(cp))
        completed.append(dict(batch_index=b,condition_id=group['condition_id'],reward_sum=sum(r['reward'] for r in rewards),
            checkpoint_path=str(cp),checkpoint_sha256=prior['sha256'],reward_path=str(reward_path),
            checkpoint_receipt=str(stage/'update/checkpoint.json')))
        print(json.dumps(dict(event='BATCH_COMPLETE',batch_index=b,reward_sum=sum(r['reward'] for r in rewards),trajectories=8,
            active=0<sum(r['reward'] for r in rewards)<8)),flush=True)
    d0_service.close()
    D.write(OUT/'training/final.json',dict(status='COMPLETE',batches=len(completed),rollouts=len(completed)*8,
        checkpoint_path=prior['path'],checkpoint_sha256=prior['sha256'],records=completed,
        selection='Predeclared final batch; no best checkpoint',sigma=config['sigma'],training_config_sha256=D.sha(config_path)))
    print('COMPLETE finite on-policy training',flush=True)

if __name__=='__main__':run()
