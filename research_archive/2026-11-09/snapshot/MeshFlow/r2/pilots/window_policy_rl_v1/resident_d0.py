"""One official frozen D0 instance serving bounded, registered training batches.

stdin: one JSON object {"job": "/absolute/path/to/d0.json"} per line.
stdout: exactly one JSON completion/error per received job. Other output goes
to stderr. EOF releases the resident model. Any failure is fatal; no retry.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from contextlib import redirect_stdout
import datetime
import json
from pathlib import Path
import sys
import time
import traceback
import uuid
import numpy as np
from . import postprocess as P
D,E=P.bind()
from r2.pilots.terminal_reward_nft_v1 import common as W
from r2.pilots.terminal_reward_nft_v1.denoise_worker import retained_d0_binding

VERSION='window_policy_rl_resident_D0_service_v1'
MAX_BATCHES=64
MAX_REQUESTS=512


def file_stat(path):
    st=D.resolve(path).stat()
    return dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,
                device=st.st_dev,inode=st.st_ino)


class Service:
    def __init__(self,service_dir):
        self.directory=P._checked(service_dir)
        if self.directory.exists():raise RuntimeError('Service session already exists; choose a new session directory')
        self.model=None;self.model_identity=None;self.predict=None;self.torch=None
        self.loads_attempted=0;self.loads_completed=0;self.forward_attempts=0;self.forward_returns=0
        self.new_batches=set();self.poisoned=False;self.closed=False
        self.binding=retained_d0_binding();self.bound_stat=file_stat(self.binding['path'])
        self.gate_path=P.OUT/'analysis/decode_gate.json'
        self.training_path=P.OUT/'inputs/training_manifest.json'
        self.manifest_path=P.OUT/'inputs/manifest.json'
        self.registration_path=P.OUT/'registration.json'
        gate,training,manifest,registration=map(D.read,(self.gate_path,self.training_path,self.manifest_path,self.registration_path))
        if gate['status']!='PASS':raise ValueError('Resident D0 training service requires calibration PASS')
        if (training['status']!='FROZEN' or training['batches']!=MAX_BATCHES
                or len(training['training_requests'])!=MAX_REQUESTS or (training['B'],training['G'])!=(1,8)):
            raise ValueError('Service budget is exactly 64 opportunities, 512 registered requests')
        if (registration['manifest_sha256']!=D.sha(self.manifest_path)
                or training['evaluation_manifest_sha256']!=registration['manifest_sha256']):
            raise ValueError('Training/evaluation registration chain differs')
        self.sigma=float(gate['selected_sigma']);self.manifest=manifest;self.training=training
        self.groups={g['batch_index']:g for g in training['groups']}
        self.requests={q['request_id']:q for q in training['training_requests']}
        if set(self.groups)!=set(range(MAX_BATCHES)) or len(self.requests)!=MAX_REQUESTS:
            raise ValueError('Training batch/request identities are not unique and exhaustive')
        source_paths=[Path(__file__),Path(P.__file__),Path(E.__file__),D.ROOT/'r2/src/meshflow_control/denoise_model.py']
        self.protected={str(p):D.sha(p) for p in [self.gate_path,self.training_path,self.manifest_path,
            self.registration_path,*source_paths]}
        self.identity=dict(version=VERSION,pid=os.getpid(),service_dir=str(self.directory),
            checkpoint=self.binding,checkpoint_initial_stat=self.bound_stat,input_source_hashes=self.protected,
            maximum_batches=MAX_BATCHES,maximum_new_D0_forward_attempts=MAX_REQUESTS,
            load_policy='Lazy first authorized uncached batch; exactly one strict official load per service process',
            numerical_contract='Unmodified load_denoiser/predict_proposal, whole FP32[N,3,3], BF16 decoder/math SDPA',
            automatic_retry=False,parent_lifetime='Read stdin until EOF; close releases owned model/context at process exit')
        D.write(self.directory/'started.json',self.identity)
        self.events=(self.directory/'events.jsonl').open('x',encoding='utf8',buffering=1)
        self.event('SERVICE_READY_NO_MODEL_LOADED')

    def event(self,name,**fields):
        row=dict(event=name,utc=W.utc(),pid=os.getpid(),**fields)
        self.events.write(json.dumps(D.clean(row),ensure_ascii=False,allow_nan=False)+'\n')
        self.events.flush();os.fsync(self.events.fileno())

    def check_binding(self):
        if retained_d0_binding()!=self.binding or file_stat(self.binding['path'])!=self.bound_stat:
            raise ValueError('Official D0 file stat or protected binding changed; no silent reload')
        P._verify(self.protected)
        if self.model is not None:
            if self.model._meshflow_denoiser_identity!=self.model_identity:
                raise ValueError('Resident official model identity changed')
            if any(m.training for m in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()):
                raise ValueError('Resident D0 must remain entirely frozen/eval')

    def ensure_model(self):
        self.check_binding()
        if self.model is not None:return 0,0.
        if self.loads_attempted:raise RuntimeError('No load retry after an attempted load')
        from meshflow_control.denoise_model import load_denoiser,predict_proposal
        import torch
        torch.set_num_threads(1)
        self.torch=torch;self.predict=predict_proposal
        self.loads_attempted+=1;self.event('MODEL_LOAD_ATTEMPT',checkpoint=self.binding)
        tick=time.perf_counter()
        self.model,self.model_identity=load_denoiser(D.resolve(self.binding['path']),device='cuda',expected_sha256=self.binding['sha256'])
        elapsed=time.perf_counter()-tick
        self.loads_completed+=1;self.check_binding()
        D.write(self.directory/'resident_model.json',dict(status='PASS',identity=self.model_identity,
            load_seconds=elapsed,model_loads_attempted=1,model_loads_completed=1,
            checkpoint_full_sha_verification='Performed exactly by unchanged strict load_denoiser before deserialization',
            subsequent_batches='File stat and protected binding checked; existing frozen instance retained'))
        self.event('MODEL_LOAD_RETURN',seconds=elapsed,model_loads_completed=self.loads_completed)
        return 1,elapsed

    def prepare_job(self,path):
        if self.poisoned or self.closed:raise RuntimeError('Stopped resident service cannot accept another job')
        self.check_binding();path=D.resolve(path);spec=D.read(path)
        jobs=spec['jobs']
        if len(jobs)!=1 or jobs[0]['label']!='RL_TRAIN' or len(jobs[0].get('requests',[]))!=8:
            raise ValueError('D0 service accepts exactly one RL_TRAIN batch of eight')
        qs=jobs[0]['requests'];batch=qs[0]['batch_index']
        if type(batch) is not int or batch not in self.groups:
            raise ValueError('Batch outside frozen64 budget')
        if [q['request_id'] for q in qs]!=self.groups[batch]['requests']:
            raise ValueError('Registered request order differs')
        if any(q!=self.requests.get(q['request_id']) or q['batch_index']!=batch for q in qs):
            raise ValueError('D0 request differs from frozen training stream')
        stage=P.OUT/'training'/f'batch_{batch:03d}'
        if D.resolve(jobs[0]['stage_dir']).resolve()!=stage.resolve():raise ValueError('Batch output directory differs')
        ledger=P._checked(spec['ledger_dir'])
        if ledger!=(stage/'d0_ledger').resolve():raise ValueError('Exactly one owned D0 ledger per registered batch')
        if (D.resolve(spec['manifest_path']).resolve()!=self.manifest_path.resolve()
                or D.resolve(spec['registration_path']).resolve()!=self.registration_path.resolve()):
            raise ValueError('D0 job registration paths differ')
        if batch:
            prior=stage.parent/f'batch_{batch-1:03d}'/'update'
            if D.read(prior/'completion.json')['status']!='COMPLETE':raise ValueError('Previous policy update incomplete')
            checkpoint=D.read(prior/'checkpoint.json')
            if checkpoint['status']!='COMPLETE' or checkpoint['batch_index']!=batch-1:
                raise ValueError('Previous checkpoint receipt differs')
        else:checkpoint=None
        rows,identity=P.prepare_jobs(jobs,manifest_path=spec['manifest_path'],registration_path=spec['registration_path'])
        for row in rows:
            receipt=row['raw'];mi=receipt['model_identity']
            if (receipt['model_mode']!='current' or receipt['sigma']!=self.sigma
                    or receipt['registration_sha256']!=D.sha(self.registration_path)
                    or mi['base_file_sha256']!=self.manifest['source_binding']['sha256']):
                raise ValueError('RAW source/current policy/sigma differs')
            if batch==0 and (mi.get('fresh_zero_lora') is not True or mi.get('init_seed')!=2026101501):
                raise ValueError('First batch must use fresh registered zero-LoRA')
            if checkpoint is not None and mi.get('adapter_checkpoint_sha256')!=checkpoint['sha256']:
                raise ValueError('D0 RAW does not bind previous completed on-policy checkpoint')
        snapshot=D.resolve(spec['resource_snapshot_path'])
        if not snapshot.is_file():raise ValueError('Current controller admission snapshot is required')
        identity.update(phase='d0',orchestration_version=VERSION,orchestration_source_sha256=D.sha(__file__),
            batch_index=batch,job_path=str(path),job_sha256=D.sha(path),official_D0=self.binding,
            checkpoint_bound_stat=self.bound_stat,resource_snapshot_path=str(snapshot),resource_snapshot_sha256=D.sha(snapshot),
            decode_gate_sha256=D.sha(self.gate_path),training_manifest_sha256=D.sha(self.training_path),
            D0_per_new_request=1,source_hashes={p:h for p,h in self.protected.items() if p.endswith('.py')})
        return spec,rows,identity,ledger,batch

    def process_job(self,path):
        spec,rows,identity,ledger,batch=self.prepare_job(path)
        cached=P._completed(ledger,identity)
        if cached is not None:
            for row in rows:P._proposal(row)
            if (cached['requests']!=8 or cached['new_D0_forwards']!=8
                    or [r['request_id'] for r in cached['rows']]!=[r['request']['request_id'] for r in rows]):
                raise ValueError('Completed D0 cache coverage differs')
            self.event('BATCH_CACHE_VERIFIED_ZERO_FORWARD',batch_index=batch,ledger_dir=str(ledger))
            return dict(cached,new_D0_forwards_this_invocation=0,resident_models_loaded_this_invocation=0)
        if batch in self.new_batches or self.forward_attempts+len(rows)>MAX_REQUESTS:
            raise RuntimeError('Resident service batch/forward budget exceeded')
        if any(Path(row['proposal_dir']).exists() for row in rows):
            raise RuntimeError('Existing partial/completed proposal without this batch completion; no silent recovery')
        self.new_batches.add(batch)
        D.write(ledger/'started.json',identity)
        events=(ledger/'events.jsonl').open('x',encoding='utf8',buffering=1)
        def event(name,**fields):
            row=dict(event=name,utc=W.utc(),pid=os.getpid(),batch_index=batch,**fields)
            events.write(json.dumps(D.clean(row),allow_nan=False)+'\n');events.flush();os.fsync(events.fileno())
            self.event(name,batch_index=batch,**fields)
        before_attempts=self.forward_attempts;before_returns=self.forward_returns;tick=time.perf_counter()
        try:
            with W.Phase(ledger/'phase',identity):
                loaded,load_seconds=self.ensure_model()
                torch=self.torch;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                results=[];files={}
                for row in rows:
                    q=row['request'];rid=q['request_id'];proposal=Path(row['proposal_dir'])
                    raw=D.arrays(row['raw']['raw_path'])['raw'].reshape(q['N'],3,3)
                    E.source_metadata(q['condition_source'],raw[:q['K']])
                    if raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('D0 input must be original finite FP32[N,3,3]')
                    D.write(proposal/'started.json',dict(request_id=rid,raw_sha256=row['raw']['raw_sha256'],
                        checkpoint_sha256=self.binding['sha256'],event='D0_FORWARD_STARTED',automatic_retry=False,
                        service_dir=str(self.directory),batch_index=batch))
                    self.forward_attempts+=1
                    event('D0_FORWARD_ATTEMPT',request_id=rid,service_attempt=self.forward_attempts)
                    ft=time.perf_counter();output=self.predict(self.model,raw);torch.cuda.synchronize()
                    self.forward_returns+=1;seconds=time.perf_counter()-ft
                    event('D0_FORWARD_RETURN',request_id=rid,service_returns=self.forward_returns,seconds=seconds)
                    if output.dtype!=np.float32 or output.shape!=raw.shape or not np.isfinite(output).all():
                        raise ValueError('Invalid official D0 proposal')
                    pp=D.save_npz(proposal/'proposal.npz',official_unconstrained_proposal=output)
                    receipt=dict(status='PASS',request_id=rid,raw_sha256=row['raw']['raw_sha256'],
                        checkpoint_sha256=self.binding['sha256'],output_sha256=D.sha(pp),model_forwards=1,
                        source='OFFICIAL_D0_RESIDENT_TRAINING_SERVICE',seconds=seconds,
                        service_dir=str(self.directory),service_pid=os.getpid(),batch_index=batch,
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        peak_scope='Reset at start of actual eight-request batch after lazy load; running batch high-water mark with resident D0')
                    D.write(proposal/'receipt.json',receipt);files.update(P._proposal(row))
                    results.append(dict(request_id=rid,label=row['label'],proposal_path=str(pp),
                        proposal_receipt=str(proposal/'receipt.json'),proposal_sha256=receipt['output_sha256'],model_forwards=1))
                self.check_binding();P._verify(files)
                if self.forward_attempts-before_attempts!=8 or self.forward_returns-before_returns!=8:
                    raise RuntimeError('Batch must complete exactly eight D0 attempts/returns')
                result=dict(status='COMPLETE',version=VERSION,identity=identity,requests=8,rows=results,files=files,
                    new_D0_forwards=8,new_D0_forwards_this_invocation=8,model_calls_this_invocation=8,
                    model_forward_attempts=8,model_forward_returns=8,resident_models_loaded=loaded,
                    resident_models_loaded_this_invocation=loaded,load_seconds=load_seconds,
                    service_models_loaded_total=self.loads_completed,service_forward_attempts_total=self.forward_attempts,
                    service_forward_returns_total=self.forward_returns,service_dir=str(self.directory),service_pid=os.getpid(),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_scope='Whole actual batch after load, resident weights included; process/global memory separately in phase/resources.json',
                    seconds=time.perf_counter()-tick,post_status='NOT_RUN_CPU_STAGE_FOLLOWS',training_updates=0,new_noise=0)
                D.write(ledger/'result.json',result)
            D.write(ledger/'completion.json',result)
            event('BATCH_COMPLETE',requests=8,model_loads_this_batch=loaded)
            return result
        except BaseException as exc:
            self.poisoned=True
            failure=dict(status='STOPPED',error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,
                model_forward_attempts=self.forward_attempts-before_attempts,model_forward_returns=self.forward_returns-before_returns,
                service_dir=str(self.directory),batch_index=batch)
            D.write(ledger/'failure.json',failure);event('BATCH_FAILED',error=repr(exc),automatic_retry=False)
            raise
        finally:events.close()

    def close(self,*,failure=None):
        if self.closed:return
        self.closed=True
        self.event('SERVICE_STOPPING',failure=failure,model_loads_attempted=self.loads_attempted,
            model_loads_completed=self.loads_completed,forward_attempts=self.forward_attempts,forward_returns=self.forward_returns)
        self.model=None
        if self.torch is not None and self.torch.cuda.is_initialized():self.torch.cuda.empty_cache()
        summary=dict(status='STOPPED_ON_ERROR' if failure else 'CLOSED_EOF',failure=failure,
            model_loads_attempted=self.loads_attempted,model_loads_completed=self.loads_completed,
            forward_attempts=self.forward_attempts,forward_returns=self.forward_returns,
            batches_started=sorted(self.new_batches),automatic_retry=False,owned_model_released=True)
        D.write(self.directory/('failure.json' if failure else 'closed.json'),summary)
        self.events.close()


def serve(service_dir,*,input_stream=None,output_stream=None):
    input_stream=sys.stdin if input_stream is None else input_stream
    output_stream=sys.stdout if output_stream is None else output_stream
    service=None;failure=None
    try:
        with redirect_stdout(sys.stderr):service=Service(service_dir)
        for line in input_stream:
            job=None
            try:
                request=json.loads(line)
                if not isinstance(request,dict) or set(request)!={'job'} or not isinstance(request['job'],str):
                    raise ValueError('Expected exactly one JSON string field: job')
                job=request['job']
                with redirect_stdout(sys.stderr):result=service.process_job(job)
                response={k:result[k] for k in ('status','requests','new_D0_forwards_this_invocation',
                    'resident_models_loaded_this_invocation','seconds','cache_verified') if k in result}
                response.update(job=job,ledger_dir=str(D.read(job)['ledger_dir']),service_pid=os.getpid())
            except BaseException as exc:
                failure=repr(exc);service.poisoned=True
                with redirect_stdout(sys.stderr):service.event('SERVICE_JOB_ERROR',job=job,error=failure,traceback=traceback.format_exc())
                response=dict(status='ERROR',job=job,error=failure,automatic_retry=False,service_pid=os.getpid())
            output_stream.write(json.dumps(response,ensure_ascii=False,allow_nan=False)+'\n');output_stream.flush()
            if failure:return 1
        return 0
    finally:
        if service is not None:
            with redirect_stdout(sys.stderr):service.close(failure=failure)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service-dir',default=None)
    args=parser.parse_args()
    tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+str(os.getpid())+'_'+uuid.uuid4().hex[:8]
    directory=args.service_dir or P.OUT/'training/resident_d0_sessions'/tag
    raise SystemExit(serve(directory))
