"""CPU execution audit for the 96-output frozen TRAIN-C check; no model or review access."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .bootstrap import bind, OLD
D,E=bind()
from r2.pilots.terminal_reward_nft_v1 import resource_summary as R
VERSION='nft_training_c_frozen_execution_audit_v1'
LABELS=('BASE','NFT','SELF_FM')
D0_SHA='6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36'
EXPECTED=dict(requests=32,conditions=8,RAW_attempts=96,RAW_complete_events=96,RAW_receipts=96,RAW_verified=96,
 forward_starts=4800,forward_returns=4800,validation_forwards=0,D0_attempts=96,D0_receipts=96,D0_verified=96,
 POST_receipts=96,POST_verified=96,score_cells=192,score_verified=192,images=32,image_verified=32,
 training_update_attempts=0,training_updates=0,backward_events=0,optimizer_events=0,failure_files=0,
 dispatch_failures=0,controller_failures=0,GPU_Phase_attempts=4,GPU_Phase_completions=4,dispatch_attempts=5,dispatch_completions=5)

class Audit:
    def __init__(self):
        self.sources={};self.cache={};self.stamps={};self.violations=[];self.missing=[];self.actual=Counter();self.cells=[]
    def path(self,path):
        p=D.resolve(path).resolve()
        if 'private' in p.parts or p.name in ('review_a.json','review_b.json','reviews_actual.json','reviews_merge_receipt.json'):
            raise ValueError('Forbidden blind-review/private input: '+str(p))
        return p
    def check(self,ok,scope,reason,**detail):
        if not ok:self.violations.append(dict(scope=str(scope),reason=reason,**detail))
        return bool(ok)
    def require(self,ok,scope,reason):
        if not ok:raise ValueError(str(scope)+': '+reason)
    def absent(self,p):
        row=dict(path=str(p),reason='MISSING_REQUIRED_ARTIFACT')
        if row not in self.missing:self.missing.append(row)
    def hash(self,path,expected=None):
        p=self.path(path);key=str(p)
        if not p.is_file():self.absent(p);return None
        if key not in self.sources:
            before=p.stat();h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1<<20),b''):h.update(block)
            after=p.stat();self.stamps[key]=(after.st_size,after.st_mtime_ns)
            self.sources[key]=dict(sha256=h.hexdigest(),bytes=after.st_size)
            self.check((before.st_size,before.st_mtime_ns)==self.stamps[key],p,'Changed during read')
        digest=self.sources[key]['sha256']
        if expected is not None:self.check(digest==expected,p,'SHA256 mismatch',expected=expected,actual=digest)
        return digest
    def read(self,path,required=True):
        p=self.path(path);key=str(p)
        if not p.is_file():
            if required:self.absent(p)
            return None
        if key not in self.cache:
            self.hash(p);self.cache[key]=json.loads(p.read_text(encoding='utf-8-sig'))
        return self.cache[key]
    def events(self,path):
        p=self.path(path)
        if not p.is_file():self.absent(p);return []
        self.hash(p);raw=p.read_bytes()
        if raw and not raw.endswith(b'\n'):
            self.missing.append(dict(path=str(p),reason='PARTIAL_JSONL_TAIL'));raw=raw.rpartition(b'\n')[0]
        return [json.loads(row) for row in raw.splitlines() if row.strip()]
    def arrays(self,path):
        self.hash(path)
        with np.load(self.path(path),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
    def safe(self,scope,fn):
        try:return fn()
        except Exception as exc:self.violations.append(dict(scope=str(scope),reason='EXCEPTION',error=repr(exc)))
    def finish(self):
        for path,stamp in self.stamps.items():
            p=Path(path)
            if not p.is_file() or (p.stat().st_size,p.stat().st_mtime_ns)!=stamp:
                self.missing.append(dict(path=path,reason='CHANGED_AFTER_READ'))

def inputs(a):
    m=a.read(D.OUT/'inputs/manifest.json');a.require(m and m['status']=='FROZEN','inputs','Frozen manifest required')
    reg=a.read(D.OUT/'registration.json');a.hash(D.OUT/'inputs/manifest.json',reg['manifest_sha256'])
    old=a.read(m['source_manifest']);a.hash(m['source_manifest'],m['source_manifest_sha256'])
    lock=a.read(D.OUT/'execution_source_lock.json')
    for section in (lock['hashes'],m['protected_hashes']):
        for p,h in section.items():a.hash(p,h)
    contract=a.read(D.OUT/'evaluation/final_controller/contract.json')
    if contract:
        a.hash(D.OUT/'execution_source_lock.json',contract['execution_source_lock_sha256'])
        a.hash(D.OUT/'inputs/manifest.json',contract['manifest_sha256'])
        for key in ('source_input_sha256','code_sha256'):
            for p,h in contract[key].items():a.hash(p,h)
    oq=a.read(OLD/'inputs/training_queries.json');queries={q['condition_id']:q for q in m['evaluation_queries']}
    a.check(queries=={q['condition_id']:q for q in oq['queries']},'queries','Original eight query descriptors differ')
    for q in queries.values():a.hash(q['path'],q['sha256'])
    qs=m['evaluation_requests'];a.actual['requests']=len(qs);counts=Counter(q['condition_id'] for q in qs);a.actual['conditions']=len(counts)
    oc={q['condition_id']:q for q in old['train_conditions']}
    a.check(len({q['request_id'] for q in qs})==32 and set(counts)==set(oc) and set(counts.values())=={4},'inputs','Expected exact eight original C x four distinct requests')
    oldqs=old['training_requests']+old['evaluation_requests']
    for field in ('noise_seed','Gaussian_sha256','free_Gaussian_sha256'):
        values=[q[field] for q in qs];previous={q[field] for q in oldqs if field in q}
        a.check(len(set(values))==32 and not set(values)&previous,field,'Noise duplicate or previously registered')
    arrays={}
    for q in qs:
        rid=q['request_id'];a.hash(q['input_path'],q['input_sha256']);a.hash(q['condition_source'],q['condition_source_sha256'])
        b=a.arrays(q['input_path']);c=a.arrays(q['condition_source'])['C'].reshape(q['K'],3,3)
        a.require(D.same(b['C'],c) and D.array_sha(b['C'])==q['C_sha256'],rid,'Source C identity differs')
        a.require(D.same(b['z'],D.make_noise(q['noise_seed'],q['N'],q['K'])) and D.array_sha(b['z'])==q['Gaussian_sha256'],rid,'Gaussian differs from frozen MT19937')
        a.require(D.array_sha(b['z'][:,q['K']:])==q['free_Gaussian_sha256'],rid,'Free Gaussian differs')
        a.require((int(b['N']),int(b['K']))==(q['N'],q['K']) and q['split']=='TRAIN' and q['condition_source']==oc[q['condition_id']]['condition_source'],rid,'N/K/TRAIN/source differs')
        arrays[rid]=b
    return m,arrays,queries

def sampling_events(a,label,qs):
    path=D.OUT/'evaluation'/label/'sampling_events.jsonl';rows=a.events(path);counts=Counter(e['event'] for e in rows)
    for key,event in [('RAW_attempts','ROLLOUT_ATTEMPT'),('RAW_complete_events','ROLLOUT_COMPLETE'),('forward_starts','FORWARD_START'),('forward_returns','FORWARD_RETURN'),('validation_forwards','VALIDATION_FORWARD_START'),('training_update_attempts','UPDATE_START'),('training_updates','UPDATE_COMPLETE')]:a.actual[key]+=counts[event]
    a.actual['backward_events']+=sum(v for k,v in counts.items() if 'BACKWARD' in k.upper())
    a.actual['optimizer_events']+=sum(v for k,v in counts.items() if 'OPTIMIZER' in k.upper())
    expected=[q['request_id'] for q in qs]
    for event in ('ROLLOUT_ATTEMPT','ROLLOUT_COMPLETE'):
        a.check([e['context']['request_id'] for e in rows if e['event']==event]==expected,path,'Wrong request sequence: '+event)
    starts=[e for e in rows if e['event']=='FORWARD_START'];returns=[e for e in rows if e['event']=='FORWARD_RETURN']
    a.check([(e.get('request_id'),e.get('euler_step')) for e in starts]==[(q['request_id'],i) for q in qs for i in range(50)],path,'Euler step sequence differs')
    a.check([e['count'] for e in starts]==list(range(1,1601)) and [e['count'] for e in returns]==list(range(1,1601)),path,'Forward counters differ')
    active=None
    for e in rows:
        if e['event']=='FORWARD_START':a.check(active is None,path,'Unreturned previous forward');active=e['count']
        elif e['event']=='FORWARD_RETURN':a.check(active==e['count'],path,'Unmatched forward return');active=None
    a.check(active is None,path,'Last forward not returned')

def cell(a,label,q,b,query,model_identity):
    rid=q['request_id'];root=D.OUT/'evaluation'/label;k,n=q['K'],q['N'];scope=label+'/'+rid
    r=a.read(root/'raw'/rid/'receipt.json')
    if not r:return
    a.actual['RAW_receipts']+=1
    a.require((r['status'],r['label'],r['request_id'])==('COMPLETE',label,rid),scope,'RAW receipt identity')
    for key in ('input_sha256','C_sha256','Gaussian_sha256','noise_seed','N','K'):a.require(r[key]==q[key],scope,'Paired input differs: '+key)
    a.require(r['model_identity']==model_identity,scope,'Within-arm model identity differs')
    for key in ('raw','ply','result'):a.hash(r[key+'_path'],r[key+'_sha256'])
    z=a.arrays(r['raw_path']);raw=z['raw'].reshape(n,3,3);trajectory=z['trajectory'].reshape(51,n,3,3)
    a.require(raw.dtype==np.float32 and trajectory.dtype==np.float32 and np.isfinite(trajectory).all(),scope,'RAW dtype/finite')
    a.require(D.same(z['z'],b['z']) and D.same(z['C'],b['C']) and all(D.same(t[:k],b['C']) for t in trajectory),scope,'Actual Gaussian/C or51-state C differs')
    a.require(D.same(trajectory[-1],raw) and D.same(raw[:k],b['C']) and D.same(z['free'],raw[k:]),scope,'Final RAW/trajectory differs')
    a.require(D.same(trajectory[0,k:].reshape(1,n-k,9),b['z'][:,k:]),scope,'Initial free state differs')
    from meshflow_control.mesh_io import read_ply
    ply,_=read_ply(D.resolve(r['ply_path']));a.require(D.same(ply,raw),scope,'PLY lost RAW bits')
    runtime=a.read(r['result_path']);a.require(runtime['nfe']==50 and runtime['known_state_checks']==51,scope,'Euler50 runtime differs')
    a.actual['RAW_verified']+=1
    proposal_dir=root/'proposal'/rid;start=a.read(proposal_dir/'started.json');proposal=a.read(proposal_dir/'receipt.json')
    if start:
        a.actual['D0_attempts']+=1;a.require((start['event'],start['request_id'],start['raw_sha256'],start['checkpoint_sha256'])==('D0_FORWARD_STARTED',rid,r['raw_sha256'],D0_SHA),scope,'D0 start identity')
    if not proposal:return
    a.actual['D0_receipts']+=1
    a.require((proposal['status'],proposal['model_forwards'],proposal['checkpoint_sha256'],proposal['raw_sha256'])==('PASS',1,D0_SHA,r['raw_sha256']),scope,'Single official D0 identity')
    a.hash(proposal_dir/'proposal.npz',proposal['output_sha256']);a.actual['D0_verified']+=1
    post=a.read(root/'post'/rid/'receipt.json')
    if not post:return
    a.actual['POST_receipts']+=1
    a.require((post['status'],post['input_sha256'],post['model_calls'])==('COMPLETE',q['input_sha256'],0),scope,'POST identity/calls')
    for p,h in post['files'].items():a.hash(p,h)
    mesh=a.arrays(post['post_path']);source=E.source_metadata(q['condition_source'],raw[:k])
    a.require(D.same(mesh['output'][:k],source['C_original']),scope,'POST original source-oriented C differs')
    a.require(D.same(mesh['vertices'][mesh['faces']],mesh['output']) and post['result']['model_forwards_this_invocation']==0,scope,'POST topology/model call differs')
    a.actual['POST_verified']+=1
    for stage in ('RAW','POST'):
        score=a.read(root/'scores'/rid/stage/'score.json')
        if not score:continue
        a.actual['score_cells']+=1;i=score['identity']
        a.require((i['request_id'],i['condition_id'],i['label'],i['stage'],i['N'],i['K'])==(rid,q['condition_id'],label,stage,n,k),scope+'/'+stage,'Score identity')
        a.require(score['model_calls']==0 and score['new_queries']==0 and i['registration_sha256']==a.hash(D.OUT/'registration.json'),scope+'/'+stage,'Score calls/registration')
        a.require(i['input_hashes'].get(query['path'])==query['sha256'],scope+'/'+stage,'Wrong original query bank')
        for section in (i['input_hashes'],i['definition_hashes'],score['files']):
            for p,h in section.items():a.hash(p,h)
        a.actual['score_verified']+=1
    a.cells.append(dict(label=label,request_id=rid,C_all51_bitwise=True,Gaussian_matches_frozen=True,POST_original_C_bitwise=True,RAW_sha256=r['raw_sha256'],POST_sha256=post['post_sha256']))

def results(a,m,arrays,queries):
    E.package();qs=m['evaluation_requests']
    expected_ids={q['request_id'] for q in qs}
    for label in LABELS:
        root=D.OUT/'evaluation'/label
        for stage,filename in (('raw','attempt.json'),('raw','receipt.json'),('proposal','started.json'),('proposal','receipt.json'),('post','receipt.json')):
            found={p.parent.name for p in (root/stage).glob('*/'+filename)}
            a.check(found==expected_ids,root/stage,filename+' inventory differs from frozen32',missing=sorted(expected_ids-found),extra=sorted(found-expected_ids))
        found={(p.parent.parent.name,p.parent.name) for p in (root/'scores').glob('*/*/score.json')}
        a.check(found=={(rid,stage) for rid in expected_ids for stage in ('RAW','POST')},root/'scores','Score inventory differs from frozen64')
    for label in LABELS:
        a.safe(label+'/events',lambda label=label:sampling_events(a,label,qs))
        index=a.read(D.OUT/'evaluation'/label/'raw_index.json')
        if not index:continue
        a.require((index['status'],index['label'],index['actual_new_rollouts'],index['actual_sampling_forwards'])==('COMPLETE',label,32,1600),label,'RAW index budget')
        a.require([r['request_id'] for r in index['receipts']]==[q['request_id'] for q in qs],label,'RAW index request sequence')
        ident=index['model_identity'];a.require(ident['base_sha256']==m['source_binding']['sha256'],label,'Wrong base model')
        if label=='BASE':a.require(ident['checkpoint_path'] is None and ident['checkpoint_sha256'] is None,label,'BASE checkpoint must be absent')
        else:
            cp=OLD/f'checkpoints/{label}/round15.pt';side=a.read(cp.with_suffix('.json'));h=a.hash(cp,side['sha256'])
            a.require(D.resolve(ident['checkpoint_path']).resolve()==cp.resolve() and ident['checkpoint_sha256']==h and (side['step'],side['round_index'])==(256,15),label,'Not the final frozen adapter')
        for q in qs:a.safe(label+'/'+q['request_id'],lambda q=q,label=label,ident=ident:cell(a,label,q,arrays[q['request_id']],queries[q['condition_id']],ident))
    batch=a.read(D.OUT/'evaluation/denoise_all_worker/batch/completion.json')
    if batch:
        a.check((batch['jobs'],batch['new_D0_forwards'],batch['resident_models_loaded'],batch['training_updates'],batch['new_noise'])==(96,96,1,0,0),'D0 batch','Budget differs')
        a.check(len(batch['receipts'])==96 and {(r['label'],r['request_id']) for r in batch['receipts']}=={(label,rid) for label in LABELS for rid in expected_ids},'D0 batch','Receipt pairing differs')
    resident=a.read(D.OUT/'evaluation/denoise_all_worker/batch/resident_model.json')
    if resident:
        i=resident['identity'];a.check(i['status']=='PASS' and i['checkpoint_sha256']==D0_SHA and i['checkpoint_bytes']==7289979546 and not i['strict_missing_keys'] and not i['strict_unexpected_keys'],'D0 loader','Official strict-loader identity differs')
        a.check(D.resolve(i['checkpoint_path']).stat().st_size==7289979546,'D0 loader','Retained weight size changed')
    cpu=a.read(D.OUT/'evaluation/cpu/post_score_render_complete.json')
    if cpu:
        a.check((cpu['status'],cpu['requests'],cpu['required_new_POST'],cpu['distinct_requests'],cpu['model_calls'],cpu['torch_imported'])==('COMPLETE',96,96,32,0,False),'CPU','Budget/completion differs')
        for p,h in cpu['files'].items():a.hash(p,h)
    gallery=a.read(D.OUT/'evaluation/blind_gallery/manifest.json')
    if gallery:
        pages=gallery['images'];a.actual['images']=len(pages)
        a.check(gallery['status']=='COMPLETE' and len(pages)==32 and {r['request_id'] for r in pages}=={q['request_id'] for q in qs},'gallery','Public image requests differ')
        for row in pages:
            p=a.path(row['image_path']);a.require(p.is_relative_to((D.OUT/'evaluation/blind_gallery/images').resolve()),'gallery','Image outside public images')
            if a.hash(p,row['image_sha256'])==row['image_sha256']:a.actual['image_verified']+=1
    gate=a.read(D.OUT/'evaluation/review_gate.json')
    if gate:a.check((gate['status'],gate['new_RAW'],gate['new_sampling_forwards'],gate['new_D0_forwards'],gate['scientific_training_updates_in_final_stage'])==('WAITING_ACTUAL_ANONYMOUS_REVIEW',96,4800,96,0),'controller','Final review gate budget differs')
    return resident

def resources(a):
    workers=[];points=[];ri=R.Inputs()
    phases=[D.OUT/'evaluation'/label/'sampling_worker' for label in LABELS]+[D.OUT/'evaluation/denoise_all_worker']
    for p in phases:
        attempt=a.read(p/'attempt.json')
        if not attempt:continue
        a.actual['GPU_Phase_attempts']+=1;c=a.read(p/'completion.json')
        if c and c['status']=='COMPLETE':a.actual['GPU_Phase_completions']+=1
        found=a.safe(p,lambda p=p:R.phase_row(ri,p/'attempt.json'))
        if found:
            row,obs=found;workers.append(row);points.extend(obs);a.check(not row['monitor_errors'],p,'Resource monitor errors')
    control=D.OUT/'evaluation/final_controller/events.jsonl';events=a.events(control)
    a.actual['controller_failures']=sum(e['event']=='FINAL_STAGE_STOPPED' for e in events)
    launches={e['name']:e for e in events if e['event']=='LAUNCH'};exits={e['name']:e for e in events if e['event']=='EXIT'}
    a.check(len(launches)==sum(e['event']=='LAUNCH' for e in events),control,'Duplicate dispatch name')
    for i,e in enumerate(events):
        if e.get('resources'):points.append(R.point(e['resources'],control,i,'CONTROLLER_POLL'))
    dispatch=[]
    for p in sorted((D.OUT/'dispatch').glob('*/admission.json')):
        name=p.parent.name;start=a.read(p);end=a.read(p.parent/'exit.json');a.actual['dispatch_attempts']+=1
        points.append(R.point(start['resources'],p,0,'ADMISSION'));a.check(start['decision']['admitted'] is True,p,'Dispatch was not admitted')
        if end:
            a.actual['dispatch_completions']+=1;a.actual['dispatch_failures']+=int(end['returncode']!=0)
            points.append(R.point(end['resources'],p.parent/'exit.json',0,'COLLECTION'))
        dispatch.append(dict(name=name,command=start['command'],exit=end,launch_utc=launches.get(name,{}).get('utc'),exit_utc=exits.get(name,{}).get('utc')))
        a.check(name in launches and name in exits,p,'Missing launch/exit event')
    failed=set()
    for pattern in ('evaluation/*/raw/*/failure.json','evaluation/*/proposal/*/failure*.json','evaluation/*/sampling_worker/failure.json','evaluation/denoise_all_worker/failure.json','evaluation/denoise_all_worker/batch/failure*.json','evaluation/cpu/failure*.json'):failed.update(D.OUT.glob(pattern))
    a.actual['failure_files']=len(failed)
    for p in failed:a.read(p)
    a.check(not (D.OUT/'training').exists(),'training','Unexpected new training directory')
    ri.finish()
    for p,v in ri.sources.items():a.hash(p,v['sha256'])
    a.missing.extend(ri.notes);a.violations.extend(ri.errors)
    cpu=a.read(D.OUT/'evaluation/cpu/post_score_render_complete.json',False)
    return dict(global_resources=R.profile(points),workers=workers,groups=R.group_rows(workers),GPU_concurrency=R.overlap(workers),dispatch=dispatch,
        CPU_stage=dict(seconds=cpu.get('seconds'),pool_workers_recorded=cpu.get('workers'),CPU_threads_per_worker=cpu.get('CPU_threads_per_worker'),actual_pool_PID_peak=None) if cpu else None,
        failure_paths=[str(p) for p in sorted(failed)],
        limitations=['Whole-device sampled extrema include unrelated processes; short spikes can be missed.',
        'WDDM process VRAM N/A remains null; device deltas and CUDA counters are not process VRAM.',
        'CUDA allocated/reserved reset scopes are retained per actual rollout or D0 call.',
        'Worker lifetime concurrency is not proof of overlapping kernels; concurrent time is not an isolated benchmark.',
        'CPU pool count is configuration; continuous child-PID/RAM peak observations may be absent.',
        'Dispatch elapsed time includes parent collection delay; Phase and CPU clocks take precedence.'])

def run():
    if 'torch' in sys.modules:raise RuntimeError('Audit must remain model-free')
    a=Audit();a.hash(__file__);a.hash(R.__file__);setup=a.safe('inputs',lambda:inputs(a));resident=None
    if setup:resident=a.safe('results',lambda:results(a,*setup))
    resource=a.safe('resources',lambda:resources(a));a.finish();checks=[]
    for key,expected in EXPECTED.items():
        actual=a.actual[key];checks.append(dict(name=key,expected=expected,actual=actual,status='PASS' if actual==expected else 'FAIL'))
        if actual>expected:a.check(False,'budget/'+key,'Unexpected recorded work',expected=expected,actual=actual)
        elif actual!=expected:a.missing.append(dict(path='budget/'+key,reason='INCOMPLETE_BUDGET',expected=expected,actual=actual))
    status='FAIL' if a.violations else 'PARTIAL' if a.missing else 'COMPLETE'
    digest=hashlib.sha256(json.dumps(a.sources,sort_keys=True).encode()).hexdigest()[:20];out=D.OUT/'audit/snapshots'/digest
    summary=dict(version=VERSION,status=status,output_dir=str(out),snapshot_path=str(out),
        budget_checks=dict(status='PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL',expected=EXPECTED,actual=dict(a.actual),checks=checks),
        protected_inputs=dict(status='PASS' if not a.violations else 'FAIL',hash_scope='Each protected generator/adapter/source freshly hashed once, cached within audit; D0 7.29GB not rehashed, saved strict-loader identity+current file size checked'),
        violations=a.violations,partial_reasons=a.missing,code_sha256=a.hash(__file__),input_digest=digest,verified_cells=len(a.cells),source_files=len(a.sources),
        new_model_calls=0,new_training_updates=0,new_scores=0,torch_imported='torch' in sys.modules,private_mapping_read=False,blind_reviews_read=False,D0_loader_identity=resident,
        no_training_evidence_scope='Zero recorded update/backward/optimizer events, no training directory, exact generation/D0 counts and unchanged execution sources; not an independent GPU profiler')
    for name,value in [('summary.json',summary),('resources.json',resource),('cells.json',a.cells),('input_sources.json',a.sources)]:D.write(out/name,value)
    latest=D.checked(D.OUT/'audit/latest.json');latest.parent.mkdir(parents=True,exist_ok=True)
    pointer=dict(status=status,summary_path=str(out/'summary.json'),summary_sha256=D.sha(out/'summary.json'),output_dir=str(out),snapshot_path=str(out),code_sha256=summary['code_sha256'])
    temp=latest.with_suffix('.json.tmp');temp.write_text(json.dumps(pointer,indent=2,sort_keys=True)+'\n');temp.replace(latest)
    print(json.dumps(dict(status=status,output=str(out),actual=dict(a.actual),violations=a.violations,partial_reasons=a.missing),sort_keys=True))
    return summary

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
