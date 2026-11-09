"""Read-only CPU final audit, outside the frozen experiment implementation.

No model is instantiated. Checkpoints are mapped to CPU solely to validate
serialized tensors/AdamW/RNG hashes. Missing local files are NA in partial mode;
--final exits successfully only when every required stage has passed.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ARMS=('H_CONT','MOMENT_XY')
LABELS=('START','H_CONT_END','MOMENT_XY_END')
START_SHA='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
LIMITS=dict(training_forward=16000,training_backward=16000,optimizer_update=2000,
    sampling=416,sampling_forward=20800,precheck_forward=160,precheck_backward=48)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def jhash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def ahash(value):
    a=np.ascontiguousarray(value);h=hashlib.sha256(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes());return h.hexdigest()


def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf8') as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(tmp,path)


class Audit:
    def __init__(self,out,final=False):
        self.out=Path(out).resolve();self.root=self.out.parents[2]
        self.chair=self.root/'native_t1/runs/chair_domain_control_v1';self.final=bool(final)
        self.dest=self.out/'independent_audit';self.failures=[];self.na=[];self.results={};self.hash_cache={}
        self.begin={};self.end={};self.exposures={};self.raw_rows=[]

    def require(self,test,message,**context):
        if not test:self.failures.append(dict(message=message,**context))
        return bool(test)

    def need(self,path):
        path=Path(path)
        if not path.is_file():self.na.append(dict(path=str(path),reason='MISSING_LOCAL_FILE'));return False
        return True

    def read(self,path):
        if not self.need(path):return None
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))

    def rows(self,path):
        if not self.need(path):return None
        result=[];lines=Path(path).read_text(encoding='utf-8-sig').splitlines()
        for index,line in enumerate(lines):
            if not line.strip():continue
            try:result.append(json.loads(line))
            except json.JSONDecodeError:
                if not self.final and index==len(lines)-1:self.na.append(dict(path=str(path),reason='ACTIVE_OR_TRUNCATED_FINAL_JSONL_LINE'));break
                raise
        return result

    def csv(self,path):
        if not self.need(path):return None
        with Path(path).open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))

    def hash(self,path):
        path=Path(path)
        if path not in self.hash_cache:self.hash_cache[path]=sha(path)
        return self.hash_cache[path]

    def complete(self,test,stage,detail):
        if test:return True
        if self.final:self.require(False,'Required final stage is incomplete',stage=stage,detail=detail)
        else:self.na.append(dict(stage=stage,reason='INCOMPLETE_STAGE',detail=detail))
        return False

    def stage(self,name,fn):
        before=len(self.failures);missing=len(self.na)
        try:value=fn()
        except Exception as exc:
            self.failures.append(dict(stage=name,message=repr(exc),traceback=traceback.format_exc()));value={}
        self.results[name]=dict(status='FAIL' if len(self.failures)>before else 'NA' if len(self.na)>missing else 'PASS',details=value)

    def ledger(self):
        files=sorted((self.out/'ledger').glob('*.jsonl'))
        if not files:self.na.append(dict(stage='ledger',reason='NO_LEDGER_FILES'));return {}
        bykind=collections.defaultdict(collections.Counter);byarm=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
        for path in files:
            for row in self.rows(path) or []:
                event=row.get('event')
                if event not in ('CALL_BEGIN','CALL_RETURN','CALL_FAILED'):continue
                cid=row['call_id'];kind=row['kind'];context=row.get('context',{})
                arm=context.get('arm',context.get('label','UNASSIGNED'));bucket=self.begin if event=='CALL_BEGIN' else self.end
                self.require(cid not in bucket,'Duplicate call reservation/terminal record',call_id=cid,event=event)
                bucket[cid]=row;bykind[kind][event]+=1;byarm[arm][kind][event]+=1
        pending=set(self.begin)-set(self.end);orphan=set(self.end)-set(self.begin)
        self.require(not orphan,'Terminal call without reservation',call_ids=sorted(orphan))
        if pending:self.complete(False,'ledger_pending',dict(call_ids=sorted(pending)))
        for cid in self.begin.keys()&self.end.keys():
            a,b=self.begin[cid],self.end[cid]
            self.require(a['kind']==b['kind'] and a.get('context',{})==b.get('context',{}),'Begin/end identity mismatch',call_id=cid)
        for kind,counts in bykind.items():
            self.require(kind in LIMITS,'Unregistered call kind; possible extra diagnostic/model work',kind=kind)
            if kind in LIMITS:self.require(counts['CALL_BEGIN']<=LIMITS[kind],'Hard call cap exceeded',kind=kind,count=counts['CALL_BEGIN'])
            self.require(counts['CALL_FAILED']==0,'Actual failed calls must not be hidden',kind=kind,count=counts['CALL_FAILED'])
        total=sum(bykind[k]['CALL_BEGIN'] for k in ('training_forward','sampling_forward','precheck_forward'))
        self.require(total<=36960,'Overall top-level forward cap exceeded',count=total)
        expected=dict(training_forward=16000,training_backward=16000,optimizer_update=2000,sampling=278,sampling_forward=13900)
        for kind,n in expected.items():self.complete(bykind[kind]['CALL_RETURN']==n,'final_call_count',dict(kind=kind,actual=bykind[kind]['CALL_RETURN'],expected=n))
        pre=self.read(self.out/'precheck/result.json')
        if pre:
            self.require(pre['status']=='PASS','Precheck gate not PASS')
            self.require(pre['actual_forwards']==bykind['precheck_forward']['CALL_RETURN'] and pre['actual_zero_update_backwards']==bykind['precheck_backward']['CALL_RETURN'],'Precheck result/ledger counts differ')
        return dict(by_kind={k:dict(v) for k,v in bykind.items()},by_arm_or_label={a:{k:dict(v) for k,v in d.items()} for a,d in byarm.items()},pending=sorted(pending),orphan=sorted(orphan),total_forward_attempts=total,no_unregistered_IP_call_kinds=not(set(bykind)-set(LIMITS)))

    def training(self):
        shared_rows=self.rows(self.out/'training/shared_batch_hashes.jsonl')
        shared={} if shared_rows is None else {int(r['moment_step']):r for r in shared_rows}
        if shared_rows is not None:self.require(len(shared)==len(shared_rows),'Duplicate shared batch registry step')
        self.complete(set(shared)==set(range(1,1001)),'shared_batches',dict(count=len(shared)))
        plan=self.read(self.out/'train_plan.json');protocol=self.read(self.out/'protocol.json')
        plan_rows={} if plan is None else {int(r['sample_index']):r for r in plan['records']}
        if plan:
            self.require(plan['seed']==20261006 and len(plan_rows)==8000,'Frozen training plan seed/count mismatch')
            if protocol:self.require(self.hash(self.out/'train_plan.json')==protocol['training_plan']['sha256'],'Frozen training plan file changed')
        summary={};expected_steps=set(range(1,1001));expected_samples=set(range(8000))
        for arm in ARMS:
            logs=self.csv(self.out/'training'/f'{arm}_training_log.csv');exposure=self.csv(self.out/'training'/f'{arm}_sample_exposure.csv')
            events=self.rows(self.out/'training'/f'{arm}_events.jsonl')
            if logs is None or exposure is None or events is None:continue
            logmap={int(r['moment_step']):r for r in logs};expmap={int(r['sample_index']):r for r in exposure};self.exposures[arm]=expmap
            self.require(len(logmap)==len(logs),'Duplicate training log step',arm=arm)
            self.require(len(expmap)==len(exposure),'Duplicate training sample exposure',arm=arm)
            self.complete(set(logmap)==expected_steps,'training_updates',dict(arm=arm,updates=len(logmap)))
            self.complete(set(expmap)==expected_samples,'training_samples',dict(arm=arm,samples=len(expmap)))
            for index,row in expmap.items():
                if index not in plan_rows:self.require(False,'Exposure has unregistered sample index',arm=arm,sample_index=index);continue
                planned=plan_rows[index]
                for field in ('uid','task_id'):
                    self.require(row[field]==planned[field],'Exposure differs from registered task identity',arm=arm,sample_index=index,field=field)
                for field in ('N','K','moment_step'):
                    self.require(int(row[field])==planned[field],'Exposure differs from registered integer field',arm=arm,sample_index=index,field=field)
                self.require(float(row['alpha'])==planned['alpha'],'Exposure alpha differs from registered FP32 alpha',arm=arm,sample_index=index)
            for kind in ('INPUT_BEGIN','INPUT_DONE','UPDATE_BEGIN','UPDATE_DONE'):
                ids=[int(r['moment_step']) for r in events if r.get('event')==kind]
                self.require(len(ids)==len(set(ids)),'Duplicate training event',arm=arm,event=kind)
                self.complete(set(ids)==expected_steps,'training_events',dict(arm=arm,event=kind,count=len(ids)))
            for step,log in logmap.items():
                indices=list(range((step-1)*8,step*8));rows=[expmap[i] for i in indices if i in expmap]
                self.require(len(rows)==8,'Logged update lacks eight exposure rows',arm=arm,step=step)
                if len(rows)!=8:continue
                self.require(all(int(r['moment_step'])==step and r['arm']==arm for r in rows),'Exposure cursor/arm mismatch',arm=arm,step=step)
                denominator=sum(int(r['free_coordinates']) for r in rows)
                self.require(denominator==int(log['free_coordinate_denominator']),'Effective FM denominator differs',arm=arm,step=step)
                self.require(all(int(r['free_coordinates'])==(int(r['N'])-int(r['K']))*9 for r in rows),'Invalid free coordinate count',arm=arm,step=step)
                for field,batchfield in (('shared_input_sha256','shared_batch_sha256'),('coupled_input_sha256','coupled_batch_sha256')):
                    h=hashlib.sha256(''.join(r[field] for r in rows).encode()).hexdigest()
                    self.require(h==log[batchfield],'Actual per-sample/batch hash mismatch',arm=arm,step=step,field=field)
                    if step in shared:self.require(h==shared[step][batchfield],'Shared registry batch mismatch',arm=arm,step=step,field=field)
                self.require(int(log['conditional_total_step'])==5000+step,'Wrong cumulative training step',arm=arm,step=step)
            for kind in ('training_forward','training_backward','optimizer_update'):
                calls=[r for r in self.begin.values() if r['kind']==kind and r.get('context',{}).get('arm')==arm]
                expected=1000 if kind=='optimizer_update' else 8000
                self.complete(len(calls)==expected,'arm_call_count',dict(arm=arm,kind=kind,count=len(calls),expected=expected))
                keys=[(r['context']['moment_step'],r['context'].get('microbatch')) for r in calls]
                self.require(len(keys)==len(set(keys)),'Duplicate training call identity',arm=arm,kind=kind)
                for r in calls:
                    c=r['context'];step=int(c['moment_step'])
                    if kind!='optimizer_update':
                        index=int(c['sample_index']);self.require(index==(step-1)*8+int(c['microbatch']),'Call microbatch/sample identity mismatch',arm=arm,call_id=r['call_id'])
                        if index in expmap:self.require(c['shared_input_sha256']==expmap[index]['shared_input_sha256'],'Model-call exposure hash differs',arm=arm,index=index)
            summary[arm]=dict(updates=len(logmap),sample_rows=len(expmap),events=len(events))
        if len(self.exposures)==2:
            a,b=(self.exposures[arm] for arm in ARMS);common=set(a)&set(b)
            for i in sorted(common):
                for field in ('shared_input_sha256','coupled_input_sha256','uid','task_id','N','K','alpha','t','free_coordinates','ot_cache_key'):
                    self.require(a[i][field]==b[i][field],'Two-arm paired actual input/target identity mismatch',sample_index=i,field=field)
            self.complete(common==expected_samples,'two_arm_pairing',dict(paired_samples=len(common)))
            summary['paired_samples']=len(common)
        return dict(arms=summary,shared_batches=len(shared),evidence_scope='Independently reconstruct actual per-sample and per-batch hashes from durable exposure/call records; training tensors themselves were not all persisted.')

    def checkpoints(self):
        # Import only the tensor library. No public loader, model constructor,
        # sampling, runtime CUDA context, or experiment update function is used.
        os.environ['CUDA_VISIBLE_DEVICES']=''
        import torch
        torch.set_num_threads(2);self.require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization in CPU audit')
        def tree_hash(value):
            h=hashlib.sha256()
            def visit(v):
                if torch.is_tensor(v):
                    t=v.detach().cpu().contiguous();h.update(str((str(t.dtype),tuple(t.shape))).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
                elif isinstance(v,dict):
                    for k in sorted(v,key=lambda k:str(k)):h.update(repr(k).encode());visit(v[k])
                elif isinstance(v,(list,tuple)):
                    h.update(type(v).__name__.encode())
                    for x in v:visit(x)
                else:h.update(repr(v).encode())
            visit(value);return h.hexdigest()
        def model_hash(state):
            h=hashlib.sha256()
            for key,value in sorted(state.items()):
                value=value.detach().cpu().contiguous();h.update(key.encode());h.update(str((tuple(value.shape),value.dtype)).encode());h.update(value.view(torch.uint8).numpy().tobytes())
            return h.hexdigest()
        manifest=self.read(self.out/'checkpoint_manifest.json');protocol=self.read(self.out/'protocol.json');rows=[]
        if manifest:
            self.require(len(manifest['endpoints'])==2,'Endpoint manifest must contain exactly two endpoints')
            start=manifest['start'];self.require(start['checkpoint_sha256']==START_SHA,'Manifest START identity differs')
            start_path=Path(start['path'])
            if self.need(start_path):self.require(self.hash(start_path)==START_SHA,'Original START checkpoint file hash changed')
        for arm in ARMS:
            path=self.out/'checkpoints'/f'{arm}_moment1000_total6000.pt';side=self.read(path.with_suffix('.json'))
            if side is None or not self.need(path):continue
            filehash=self.hash(path);self.require(filehash==side['checkpoint_sha256'],'Endpoint file hash differs',arm=arm)
            payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True);m=payload['metadata'];model=payload['model'];opt=payload['optimizer']
            self.require(payload['schema']=='chair_moment_channel_checkpoint_v2' and payload['update_in_progress'] is False,'Endpoint not a completed V2 payload',arm=arm)
            self.require(m==side['metadata'],'Endpoint metadata/sidecar mismatch',arm=arm)
            self.require((m['arm'],m['moment_step'],m['conditional_total_step'],m['completed_updates'])==(arm,1000,6000,1000),'Endpoint cursor mismatch',arm=arm)
            self.require(m['source_checkpoint_sha256']==START_SHA and m['full_recovery_state'] and not m['optimizer_reset'],'Endpoint origin/recovery contract mismatch',arm=arm)
            self.require(m['lambda_variance']==.25 and m['loss_recipe']=='fm' and m['context_encoder']=='geo','Endpoint scientific configuration mismatch',arm=arm)
            self.require(m['model_state_sha256']==model_hash(model),'Serialized model hash mismatch',arm=arm)
            self.require(m['optimizer_state_sha256']==tree_hash(opt),'Serialized full AdamW hash mismatch',arm=arm)
            self.require(m['rng_sha256']==tree_hash(payload['rng']),'Serialized RNG hash mismatch',arm=arm)
            self.require(m['stream_state_sha256']==jhash(payload['stream_state']),'Serialized stream hash mismatch',arm=arm)
            self.require(m['protocol_sha256']==jhash(payload['protocol'])==jhash(protocol),'Serialized/frozen protocol mismatch',arm=arm)
            ss=payload['stream_state'];self.require((ss['batch_index'],ss['sample_index'],ss['seed'],ss['arm'])==(1000,8000,20261006,arm),'Endpoint stream cursor differs',arm=arm)
            self.require(ss['plan_sha256']==protocol['training_plan']['sha256'],'Endpoint stream references a different training plan',arm=arm)
            groups=opt['param_groups'];names=payload['optimizer_parameter_names'];seen=set();seen_names=set();new_count=0
            self.require(len(groups)==len(names)==(1 if arm=='H_CONT' else 2),'Endpoint optimizer group count mismatch',arm=arm)
            for gi,(group,group_names) in enumerate(zip(groups,names)):
                self.require((group['lr'],tuple(group['betas']),group['weight_decay'])==(1e-5,(.9,.95),0.),'AdamW hyperparameters changed',arm=arm,group=gi)
                self.require(len(group['params'])==len(group_names),'AdamW parameter name map length differs',arm=arm,group=gi)
                for pid,name in zip(group['params'],group_names):
                    self.require(pid not in seen and name not in seen_names,'Duplicate optimizer ownership',arm=arm,name=name);seen.add(pid);seen_names.add(name)
                    state=opt['state'][pid];isnew=name.startswith('relation_modules.')
                    self.require(isnew==(gi==1),'New parameter mapped to wrong optimizer group',arm=arm,name=name)
                    self.require(int(state['step'])==(1000 if isnew else 6000),'AdamW step not fully continued',arm=arm,name=name)
                    self.require(model[name].dtype==torch.float32 and bool(torch.isfinite(model[name]).all()),'Invalid serialized parameter',arm=arm,name=name)
                    for key in ('exp_avg','exp_avg_sq'):
                        value=state[key];self.require(value.shape==model[name].shape and value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'Invalid serialized AdamW moment',arm=arm,name=name,state=key)
                    if isnew:new_count+=model[name].numel()
            self.require(seen==set(opt['state']),'Unmapped AdamW state',arm=arm)
            expected={f'relation_modules.layer_{l:02d}.{fc}.{p}' for l in (3,6,9,12) for fc in ('fc1','fc2') for p in ('weight','bias')} if arm=='MOMENT_XY' else set()
            self.require({n for n in seen_names if n.startswith('relation_modules.')}==expected,'Unexpected relation parameter namespace',arm=arm)
            self.require(new_count==(104496 if arm=='MOMENT_XY' else 0),'Unexpected new parameter count',arm=arm)
            if manifest:
                match=[r for r in manifest['endpoints'] if r['metadata']['arm']==arm]
                self.require(len(match)==1 and match[0]['checkpoint_sha256']==filehash,'Endpoint authoritative manifest differs',arm=arm)
            rows.append(dict(arm=arm,path=str(path),checkpoint_sha256=filehash,model_parameters_mapped=len(seen_names),new_parameters=new_count,full_AdamW_verified=True,moment_step=1000,total_step=6000))
            del payload,model,opt;gc.collect()
        self.require(not torch.cuda.is_initialized(),'CPU endpoint audit initialized CUDA')
        return dict(endpoints=rows,checkpoint_loading='torch.load(weights_only=True,mmap=True,map_location=cpu), no model instantiation')

    def generation(self):
        requests=self.read(self.out/'registration/evaluation_requests.json');records=self.read(self.out/'registration/evaluation_records.json')
        if requests is None or records is None:return {}
        self.require(len(requests)==432 and sum(r['status']=='READY' for r in requests)==414,'Registered request/ready count differs')
        self.require(sum(r['status']!='READY' for r in requests)==18,'Registered NA denominator differs')
        identities={(r['checkpoint_label'],r['case_id']) for r in requests};self.require(len(identities)==432,'Duplicate registered evaluation identity')
        base={r['case_id']:r for r in records};self.require(len(base)==144,'Original evaluation panel identity count differs')
        labels={};noise_groups=collections.defaultdict(dict);allraw=[]
        for label in LABELS:
            result=self.read(self.out/'evaluation'/label/'generated_rows.json')
            if result is None:continue
            rows=result['rows'];ids={r['case_id'] for r in rows}
            self.require(len(rows)==len(ids)==144 and ids==set(base),'Generated panel identity/denominator mismatch',label=label)
            counts=collections.Counter(r['status'] for r in rows);expected={'CACHED':138,'NOT_APPLICABLE':6} if label=='START' else {'SAMPLED':138,'NOT_APPLICABLE':6}
            self.require(dict(counts)==expected,'Generated status counts differ',label=label,counts=dict(counts))
            labels[label]=dict(counts)
            for row in rows:
                rec=base[row['case_id']]
                if rec['status']!='READY':self.require(row['status']=='NOT_APPLICABLE' and row.get('model_calls',0)==0,'NA request consumed model work',label=label,case_id=row['case_id']);continue
                allraw.append((label,row))
        replay=self.read(self.out/'initial_replay/result.json')
        registered_replays=self.read(self.out/'registration/initialization_replays.json')
        if replay:
            self.require(replay['status']=='PASS' and len(replay['rows'])==2,'Initial replay gate incomplete')
            if registered_replays:self.require({r['case_id'] for r in replay['rows']}=={r['case_id'] for r in registered_replays},'Initial replay identities changed')
            for row in replay['rows']:
                self.require(row['bitwise_equal'] and row['max_abs']==0,'Initial replay differs from START')
                allraw.append(('MOMENT_ZERO_REPLAY',row))
        raw_audit=[]
        for label,row in allraw:
            cid=row['case_id'];bankpath=Path(row['input_path']);bankpath=bankpath if bankpath.is_absolute() else self.chair/bankpath
            rawpath=Path(row['raw_path'])
            if not self.need(bankpath) or not self.need(rawpath):continue
            self.require(self.hash(bankpath)==row['input_sha256'],'Frozen bank file hash changed',label=label,case_id=cid)
            self.require(self.hash(rawpath)==row['raw_sha256'],'Raw output file hash changed',label=label,case_id=cid)
            with np.load(bankpath,allow_pickle=False) as bank,np.load(rawpath,allow_pickle=False) as z:
                raw=z['raw'];N,K=int(row['N']),int(row['K'])
                self.require(raw.shape==(N,9) and raw.dtype==np.float32 and np.isfinite(raw).all(),'Raw finite/N/dtype contract failed',label=label,case_id=cid)
                self.require(ahash(raw)==row['raw_array_sha256'],'Raw array hash differs',label=label,case_id=cid)
                for a,b in (('noise','z'),('C','C'),('known_mask','known_mask')):
                    self.require(same(z[a],bank[b]),'Raw input copy differs from original bank bytes',label=label,case_id=cid,array=a)
                self.require(same(raw[:K],bank['C'].reshape(-1,9)),'Final raw C bytes changed',label=label,case_id=cid)
                self.require(np.array_equal(bank['known_mask'].reshape(-1),np.arange(N)<K),'Original bank known slots differ',label=label,case_id=cid)
                noisehash=ahash(bank['z']);conditionhash=ahash(bank['C'].reshape(-1,9))
                if row.get('table')=='B':noise_groups[(label,row['uid'],row['task_id'],int(row['noise_index']))][str(row['alpha'])]=noisehash
                if row['status']=='CACHED':
                    cache=row.get('metadata',{})
                    self.require(cache.get('cache_identity_verified') is True and cache.get('new_model_calls')==0
                        and cache.get('original_nfe')==50 and cache.get('original_known_all51_bitwise') is True,
                        'START cache verification marker differs',label=label,case_id=cid)
                    meta=row.get('historical_metadata')
                else:meta=row.get('metadata')
                self.require(meta is not None,'Missing actual/historical sampler metadata',label=label,case_id=cid)
                if meta:
                    self.require(meta['nfe']==50 and meta['solver']=='original_Euler50' and meta['known_all51_bitwise'] and meta['known_state_checks']==51,'Euler/C all-step audit failed',label=label,case_id=cid)
                    self.require(meta['forward_dtype']=='bfloat16' and meta['integration_dtype']=='float32' and meta['geometry_repairs']==0 and not meta['GT_model_inputs'],'Sampler scientific contract differs',label=label,case_id=cid)
                    self.require(meta['noise_sha256']==noisehash and meta['condition_sha256']==conditionhash,'Sampler input hash differs',label=label,case_id=cid)
                if label=='MOMENT_ZERO_REPLAY':
                    historic=Path(row['historical_raw_path'])
                    if self.need(historic):
                        self.require(self.hash(historic)==row['historical_raw_sha256'],'Historical replay source changed',case_id=cid)
                        with np.load(historic,allow_pickle=False) as old:self.require(same(raw,old['raw']),'Initial replay raw bytes independently differ',case_id=cid)
            raw_audit.append(dict(label=label,case_id=cid,N=N,K=K,raw_sha256=row['raw_sha256'],noise_sha256=noisehash,condition_sha256=conditionhash))
        for group,values in noise_groups.items():self.require(len(values)==3 and len(set(values.values()))==1,'Original Gaussian bytes vary across alpha or curve missing',group=list(group),values=values)
        requests_by_id={(r.get('context',{}).get('label'),r.get('context',{}).get('case_id')):r for r in self.begin.values() if r['kind']=='sampling'}
        sampling_calls=[r for r in self.begin.values() if r['kind']=='sampling'];self.require(len(requests_by_id)==len(sampling_calls),'Duplicate actual full sampling request')
        for key,row in requests_by_id.items():
            forwards=[r for r in self.begin.values() if r['kind']=='sampling_forward' and (r['context'].get('label'),r['context'].get('case_id'))==key]
            self.require(len(forwards)==50 and {r['context']['euler_step'] for r in forwards}==set(range(50)),'Sampling NFE or Euler-step ledger mismatch',identity=key)
        self.complete(len(raw_audit)==416,'all_applicable_raw_plus_replays',dict(actual=len(raw_audit),expected=416))
        self.raw_rows=raw_audit
        expected_actual={(label,row['case_id']) for label,row in allraw if label!='START'}
        self.complete(set(requests_by_id)==expected_actual,'generated_vs_actual_sampling_identities',dict(actual=len(requests_by_id),generated=len(expected_actual)))
        return dict(requests=len(requests),ready=414,not_applicable=18,labels=labels,verified_raw_outputs=len(raw_audit),raw_rows=raw_audit,cross_alpha_noise_groups=len(noise_groups),expected_new_samples=276,expected_replays=2,expected_actual_sampling=278,expected_sampling_NFE=13900,
            C_all_steps_evidence='Raw final C bytes checked directly; all51 persistence is supported by stored actual sampler checks, since no trajectories were requested in V2.')

    def identities(self):
        before=self.read(self.out/'registration/source_hashes_before.json');after={}
        if before:
            for rel,want in before.items():
                path=self.root/rel
                if not self.need(path):continue
                actual=self.hash(path);after[rel]=actual;self.require(actual==want,'Protected source/data/report changed',path=rel)
            write(self.dest/'protected_hashes_after.json',after)
        frozen=self.read(self.out/'registration/training_implementation_hashes.json')
        if frozen:
            for rel,want in frozen.items():
                p=self.root/rel
                if self.need(p):self.require(self.hash(p)==want,'Frozen V2 implementation changed during training',path=rel)
        protocol=self.read(self.out/'protocol.json');hashpath=self.out/'protocol.sha256'
        if protocol is not None and self.need(hashpath):self.require(self.hash(self.out/'protocol.json')==hashpath.read_text().split()[0],'Frozen protocol file changed')
        diagnostic_paths=[str(p.relative_to(self.out)) for p in (self.out/'evaluation').glob('*/diagnostics/results.json')]
        self.require(not diagnostic_paths,'Unexpected I/P diagnostic artifact in V2',paths=diagnostic_paths)
        return dict(protected_files=len(after),frozen_V2_files=len(frozen or {}),computed_after_manifest=str(self.dest/'protected_hashes_after.json'),no_IP_results=not diagnostic_paths)

    def run(self):
        tick=time.perf_counter()
        for name,fn in (('ledger',self.ledger),('training',self.training),('checkpoints',self.checkpoints),('generation',self.generation),('identities',self.identities)):
            self.stage(name,fn);print('AUDIT_STAGE '+name+' '+self.results[name]['status'],flush=True)
        status='FAIL' if self.failures else 'INCOMPLETE' if self.na else 'PASS'
        result=dict(status=status,final_mode=self.final,stages=self.results,failures=self.failures,NA=self.na,
            seconds=time.perf_counter()-tick,audit_source_sha256=sha(__file__),no_model_instantiation=True,no_GPU_calls=True,
            output_root=str(self.out),source_hashes_verified=len(self.hash_cache))
        name='final_audit.json' if self.final else 'partial_audit.json';write(self.dest/name,result)
        lines=['# Independent V2 final audit','',f'Status: **{status}**. CPU read-only inspection; no model or GPU execution.','']
        lines.extend(f'- {name}: {value["status"]}' for name,value in self.results.items())
        lines+=['',f'Failures: {len(self.failures)}. Missing/incomplete items: {len(self.na)}. Detailed checks are in `{name}`.','']
        (self.dest/('final_audit.md' if self.final else 'partial_audit.md')).write_text('\n'.join(lines),encoding='utf8')
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--final',action='store_true')
    args=parser.parse_args();result=Audit(args.out,args.final).run()
    print(json.dumps(dict(status=result['status'],failures=len(result['failures']),NA=len(result['NA']))))
    if result['status']=='FAIL' or (args.final and result['status']!='PASS'):raise SystemExit(1)


if __name__=='__main__':main()
