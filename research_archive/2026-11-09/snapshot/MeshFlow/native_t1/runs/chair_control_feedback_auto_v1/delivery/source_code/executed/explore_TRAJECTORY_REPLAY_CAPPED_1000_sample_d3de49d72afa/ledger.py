"""Durable bounded real-call accounting. Single GPU owner, CPU scores separate."""
from collections import Counter
from pathlib import Path
import time
from .common import *
LIMITS=dict(training_forward=100000,training_backward=100000,optimizer_update=6000,
    sampling=512,sampling_forward=25600,precheck_forward=64,precheck_backward=32)
FORWARDS=('training_forward','sampling_forward','precheck_forward')
class Budget:
    def __init__(self,out,stage):
        self.out=Path(out);self.stage=stage;self.path=self.out/'ledger'/('calls_'+stage+'.jsonl')
        self.attempts=Counter();self.returns=Counter();self.failed=Counter();self.seconds=Counter();self.pending={}
        self.feedback_samples=0;self.sample_equivalents=0
        for p in sorted((self.out/'ledger').glob('calls_*.jsonl')):
            for r in ledger_rows(p):
                if r['event']=='CALL_BEGIN':
                    self.attempts[r['kind']]+=1;self.pending[r['call_id']]=r
                    if r['kind'] in FORWARDS:self.sample_equivalents+=r.get('batch_size',1)
                    if r['kind']=='sampling' and r.get('context',{}).get('split')=='TRAIN_FEEDBACK':self.feedback_samples+=1
                elif r['event'] in ('CALL_RETURN','CALL_FAILED'):
                    self.pending.pop(r['call_id'],None)
                    (self.returns if r['event']=='CALL_RETURN' else self.failed)[r['kind']]+=1
                    self.seconds[r['kind']]+=r.get('wall_seconds',0)
        occupancy=sum(r.get('seconds',0) for r in ledger_rows(self.out/'ledger/gpu_occupancy.jsonl') if r['event']=='GPU_RELEASE')
        self.gpu_deadline=time.time()+max(0,8*3600-occupancy)
        self.serial=sum(self.attempts.values())
        if self.pending:raise RuntimeError('Unresolved calls require engineering reconciliation, not blind replay: '+str(self.pending))
    def begin(self,kind,context=None,batch_size=1):
        if time.time()>=self.gpu_deadline:raise RuntimeError('Eight GPU-hour occupancy ceiling reached')
        if kind not in LIMITS or self.attempts[kind]>=LIMITS[kind]:raise RuntimeError('Hard call budget: '+kind)
        if kind in FORWARDS and self.sample_equivalents+batch_size>160000:raise RuntimeError('160000 sample-equivalent forward budget')
        context=context or {}
        if kind=='sampling' and context.get('split')=='TRAIN_FEEDBACK' and self.feedback_samples>=128:raise RuntimeError('128 feedback rollout budget')
        self.serial+=1;cid=self.stage+':'+str(self.serial)+':'+kind
        row=dict(event='CALL_BEGIN',call_id=cid,kind=kind,context=context,batch_size=batch_size)
        append_event(self.path,row);self.pending[cid]=row;self.attempts[kind]+=1
        if kind in FORWARDS:self.sample_equivalents+=batch_size
        if kind=='sampling' and context.get('split')=='TRAIN_FEEDBACK':self.feedback_samples+=1
        return cid,time.perf_counter()
    def end(self,token,error=None):
        cid,t=token;row=self.pending.pop(cid);kind=row['kind'];seconds=time.perf_counter()-t
        append_event(self.path,dict(event='CALL_RETURN' if error is None else 'CALL_FAILED',call_id=cid,
            kind=kind,context=row['context'],wall_seconds=seconds,error=None if error is None else repr(error)))
        (self.returns if error is None else self.failed)[kind]+=1;self.seconds[kind]+=seconds
    def call(self,kind,fn,*args,context=None,**kwargs):
        token=self.begin(kind,context)
        try:result=fn(*args,**kwargs)
        except BaseException as e:self.end(token,e);raise
        self.end(token);return result
    def snapshot(self):
        result=dict(attempts=dict(self.attempts),returns=dict(self.returns),failed=dict(self.failed),
            pending=list(self.pending),sample_equivalent_forwards=self.sample_equivalents,
            batch_forward_calls=sum(self.attempts[x] for x in FORWARDS),train_feedback_rollouts=self.feedback_samples,
            phase_call_seconds=dict(self.seconds),limits=LIMITS,forward_equivalent_limit=160000,
            note='Sampling call seconds include its nested forwards; do not add them. All real model batches are size1. No gradient checkpointing used.')
        atomic_json(self.out/'accounting.json',result);return result
