"""Strict 32 rollout / 1600 actual-forward journal; no training kinds allowed."""
from collections import Counter
import time
from .common import *
LIMITS={'sampling':32,'sampling_forward':1600}
class Budget:
    def __init__(self,out=OUT,stage='half_sample'):
        self.out=Path(out);self.path=self.out/'ledger/calls.jsonl';self.stage=stage
        self.attempts=Counter();self.returns=Counter();self.failed=Counter();self.seconds=Counter();self.pending={};self.per=Counter()
        for r in ledger_rows(self.path):
            k=r['kind'];cid=r['call_id']
            if r['event']=='CALL_BEGIN':
                self.attempts[k]+=1;self.pending[cid]=r
                if k=='sampling_forward':self.per[r['context']['request_id']]+=1
            else:
                self.pending.pop(cid,None)
                (self.returns if r['event']=='CALL_RETURN' else self.failed)[k]+=1
                self.seconds[k]+=r.get('wall_seconds',0)
        if self.pending:raise RuntimeError('Unclosed calls: no automatic replay')
        self.serial=sum(self.attempts.values())
    def call(self,kind,fn,*args,context=None,**kwargs):
        if kind not in LIMITS or self.attempts[kind]>=LIMITS[kind]:raise RuntimeError('Hard budget exhausted or forbidden call: '+kind)
        ctx=dict(context or {});assert ctx.get('label')==LABEL and ctx.get('split') in ('OLD','FRESH')
        if kind=='sampling_forward':
            rid=ctx['request_id']
            if self.per[rid]>=50:raise RuntimeError('No more than 50 forwards per frozen request')
            self.per[rid]+=1
        self.serial+=1;cid=str(self.serial);row=dict(event='CALL_BEGIN',kind=kind,call_id=cid,context=ctx)
        append_event(self.path,row);self.pending[cid]=row;self.attempts[kind]+=1;tick=time.perf_counter()
        try:
            value=fn(*args,**kwargs)
        except BaseException as exc:
            elapsed=time.perf_counter()-tick;self.failed[kind]+=1
            append_event(self.path,dict(event='CALL_FAILED',kind=kind,call_id=cid,context=ctx,wall_seconds=elapsed,error=repr(exc)))
            self.pending.pop(cid);self.seconds[kind]+=elapsed;raise
        elapsed=time.perf_counter()-tick;self.returns[kind]+=1
        append_event(self.path,dict(event='CALL_RETURN',kind=kind,call_id=cid,context=ctx,wall_seconds=elapsed))
        self.pending.pop(cid);self.seconds[kind]+=elapsed;return value
    def snapshot(self):
        result=dict(attempts=dict(self.attempts),returns=dict(self.returns),failed=dict(self.failed),pending=list(self.pending),
            limits=LIMITS,per_request_forward_counts=dict(self.per),phase_call_seconds=dict(self.seconds),
            training_updates=0,backward_calls=0,optimizer_updates=0,baseline_new_generation=0,additional_model_calls=0)
        atomic_json(self.out/'accounting.json',result);return result
