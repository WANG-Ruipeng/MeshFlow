"""Global durable budget. Every attempted real call consumes its fixed allowance."""
from collections import Counter
from pathlib import Path
import time
from .common import append_event,ledger_rows,atomic_json,read_json

LIMITS=dict(training_forward=32000,training_backward=32000,optimizer_update=4000,
            sampling_forward=20100,sampling=402,precheck_forward=96,precheck_backward=16)
FORWARD_KINDS=('training_forward','sampling_forward','precheck_forward')


class Budget:
    def __init__(self,out,stage):
        self.out=Path(out);self.stage=stage;self.path=self.out/'ledger'/f'{stage}.jsonl'
        self.attempts=Counter();self.returns=Counter();self.failed=Counter();self.seconds=Counter()
        self.pending={};self.arm_updates=Counter();self.per_output=Counter()
        for p in sorted((self.out/'ledger').glob('*.jsonl')):
            for r in ledger_rows(p):
                if r.get('event')=='CALL_BEGIN':
                    self.attempts[r['kind']]+=1;self.pending[r['call_id']]=r
                    if r['kind']=='optimizer_update':self.arm_updates[r['context']['arm']]+=1
                    if r['kind']=='sampling_forward':self.per_output[r['context']['request_id']]+=1
                elif r.get('event') in ('CALL_RETURN','CALL_FAILED'):
                    self.pending.pop(r['call_id'],None)
                    (self.returns if r['event']=='CALL_RETURN' else self.failed)[r['kind']]+=1
                    self.seconds[r['kind']]+=r.get('wall_seconds',0.)
        self.serial=sum(self.attempts.values())
        if self.pending:raise RuntimeError('Unresolved actual calls; inspect recovery before continuing: '+str(list(self.pending)))

    def begin(self,kind,context=None):
        if kind not in LIMITS or self.attempts[kind]>=LIMITS[kind]:raise RuntimeError('Forbidden/exhausted call '+kind)
        if kind in FORWARD_KINDS and sum(self.attempts[k] for k in FORWARD_KINDS)>=52196:raise RuntimeError('Total forward limit')
        context=dict(context or {})
        if kind=='optimizer_update':
            if self.arm_updates[context['arm']]>=1000:raise RuntimeError('Per-arm update limit')
            self.arm_updates[context['arm']]+=1
        if kind=='sampling_forward':
            rid=context['request_id']
            if self.per_output[rid]>=50:raise RuntimeError('Per-output Euler50 exceeded')
            self.per_output[rid]+=1
        self.serial+=1;cid=f'{self.stage}:{self.serial}:{kind}'
        r=dict(event='CALL_BEGIN',call_id=cid,kind=kind,stage=self.stage,context=context)
        append_event(self.path,r);self.attempts[kind]+=1;self.pending[cid]=r
        return cid,time.perf_counter()

    def end(self,token,error=None):
        cid,tick=token;r=self.pending.pop(cid);kind=r['kind'];sec=time.perf_counter()-tick
        append_event(self.path,dict(event='CALL_RETURN' if error is None else 'CALL_FAILED',call_id=cid,
            kind=kind,stage=self.stage,context=r['context'],wall_seconds=sec,error=None if error is None else repr(error)))
        (self.returns if error is None else self.failed)[kind]+=1;self.seconds[kind]+=sec

    def call(self,kind,fn,*args,context=None,**kwargs):
        token=self.begin(kind,context)
        try:value=fn(*args,**kwargs)
        except BaseException as exc:self.end(token,exc);raise
        self.end(token);return value

    def snapshot(self):
        value=dict(attempts=dict(self.attempts),returns=dict(self.returns),failed=dict(self.failed),
            pending=list(self.pending),total_model_forward_attempts=sum(self.attempts[k] for k in FORWARD_KINDS),
            batch_forward_attempts=sum(self.attempts[k] for k in FORWARD_KINDS),
            per_arm_optimizer_attempts=dict(self.arm_updates),wall_seconds_by_kind=dict(self.seconds),limits=LIMITS,
            total_forward_limit=52196,leaf_gradient_note='Calibration detached output leaf gradients recorded separately; not network backward',
            timing_note='Rollout and its forward timers overlap; never add them as independent wall time')
        atomic_json(self.out/'accounting.json',value);return value
