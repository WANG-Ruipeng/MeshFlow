"""Actual calls counted before execution; single locked GPU owner."""
from collections import Counter
from pathlib import Path
import time
from .common import append_event, ledger_rows, atomic_json

LIMITS = dict(training_forward=27000, training_backward=27000, optimizer_update=3000,
              sampling=290, sampling_forward=14500, precheck_forward=24,
              precheck_backward=12, precheck_optimizer_update=1)
FORWARDS = ('training_forward', 'sampling_forward', 'precheck_forward')


class Budget:
    def __init__(self, out, stage, acquired_at=None):
        self.out=Path(out); self.stage=stage
        self.path=self.out/'ledger'/('calls_'+stage+'.jsonl')
        self.attempts=Counter();self.returns=Counter();self.failed=Counter();self.seconds=Counter();self.pending={}
        self.per_request=Counter()
        for path in sorted((self.out/'ledger').glob('calls_*.jsonl')):
            for row in ledger_rows(path):
                kind=row['kind'];cid=row['call_id']
                if row['event']=='CALL_BEGIN':
                    self.attempts[kind]+=1;self.pending[cid]=row
                    if kind=='sampling':
                        ctx=row['context'];self.per_request[(ctx['label'],ctx['request_id'])]=0
                    if kind=='sampling_forward':
                        ctx=row['context'];self.per_request[(ctx['label'],ctx['request_id'])]+=1
                elif row['event'] in ('CALL_RETURN','CALL_FAILED'):
                    self.pending.pop(cid,None)
                    (self.returns if row['event']=='CALL_RETURN' else self.failed)[kind]+=1
                    self.seconds[kind]+=row.get('wall_seconds',0)
        if self.pending:raise RuntimeError('Unresolved calls: require explicit engineering reconciliation, never blind replay')
        used=sum(x.get('seconds',0) for x in ledger_rows(self.out/'ledger/gpu_occupancy.jsonl') if x['event']=='GPU_RELEASE')
        self.gpu_deadline=(acquired_at or time.time())+max(0,14400-used)
        self.serial=sum(self.attempts.values())

    def ensure_stage(self, seconds=0):
        if time.time()+seconds>=self.gpu_deadline:raise RuntimeError('Four GPU-hour occupancy budget reached')

    def begin(self,kind,context=None,batch_size=1):
        self.ensure_stage()
        if kind not in LIMITS or self.attempts[kind]>=LIMITS[kind]:raise RuntimeError('Hard call budget: '+kind)
        if kind in FORWARDS and sum(self.attempts[k] for k in FORWARDS)>=41524:raise RuntimeError('Top-level forward budget')
        context=context or {}
        if context.get('split') in ('TRAIN_FEEDBACK','CONFIRM'):raise RuntimeError('Feedback refresh and CONFIRM generation forbidden')
        if kind=='sampling':
            self.per_request[(context['label'],context['request_id'])]=0
        if kind=='sampling_forward':
            key=(context['label'],context['request_id'])
            if self.per_request[key]>=50:raise RuntimeError('Euler50 per-output limit')
            self.per_request[key]+=1
        self.serial+=1;cid=self.stage+':'+str(self.serial)+':'+kind
        row=dict(event='CALL_BEGIN',kind=kind,call_id=cid,context=context,batch_size=batch_size)
        append_event(self.path,row);self.attempts[kind]+=1;self.pending[cid]=row
        return cid,time.perf_counter()

    def end(self,token,error=None):
        cid,tick=token;row=self.pending.pop(cid);kind=row['kind'];elapsed=time.perf_counter()-tick
        append_event(self.path,dict(event='CALL_RETURN' if error is None else 'CALL_FAILED',call_id=cid,
            kind=kind,context=row['context'],wall_seconds=elapsed,error=None if error is None else repr(error)))
        (self.returns if error is None else self.failed)[kind]+=1;self.seconds[kind]+=elapsed

    def call(self,kind,fn,*args,context=None,**kwargs):
        token=self.begin(kind,context)
        try:result=fn(*args,**kwargs)
        except BaseException as exc:self.end(token,exc);raise
        self.end(token);return result

    def snapshot(self):
        result=dict(attempts=dict(self.attempts),returns=dict(self.returns),failed=dict(self.failed),
            pending=list(self.pending),limits=LIMITS,top_level_forward_limit=41524,
            top_level_forward_attempts=sum(self.attempts[k] for k in FORWARDS),
            phase_call_seconds=dict(self.seconds),GPU_workers=1,feedback_refresh=0,
            note='Sampling timer includes nested forward calls. CPU scoring/rendering independent; precheck optimizer copy discarded.')
        atomic_json(self.out/'accounting.json',result);return result
