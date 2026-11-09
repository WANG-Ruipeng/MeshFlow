"""Reserve revision calls against the original consumed + new hard budget."""
from collections import Counter
from pathlib import Path
import time
from .common import *

LIMITS={'precheck_forward':96,'precheck_backward':24}

def scan(folder):
    attempts=Counter();returns=Counter();failed=Counter();pending={};seconds=Counter()
    for path in sorted((folder/'ledger').glob('*.jsonl')):
        for r in ledger_rows(path):
            if r.get('event')=='CALL_BEGIN':attempts[r['kind']]+=1;pending[r['call_id']]=r
            elif r.get('event') in ('CALL_RETURN','CALL_FAILED'):
                assert r['call_id'] in pending;pending.pop(r['call_id'])
                (returns if r['event']=='CALL_RETURN' else failed)[r['kind']]+=1
                seconds[r['kind']]+=r.get('wall_seconds',0.)
    return attempts,returns,failed,pending,seconds

class Budget:
    def __init__(self,out,stage):
        self.out=Path(out);self.stage=stage;self.path=self.out/'ledger'/f'{stage}.jsonl'
        self.historical,self.historical_returns,self.historical_failed,oldpending,self.old_seconds=scan(OLD)
        self.attempts,self.returns,self.failed,self.pending,self.seconds=scan(self.out)
        assert self.historical['precheck_forward']==10 and self.historical['precheck_backward']==1
        if oldpending or self.pending:raise RuntimeError('Unresolved call reservation; no uncounted replay')
        self.serial=sum(self.attempts.values())

    def begin(self,kind,context=None):
        if kind not in LIMITS:raise ValueError('Only zero-update precheck calls authorized in this repair')
        if self.historical[kind]+self.attempts[kind]>=LIMITS[kind]:raise RuntimeError('Cumulative original hard budget exhausted: '+kind)
        self.serial+=1;cid=f'r1:{self.stage}:{self.serial}:{kind}'
        row=dict(event='CALL_BEGIN',call_id=cid,kind=kind,stage=self.stage,context=context or {})
        append_event(self.path,row);self.attempts[kind]+=1;self.pending[cid]=row
        return cid,time.perf_counter()

    def end(self,token,error=None):
        cid,tick=token;r=self.pending.pop(cid);seconds=time.perf_counter()-tick;kind=r['kind']
        append_event(self.path,dict(event='CALL_RETURN' if error is None else 'CALL_FAILED',call_id=cid,
            kind=kind,stage=self.stage,context=r['context'],wall_seconds=seconds,error=None if error is None else repr(error)))
        (self.returns if error is None else self.failed)[kind]+=1;self.seconds[kind]+=seconds

    def call(self,kind,fn,*args,context=None,**kwargs):
        token=self.begin(kind,context)
        try:value=fn(*args,**kwargs)
        except BaseException as exc:self.end(token,exc);raise
        self.end(token);return value

    def snapshot(self):
        totals=self.historical+self.attempts
        summary=dict(precheck_forwards=self.attempts['precheck_forward'],precheck_zero_update_backwards=self.attempts['precheck_backward'],
            cumulative_precheck_forwards=totals['precheck_forward'],cumulative_precheck_backwards=totals['precheck_backward'],
            optimizer_updates=0,training_forwards=0,new_samples=0,new_sampling_forwards=0,
            failed_actual_calls=sum(self.failed.values()),historical_failed_calls=sum(self.historical_failed.values()),pending=len(self.pending))
        value=dict(summary=summary,attempts=dict(self.attempts),returns=dict(self.returns),failed=dict(self.failed),
            historical_attempts=dict(self.historical),cumulative_attempts=dict(totals),wall_seconds_by_kind=dict(self.seconds),
            pending=list(self.pending),original_hard_limits=LIMITS,
            historical_note='Old incomplete cost consumes one backward but supplies no complete cost measurement; no reconstruction of missing timers')
        atomic_json(self.out/'accounting.json',value);return value
