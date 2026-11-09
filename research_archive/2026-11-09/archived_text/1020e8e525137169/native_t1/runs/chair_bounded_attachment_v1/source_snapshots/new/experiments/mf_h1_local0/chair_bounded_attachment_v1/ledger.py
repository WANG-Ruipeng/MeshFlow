"""Durable actual attempts, no hidden retry, zero training authorization."""
from collections import Counter,defaultdict
import time
from .common import *

LIMITS=dict(sampling=34,sampling_forward=1700,extra_forward=8,
    projection_endpoint=16,projection_trajectory=48,
    model_backward=0,optimizer_update=0,training_forward=0,training_backward=0)


class Budget:
    def __init__(self,out=OUT):
        self.out=Path(out);self.path=self.out/'ledger/calls.jsonl'
        self.begins={};self.ends={};self.started={};self.counts=Counter();self.per_output=Counter()
        for row in ledger_rows(self.path):
            token=row['call_id']
            if row['event']=='CALL_BEGIN':
                if token in self.begins:raise RuntimeError('Duplicate call id')
                self.begins[token]=row;self.counts[row['kind']]+=1
                if row['kind']=='sampling_forward':self.per_output[row['context']['request_id']]+=1
            elif row['event'] in ('CALL_RETURN','CALL_FAILED'):
                if token in self.ends:raise RuntimeError('Duplicate completion')
                self.ends[token]=row
        if set(self.begins)-set(self.ends):raise RuntimeError('Unresolved actual call; preserve and investigate, never auto-replay')
        if any(v['event']=='CALL_FAILED' for v in self.ends.values()):raise RuntimeError('Previous failed model/projection call requires explicit engineering audit before any retry')

    def begin(self,kind,context=None):
        context=dict(context or {})
        if kind not in LIMITS or self.counts[kind]>=LIMITS[kind]:raise RuntimeError('Budget exceeded: '+kind)
        if kind in ('sampling_forward','extra_forward') and self.counts['sampling_forward']+self.counts['extra_forward']>=1708:
            raise RuntimeError('Total forward limit exceeded')
        if kind=='projection_trajectory' or context.get('label')=='TRAJECTORY':
            if read_json(self.out/'phase1_gate.json')['status']!='PASS':raise RuntimeError('Phase 2 not authorized by frozen phase 1 gate')
        if kind=='sampling_forward' and self.per_output[context['request_id']]>=50:raise RuntimeError('Per output 50 NFE cap')
        token=str(len(self.begins)+1).zfill(6)
        row=dict(event='CALL_BEGIN',call_id=token,kind=kind,context=context)
        append_event(self.path,row);self.begins[token]=row;self.counts[kind]+=1;self.started[token]=time.perf_counter()
        if kind=='sampling_forward':self.per_output[context['request_id']]+=1
        return token

    def end(self,token,error=None):
        row=dict(event='CALL_FAILED' if error is not None else 'CALL_RETURN',call_id=token,
            kind=self.begins[token]['kind'],seconds=time.perf_counter()-self.started.pop(token),
            error=repr(error) if error is not None else None)
        append_event(self.path,row);self.ends[token]=row

    def call(self,kind,function,*args,context=None,**kwargs):
        token=self.begin(kind,context)
        try:result=function(*args,**kwargs)
        except BaseException as exc:self.end(token,exc);self.snapshot();raise
        self.end(token);return result

    def snapshot(self):
        failed=Counter(r['kind'] for r in self.ends.values() if r['event']=='CALL_FAILED')
        returned=Counter(r['kind'] for r in self.ends.values() if r['event']=='CALL_RETURN')
        value=dict(attempts=dict(self.counts),returns=dict(returned),failed=dict(failed),
            pending=sorted(set(self.begins)-set(self.ends)),limits=LIMITS,
            total_model_forwards=self.counts['sampling_forward']+self.counts['extra_forward'],
            model_backward=0,optimizer_updates=0,training_updates=0,
            per_output_NFE=dict(self.per_output),timing_note='Rollout contains model and project time; do not sum nested timers')
        atomic_json(self.out/'accounting.json',value);return value
