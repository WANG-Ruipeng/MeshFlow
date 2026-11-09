"""Reserve before every actual call; failure and unknown returns consume budget."""
from collections import Counter
from pathlib import Path
import time
from .common import append_event, ledger_rows, atomic_json, read_json

LIMITS = dict(training_forward=24000, training_backward=24000, optimizer_update=3000,
              sampling_forward=28900, sampling=578, precheck_forward=96, precheck_backward=24)
FORWARD_KINDS = ('training_forward','sampling_forward','precheck_forward')


class Budget:
    def __init__(self, out, stage):
        self.out=Path(out); self.stage=stage; self.path=self.out/'ledger'/f'{stage}.jsonl'
        self.attempts=Counter(); self.returns=Counter(); self.failed=Counter()
        self.seconds=Counter(); self.pending={}
        for path in sorted((self.out/'ledger').glob('*.jsonl')):
            for r in ledger_rows(path):
                if r.get('event')=='CALL_BEGIN':
                    self.attempts[r['kind']]+=1; self.pending[r['call_id']]=r
                elif r.get('event') in ('CALL_RETURN','CALL_FAILED'):
                    self.pending.pop(r['call_id'],None)
                    (self.returns if r['event']=='CALL_RETURN' else self.failed)[r['kind']]+=1
                    self.seconds[r['kind']]+=r.get('wall_seconds',0.)
        self.serial=sum(self.attempts.values())
        if self.pending:
            raise RuntimeError('Unresolved reserved calls; no automatic replay: '+str(list(self.pending)))

    def begin(self, kind, context=None):
        if kind not in LIMITS: raise ValueError('Unknown budget kind '+kind)
        if self.attempts[kind]>=LIMITS[kind]: raise RuntimeError('Hard budget exhausted: '+kind)
        if kind in FORWARD_KINDS and sum(self.attempts[k] for k in FORWARD_KINDS)>=52996:
            raise RuntimeError('Hard overall forward budget exhausted')
        self.serial+=1; cid=f'{self.stage}:{self.serial}:{kind}'
        row=dict(event='CALL_BEGIN',call_id=cid,kind=kind,stage=self.stage,context=context or {})
        append_event(self.path,row); self.attempts[kind]+=1; self.pending[cid]=row
        return cid,time.perf_counter()

    def end(self, token, error=None):
        cid,tick=token; row=self.pending.pop(cid); kind=row['kind']; seconds=time.perf_counter()-tick
        event='CALL_RETURN' if error is None else 'CALL_FAILED'
        append_event(self.path,dict(event=event,call_id=cid,kind=kind,stage=self.stage,
                                   context=row['context'],wall_seconds=seconds,error=None if error is None else repr(error)))
        (self.returns if error is None else self.failed)[kind]+=1;self.seconds[kind]+=seconds

    def call(self, kind, fn, *args, context=None, **kwargs):
        token=self.begin(kind,context)
        try: result=fn(*args,**kwargs)
        except BaseException as exc:
            self.end(token,exc);raise
        self.end(token);return result

    def snapshot(self):
        value=dict(attempts=dict(self.attempts),returns=dict(self.returns),failed=dict(self.failed),
                   total_forward_attempts=sum(self.attempts[k] for k in FORWARD_KINDS),
                   wall_seconds_by_kind=dict(self.seconds),pending=list(self.pending),limits=LIMITS,
                   wall_time_note='CPU wall time around actual call; aggregate nested sampling and its forwards separately, never sum nested categories.')
        generation_rows=[]
        for path in (self.out/'evaluation').glob('*/generated_rows.json'):
            generation_rows.extend(read_json(path)['rows'])
        value['summary']=dict(optimizer_updates=self.returns['optimizer_update'],
            registered_unique_generation_requests=len(generation_rows),
            applicable_unique_generation_requests=sum(r['status']!='NOT_APPLICABLE' for r in generation_rows),
            actual_full_sampling_attempts=self.attempts['sampling'],actual_full_sampling_returns=self.returns['sampling'],
            historical_raw_cache_hits=sum(r['status']=='CACHED' for r in generation_rows),
            generation_not_applicable=sum(r['status']=='NOT_APPLICABLE' for r in generation_rows),
            precheck_forwards=self.attempts['precheck_forward'],precheck_zero_update_backwards=self.attempts['precheck_backward'],
            training_microbatch_forwards=self.attempts['training_forward'],training_backwards=self.attempts['training_backward'],
            sampling_forwards=self.attempts['sampling_forward'],total_model_forwards=sum(self.attempts[k] for k in FORWARD_KINDS),
            failed_actual_calls=sum(self.failed.values()),unresolved_calls=len(self.pending))
        requests=read_json(self.out/'registration/evaluation_requests.json') if (self.out/'registration/evaluation_requests.json').exists() else None
        if requests is not None:
            value['summary']['processed_generation_records']=len(generation_rows)
            value['summary']['processed_generation_not_applicable']=sum(r['status']=='NOT_APPLICABLE' for r in generation_rows)
            value['summary']['registered_unique_generation_requests']=len(requests)
            value['summary']['applicable_unique_generation_requests']=sum(r['status']=='READY' for r in requests)
            value['summary']['generation_not_applicable']=sum(r['status']!='READY' for r in requests)
        value['summary']['registered_generation_request_limit']=576
        value['summary']['scientific_optimizer_update_limit']=3000
        atomic_json(self.out/'accounting.json',value);return value
