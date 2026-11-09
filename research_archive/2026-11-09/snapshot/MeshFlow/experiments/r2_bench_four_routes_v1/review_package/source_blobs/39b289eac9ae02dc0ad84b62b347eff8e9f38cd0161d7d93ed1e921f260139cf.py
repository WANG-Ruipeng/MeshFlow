"""Read completed ledgers and identities; zero model/scorer/selector calls."""
from __future__ import annotations
import collections,json
from datetime import datetime, timezone
from .common import *


def lines(path):
    return [json.loads(s) for s in resolve(path).read_text().splitlines() if s.strip()]


def run():
    from .run import all_raw_barrier,selected,new_selected
    registered();all_raw_barrier()
    result=read(OUT/'results/completion.json');verify(result['files']);verify(result['sources'])
    records=[];last_raw=0.;first_d0=None;ledger={}
    for q in new_requests():
        p=raw_dir(q)/'calls.jsonl';rows=lines(p)
        counts=collections.Counter(r['event'] for r in rows)
        if counts!={'FORWARD_STARTED':50,'FORWARD_COMPLETE':50}:raise ValueError('Generator actual call count')
        for event in ('FORWARD_STARTED','FORWARD_COMPLETE'):
            if [r['step'] for r in rows if r['event']==event]!=list(range(50)):raise ValueError('Euler step ledger')
        last_raw=max(last_raw,max(r['unix'] for r in rows));ledger[str(p)]=sha(p)
        records.append(dict(request_id=q['request_id'],new_generator_forwards=50,new_D0_forwards=0))
    winners=selected();new=new_selected()
    for q in new:
        p=post_dir(q)/'calls.jsonl';rows=lines(p);counts=collections.Counter(r['event'] for r in rows)
        if counts!={'D0_FORWARD_STARTED':1,'D0_FORWARD_RETURNED':1}:raise ValueError('Single D0 budget')
        start=datetime.fromisoformat(next(r['utc'] for r in rows if r['event']=='D0_FORWARD_STARTED')).timestamp()
        first_d0=start if first_d0 is None else min(first_d0,start)
        ledger[str(p)]=sha(p)
    if first_d0 is not None and first_d0<=last_raw:raise ValueError('D0 preceded completion of new RAW wave')
    candidate_rows=read(OUT/'selection/candidates.json');selection=read(OUT/'selection/complete.json')
    if len(candidate_rows)!=92 or len(winners)>23 or len(new)>23:raise ValueError('Frozen cardinality')
    if result['logical_rows']!=207 or result['numeric_fields']!=177:raise ValueError('Comparison table scope')
    actual_new_posts=list((OUT/'evaluation'/LABEL).glob('*/post/receipt.json'))
    if len(actual_new_posts)!=len(new):raise ValueError('Unselected D0 output exists')
    own_final=list((OUT/'cpu'/LABEL).glob('*/complete.json'))
    if len(own_final)!=len(new):raise ValueError('Unselected final CPU output exists')
    record=dict(status='COMPLETE',UTC=utc(),generator='NATURAL_HIGH_B_global22500',generator_sha256=GENERATOR_SHA,
        total_candidates=92,new_RAW=46,reused_RAW=46,new_generator_forwards=2300,new_D0_forwards=len(new),
        reused_winner_D0=len(winners)-len(new),selected=len(winners),failed_conditions=23-len(winners),
        original_RAW_scores_reused=46,new_RAW_scores=46,new_POST_score_rows=2*len(new),logical_comparison_rows=207,
        first_new_D0_UTC=None if first_d0 is None else datetime.fromtimestamp(first_d0,timezone.utc).isoformat(),
        last_new_generator_return_UTC=datetime.fromtimestamp(last_raw,timezone.utc).isoformat(),
        all_new_RAW_precede_any_new_D0=True,selector_failures=selection['failed_conditions'],
        interpretation='Inference budget plus fixed selection; no model-weight improvement claim',
        no_Y_no_visual_selection=True,files=ledger,results_sha256=sha(OUT/'results/completion.json'),
        visual_review='ROOT_UNIFIED_FIVE_AXES_SEPARATE',training_updates=0,head_calls=0,OT_calls=0)
    immutable(OUT/'audits/execution_closeout.json',record);return record


if __name__=='__main__':
    print(json.dumps({k:v for k,v in run().items() if k!='files'},indent=2))
