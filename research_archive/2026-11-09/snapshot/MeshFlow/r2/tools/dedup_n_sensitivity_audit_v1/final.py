"""Read-only independent budget, immutable identity and score-row audit."""
from __future__ import annotations
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import numpy as np
from legacy import ROOT, OUT, sha, save, table

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
    exp=OUT.parent;ev=exp/'evaluation';reg=read(exp/'registration.json')
    pf=read(OUT/'preflight.json')
    assert sha(exp/'registration.json')==pf['registration_sha256']
    assert sha(reg['gate_path'])==reg['gate_sha256']
    frozen=read(exp/'evaluation_freeze.json')
    assert frozen['registration_sha256']==pf['registration_sha256']
    protected={str(exp/'registration.json'):sha(exp/'registration.json'),reg['gate_path']:reg['gate_sha256']}
    checked={}
    def verify(path,h):
        path=str(path)
        if path not in checked:
            assert sha(path)==h,path
            checked[path]=h
        else:assert checked[path]==h,path
    for p,h in frozen['evaluation_code_hashes'].items():verify(p,h)
    for p,h in pf['protected_small_inputs'].items():verify(p,h)
    low=('ORIGINAL_LOW_N','DEDUP_LOW_N');high=('ORIGINAL_ORIG_N','DEDUP_ORIG_N')
    requests={q['request_id']:q for q in reg['requests']}
    original={q['request_id']:q for q in reg['original_requests']}
    rawrows=[];scores=[];rolltimes=[];rollstarts=[];envelopes={}
    for label in low:
        assert len(list((ev/'raw'/label).glob('*/receipt.json')))==24
        assert len(list((ev/'post'/label).glob('*/receipt.json')))==24
        arm_starts=[];arm_ends=[];d0_reservations=[];d0_receipt_writes=[]
        for rid,q in requests.items():
            rp=ev/'raw'/label/rid/'receipt.json';pr=ev/'post'/label/rid/'receipt.json'
            rr=read(rp);post=read(pr)
            assert rr['status']==post['status']=='COMPLETE'
            assert rr['label']==post['label']==label and rr['request_id']==post['request_id']==rid
            assert rr['N']==post['N']==q['N'] and rr['K']==post['K']==q['K']
            assert rr['new_rollouts']==1 and rr['new_sampling_F']==50
            assert post['new_D0_F']==1 and post['GEO_calls']==1
            assert rr['checkpoint_sha256']==reg['models'][label]['sha256']
            for key in ('input_sha256','C_sha256','Gaussian_sha256','condition_source_sha256'):
                assert rr[key]==post[key]==q[key]
            verify(rr['raw_path'],rr['raw_sha256']);verify(post['post_path'],post['post_sha256'])
            for path,h in post['files'].items():verify(path,h)
            a=arrays(rr['raw_path']);inp=arrays(q['input_path']);n,k=int(q['N']),int(q['K'])
            world=a['trajectory'].reshape(51,n,3,3);c=inp['C'].reshape(k,3,3)
            assert world.dtype==np.float32 and np.isfinite(world).all()
            assert all(x[:k].tobytes()==c.tobytes() for x in world)
            assert a['z'].dtype==inp['z'].dtype and a['z'].tobytes()==inp['z'].tobytes()
            assert world[-1].tobytes()==a['raw'].reshape(n,3,3).tobytes()
            assert world[0,k:].tobytes()==inp['z'].reshape(n,3,3)[k:].tobytes()
            calls=lines(rp.parent/'calls.jsonl');cs=Counter((x.get('event'),x.get('kind')) for x in calls)
            assert cs['RESERVE','rollout_attempt']==1
            assert cs['RESERVE','sampling_forward']==cs['RETURN','sampling_forward']==50
            assert not any(x.get('event') in ('FAIL','ERROR') for x in calls)
            complete=[x for x in calls if x.get('event')=='ROLLOUT_COMPLETE'];assert len(complete)==1
            assert complete[0]['nfe']==50 and complete[0]['known_all51_bitwise']
            rolltimes.append(complete[0]['time_unix'])
            start=[x['time_unix'] for x in calls if x.get('event')=='RESERVE' and x.get('kind')=='rollout_attempt']
            rollstarts += start;arm_starts+=start;arm_ends.append(complete[0]['time_unix'])
            proposal=read(post['proposal_receipt'])
            assert proposal['model_forwards']==1 and proposal['checkpoint_sha256']==reg['d0_checkpoint']['sha256']
            assert proposal['raw_sha256']==rr['raw_sha256']
            d0_reservations.append((pr.parent/'D0_started.json').stat().st_mtime)
            d0_receipt_writes.append(Path(post['proposal_receipt']).stat().st_mtime)
            rawrows.append(dict(label=label,request_id=rid,N=n,K=k,rollout_attempts=1,sampling_forwards=50,D0_forwards=1,GEO_calls=1,
                                C_all51=True,frozen_noise_exact=True,raw_receipt_sha256=sha(rp),post_receipt_sha256=sha(pr)))
        envelopes[label]=dict(rollout_first_reserve=min(arm_starts),rollout_last_complete=max(arm_ends),
            D0_first_reservation_file_mtime=min(d0_reservations),D0_last_proposal_receipt_mtime=max(d0_receipt_writes))
    events=[]
    for path in sorted((exp/'execution').glob('*events.jsonl')):events+=lines(path)
    ds=[x for x in events if x.get('event')=='D0_FORWARD_RESERVED']
    de=[x for x in events if x.get('event')=='D0_FORWARD_RETURNED']
    assert len(ds)==len(de)==48
    assert Counter(x['request_id'] for x in ds)==Counter({rid:2 for rid in requests})
    assert Counter(x['request_id'] for x in de)==Counter(x['request_id'] for x in ds)
    start_d0=min(datetime.fromisoformat(x['utc']).timestamp() for x in ds)
    assert max(rolltimes)<start_d0,'D0 began before final RAW completion'
    ledger=lines(exp/'ledger.jsonl');freeze=[x for x in ledger if x['event']=='REGISTRATION_FROZEN']
    assert len(freeze)==1 and freeze[0]['sha256']==pf['registration_sha256']
    assert freeze[0]['gate_sha256']==reg['gate_sha256']
    assert datetime.fromisoformat(freeze[0]['utc']).timestamp()<min(rollstarts)
    query_by={q['condition_id']:q for q in reg['queries']}
    for label in low+high:
        completion=read(ev/'score_completion'/(label+'.json'))
        assert completion['status']=='COMPLETE' and completion['model_calls']==0
        refs=completion['receipts'];assert len(refs)==48
        assert {(x['request_id'],x['stage']) for x in refs}=={(rid,s) for rid in requests for s in ('RAW','POST')}
        expected_new=48 if label in low else 0
        assert sum(x['new_quality_scans_this_call'] for x in refs)==expected_new
        for ref in refs:
            verify(ref['path'],ref['sha256']);score=read(ref['path'])
            q=(requests if label in low else original)[ref['request_id']]
            assert score['N']==q['N'] and score['K']==q['K']
            assert score['model_calls']==0 and score['C_query_count']==2048
            assert score['C_preserved_exact'] is True
            for key in ('input_sha256','C_sha256','Gaussian_sha256','condition_source_sha256'):
                assert score['identity'][key]==q[key]
            assert score['identity']['query_sha256']==query_by[q['condition_id']]['sha256']
            assert bool(ref['reused'])==(label in high)
            if label in high:
                reuse=read(ref['reuse_receipt_path'])
                assert reuse['new_quality_scans']==0 and reuse['model_calls']==0
                assert reuse['score_path']==ref['path'] and reuse['score_sha256']==ref['sha256']
            scores.append(dict(label=label,request_id=ref['request_id'],stage=ref['stage'],N=score['N'],K=score['K'],
                free_denominator=score['free_face_denominator'],C_query_count=score['C_query_count'],
                C_preserved_exact=True,new_quality_scans=int(ref['new_quality_scans_this_call']),reused=bool(ref['reused']),
                score_path=ref['path'],score_sha256=ref['sha256']))
    tablepath=ev/'tables/per_output.csv'
    with tablepath.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==192
    rows_by={(x['label'],x['request_id'],x['stage']):x for x in rows};assert len(rows_by)==192
    scalarchecks=0
    for x in scores:
        row=rows_by[x['label'],x['request_id'],x['stage']];s=read(x['score_path'])
        assert int(row['N'])==x['N'] and int(row['K'])==x['K']
        assert int(row['free_face_denominator'])==x['free_denominator']
        assert row['score_sha256']==x['score_sha256']
        if x['stage']=='RAW':assert x['free_denominator']==x['N']-x['K']
        metrics={**s['metrics'],**{k:v for k,v in s['topology'].items() if isinstance(v,(int,float,bool)) or v is None}}
        for key,value in metrics.items():
            if key not in row or value is None or isinstance(value,bool):continue
            if isinstance(value,(int,float)) and math.isfinite(value):
                assert math.isclose(float(row[key]),float(value),rel_tol=1e-12,abs_tol=1e-12),(x['label'],x['request_id'],key)
                scalarchecks+=1
    for p,h in pf['protected_small_inputs'].items():assert sha(p)==h,p
    for p,h in frozen['evaluation_code_hashes'].items():assert sha(p)==h,p
    table('execution_budget_per_request.csv',rawrows);table('score_identity_audit.csv',scores)
    result=dict(status='PASS',version='independent_n_sensitivity_execution_audit_v1',source_sha256=sha(__file__),
        registration_sha256=sha(exp/'registration.json'),evaluation_freeze_sha256=sha(exp/'evaluation_freeze.json'),
        B_gate_sha256=reg['gate_sha256'],B_gate_before_first_model_call=True,
        counts=dict(new_rollouts=48,new_sampling_forwards=2400,new_D0_forwards=48,new_GEO=48,
                    new_quality_scans=96,cached_quality_rows=96,total_score_rows=192,training_updates=0,backward=0),
        C_preserved_all_51_states=True,fixed_noise_exact=True,all_RAW_barrier=True,
        final_RAW_completion_unix=max(rolltimes),first_D0_reserve_unix=start_d0,
        execution_envelopes=envelopes,
        D0_bookkeeping_envelopes_nonoverlap=(envelopes[low[0]]['D0_last_proposal_receipt_mtime']<envelopes[low[1]]['D0_first_reservation_file_mtime'] or
            envelopes[low[1]]['D0_last_proposal_receipt_mtime']<envelopes[low[0]]['D0_first_reservation_file_mtime']),
        timing_scope='Rollout call-ledger intervals and D0 filesystem bookkeeping envelopes, not CUDA kernel timing; concurrent generator times are not isolated compute benchmarks.',
        C_query_count=2048,checked_scalar_table_values=scalarchecks,checked_file_hashes=checked,
        interpretation='Frozen identity and execution-accounting audit. No extra model, scoring, intersection, or geometry processing. Visual verdict and gate decision remain separately documented.')
    save('final.json',result)
    print(json.dumps({'status':'PASS','counts':result['counts'],'scalar_checks':scalarchecks,'output':str(OUT/'final.json')}))

if __name__=='__main__':main()
