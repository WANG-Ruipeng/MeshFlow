"""Read-only CPU accounting/learning curves for the bounded ABC experiment.

This entry never imports a trainer, model, torch, or a scorer. Training-state
fit is not rollout quality. C's independent-variable FM is labelled separately
from the original all-free-coordinate diagnostic partition.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[4]
ARMS=('A_COUPLE','B_POOL47','C_SHARED_TIP')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x] if path.exists() else []
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write_csv(path,table):
    keys=list(dict.fromkeys(k for r in table for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(table)
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def analyze(root,out,partial=False):
    root=Path(root).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Immutable analysis output already exists: '+str(out))
    out.mkdir(parents=True)
    folders={a:root/'training'/a for a in ARMS}
    folders.update(LONG=ROOT/'experiments/r2_seam_quality_four_v1/training/LONG',
                   SEAM200=ROOT/'experiments/r2_seam_slot_c200_v1/training/SEAM_SLOT_C200')
    audit=[];updates=[];partitions=[];headrows=[];pointerrows=[];exposure={};states={};callrows=[];weights=[];pointerwindows=[];headwindows=[]
    input_hashes={}
    for arm,folder in folders.items():
        if not (folder/'status.json').exists():
            audit.append(dict(arm=arm,check='status',status='PENDING'));continue
        states[arm]=read(folder/'status.json')
        log=rows(folder/'training_log.jsonl');geo=rows(folder/'geometry_diagnostics.jsonl')
        exposure[arm]=rows(folder/'exposure.jsonl');calls=rows(folder/'calls.jsonl')
        target=200 if arm=='SEAM200' else 800
        complete=states[arm]['status']=='COMPLETE'
        cursor=len(log)
        ok=[r['local_step'] for r in log]==list(range(1,cursor+1)) and (not complete or cursor==target)
        audit.append(dict(arm=arm,check='update_cursor',status='PASS' if ok else 'FAIL',actual=cursor,expected=target,run_status=states[arm]['status']))
        begins=Counter(r['kind'] for r in calls if r['event']=='CALL_BEGIN')
        dones=Counter(r['kind'] for r in calls if r['event']=='CALL_DONE')
        failures=[r for r in calls if r['event']=='CALL_FAILED']
        expected={k:cursor*8 for k in ('train_forward','train_backward','head_forward','head_backward')}
        expected.update(optimizer_update=cursor,head_optimizer_update=cursor)
        if arm=='C_SHARED_TIP':expected.update(pointer_forward=cursor*8,pointer_backward=cursor*8,pointer_optimizer_update=cursor)
        for kind in sorted(set(begins)|set(expected)):
            wanted=expected.get(kind,0 if kind=='preflight_forward' and arm!='SEAM200' else None)
            verdict='PASS' if begins[kind]==dones[kind] and (wanted is None or dones[kind]==wanted) else ('PENDING' if partial and not complete else 'FAIL')
            callrows.append(dict(arm=arm,kind=kind,attempted=begins[kind],successful=dones[kind],expected=wanted,status=verdict))
        audit.append(dict(arm=arm,check='failed_calls',status='PASS' if not failures else 'FAIL',actual=len(failures)))
        if arm in ARMS:
            reg=read(root/'training_inputs'/arm/'registration.json')
            inits=[read(p) for p in sorted((folder/'attempts').glob('*/initialization.json'))]
            initials=[r for r in inits if r['global_step']==12200]
            initial_ok=len(initials)==1 and all(r.get('exact_generator_optimizer_RNG_restoration') is True for r in inits)
            if initials:
                initial_ok=initial_ok and all(initials[0][key]==reg[expected] for key,expected in (
                    ('generator_state_sha256','source_generator_sha256'),('optimizer_state_sha256','source_optimizer_sha256'),
                    ('rng_sha256','source_rng_sha256'),('head_state_sha256','source_head_sha256')))
            audit.append(dict(arm=arm,check='exact_source_model_AdamW_RNG_head_restore',status='PASS' if initial_ok else 'FAIL',initializations=len(inits)))
            if complete:
                audit.append(dict(arm=arm,check='fixed_final_global_step',status='PASS' if states[arm]['global_step']==13000 else 'FAIL',actual=states[arm]['global_step']))
            registered=read(root/'training_inputs'/arm/'data_manifest.json')['steps']
            expected_exposure=[(r['step'],m,x['uid'],x['sample_index'],x['alpha']) for r in registered for m,x in enumerate(r['samples'][arm])]
            actual_exposure=[(r['local_step'],r['microbatch'],r['uid'],r['sample_index'],r['alpha']) for r in exposure[arm]]
            identity=actual_exposure==expected_exposure[:len(actual_exposure)] and (not complete or len(actual_exposure)==6400)
            audit.append(dict(arm=arm,check='registered_parent_condition_exposure',status='PASS' if identity else 'FAIL',samples=len(actual_exposure)))
        for r in log:
            updates.append(dict(arm=arm,seam_step=r['local_step']+(0 if arm=='SEAM200' else 200),
                                FM_domain='independent corner variables' if arm=='C_SHARED_TIP' else 'all nonanchor corner variables',
                                **{k:v for k,v in r.items() if not isinstance(v,(dict,list))}))
        for name,file,table in [('head','occupancy_training.jsonl',headrows),('pointer','pointer_training.jsonl',pointerrows)]:
            for r in rows(folder/file):table.append(dict(arm=arm,**{k:v for k,v in r.items() if not isinstance(v,(dict,list))}))
        # Always compare within the same t-bin; source endpoints already occur
        # in training interpolation inputs, so these are not free-rollout tests.
        windows=[('first50',1,min(50,cursor)),('last50',max(1,cursor-49),cursor)]
        if cursor>=200:windows.append(('all',1,cursor))
        for label,lo,hi in windows:
            hs=[r for r in headrows if r['arm']==arm and lo<=r['local_step']<=hi]
            if hs:
                totals={key:sum(r[key] for r in hs) for key in ('edge_count','positive_count','null_count','true_positive','true_negative','false_positive','false_negative')}
                headwindows.append(dict(arm=arm,window=label,local_step_lo=lo,local_step_hi=hi,**totals,
                    positive_recall=totals['true_positive']/totals['positive_count'] if totals['positive_count'] else None,
                    Null_accuracy=totals['true_negative']/totals['null_count'] if totals['null_count'] else None))
            ps=[r for r in pointerrows if r['arm']==arm and lo<=r['local_step']<=hi]
            if ps:
                fields=('query_count','source_nonnull','predicted_nonnull','correct_nonnull','missed_nonnull','false_positive_null','correct_null','same_face_conflict_rejections')
                totals={key:sum(r[key] for r in ps) for key in fields}
                totals['source_null']=totals['query_count']-totals['source_nonnull']
                totals['wrong_nonnull_owner']=totals['predicted_nonnull']-totals['correct_nonnull']-totals['false_positive_null']
                pointerwindows.append(dict(arm=arm,window=label,local_step_lo=lo,local_step_hi=hi,**totals,
                    exact_owner_nonnull_recall=totals['correct_nonnull']/totals['source_nonnull'] if totals['source_nonnull'] else None,
                    Null_accuracy=totals['correct_null']/totals['source_null'] if totals['source_null'] else None,
                    overall_accuracy=(totals['correct_nonnull']+totals['correct_null'])/totals['query_count'],
                    interpretation='teacher source occupancy candidates on training states; not free-rollout accuracy'))
            for bi in [-1,0,1,2,3,4]:
                subset=[r for r in geo if lo<=r['local_step']<=hi and (bi<0 or r['time_bin']==bi)]
                for partition in ('source_interface_third','source_interface_other_free','bulk'):
                    den=sum(r[partition+'_scalar_count'] for r in subset)
                    sse=sum(r[partition+'_velocity_sse'] for r in subset)
                    es=sum(r[partition+'_endpoint_sse'] for r in subset)
                    partitions.append(dict(arm=arm,window=label,local_step_lo=lo,local_step_hi=hi,
                        seam_step_lo=lo+(0 if arm=='SEAM200' else 200),seam_step_hi=hi+(0 if arm=='SEAM200' else 200),
                        time_bin='ALL' if bi<0 else bi,time_bin_lower=None if bi<0 else bi*.2,
                        time_bin_upper=None if bi<0 else (bi+1)*.2,partition=partition,scalar_count=den,
                        velocity_sse=sse,velocity_mse=sse/den if den else None,
                        endpoint_sse=es,endpoint_mse=es/den if den else None,microbatches=len(subset),
                        interpretation='training interpolation; original source-partition coordinates; not rollout'))
        for p in folder.glob('*.json*'):input_hashes[str(p)]=sha(p)
        if arm in ARMS:
            for p in (root/'checkpoints').glob(arm+'*.pt'):
                weights.append(dict(arm=arm,path=str(p),bytes=p.stat().st_size,kind='rolling' if '_rolling' in p.name else 'final',deleted=False))
    # A/C must replay old LONG sample identities; B follows its separately
    # frozen 47-parent balanced manifest. Coordinates differ only by mechanisms.
    keys=('local_step','microbatch','uid','sample_index','alpha','t','N','K','path')
    for arm in ('A_COUPLE','C_SHARED_TIP'):
        if arm not in exposure or 'LONG' not in exposure:continue
        actual=[tuple(r.get(k) for k in keys) for r in exposure[arm]]
        expected=[tuple(r.get(k) for k in keys) for r in exposure['LONG'][:len(actual)]]
        audit.append(dict(arm=arm,check='LONG_exposure_identity',status='PASS' if actual==expected else 'FAIL',samples=len(actual)))
    b=exposure.get('B_POOL47',[])
    if b:
        counts=Counter(r['uid'] for r in b)
        audit.append(dict(arm='B_POOL47',check='balanced_parent_exposure',status='PASS' if len(b)<6400 or len(counts)==47 and max(counts.values())-min(counts.values())<=2 else 'FAIL',
                          samples=len(b),parents=len(counts),minimum=min(counts.values()),maximum=max(counts.values())))
        write_csv(out/'B_parent_exposure.csv',[dict(parent_id=p,exposures=n) for p,n in sorted(counts.items())])
    headhashes={a:s.get('occupancy_export',{}).get('head_state_sha256') for a,s in states.items()}
    shared=[headhashes.get(a) for a in ('LONG','A_COUPLE','C_SHARED_TIP')]
    audit.append(dict(arm='LONG/A/C',check='final_occupancy_head_identity',status='PASS' if all(shared) and len(set(shared))==1 else ('PENDING' if partial or not all(shared) else 'FAIL'),hashes=shared))
    activity=[]
    for arm in ARMS:
        log=[r for r in updates if r['arm']==arm]
        if not log:continue
        for field in ('coupling','pointer_CE','pointer_ce','occupancy_BCE'):
            values=[r[field] for r in log if field in r]
            if values:activity.append(dict(arm=arm,field=field,updates=len(values),nonzero=sum(v!=0 for v in values),mean=sum(values)/len(values),maximum=max(values)))
    for file,table in [('updates.csv',updates),('partition_windows.csv',partitions),('occupancy_micro.csv',headrows),
                       ('pointer_micro.csv',pointerrows),('pointer_windows.csv',pointerwindows),('occupancy_windows.csv',headwindows),
                       ('calls.csv',callrows),('weights_inventory.csv',weights),('auxiliary_activity.csv',activity)]:write_csv(out/file,table)
    write_json(out/'audit.json',audit)
    status='PASS' if all(r['status']=='PASS' for r in audit+callrows) else ('PARTIAL' if partial and not any(r['status']=='FAIL' for r in audit+callrows) else 'FAIL')
    result=dict(status=status,partial=partial,arms=list(ARMS),audit_rows=len(audit),head_hashes=headhashes,
                input_hashes=input_hashes,code_sha256=sha(Path(__file__)),no_model_calls=True,no_scoring=True,
                notes=['C FM denominator differs; use original partition windows for matched-coordinate diagnostics.',
                       'A coupling and C pointer CE are separate objectives, never combined as an interchangeable loss.',
                       'Window samples are stochastic training states, not identical fixed-query generalization tests.',
                       'Weight inventory is read-only; no checkpoint was deleted.'])
    write_json(out/'completion.json',result);print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='experiments/r2_seam_abc_v1');p.add_argument('--output',required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();analyze(a.root,a.output,a.partial)
