"""Only fixed V2 endpoint generations and the two frozen initialization replays."""
from pathlib import Path
import numpy as np
from .common import *
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.evaluation import bank_for
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.sampling import sample_case

def one(out,where,record,label,model,budget,checkpoint_sha):
    cid=record['case_id'];path=where/'cases'/(cid+'.json')
    if path.exists():
        row=read_json(path)
        if row['status'] in ('RUNNING','FAILED'):raise RuntimeError('An attempted sampling cannot silently replay: '+cid)
        if row.get('raw_path'):assert digest(row['raw_path'])==row['raw_sha256']
        return row
    row=dict(record,label=label,checkpoint_sha256=checkpoint_sha)
    if record['status']!='READY':
        row.update(status='NOT_APPLICABLE',model_calls=0);atomic_json(path,row);return row
    bank=bank_for(record);atomic_json(path,dict(row,status='RUNNING'))
    token=budget.begin('sampling',dict(label=label,case_id=cid))
    try:
        result=sample_case(model,bank,False,budget,context=dict(label=label,case_id=cid))
    except BaseException as exc:
        budget.end(token,exc);atomic_json(path,dict(row,status='FAILED',error=repr(exc)));budget.snapshot();raise
    # CALL_RETURN describes successful model execution even if later file writing fails.
    budget.end(token)
    rawpath=where/'raw'/(cid+'.npz')
    sha=save_npz(rawpath,raw=result['raw'],noise=bank['z'],C=bank['C'],known_mask=bank['known_mask'])
    row.update(status='SAMPLED',raw_path=str(rawpath),raw_sha256=sha,raw_array_sha256=array_hash(result['raw']),model_calls=50,metadata=result['metadata'],saved_trajectory=False)
    atomic_json(path,row);return row

def initial_replay(out,model,budget):
    out=Path(out);where=out/'initial_replay';requests=read_json(out/'registration/initialization_replays.json')
    cache={r['case_id']:r for r in read_json(out/'evaluation/START/generated_rows.json')['rows']};rows=[]
    for rec in requests:
        row=one(out,where,rec,'MOMENT_ZERO_REPLAY',model,budget,START_SHA)
        source=cache[rec['case_id']];assert source['status']=='CACHED'
        with np.load(row['raw_path'],allow_pickle=False) as a,np.load(source['raw_path'],allow_pickle=False) as b:
            equal=a['raw'].dtype==b['raw'].dtype and a['raw'].shape==b['raw'].shape and a['raw'].tobytes()==b['raw'].tobytes()
            delta=float(np.abs(a['raw'].astype(np.float64)-b['raw']).max())
        rows.append(dict(row,bitwise_equal=equal,max_abs=delta,historical_raw_path=source['raw_path'],historical_raw_sha256=source['raw_sha256']))
        atomic_json(where/'result.json',dict(status='RUNNING',rows=rows,complete_replays=len(rows)))
        print('INITIAL_REPLAY '+rec['case_id']+' exact='+str(equal),flush=True)
    passed=len(rows)==2 and all(r['bitwise_equal'] for r in rows)
    result=dict(status='PASS' if passed else 'BLOCKED',rows=rows,complete_replays=2,sampling_NFE=100)
    atomic_json(where/'result.json',result);budget.snapshot()
    if not passed:raise RuntimeError('Initialization complete replay must be bitwise equal')
    return result

def run(out,label,model,budget,checkpoint_sha):
    assert label in ('H_CONT_END','MOMENT_XY_END')
    out=Path(out);where=out/'evaluation'/label;rows=[]
    for record in read_json(out/'registration/evaluation_records.json'):
        rows.append(one(out,where,record,label,model,budget,checkpoint_sha))
        if len(rows)%8==0:
            print(f'{label}: {len(rows)}/144 registered requests',flush=True);budget.snapshot()
    result=dict(status='COMPLETE',label=label,requested=144,rows=rows,
        sampled=sum(r['status']=='SAMPLED' for r in rows),cached=0,not_applicable=sum(r['status']=='NOT_APPLICABLE' for r in rows))
    assert result['sampled']==138 and result['not_applicable']==6
    atomic_json(where/'generated_rows.json',result);budget.snapshot();return result
