"""Finite registered generations and trajectory diagnostics; no new inputs."""
from pathlib import Path
import numpy as np
from .common import *
from .registration import resolve,records
from .sampling import sample_case,validate_bank
from .diagnostics import select_diagnostic_curves,diagnose_pair


def bank_for(record):
    p=resolve(record['input_path'])
    if digest(p)!=record['input_sha256']:raise ValueError('Frozen generation input hash changed')
    with np.load(p,allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
    validate_bank(bank);return bank


def verified_old(record,oldrow):
    if oldrow['status']!='PASS' or oldrow['input_sha256']!=record['input_sha256']:
        raise ValueError('Historical raw has a different input or failure status')
    audit=oldrow['audit']
    if audit['solver']!='original_Euler50' or audit['nfe']!=50 or not audit['known_all51_bitwise']:
        raise ValueError('Historical sampler contract differs')
    if audit['forward_dtype']!='bfloat16' or audit['integration_dtype']!='float32':
        raise ValueError('Historical dtype contract differs')
    p=resolve(oldrow['raw_path'])
    if digest(p)!=oldrow['raw_sha256']:raise ValueError('Historical raw file changed')
    bank=bank_for(record)
    with np.load(p,allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
    for old,new in (('noise','z'),('C','C'),('known_mask','known_mask')):
        if raw[old].shape!=bank[new].shape or raw[old].dtype!=bank[new].dtype or raw[old].tobytes()!=bank[new].tobytes():
            raise ValueError('Historical cache tensor identity differs: '+old)
    if raw['raw'].shape!=(record['N'],9) or not np.isfinite(raw['raw']).all():raise ValueError('Invalid historical raw')
    if raw['raw'][:record['K']].tobytes()!=bank['C'].reshape(-1,9).tobytes():raise ValueError('Historical C changed')
    return p,raw,bank


def run(out,label,model,budget,model_hashes):
    out=Path(out);where=out/'evaluation'/label
    (where/'cases').mkdir(parents=True,exist_ok=True)
    panels=read_json(CHAIR/'panels.json');requests=records();diag=select_diagnostic_curves(panels)
    diag_ids={p[s+'_record']['case_id'] for p in diag for s in ('minus','plus')}
    oldledger=read_json(CHAIR/'evaluation/phase3000/Geo/ledger.json')
    assert oldledger['identity']['checkpoint_sha256']==START_SHA
    assert oldledger['identity']['panels_sha256']==digest(CHAIR/'panels.json')
    assert oldledger['identity']['protocol_sha256']==digest(CHAIR/'protocol.json')
    oldrows={r['case_id']:r for r in oldledger['rows']};rows=[]
    for record in requests:
        cid=record['case_id'];casepath=where/'cases'/(cid+'.json')
        if casepath.exists():
            row=read_json(casepath)
            if row['status'] in ('RUNNING','FAILED'):raise RuntimeError('Prior affected request cannot replay: '+cid)
            if row.get('raw_path') and digest(row['raw_path'])!=row['raw_sha256']:raise ValueError('Saved raw changed')
            rows.append(row);continue
        row=dict(record,label=label,checkpoint_sha256=model_hashes['checkpoint_sha256'])
        if record['status']!='READY':
            row.update(status='NOT_APPLICABLE',model_calls=0);atomic_json(casepath,row);rows.append(row);continue
        oldraw=bank=None
        if label=='START':
            p,oldraw,bank=verified_old(record,oldrows[cid])
            row.update(historical_raw_path=str(p),historical_raw_sha256=oldrows[cid]['raw_sha256'],
                       historical_score_path=str(resolve(oldrows[cid]['score_path'])),historical_score_sha256=oldrows[cid]['score_sha256'])
            if cid not in diag_ids:
                row.update(status='CACHED',raw_path=str(p),raw_sha256=oldrows[cid]['raw_sha256'],
                           raw_array_sha256=array_hash(oldraw['raw']),model_calls=0,metadata=oldrows[cid]['audit'])
                atomic_json(casepath,row);rows.append(row);continue
        if bank is None:bank=bank_for(record)
        atomic_json(casepath,dict(row,status='RUNNING'))
        token=budget.begin('sampling',dict(label=label,case_id=cid,reason='diagnostic_velocity_replay' if label=='START' else 'fixed_endpoint'))
        try:
            result=sample_case(model,bank,cid in diag_ids,budget,context=dict(label=label,case_id=cid))
            p=where/'raw'/(cid+'.npz')
            arrays=dict(raw=result['raw'],noise=bank['z'],C=bank['C'],known_mask=bank['known_mask'])
            if result['states'] is not None:arrays['states']=result['states']
            if result['velocities'] is not None:arrays['velocities']=result['velocities']
            sha=save_npz(p,**arrays)
            row.update(status='SAMPLED',raw_path=str(p),raw_sha256=sha,raw_array_sha256=array_hash(result['raw']),
                       model_calls=50,metadata=result['metadata'],saved_trajectory=cid in diag_ids)
            if oldraw is not None:
                row['replay_vs_historical']=dict(bitwise_equal=result['raw'].tobytes()==oldraw['raw'].tobytes(),
                    max_abs=float(np.max(np.abs(result['raw'].astype(np.float64)-oldraw['raw']))))
                if not row['replay_vs_historical']['bitwise_equal']:
                    # Numeric evidence is retained; no additional replay or selection is allowed.
                    row['historical_score_reuse_allowed']=False
            atomic_json(casepath,row);budget.end(token)
        except BaseException as exc:
            budget.end(token,exc)
            partial=getattr(exc,'partial_sampling',None)
            if partial:
                arr={k:v for k,v in partial.items() if isinstance(v,np.ndarray)}
                if arr:save_npz(where/'failed'/(cid+'.npz'),**arr)
            atomic_json(casepath,dict(row,status='FAILED',error=repr(exc)));budget.snapshot();raise
        rows.append(row)
        if len(rows)%16==0:
            print(f'{label}: {len(rows)}/144 registered generation rows complete',flush=True)
            budget.snapshot()
    atomic_json(where/'generated_rows.json',dict(label=label,rows=rows,status='COMPLETE',requested=144,
                sampled=sum(r['status']=='SAMPLED' for r in rows),cached=sum(r['status']=='CACHED' for r in rows),
                not_applicable=sum(r['status']=='NOT_APPLICABLE' for r in rows)))
    lookup={r['case_id']:r for r in rows};all_rows=[];summaries=[];fp32=[]
    for p in diag:
        key=f"{p['split']}_{p['uid']}_n{p['noise_index']}";target=where/'diagnostics'/f'{key}.json'
        if target.exists():
            d=read_json(target)
            if d['status']!='PASS':raise RuntimeError('Incomplete diagnostic cannot replay: '+key)
        else:
            pair=dict(p)
            for side in ('minus','plus'):
                rec=p[side+'_record'];pair[side+'_bank']=bank_for(rec)
                with np.load(lookup[rec['case_id']]['raw_path'],allow_pickle=False) as z:
                    pair[side+'_result']={k:z[k].copy() for k in ('raw','states','velocities')}
            atomic_json(target,dict(status='RUNNING',uid=p['uid'],noise_index=p['noise_index']))
            try:
                d=diagnose_pair(model,pair,budget,context=dict(label=label,pair=key))
                vectors=d.pop('vectors');vector_path=target.with_suffix('.npz')
                d.update(status='PASS',vector_path=str(vector_path),vector_sha256=save_npz(vector_path,**vectors))
                atomic_json(target,d)
            except BaseException as exc:
                atomic_json(target,dict(status='FAILED',error=repr(exc)));budget.snapshot();raise
        all_rows.extend(d['rows']);summaries.append(d['summary']);fp32.extend(d['fp32'])
        budget.snapshot();print(f'{label}: diagnostic pair {key} complete',flush=True)
    atomic_json(where/'diagnostics/results.json',dict(status='PASS',rows=all_rows,summaries=summaries,fp32=fp32))
    budget.snapshot()
    return dict(status='COMPLETE',requests=144,sampled=sum(r['status']=='SAMPLED' for r in rows),
                cached=sum(r['status']=='CACHED' for r in rows),not_applicable=6,diagnostic_pairs=len(summaries))
