"""Independent final artifact audit. NumPy/stdlib only; no model/scorer/OT calls."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
CACHE={}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    p=Path(p)
    if p.suffix=='.pt':raise RuntimeError('No repeated large checkpoint hashing')
    if p not in CACHE:CACHE[p]=hashlib.sha256(p.read_bytes()).hexdigest()
    return CACHE[p]
def hashed(p,h):
    assert sha(p)==h,('HASH_MISMATCH',str(p))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert np.isclose(a,b,atol=1e-12,rtol=1e-10),(a,b)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def jlines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def aggregate(rs,keys,metrics):
    groups=defaultdict(list)
    for r in rs:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,group in sorted(groups.items()):
        r=dict(zip(keys,key))
        for m in metrics:
            values=[float(g[m]) for g in group if g.get(m) not in ('',None)]
            r[m]=statistics.fmean(values) if values else None
        out.append(r)
    return out
def run():
    started=time.perf_counter();reg=read(OUT/'registration.json')
    package_path=OUT/'audits/source_contract/probe_manifest.json';pkg=read(package_path)
    contract=read(OUT/'audits/source_contract/manifest.json')
    analysis=read(OUT/'probe_analysis_v1/COMPLETE.json')
    trajectory=read(OUT/'trajectory/completion.json')
    for p,h in reg['protected'].items():hashed(p,h)
    for x in contract['source_inventory']:hashed(x['path'],x['sha256'])
    hashed(pkg['registration_path'],pkg['registration_sha256'])
    hashed(pkg['pair_preparation_module'],pkg['pair_preparation_module_sha256'])
    assert Counter(r['representation'] for r in pkg['rows'])=={'NATIVE':12,'SEAM':4}
    for r in pkg['rows']:
        for key,hkey in [('trace_path','trace_sha256'),('case_path','case_sha256'),('noise_npz','noise_npz_sha256'),('pair_path','pair_sha256')]:hashed(r[key],r[hkey])
        a=read(r['audit_json']);assert a['status']=='PASS' and all(a['checks'].values())
        assert a['production_OT_calls']==1 and a['model_forwards']==a['backwards']==0
    ot=jlines(OUT/'audits/source_contract/ledger.jsonl')
    assert sum(x['event']=='OT_STARTED' for x in ot)==sum(x['event']=='OT_COMPLETED' for x in ot)==16
    assert not any('FAIL' in x['event'] for x in ot)
    assert contract['new_noise_draws']==0 and contract['new_epsilon_arrays']==12
    source=ROOT/'r2/pilots/bench_flow_integrity_v1/probe.py';source_text=source.read_text()
    assert 'assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())' in source_text
    assert 'assert state_hash(model.state_dict())==frozen_hash' in source_text
    assert 'with torch.no_grad()' in source_text
    summaries=[];total=0;prediction_files=0
    for label,spec in reg['models'].items():
        directory=OUT/'probes'/label;attempt=read(directory/'attempt.json');done=read(directory/'completion.json')
        assert not (directory/'failure.json').exists()
        assert attempt['weights']==spec and done['status']=='COMPLETE'
        hashed(source,attempt['probe_code_sha256']);hashed(package_path,done['manifest_sha256'])
        assert attempt['manifest_sha256']==done['manifest_sha256']
        side=Path(spec['path']).with_suffix('.json');hashed(side,spec['sidecar_sha256'])
        metadata=read(side);assert metadata['sha256']==spec['sha256']
        assert Path(spec['path']).stat().st_size==spec['bytes']
        if 'generator_state_sha256' in metadata:assert metadata['generator_state_sha256']==done['frozen_tensor_sha256']
        selected=[r for r in pkg['rows'] if r['representation']==('SEAM' if label=='SEAM_S3000' else 'NATIVE')]
        calls=jlines(directory/'calls.jsonl')
        assert len(calls)==spec['probe_forwards']*2
        for index,entry in enumerate((r,ti,t) for r in selected for ti,t in enumerate(pkg['times'])):
            r,ti,t=entry;start,finish=calls[2*index:2*index+2]
            assert start['type']=='FORWARD_STARTED' and finish['type']=='FORWARD_COMPLETE'
            for event in (start,finish):
                assert event['request_id']==r['request_id'] and event['time_index']==ti and event['representation']==r['representation']
                close(event['t'],t)
            assert start['utc']<=finish['utc']
        assert done['forwards']==spec['probe_forwards'] and done['backwards']==done['updates']==done['head_forwards']==0
        assert len(done['files'])==len(selected)
        assert {str(p) for p in directory.glob('*.npz')}==set(done['files'])
        for p,h in done['files'].items():
            hashed(p,h);pred=load(p);assert len(pred['pred'])==5 and np.isfinite(pred['pred']).all()
            np.testing.assert_allclose(pred['times'],pkg['times'],rtol=0,atol=1e-7)
        prediction_files+=len(done['files']);total+=done['forwards']
        summaries.append(dict(label=label,forwards=done['forwards'],prediction_files=len(done['files']),
            frozen_tensor_sha256=done['frozen_tensor_sha256'],weights_sha256=spec['sha256'],
            seconds=done['seconds'],peak_allocated=done['peak_allocated'],peak_reserved=done['peak_reserved']))
    assert total==200 and prediction_files==40
    dispatch=read(OUT/'dispatch/completion.json')
    assert dispatch['status']=='COMPLETE' and dispatch['forwards']==200 and dispatch['backwards']==dispatch['updates']==0
    assert set(dispatch['jobs'])==set(reg['models'])
    assert sum(x['event']=='DISPATCH' for x in dispatch['history'])==4
    assert all(x['code']==0 for x in dispatch['history'] if x['event']=='EXIT')
    assert trajectory['trajectories']==108 and len(trajectory['source_trajectories'])==108
    for p,h in trajectory['source_trajectories'].items():hashed(p,h)
    for key in ('model_calls','scoring_calls','new_noise','new_OT'):assert trajectory[key]==0
    gate=read(OUT/'trajectory/risk_gate.json')
    assert gate['status']=='NOT_TRIGGERED' and len(gate['rows'])==4 and gate['threshold']==.005
    assert all(r['B_velocity_variation_proxy']<=.005 for r in gate['rows'])
    identities=read(OUT/'audits/trajectory_inventory/checkpoint_identities.json')
    aliased=next(x for x in identities if x['label']=='COVERAGE_FINAL')
    assert aliased['path']==reg['models']['COVERAGE_S1500']['path']
    assert aliased['registered_sha256']==reg['models']['COVERAGE_S1500']['sha256']
    assert analysis['registered_forward_artifacts']==analysis['identity_rows']==200 and analysis['model_calls_here']==0
    for key in ('source_hashes','protected_input_hashes','output_hashes'):
        for p,h in analysis[key].items():hashed(p,h)
    scalars=rows(OUT/'probe_analysis_v1/per_time_stratum.csv')
    metrics=('velocity_mse','velocity_mse_L2','E_shared_pair_mean','endpoint_mse','endpoint_rms_L','endpoint_gap_L_mean','endpoint_gap_L_p95')
    keys=('model','representation','kind','stratum')
    req=aggregate(scalars,keys+('parent_id','condition_id','request_id'),metrics)
    cond=aggregate(req,keys+('parent_id','condition_id'),metrics)
    parents=aggregate(cond,keys+('parent_id',),metrics)
    means=aggregate(parents,keys,metrics)
    saved=rows(OUT/'probe_analysis_v1/parent_equal_summary.csv')
    ix={tuple(r[k] for k in keys):r for r in saved};checks=0
    for r in means:
        other=ix[tuple(r[k] for k in keys)]
        for m in metrics:
            if r[m] is None:assert other[m]==''
            else:close(r[m],float(other[m]));checks+=1
    paircsv=rows(OUT/'probe_analysis_v1/native_paired_per_parent.csv')
    pi={(r['model'],r['kind'],r['stratum'],r['parent_id']):r for r in parents}
    for r in paircsv:
        first=pi[(r['first_model'],r['kind'],r['stratum'],r['parent_id'])]
        second=pi[(r['second_model'],r['kind'],r['stratum'],r['parent_id'])]
        for m in metrics:
            a,b=first[m],second[m]
            if a is None or b is None:assert r[m+'_delta']==''
            else:close(float(r[m+'_delta']),b-a);checks+=1
    scalarix={(r['model'],r['request_id'],int(r['time_index']),r['kind'],r['stratum']):r for r in scalars}
    spot=[]
    for label in reg['models']:
        rep='SEAM' if label=='SEAM_S3000' else 'NATIVE';prefix='6b894c' if rep=='SEAM' else 'a8e987'
        row=next(r for r in pkg['rows'] if r['representation']==rep and r['parent_id'].startswith(prefix) and r['noise_index']==0)
        sample=load(row['trace_path']);prediction=load(OUT/'probes'/label/(row['request_id']+'.npz'))
        pairmap=read(row['pair_path'])['pairs'];n=int(sample['N'])
        fixed=sample['anchor_ids']>=0;free=sample['valid_mask'][:,None]&~fixed
        for ti in (0,2,4):
            v=prediction['pred'][ti].reshape(n,3,3).astype(np.float64);v[fixed]=0
            residual=v-sample['u'].reshape(n,3,3).astype(np.float64)
            calc=float(np.square(residual[free]).mean())
            close(calc,float(scalarix[(label,row['request_id'],ti,'corner','all_free')]['velocity_mse']))
            for name,pairs in [('all_both_free',[p for p in pairmap if p['both_free']]),('all_one_fixed',[p for p in pairmap if not p['both_free'] and not p['both_fixed']])]:
                differences=[residual[p['slot_a'],p['corner_a']]-residual[p['slot_b'],p['corner_b']] for p in pairs]
                expected=scalarix[(label,row['request_id'],ti,'shared_source_pair',name)]
                if differences:close(float(np.square(np.asarray(differences)).mean()),float(expected['velocity_mse']))
                else:assert expected['velocity_mse']==''
            spot.append(dict(label=label,request_id=row['request_id'],time_index=ti,all_free_velocity_mse=calc))
    groupmap={r['parent_id']:r['eval_group'] for r in pkg['rows']}
    core=[]
    for r in means:
        if (r['kind'],r['stratum']) in (('corner','all_free'),('shared_source_pair','all_both_free'),('shared_source_pair','all_one_fixed')):
            core.append(r)
    result=dict(status='PASS',audit_utc=datetime.now(timezone.utc).isoformat(),audit_scope='Independent CPU artifact/accounting and reduction audit; no new scoring/model/OT',
        actual_budget=dict(frozen_forwards=total,backwards=0,optimizer_updates=0,head_forwards=0,new_rollouts=0,D0=0,CPU_OT=16,cached_trajectories=108,new_generation_noise_arrays=0,new_diagnostic_epsilon_arrays=12),
        probe_models=summaries,dispatch_max_workers=dispatch['actual_max_workers'],prediction_file_hashes_verified=prediction_files,
        RNG_and_frozen_state_check='Executed loader/post-forward assertions precede COMPLETE in exact attempt-hashed probe.py; RNG tensors were not persisted for independent replay',
        checkpoint_hash_policy='Use registered and loader-verified SHA/state hashes plus fresh small sidecar hashes and file size; no repeated checkpoint read/load',
        coverage_alias=dict(trajectory='COVERAGE_FINAL',probe='COVERAGE_S1500',same_path=aliased['path'],same_sha256=aliased['registered_sha256']),
        source_and_artifact_hashes_verified=len(CACHE),source_hash_status='All registered protected sources, inputs, predictions and analyzed output hashes match at audit time',
        risk_gate=gate,summary_numeric_checks=checks,spot_prediction_reductions=spot,parent_equal_core=core,
        metadata_note='probe scalar split=UNSPECIFIED; frozen eval_group resolved by exact parent_id in probe manifest, not inferred from scores',
        eval_group_by_parent=groupmap,
        interpretation_limits=['No Euler100 was triggered; integration error was not excluded.',
          'Lower source-bridge MSE does not establish autonomous generation improvement.',
          'SEAM probe has two parents and target-derived source-owner layout; not same-input comparison with six-parent Native.',
          'Reported both-free differential MSE averages source-legal adjacent-edge endpoint pairs, not arbitrary same-vertex copies or all generated edges.'],
        seconds=time.perf_counter()-started,audit_code_sha256=sha(__file__))
    target=OUT/'audits/final_check.json'
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(dict(status=result['status'],forwards=total,hashes=len(CACHE),checks=checks,spot=len(spot),seconds=result['seconds'])))
if __name__=='__main__':run()
