"""CPU scoring adapter for four frozen finite-surface checkpoints.

Reads the completed Hybrid scorer unchanged. No model imports, sampling, I/P
forward calls, GT query resampling or attention execution are performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
import numpy as np

from experiments.mf_h1_local0.chair_hybrid_coupling_v1 import scoring as old
from experiments.mf_h1_local0.gpu_guidance_evaluation import shape_quality

ROOT=Path(__file__).resolve().parents[3]
CHAIR=ROOT/'native_t1/runs/chair_domain_control_v1'
HYBRID=ROOT/'native_t1/runs/chair_hybrid_coupling_v1'
LABELS=('START','H_CONT','MOMENT_XY','SURFACE_XY')
ARMS=LABELS[1:]
START_SHA='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'
read_json,write_json,write_csv,sha=old.read_json,old.write_json,old.write_csv,old.sha
arrays,resolve=old.arrays,old.resolve
DEFINITION=dict(old.DEFINITION,
    experiment='NATIVE_CHAIR_FINITE_SURFACE_ATTENTION_V1',
    fixed_query_MSE='mean((d(q,F)/L)^2) inherited field; A20 alpha=1 has L=L0',
    primary_fixed_query_MSE='mean((d(original_512_coverage_points,F)/L0)^2); A20 validation alpha=1',
    GT_to_F_original_RMS_L='R12_to_free_RMS_bbox_pct / 100; original frozen seed9911 target points',
    F_to_GT_original_RMS_L='free_to_R12_RMS_bbox_pct / 100; original predicted-area points, distinct direction',
    GT_angle_reference='same gt_free complement, original shape_quality formula; all corner angles and per-face minima',
    trajectory_quadrature='offline relation accuracy on pre-registered existing SURFACE_XY states k=0,25,49; not geometric endpoint MSE',
    I_P_diagnostic='NOT_RUN_BY_PROTOCOL')


def records(source=CHAIR):
    panels=read_json(Path(source)/'panels.json')
    return [dict(r,table=t) for t in ('A','B','C') for r in panels[t]]


def gt_angle_reference(bank):
    """Preserve the original zero-edge convention; no new angle loss or repair."""
    gt=np.asarray(bank['gt_free'],np.float64).reshape(-1,3,3);L=float(bank['L'])
    normalized=gt/L;angles=[]
    for corner in range(3):
        a=normalized[:,(corner+1)%3]-normalized[:,corner]
        b=normalized[:,(corner+2)%3]-normalized[:,corner]
        den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
        cosine=np.divide((a*b).sum(-1),den,out=np.ones_like(den),where=den>0)
        angles.append(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    angles=np.stack(angles,axis=-1);quality=shape_quality(gt,L)
    minima=angles.min(axis=1)
    assert float(np.mean(minima<5))==quality['min_angle_lt5_fraction']
    assert float(np.quantile(angles,.05))==quality['angle_p05_degrees']
    result=dict(status='PASS',GT_sha256=old.array_sha(bank['gt_free']),GT_faces=len(gt),
        GT_minimum_angle_lt5_fraction=quality['min_angle_lt5_fraction'],
        GT_exact_zero_area_count=quality['exact_zero_area_count'],GT_near_degenerate_count=quality['near_degenerate_area_count'],
        GT_corner_angle_p05_degrees=quality['angle_p05_degrees'],
        GT_corner_angle_p50_degrees=float(np.quantile(angles,.5)),GT_corner_angle_p95_degrees=float(np.quantile(angles,.95)),
        GT_face_minimum_angle_p05_degrees=float(np.quantile(minima,.05)),
        GT_face_minimum_angle_p50_degrees=float(np.quantile(minima,.5)),
        formula='original shape_quality: normalized vectors, arccos clipped cosine; zero denominator cosine=1',
        zero_area_interpretation='numerical convention only; zero-area face is not a valid area surface')
    return result,angles


def _reuse_start_score(out,folder,record,generated,source,start_score_root):
    """Only the completed OT_HYBRID checkpoint may supply this experiment START."""
    target=folder/'scores'/(record['case_id']+'.json')
    if target.exists() or not generated or generated.get('status') not in ('PASS','CACHED','SAMPLED'):return
    previous=Path(start_score_root)/'evaluation/OT_HYBRID/scoring/scores'/(record['case_id']+'.json')
    if not previous.exists():return
    manifest=read_json(Path(start_score_root)/'checkpoint_manifest.json')
    endpoint=next(r for r in manifest['endpoints'] if r['metadata']['arm']=='OT_HYBRID')
    if endpoint['checkpoint_sha256']!=START_SHA:raise ValueError('Wrong Hybrid endpoint for START score cache')
    metadata=endpoint['metadata']
    if (metadata['pilot_step'],metadata['conditional_total_step'],metadata['lambda_variance'])!=(1000,5000,.25):
        raise ValueError('Wrong Hybrid checkpoint lineage')
    bank,ref=old.bank_reference(source,record)
    rawpath=resolve(out,generated['raw_path']);rawhash=sha(rawpath)
    if rawhash!=generated['raw_sha256']:raise ValueError('New registered raw identity changed')
    raw=arrays(rawpath)['raw'];K=len(bank['C']);saved=read_json(previous);score=saved['score']
    if score['status']!='PASS':return
    if raw.shape!=(int(bank['N']),9) or not np.array_equal(raw[:K].reshape(-1,3,3),bank['C']):return
    if old.array_sha(raw[K:].reshape(-1,3,3))!=score['detail']['predicted_free_sha256']:return
    if score['detail']['reference']!=ref['audit']:raise ValueError('Hybrid cached reference identity changed')
    if saved['identity']['input_sha256']!=record['input_sha256']:raise ValueError('Hybrid input identity changed')
    sources=old.source_identity()
    if saved['identity']['score_sources']!=sources:raise ValueError('Original scorer source changed')
    identity=dict(raw_sha256=rawhash,input_sha256=record['input_sha256'],
        ref_sha256=sha(resolve(source,record['ref_npz'])),score_sources=sources)
    write_json(target,dict(identity=identity,score=score,original_score_reused=True,
        original_score_path=str(previous),original_score_sha256=sha(previous),
        source_checkpoint_sha256=START_SHA,source_experiment='chair_hybrid_coupling_v1/OT_HYBRID'))
    old._event(out,dict(kind='surface_START_score_cache',case_id=record['case_id'],source=str(previous),source_sha256=sha(previous)))


def _add_geometry_disclosures(row,state,folder):
    if row['status']!='PASS' or state is None:return
    row.update(GT_to_F_original_RMS_L=row['R12_to_free_RMS_bbox_pct']/100,
        F_to_GT_original_RMS_L=row['free_to_R12_RMS_bbox_pct']/100,
        bidirectional_original_RMS_L=row['remainder_symmetric_RMS_bbox_pct']/100)
    bank=state['bank'];key=old.array_sha(bank['gt_free'])
    path=folder/'gt_angle_references'/(key+'.json')
    if path.exists():gt=read_json(path)
    else:
        gt,angles=gt_angle_reference(bank);write_json(path,gt)
        np.savez_compressed(path.with_suffix('.npz'),corner_angles_degrees=angles,face_minimum_angles_degrees=angles.min(axis=1))
    row.update({k:v for k,v in gt.items() if k.startswith('GT_')})
    row.update(GT_angle_reference_path=str(path),GT_angle_reference_sha256=sha(path),
        small_angle_excess_over_same_GT=row['free_minimum_angle_lt5_fraction']-gt['GT_minimum_angle_lt5_fraction'])


def run_checkpoint(out,checkpoint_label,records,generated_rows,source_out=CHAIR,start_score_root=HYBRID):
    if checkpoint_label not in LABELS:raise ValueError('Unregistered surface checkpoint label')
    out,source=Path(out),Path(source_out);folder=out/'evaluation'/checkpoint_label/'scoring'
    folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    if len({r['case_id'] for r in records})!=len(records):raise ValueError('Duplicate registered case')
    index={r['case_id']:r for r in generated_rows}
    if len(index)!=len(generated_rows):raise ValueError('Duplicate generated case')
    checks={k:sha(ROOT/k)==v for k,v in read_json(source/'protocol.json')['source_scorer_identity'].items()}
    if not all(checks.values()):raise ValueError('Historical scorer source identity changed')
    rows=[];width_state=defaultdict(dict);expected=defaultdict(list)
    for rec in records:
        if checkpoint_label=='START' and rec.get('status')=='READY':
            _reuse_start_score(out,folder,rec,index.get(rec['case_id']),source,start_score_root)
        row,state=old.score_case(source,checkpoint_label,rec,index.get(rec['case_id']),folder,out)
        _add_geometry_disclosures(row,state,folder);rows.append(row)
        if rec['table']=='B':
            expected[(rec['uid'],rec['noise_index'])].append(rec)
            if state and row['status']=='PASS':
                state['source']=source;width_state[(rec['uid'],rec['noise_index'])][float(rec['alpha'])]=state
        write_json(folder/'per_output.json',rows)
    wo=[];wc=[];matrix=[];columns=[]
    for key,recs in sorted(expected.items()):
        if set(width_state[key])!=set(old.ALPHAS):
            for candidate in ('MODEL','No-response','Post-scale'):
                wc.append(dict(checkpoint_label=checkpoint_label,uid=key[0],noise_index=key[1],split=recs[0]['split'],candidate=candidate,
                    status='BLOCKED',reason='Incomplete registered alpha triplet'))
            continue
        cache=folder/'width_groups'/(key[0]+'_n'+str(key[1])+'.json')
        identity={str(a):dict(raw_array_sha256=old.array_sha(v['rawfile']['raw']),input_sha256=v['record']['input_sha256'],
            query_sha256=v['record']['control_points_sha256']) for a,v in width_state[key].items()}
        if cache.exists():
            saved=read_json(cache)
            if saved['identity']!=identity:raise ValueError('Width cache identity changed')
            o,c,m,a=(saved[n] for n in ('outputs','curves','matrices','columns'))
        else:
            o,c,m,a=old.score_width_group(checkpoint_label,width_state[key])
            write_json(cache,dict(identity=identity,outputs=o,curves=c,matrices=m,columns=a))
            old._event(out,dict(kind='surface_width_group',checkpoint_label=checkpoint_label,uid=key[0],noise_index=key[1],surface_distance_calls=len(m)))
        wo+=o;wc+=c;matrix+=m;columns+=a
    tables=dict(per_output=rows,width_per_output=wo,width_per_curve=wc,cross_matrix=matrix,target_column_audit=columns)
    for name,data in tables.items():write_json(folder/(name+'.json'),data);write_csv(folder/(name+'.csv'),data)
    summary=dict(checkpoint_label=checkpoint_label,status='COMPLETE' if all(r['status'] in ('PASS','NOT_APPLICABLE') for r in rows) else 'INCOMPLETE',
        requested=len(records),availability=dict(Counter(r['status'] for r in rows)),row_counts={k:len(v) for k,v in tables.items()},
        elapsed_seconds=time.perf_counter()-started,definition=DEFINITION,historical_scoring_source_checks=checks,
        source_adapter_sha256={str(Path(old.__file__).relative_to(ROOT)):sha(old.__file__)},
        model_forwards=0,optimizer_updates=0,GT_query_resampling_calls=0,I_P_extra_forwards=0)
    write_json(folder/'summary.json',summary);return summary


def trajectory_quadrature_inputs(out,generated_rows,registered_case_ids):
    """Return frozen-state inputs only; root/relationship module owns computation.

    Caller supplies pre-registered IDs. Missing states remain BLOCKED, and this
    function never generates or reconstructs a trajectory.
    """
    index={r['case_id']:r for r in generated_rows};result=[]
    for cid in registered_case_ids:
        row=index.get(cid,{});base=dict(case_id=cid,checkpoint_label='SURFACE_XY',steps=[0,25,49])
        if row.get('status') not in ('SAMPLED','CACHED','PASS'):
            result.append(dict(base,status='BLOCKED',reason='Registered SURFACE output absent'));continue
        path=resolve(out,row['raw_path'])
        if sha(path)!=row['raw_sha256']:raise ValueError('Trajectory raw identity changed')
        data=arrays(path)
        # Both whole accepted trajectories and explicitly saved selected states
        # are allowed; the relation worker must verify the actual step labels.
        if 'states' not in data and 'trajectory_states' not in data:
            result.append(dict(base,status='BLOCKED',reason='Existing raw lacks saved states'));continue
        result.append(dict(base,status='READY',raw_path=str(path),raw_sha256=row['raw_sha256'],
            uid=row.get('uid'),noise_index=row.get('noise_index'),input_sha256=row.get('input_sha256'),
            relation_candidate_nodes=192,reference_nodes=12288,convergence_nodes=49152,
            geometry_endpoint_scores_unchanged=True,additional_model_forwards=0))
    write_json(Path(out)/'trajectory_quadrature/registered_inputs.json',result);return result


def cpu_tests(out):
    tests=old.cpu_tests(out)
    gt=np.array([[[0.,0,0],[1.,0,0],[0,1.,0]],[[0.,0,0],[1.,0,0],[2.,1e-6,0]]],np.float64)
    ref,angles=gt_angle_reference(dict(gt_free=gt,L=np.array(3.)))
    assert ref['GT_minimum_angle_lt5_fraction']==.5 and angles.shape==(2,3)
    result=dict(status='PASS',inherited_positive_controls=tests['status'],GT_angle_reference=ref,
        zero_model_forwards=True,GT_query_resampling_calls=0)
    write_json(Path(out)/'tests/surface_scoring_cpu_tests.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--label',choices=LABELS)
    p.add_argument('--source',default=str(CHAIR));p.add_argument('--generated');p.add_argument('--records');p.add_argument('--cpu-tests',action='store_true')
    a=p.parse_args()
    if a.cpu_tests:print(cpu_tests(a.out))
    else:
        payload=read_json(a.generated);generated=payload if isinstance(payload,list) else payload['rows']
        reg=read_json(a.records) if a.records else records(a.source)
        if isinstance(reg,dict):reg=reg.get('records',reg.get('rows'))
        print(run_checkpoint(a.out,a.label,reg,generated,a.source))
