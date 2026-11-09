"""Read-only preflight, exclusive archival and fresh original-CPU replay.

Archive integrity is separate from scientific success. Failed/missing attempts,
original-A GPU guard gaps, and final CPU/CUDA discrepancies remain explicit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime, hashlib, json, os
from pathlib import Path, PurePosixPath
import shutil, subprocess, sys, zipfile
import numpy as np
import torch
from ..exporter import ASSETS, RadialSupport
from ..io_utils import array_hash, file_hash, write_json
from ..verify_repo import verify
from ..layout_student.deployment import named_quality, qualification
from ..layout_student.integrity import mesh_identity
from ..layout_student.prepare import split_uids
from ..layout_student.train import model_sha256
from ..layout_student.package_results import inside, mesh_arrays, verify_manifest, write_obj
from .evaluate import MODEL_NAMES, POLICIES, SEED_BASES, POINTS
from .train import load_train_archive


def require(ok, message):
    if not ok: raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def exact_equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys()==b.keys() and all(exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return isinstance(b, type(a)) and len(a)==len(b) and all(exact_equal(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def check_paired_sequences(states):
    records={}
    for seed in (10,11):
        reference=states[('c0',seed)]
        pairs=[(r['uids'],r['rng_matched_k']) for r in reference['trace']]
        for arm in ('c1','c2'):
            state=states[(arm,seed)]
            require(pairs==[(r['uids'],r['rng_matched_k']) for r in state['trace']],f'Parent/RNG sequence changed: {arm}_{seed}')
            require(exact_equal(reference['sampler'],state['sampler']),f'Sampler mismatch: {arm}_{seed}')
            require(exact_equal(reference['rng'],state['rng']),f'RNG progression mismatch: {arm}_{seed}')
        records[str(seed)]={'updates':len(pairs),'same_parent_sequence':True,'same_rng_matched_k':True,
                           'same_final_rng_and_sampler':True,'equal_compute_claim':False}
    return records


def attempt_accounting(ledger, attempts, uids):
    expected_attempts={(u,m) for u in uids for m in MODEL_NAMES}
    expected_rows={(u,m,p) for u,m in expected_attempts for p in POLICIES[m]}
    ids=[(a['uid'],a['model']) for a in attempts]
    require(len(ids)==len(set(ids)) and set(ids)<=expected_attempts,'Duplicate/unregistered attempts')
    flattened=[];errors=[]
    for a in attempts:
        require(a['status'] in ('success','error'),'Unfinalized attempt status')
        if a['status']=='success':
            rr=a['rows']
            require(len(rr)==len(POLICIES[a['model']]) and {r['policy'] for r in rr}==set(POLICIES[a['model']]),'Successful attempt omits/duplicates policy')
            require(all(r['uid']==a['uid'] and r['model']==a['model'] and r['status']=='success' for r in rr),'Attempt/row identity mismatch')
            flattened.extend(rr)
        else:
            require(a.get('rows',[])==[],'Error attempt contains successful rows');errors.append(a)
    key=lambda r:(r['uid'],r['model'],r['policy'])
    rows=ledger['rows'];row_keys=[key(r) for r in rows]
    require(len(row_keys)==len(set(row_keys)),'Duplicate aggregate rows')
    require(sorted(rows,key=key)==sorted(flattened,key=key),'Aggregate changed/lost attempt rows')
    require(sorted(ledger['errors'],key=lambda a:(a['uid'],a['model']))==sorted(errors,key=lambda a:(a['uid'],a['model'])),'Aggregate discarded errors')
    failed={(a['uid'],a['model'],p) for a in errors for p in POLICIES[a['model']]}
    missing_attempts=expected_attempts-set(ids)
    missing={(u,m,p) for u,m in missing_attempts for p in POLICIES[m]}
    require(not set(row_keys)&failed and not set(row_keys)&missing,'Conflicting successful/failure slots')
    require(set(row_keys)|failed|missing==expected_rows,'Planned result slots disappeared')
    require(ledger['attempt_count']==len(ids),'Attempt count mismatch')
    require(ledger['expected_attempt_count']==len(expected_attempts) and ledger['expected_row_count']==len(expected_rows),'Planned denominator changed')
    require({tuple(x) for x in ledger['failed_row_slots']}==failed,'Failed slots not retained')
    require({tuple(x) for x in ledger['unattempted']}==missing_attempts,'Missing attempts not retained')
    require({tuple(x) for x in ledger['missing_success_rows']}==expected_rows-set(row_keys),'Missing result slots not retained')
    return {'planned_attempts':len(expected_attempts),'observed_attempts':len(attempts),
            'planned_row_slots':len(expected_rows),'saved_success_rows':len(rows),
            'error_attempts_retained':len(errors),'error_accounted_row_slots':len(failed),
            'missing_attempts_retained':sorted(missing_attempts),'missing_row_slots_retained':sorted(missing),
            'all_planned_slots_explicitly_accounted':True,'all_planned_attempts_completed':not missing_attempts}


def compare_prior_mesh(old_mesh,new_mesh,faces,old_row,old_identity,row):
    """Compare geometry exactly; keep historical/storage index dtype semantics."""
    for key in ('vertices','u','r'):
        require(old_mesh[key].dtype==new_mesh[key].dtype and np.array_equal(old_mesh[key],new_mesh[key]),'Frozen prior geometry changed: '+key)
    require(np.array_equal(old_mesh['faces'],new_mesh['faces']) and np.array_equal(new_mesh['faces'],faces),'Prior connectivity changed')
    require(array_hash(old_mesh['vertices'],faces)==old_row['mesh_hash'],'Prior logical hash mismatch')
    require(array_hash(old_mesh['vertices'],old_mesh['faces'])==old_identity['stored_array_hash'],'Prior stored hash mismatch')
    require(array_hash(new_mesh['vertices'],faces)==row['mesh_hash'],'New logical hash mismatch')
    require(array_hash(new_mesh['vertices'],new_mesh['faces'])==row['stored_array_hash'],'New stored hash mismatch')
    if 'alpha' in old_mesh:
        require('alpha' in new_mesh and np.array_equal(old_mesh['alpha'],new_mesh['alpha']),'Prior guard alpha changed')
    return {'source_logical_hash':old_row['mesh_hash'],'source_stored_array_hash':old_identity['stored_array_hash'],
            'logical_hash':row['mesh_hash'],'stored_array_hash':row['stored_array_hash'],
            'source_faces_dtype':str(old_mesh['faces'].dtype),'saved_faces_dtype':str(new_mesh['faces'].dtype),
            'geometry_bitwise_equal':True,'dtype_semantics_preserved':True}


def parity_audit(run,rows,models):
    path=run/'FINAL_DEPLOYMENT_PARITY.json';p=read_json(path)
    require(p['results_sha256']==file_hash(run/'results.json'),'Parity used another ledger')
    require(p['rollout_source_sha256']==file_hash(Path(__file__).with_name('rollout.py')),'Parity rollout changed')
    require(p['source_sha256']==file_hash(Path(__file__).with_name('final_parity.py')),'Parity audit source changed')
    require(p['main_table_uses_original_CPU_only'] and not p['cuda_bitwise_equal_claim'],'Parity limitations hidden')
    row_map={(r['uid'],r['model'],r['policy']):r for r in rows}
    ids=[(r['uid'],r['model'],r.get('device')) for r in p['records']]
    require(len(ids)==len(set(ids)),'Duplicate parity records')
    uids=read_json(run/'RUN_PROTOCOL.json')['uids']
    expected_slots={(uid,model,device) for uid in uids for model in models
                    if model.startswith(('c0_','c1_','c2_')) for device in ('cpu','cuda')}
    require(set(ids)==expected_slots,'Missing final CPU/CUDA parity slots')
    for r in p['records']:
        require(r['device'] in ('cpu','cuda'),'Parity device missing')
        if r['status']=='MISSING_CPU_RESULT':
            require((r['uid'],r['model'],'raw') not in row_map,'False missing parity record');continue
        require(r['status']=='MEASURED','Unknown parity status')
        cpu=row_map[(r['uid'],r['model'],'raw')]
        require(r['checkpoint_sha256']==models[r['model']]['checkpoint_sha256'],'Parity model mismatch')
        require(r['cpu_mesh_file_sha256']==cpu['mesh_npz_sha256'],'Parity CPU mesh mismatch')
        require(r['cpu_qualified']==cpu['qualified'] and r['qualification_changed']==(r['mirror_qualified']!=cpu['qualified']),'Hidden qualification drift')
        require(r['cpu_failure_reasons']==cpu['failure_reasons'] and
                r['failure_reasons_changed']==(r['mirror_failure_reasons']!=cpu['failure_reasons']),'Hidden failure-reason drift')
        numerical=(r['direction_max_abs_difference']<=1e-6 and r['guard_alpha_disagreements']==0 and not r['qualification_changed'])
        require(r['numerical_parity_within_direction_atol_1e_minus_6_and_same_alpha_gate']==numerical,'Parity acceptance flag inconsistent')
    for device in ('cpu','cuda'):
        rr=[r for r in p['records'] if r.get('device')==device and r['status']=='MEASURED']
        expected={(r['uid'],r['model']) for r in rows if r['policy']=='raw' and r['model'].startswith(('c0_','c1_','c2_'))}
        require({(r['uid'],r['model']) for r in rr}==expected,'Parity lacks frozen final output')
        s=p['statistics'][device]
        require(s['recorded']==len(rr) and s['expected']==96,'Parity denominator changed')
        checks={'alpha_disagreement_cases':sum(r['guard_alpha_disagreements']>0 for r in rr),
                'qualification_changed_cases':sum(r['qualification_changed'] for r in rr),
                'failure_reasons_changed_cases':sum(r['failure_reasons_changed'] for r in rr),
                'mirror_invalid_count':sum(not r['mirror_exact_audit_valid'] for r in rr),
                'direction_tolerance_exceeded_cases':sum(r['direction_max_abs_difference']>1e-6 for r in rr),
                'per_row_parity_failure_count':sum(not r['numerical_parity_within_direction_atol_1e_minus_6_and_same_alpha_gate'] for r in rr)}
        for key,value in checks.items():require(s[key]==value,'Parity summary hides discrepancies: '+key)
        require(s['missing_cases']==96-len(rr),'Missing parity cases hidden')
        maxima={'direction_max_abs_difference':max((r['direction_max_abs_difference'] for r in rr),default=None),
                'radius_max_abs_difference':max((r['radius_max_abs_difference'] for r in rr),default=None),
                'angle_max_abs_difference_deg':max((abs(x) for r in rr for x in r['angle_difference_deg'].values()),default=None)}
        for key,value in maxima.items():require(s[key]==value,'Parity maximum hides drift: '+key)
    status=('NUMERICAL_DEPLOYMENT_GAP' if any(s['per_row_parity_failure_count'] or s['recorded']!=96 for s in p['statistics'].values())
            else 'MEASURED_WITHIN_DIRECTION_TOLERANCE_SAME_ALPHA_AND_QUALIFICATION')
    require(p['status']==status,'Final parity conclusion inconsistent with measured gaps')
    return {'sha256':file_hash(path),'reported_status_preserved':p['status'],'statistics':p['statistics'],
            'record_count':len(p['records']),'not_promoted_to_pass_by_packaging':True}


def audit_run(run,smoke_run):
    """Read-only complete preflight. A failure creates no sealing artifacts."""
    run=Path(run).resolve();smoke_run=Path(smoke_run).resolve();root=Path(__file__).resolve().parents[1]
    protocol=read_json(run/'RUN_PROTOCOL.json');ph=file_hash(run/'RUN_PROTOCOL.json')
    require(protocol['experiment']=='A-ENDPOINT-v1','Wrong experiment')
    uids=sorted(protocol['uids']);train=protocol['train_uids'];check=protocol['internal_check_uids']
    require((train,check)==split_uids(uids),'Parent split changed')
    require(protocol['training']['arms']==['c0','c1','c2'] and protocol['training']['seeds']==[10,11] and protocol['training']['updates']==1000,'Run set/budget changed')
    require(not protocol['heldout_generalization'] and protocol['training']['equal_updates_not_equal_compute'],'Scope/compute qualification removed')
    config_hash=hashlib.sha256(json.dumps({'training':protocol['training'],'loss':protocol['loss']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(config_hash==protocol['hashes']['config_sha256'],'Frozen config changed')
    dm=read_json(run/'DATA_MANIFEST.json')['files']
    require(file_hash(run/'DATA_MANIFEST.json')==protocol['hashes']['data_manifest_sha256'] and len(dm)==20,'Prepared data manifest changed')
    data_count=verify_manifest(run,dm)
    archive_uids,_,_,_=load_train_archive(run,protocol);require(archive_uids==train,'Training archive split mismatch')
    for rel,h in protocol['fixed_source_hashes'].items():require(file_hash(inside(root,rel))==h,'Fixed source changed: '+rel)
    old=Path(protocol['source_run']).resolve()
    require(file_hash(old/'MANIFEST_SHA256.json')==protocol['source_run_manifest_sha256']==protocol['hashes']['source_run_manifest_sha256'],'Prior v1 manifest changed')
    om=read_json(old/'MANIFEST_SHA256.json')['files'];require(len(om)==877,'Prior v1 manifest count changed')
    old_count=verify_manifest(old,om);op=read_json(old/'RUN_PROTOCOL.json')
    require(file_hash(ASSETS/'original_A_10.pt')==op['hashes']['original_a_sha256'],'Original A changed')
    provenance=read_json(run/'SOURCE_PROVENANCE.json')
    require(provenance['verified_manifest_files']==om and provenance['source_manifest_sha256']==file_hash(old/'MANIFEST_SHA256.json'),'Prepared provenance differs from prior evidence')
    upstream=verify(Path(__file__).resolve().parents[3]);require(upstream['pass'] and upstream['protected_files_checked']==11,'Official core identity failed')
    impl=read_json(run/'IMPLEMENTATION_GATE.json')
    require(impl['strict_all_model_all_device_correctness_pass'] is False and impl['not_a_claim_of_full_GPU_equivalence'],'Numerical guard limit hidden')
    for name,h in impl['evidence'].items():require(file_hash(run/'correctness_evidence'/name)==h,'Correctness evidence changed')
    require(impl['cpu_legacy_cases_passed']==51 and impl['cpu_single_and_batch4_cases_passed']==48 and impl['actual_initialization_cuda_cases_passed']==32,'Wrong scoped implementation gate')
    gate=read_json(run/'TRAINING_GATE.json')
    require(gate['status']=='APPROVED_FOR_PREDECLARED_V1_INITIALIZED_STUDY' and gate['protocol_sha256']==ph,'Training gate absent/different')
    require(gate['all_model_all_device_correctness_pass'] is False and gate['original_A_GPU_pressure_limit_retained'],'Training gate hides original-A CUDA failures')
    require(gate['implementation_gate_sha256']==file_hash(run/'IMPLEMENTATION_GATE.json'),'Implementation gate changed')
    for rel,h in gate['training_source_hashes'].items():require(file_hash(inside(root,rel))==h,'Frozen training source changed: '+rel)
    smoke=read_json(smoke_run/'GPU_SMOKE.json')
    require(smoke['status']=='PASS' and smoke['formal_protocol_sha256']==ph,'CUDA smoke protocol mismatch')
    require(file_hash(smoke_run/'GPU_SMOKE.json')==gate['cuda_resume_smoke_sha256'],'Smoke evidence changed')
    require(smoke['train_source_sha256']==file_hash(root/'layout_endpoint/train.py') and smoke['rollout_source_sha256']==file_hash(root/'layout_endpoint/rollout.py') and smoke['smoke_source_sha256']==file_hash(smoke_run/'SMOKE_SOURCE.py'),'Tested CUDA source changed')
    require(smoke['all_semantic_training_state_bitwise_equal'] and smoke['parent_sequence_and_RNG_progression_equal_across_arms'] and smoke['internal_check_teacher_reads']==0,'Resume/label-isolation smoke failed')
    verify_manifest(smoke_run,read_json(smoke_run/'DATA_MANIFEST.json')['files'])
    diagnostic=run/'diagnostics/SINGLE_OBJECT_DIAGNOSTIC.json'
    require(file_hash(diagnostic)==gate['diagnostic_sha256'] and not read_json(diagnostic)['diagnostic_weights_used_for_formal_training'],'Single-object evidence changed/reused')
    states={};training_checks={};models={}
    for seed in (10,11):
        initial=inside(run,protocol['paths']['initial_checkpoints'][str(seed)])
        require(file_hash(initial)==protocol['hashes']['initial_checkpoint_sha256'][str(seed)]==file_hash(old/'training'/f'seed_{seed}'/'final.pt'),'Initial checkpoint changed')
        initial_state=torch.load(initial,map_location='cpu',weights_only=False)
        require(initial_state['update']==3000 and initial_state['is_predeclared_final'] and model_sha256(initial_state['model'])==protocol['hashes']['initial_model_sha256'][str(seed)],'Wrong v1 initial final')
        models[f'v1_{seed}']={'model_sha256':model_sha256(initial_state['model']),'checkpoint_sha256':file_hash(initial)}
        for arm in ('c0','c1','c2'):
            folder=run/'training'/arm/f'seed_{seed}';checkpoint=folder/'final.pt'
            state=torch.load(checkpoint,map_location='cpu',weights_only=False);report=read_json(folder/'TRAIN_REPORT.json')
            c=state['config'];trace=state['trace']
            require(state['update']==1000 and state['is_predeclared_final'] and not state['diagnostic_only'],'Not fixed formal final')
            require(c==read_json(folder/'TRAIN_CONFIG.json'),'Saved training config mismatch')
            require(c['arm']==arm and c['seed']==seed and c['total_updates']==1000 and c['batch_size']==4 and c['parameter_count']==29907,'Architecture/budget/seed changed')
            require(c['train_uids']==train and c['active_training_uids']==train and c['diagnostic_uid'] is None,'Formal training wrong parents/diagnostic subset')
            require(not c['mainfield_network_input'] and c['quality_validation_tolerance_deg']==1e-6,'Student input/tolerance changed')
            require(c['optimizer_initialization']=='fresh_empty_no_v1_Adam_inheritance','Old optimizer reused')
            require(state['protocol_sha256']==c['protocol_sha256']==ph and state['training_labels_sha256']==protocol['hashes']['training_labels_sha256'],'Training data/protocol mismatch')
            require(state['initial_checkpoint_sha256']==c['initial_checkpoint_sha256']==file_hash(initial),'Arm initialized from other weights')
            for rel,h in c['source_sha256'].items():require(file_hash(inside(root,rel))==h,'Training code changed: '+rel)
            require(trace==read_json(folder/'TRAIN_TRACE.json') and len(trace)==1000 and [r['update'] for r in trace]==list(range(1,1001)),'Update trace incomplete')
            exposures=Counter()
            for step in trace:
                require(len(step['uids'])==4 and set(step['uids'])<=set(train),'Check/unknown UID entered training')
                require(len(step['rng_matched_k'])==4 and all(isinstance(k,int) and 0<=k<32 for k in step['rng_matched_k']),'Invalid RNG draws')
                require(np.isfinite(step['loss']) and np.isfinite(step['gradient_l2']),'Nonfinite trace')
                exposures.update(step['uids'])
            require(sum(exposures.values())==4000 and set(exposures)==set(train) and set(exposures.values())<={333,334},'Unbalanced parent exposure')
            require(state['sampler']['exposures']==[exposures[u] for u in train] and report['parent_exposures']==dict(exposures),'Sampler/report exposure mismatch')
            require({'optimizer','scheduler','rng','sampler'}<=state.keys(),'Checkpoint cannot fully resume')
            require({'python','numpy','torch_cpu','torch_cuda'}<=state['rng'].keys() and len(state['rng']['torch_cuda'])>0,'Incomplete GPU RNG')
            require(state['optimizer']['state'] and all(float(v['step'])==1000 for v in state['optimizer']['state'].values()),'Wrong Adam update count')
            require(all(torch.isfinite(v).all() for v in state['model'].values()),'Nonfinite final model')
            mh,ch=model_sha256(state['model']),file_hash(checkpoint)
            require(mh==state['model_sha256']==report['model_sha256'] and ch==report['checkpoint_sha256'],'Final model/checkpoint hash mismatch')
            require(report['status']=='COMPLETE' and not report['selection_uses_internal_check'] and report['internal_check_teacher_files_opened']==0,'Selection/label isolation failed')
            models[f'{arm}_{seed}']={'model_sha256':mh,'checkpoint_sha256':ch};states[(arm,seed)]=state
            training_checks[f'{arm}_{seed}']={**models[f'{arm}_{seed}'],'updates':1000,'parent_exposures':dict(exposures),'full_resumable_state_verified':True,'source_hashes_checked':len(c['source_sha256'])}
    paired=check_paired_sequences(states)
    attempts=[read_json(p) for p in sorted((run/'attempts').glob('*/*/result.json'))]
    ledger=read_json(run/'results.json');rows=ledger['rows'];accounting=attempt_accounting(ledger,attempts,uids)
    require(accounting['planned_attempts']==160 and accounting['planned_row_slots']==288,'Wrong full denominator')
    meshes={};prior_checks=[];fallback_count=0
    old_rows={(r['uid'],r['model'],r['policy']):r for r in read_json(old/'results.json')['rows']}
    old_integrity={(r['uid'],r['model'],r['policy']):r for r in read_json(old/'MESH_INTEGRITY.json')['rows']}
    support=RadialSupport();row_map={(r['uid'],r['model'],r['policy']):r for r in rows}
    for uid in uids:
        data=mesh_arrays(run/'data'/f'{uid}.npz');y,base,faces=data['logr'],data['base_u'],data['faces']
        require(file_hash(run/'data'/f'{uid}.npz')==om[f'data/{uid}.npz'],'Prior prepared parent changed')
        require(faces.dtype==np.int32 and faces.shape==(512,3),'Logical int32 template changed')
        fr=support.query(y,base);fv=base*fr[:,None];floor=named_quality(fv,faces)
        for row in [r for r in rows if r['uid']==uid]:
            require(row['split']==('train' if uid in train else 'internal_check'),'Output split mismatch')
            require(row['mainfield_hash']==array_hash(y) and row['data_file_sha256']==dm[f'data/{uid}.npz'],'Output mainfield/data mismatch')
            path=inside(run,row['mesh_file']);mesh_id=mesh_identity(path,faces)
            for key,value in mesh_id.items():require(row[key]==value,'Saved identity mismatch: '+key)
            mesh=mesh_arrays(path);v,f,u,radius=[mesh[k] for k in ('vertices','faces','u','r')]
            require(v.shape==u.shape==(258,3) and radius.shape==(258,) and v.dtype==u.dtype==radius.dtype==np.float64,'Saved geometry shape/dtype changed')
            require(np.array_equal(f,faces) and np.allclose(np.linalg.norm(u,axis=1),1,rtol=0,atol=1e-12),'Connectivity/unit direction invalid')
            queried=support.query(y,u)
            require(np.array_equal(radius,queried) and np.array_equal(v,u*queried[:,None]),'Stale accepted-direction query')
            q=named_quality(v,f)
            for key in ('min_triangle_angle_deg','triangle_min_angle_p05_deg'):
                require(abs(q[key]-row['quality'][key])<=1e-10 and abs(floor[key]-row['quality_floor'][key])<=1e-10,'Actual quality/floor mismatch')
            require(row['audit']['exact_requested'] and row['audit']['exact_executed'],'Exact endpoint audit missing')
            qr=qualification(q,row['audit'],floor)
            require(qr['qualified']==row['qualified'] and qr['failure_reasons']==row['failure_reasons'],'Qualification mismatch')
            if row['model'] in models:
                require(row['identity']['model_sha256']==models[row['model']]['model_sha256'] and row['identity']['file_sha256']==models[row['model']]['checkpoint_sha256'],'Output model identity mismatch')
            if row['model']=='fixed':require(np.array_equal(u,base) and np.array_equal(v,fv),'Fixed baseline changed')
            if row['fallback_to_fixed']:
                require(row['policy']=='quality_fallback' and np.array_equal(u,base) and np.array_equal(v,fv) and np.array_equal(radius,fr),'Fallback is not same-object Fixed');fallback_count+=1
            if row['policy'] in ('raw','offline_teacher') and row['model'] in ('fixed','teacher_reference','v1_10','v1_11'):
                old_key=(uid,row['model'].replace('v1_','student_'),row['policy']);prior=old_rows[old_key]
                prior_mesh=mesh_arrays(inside(old,prior['mesh_file']))
                require(file_hash(inside(old,prior['mesh_file']))==old_integrity[old_key]['mesh_npz_sha256'],'Prior mesh bytes changed')
                prior_checks.append({'uid':uid,'model':row['model'],**compare_prior_mesh(prior_mesh,mesh,faces,prior,old_integrity[old_key],row)})
            metrics=row['metrics'];seeds=[b+uids.index(uid) for b in SEED_BASES]
            require(len(metrics)==2 and [m['seed'] for m in metrics]==seeds and all(m['points_per_direction']==POINTS and np.isfinite(m['rms_bbox_percent']) and m['rms_bbox_percent']>=0 for m in metrics),'Evaluation seed/count/metric mismatch')
            require(row['independent_evaluation_not_used_for_deployment'],'Error-dependent selection')
            meshes[(uid,row['model'],row['policy'])]=mesh
    for row in rows:
        if row['policy']=='quality_fallback':
            raw=row_map[(row['uid'],row['model'],'raw')]
            require(row['fallback_to_fixed']==(not raw['qualified']) and row['fallback_reasons']==raw['failure_reasons'],'Undeclared fallback criterion')
            if raw['qualified']:require(row['stored_array_hash']==raw['stored_array_hash'],'Qualified raw changed in delivery')
    parity=parity_audit(run,rows,models);replay_uid=uids[0]
    require((replay_uid,'c2_10','raw') in meshes and (replay_uid,'c2_10','quality_fallback') in meshes,'Predeclared replay unavailable; no substitution allowed')
    prior_files={rel:h for rel,h in om.items() if (rel in ('RUN_PROTOCOL.json','FINAL_AUDIT.json','DATA_MANIFEST.json','TEACHERS.json','MESH_INTEGRITY.json','results.json','SOURCE_PROVENANCE.json','FIT_DIAGNOSTIC.json','ORACLE_NUMERICS.json','SUMMARY.json') or rel.startswith('teacher_evidence/') or (rel.startswith('attempts/') and any('/'+name+'/' in rel for name in ('fixed','teacher_reference','student_10','student_11'))) or (rel.startswith('training/') and rel.endswith(('TRAIN_REPORT.json','TRAIN_CONFIG.json'))))}
    audit={**accounting,'saved_mesh_hashes_checked':len(meshes),'logical_and_storage_hashes_checked_separately':True,
           'accepted_direction_query_exact_checks':len(meshes),'fallback_same_uid_fixed_checks':fallback_count,
           'frozen_v1_geometry_comparisons':prior_checks,'data_manifest_files_checked':data_count,
           'prior_v1_manifest_files_unchanged':old_count,'prior_v1_manifest_sha256':file_hash(old/'MANIFEST_SHA256.json'),
           'training_checkpoints':training_checks,'paired_training_sequences':paired,'equal_compute_claim':False,
           'gpu_resume_smoke':'PASS','implementation_strict_all_cases_pass':False,'implementation_scope_preserved':True,
           'final_cpu_gpu_diagnostic':parity,'upstream':upstream,'original_A_hash_unchanged':True,
           'predeclared_replay':{'model':'c2_10','uid':replay_uid},'frozen_source_files_checked':len(protocol['fixed_source_hashes']),
           'training_gate_sha256':file_hash(run/'TRAINING_GATE.json'),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    return audit,rows,meshes,old,prior_files


REPLAY_SCRIPT = r'''"""Fresh extracted original CPU replay, using extracted artifacts only."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--report',required=True);a=p.parse_args()
run=Path(a.run).resolve();sys.path.insert(0,str(run/'source_snapshot'))
import numpy as np
import torch
from experiments.a_radial import exporter
from experiments.a_radial.layout_student import deployment
from experiments.a_radial.layout_student.train import model_sha256
from experiments.a_radial.legacy.networks import DomainVelocity
from experiments.a_radial.io_utils import array_hash,file_hash
assert Path(exporter.__file__).resolve().is_relative_to(run/'source_snapshot')
assert Path(deployment.__file__).resolve().is_relative_to(run/'source_snapshot')
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
protocol=json.loads((run/'RUN_PROTOCOL.json').read_text());uid=sorted(protocol['uids'])[0]
checkpoint=run/'training/c2/seed_10/final.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert state['update']==1000 and state['is_predeclared_final'] and not state['diagnostic_only']
assert state['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json')
assert state['model_sha256']==model_sha256(state['model'])
initial=run/protocol['paths']['initial_checkpoints']['10']
assert file_hash(initial)==state['initial_checkpoint_sha256']==protocol['hashes']['initial_checkpoint_sha256']['10']
model=DomainVelocity().cpu();model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
with np.load(run/'data'/f'{uid}.npz',allow_pickle=False) as z:
    y=z['logr'];base=z['base_u'];faces=z['faces']
support=exporter.RadialSupport(run/'source_snapshot/experiments/a_radial/assets/support.npz')
result=deployment.deploy(model,y,support,base,faces)
rows=json.loads((run/'attempts'/uid/'c2_10/result.json').read_text())['rows'];checks={}
for policy,prefix in [('raw','raw'),('quality_fallback','delivered')]:
    row=next(r for r in rows if r['policy']==policy)
    mesh=result[prefix+'_mesh'];audit=result[prefix+'_audit'];quality=result[prefix+'_quality']
    assert array_hash(mesh['vertices'],faces)==row['mesh_hash']
    assert array_hash(mesh['vertices'],mesh['faces'])==row['stored_array_hash']
    assert file_hash(run/row['mesh_file'])==row['mesh_npz_sha256']
    with np.load(run/row['mesh_file'],allow_pickle=False) as saved:
        for key in ('vertices','faces','u','r'):
            assert np.array_equal(mesh[key],saved[key]),(policy,key)
        if 'alpha' in mesh:assert np.array_equal(mesh['alpha'],saved['alpha'])
    radius=support.query(y,mesh['u'])
    assert np.array_equal(mesh['r'],radius)
    assert np.array_equal(mesh['vertices'],mesh['u']*radius[:,None])
    gate=deployment.qualification(quality,audit,result['floor'])
    assert gate['qualified']==row['qualified'] and gate['failure_reasons']==row['failure_reasons']
    assert audit['exact_requested'] and audit['exact_executed']
    for key in ('valid','selfintersection_count'):assert audit[key]==row['audit'][key]
    for key in ('min_triangle_angle_deg','triangle_min_angle_p05_deg'):assert abs(quality[key]-row['quality'][key])<=1e-10
    checks[policy]={'mesh_hash':row['mesh_hash'],'stored_array_hash':row['stored_array_hash'],
        'mesh_npz_sha256':row['mesh_npz_sha256'],'geometry_and_accepted_query_bitwise_equal':True,
        'audit_and_qualification_match':True,'qualified':gate['qualified']}
assert result['delivery']['fallback_to_fixed']==next(r for r in rows if r['policy']=='quality_fallback')['fallback_to_fixed']
report={'status':'PASS','model':'c2_10','uid':uid,'updates':1000,'device':'cpu',
    'loaded_from_fresh_extracted_source':True,'teacher_label_read_for_prediction':False,
    'initial_checkpoint_sha256_verified':True,'checkpoint_sha256':file_hash(checkpoint),
    'checks':checks,'fallback_to_fixed':result['delivery']['fallback_to_fixed']}
with Path(a.report).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report),flush=True)
'''


def package_run(run,smoke_run):
    run=Path(run).resolve();smoke_run=Path(smoke_run).resolve()
    archive=run.with_name(run.name+'_results.zip');fresh=run.with_name(run.name+'_fresh_extract')
    archive_report=archive.with_suffix('.zip.json');fresh_report=archive.with_suffix('.zip.fresh.json')
    destinations=[run/'FINAL_AUDIT.json',run/'MANIFEST_SHA256.json',run/'meshes_obj',run/'source_snapshot',
                  run/'gpu_smoke_evidence',run/'prior_v1_evidence',run/'REPLAY_FRESH.py',archive,fresh,archive_report,fresh_report]
    for path in destinations:require(not path.exists(),'Exclusive sealing destination exists: '+str(path))
    audit,rows,meshes,old,prior_files=audit_run(run,smoke_run)
    compile(REPLAY_SCRIPT,'<fresh-replay>','exec')
    # All static scientific/provenance assertions complete before the first write.
    for row in rows:
        mesh=meshes[(row['uid'],row['model'],row['policy'])]
        write_obj(run/'meshes_obj'/row['uid']/(row['model']+'__'+row['policy']+'.obj'),mesh['vertices'],mesh['faces'])
    source=Path(__file__).resolve().parents[1]
    ignored=shutil.ignore_patterns('__pycache__','.pytest_cache','build','*.pyc','*.nbi','*.nbc')
    shutil.copytree(source,run/'source_snapshot/experiments/a_radial',ignore=ignored)
    shutil.copytree(smoke_run,run/'gpu_smoke_evidence',ignore=ignored)
    evidence=run/'prior_v1_evidence';evidence.mkdir()
    shutil.copy2(old/'MANIFEST_SHA256.json',evidence/'MANIFEST_SOURCE_FULL.json')
    for rel,h in prior_files.items():
        dest=inside(evidence,rel);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(inside(old,rel),dest);require(file_hash(dest)==h,'Copied prior evidence changed')
    write_json(evidence/'COPIED_FILES.json',{'source_run':str(old),'source_manifest_sha256':file_hash(old/'MANIFEST_SHA256.json'),
        'files':prior_files,'scope':'Referenced subset; full original manifest retained, omitted artifacts not claimed present.'})
    with (run/'REPLAY_FRESH.py').open('x') as out:out.write(REPLAY_SCRIPT)
    audit['exported_obj_count']=len(rows);audit['prior_v1_files_copied']=len(prior_files)
    write_json(run/'FINAL_AUDIT.json',audit)
    manifest={p.relative_to(run).as_posix():file_hash(p) for p in sorted(run.rglob('*')) if p.is_file()}
    write_json(run/'MANIFEST_SHA256.json',{'files':manifest})
    members={**manifest,'MANIFEST_SHA256.json':file_hash(run/'MANIFEST_SHA256.json')}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in sorted(members):z.write(run/rel,arcname=(PurePosixPath(run.name)/rel).as_posix())
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None,'Archive CRC failed')
        expected_names={(PurePosixPath(run.name)/rel).as_posix() for rel in members}
        require(len(z.namelist())==len(expected_names) and set(z.namelist())==expected_names,'Unexpected archive member')
        for rel,h in members.items():require(hashlib.sha256(z.read((PurePosixPath(run.name)/rel).as_posix())).hexdigest()==h,'Archive member changed')
        fresh.mkdir()
        for name in z.namelist():inside(fresh,name)
        z.extractall(fresh)
    extracted=fresh/run.name
    require(read_json(extracted/'MANIFEST_SHA256.json')['files']==manifest,'Fresh manifest differs')
    verify_manifest(extracted,members)
    env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
    replay=subprocess.run([sys.executable,'-I','-B',str(extracted/'REPLAY_FRESH.py'),'--run',str(extracted),'--report',str(fresh_report)],
                          cwd=fresh,env=env,capture_output=True,text=True)
    if replay.returncode:
        failed=archive.with_suffix('.zip.replay_failed.json')
        write_json(failed,{'status':'FAIL','returncode':replay.returncode,'stdout':replay.stdout,'stderr':replay.stderr,'fresh_manifest_passed':True})
        raise RuntimeError('Fresh extracted replay failed; evidence retained: '+str(failed))
    require(read_json(fresh_report)['status']=='PASS','Fresh replay did not pass')
    verify_manifest(run,members);verify_manifest(extracted,members)
    for sealed in (run,extracted):
        present={p.relative_to(sealed).as_posix() for p in sealed.rglob('*') if p.is_file()}
        require(present==set(members),'Sealed/fresh file set changed during replay')
    verify_manifest(old,read_json(old/'MANIFEST_SHA256.json')['files'])
    require(file_hash(old/'MANIFEST_SHA256.json')==audit['prior_v1_manifest_sha256'],'Prior manifest changed during sealing')
    result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':file_hash(archive),
        'crc_verified':True,'member_hashes_verified_including_manifest':len(members),
        'fresh_extracted_manifest_verified':True,'fresh_original_cpu_replay':'PASS','fresh_replay_report':str(fresh_report),
        'sealed_run_and_fresh_artifacts_unchanged':True,'prior_v1_877_files_unchanged_after_sealing':True,
        'saved_meshes_and_objs':len(rows),'error_attempts_retained':audit['error_attempts_retained'],
        'missing_attempts_retained':audit['missing_attempts_retained'],
        'scientific_pass_not_inferred_from_archive_integrity':True,'implementation_strict_all_cases_pass':False,
        'final_cpu_gpu_diagnostic_status':audit['final_cpu_gpu_diagnostic']['reported_status_preserved']}
    write_json(archive_report,result);print(json.dumps(result,indent=2),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--smoke-run',required=True)
    p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if a.audit_only:
        audit,_,_,_,_=audit_run(a.run,a.smoke_run);print(json.dumps(audit,indent=2),flush=True)
    else:package_run(a.run,a.smoke_run)


if __name__=='__main__':main()


