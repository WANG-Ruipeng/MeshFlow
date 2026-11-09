"""Gated frozen Native geometry validation; optimizer/model backward are forbidden."""
from .stable_runtime import configure_stable_runtime
import argparse,copy,gc,hashlib,json,platform,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
import yaml
from .official_wrapper import state_sha256
from .prompt_dataset import file_sha256
from .learning_data import array_hash as archive_array_hash
from .nxc112_evaluation import array_hash

PROJECT=Path(__file__).resolve().parent
REPO=PROJECT.parents[1]
SOURCE=PROJECT/'outputs/native_patch_20260929_010000'
CHECKPOINT=SOURCE/'Native_correct/endpoint_step500.pt'
EXPECTED_CHECKPOINT_SHA='7928d77507bf1fe610876f61b5b11e614263323b5fc0f595cc6740ffd24af634'
EXPECTED_STATE_SHA='37e31fe5925b8f870b9a4d3bbbbc84edc7bf30177946a6f67f4529a7f8e97172'
CONFIG=REPO/'configs/snet/base-120m-ot-v-chair.yaml'
AUTH=Path('/mnt/c/Users/warpwang/.codex/attachments/0bdf9c8f-b1b2-4354-a128-4a79ecfa3206/Pasted text.txt')
DEVELOPMENT=(9201,9202,9203,9204)
CONFIRMATION=(9301,9302,9303,9304)
UIDS=('2af09bd8df40506c9e646678ef50aa3d','5b0e833cf2ea465e42bd82e95587d8af')
FACE_IDS=([30,33,21,17,18,20,109,110,19,57,51,56],[24,27,30,23,17,18,19,22,59,60,95,109])


def atomic_json(path,value):
    temporary=Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def record(path):
    path=Path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha256(path))


def metadata_protection(out):
    """Read file metadata for all historical assets; hash only used critical inputs."""
    paths=[p for p in (PROJECT/'outputs').rglob('*') if p.is_file() and not p.is_relative_to(out)]
    paths.extend(PROJECT.glob('*.py'))
    paths.extend(p for p in (REPO/'models').rglob('*.py'))
    paths.extend([CONFIG,Path('/mnt/c/Users/warpwang/Downloads/last.pt')])
    return {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(set(paths)) if not p.name.startswith('guided_native_geometry_')}


def prepare_assets():
    started=time.perf_counter()
    old=json.loads((SOURCE/'results.json').read_text(encoding='utf-8-sig'))
    if old['status']!='PASS_EXECUTION':raise RuntimeError('Source run incomplete')
    expected=old['arms']['Native_correct']['checkpoint']
    if expected['sha256']!=EXPECTED_CHECKPOINT_SHA or expected['state_sha256']!=EXPECTED_STATE_SHA:
        raise RuntimeError('Source T1 identity differs')
    ck=record(CHECKPOINT)
    if ck['sha256']!=EXPECTED_CHECKPOINT_SHA:raise RuntimeError('T1 checkpoint file hash mismatch')
    source_input=old['assets']['inputs'];inp=Path(source_input['path'])
    if file_sha256(inp)!=source_input['sha256']:raise RuntimeError('Source input hash mismatch')
    required=[f'noise_seed{s}' for s in DEVELOPMENT]
    for p in (0,1):required.extend([f'parent{p}_target',f'parent{p}_K12_constraints',f'parent{p}_K12_source_face_ids',f'parent{p}_K12_free_target'])
    with np.load(inp,allow_pickle=False) as archive:
        arrays={key:archive[key].copy() for key in required}
    for key,value in arrays.items():
        if archive_array_hash(value)!=source_input['arrays'][key]['sha256']:raise RuntimeError('Archived tensor hash differs: '+key)
    cases={}
    for p in (0,1):
        c=next(q for q in old['assets']['cases'] if q['parent_index']==p and q['K']==12)
        ids=arrays[f'parent{p}_K12_source_face_ids'].tolist()
        if c['parent_uid']!=UIDS[p] or ids!=FACE_IDS[p] or c['source_face_ids']!=ids:raise RuntimeError('Wrong parent or K12 patch')
        C=arrays[f'parent{p}_K12_constraints'];target=arrays[f'parent{p}_target'];free=arrays[f'parent{p}_K12_free_target']
        if C.dtype!=np.float32 or C.shape!=(12,3,3) or target.shape!=(112,9) or free.shape!=(100,9):raise RuntimeError('Unexpected source shapes')
        if not np.array_equal(C,target[ids].reshape(12,3,3)):raise RuntimeError('C differs from archived source faces')
        if not np.array_equal(free,target[[i for i in range(112) if i not in ids]]):raise RuntimeError('R12 archive mismatch')
        cases[(p,12)]=dict(parent_index=p,parent_uid=c['parent_uid'],K=12,N=112,constraints=C,
            bbox_diagonal=c['bbox_diagonal'],source_face_ids=ids,full_target=target,free_target=free)
    raw_records={};paths={};source_scores={}
    for p in (0,1):
        C=cases[p,12]['constraints'].reshape(12,9)
        for seed in DEVELOPMENT:
            row=next(q for q in old['generated'] if (q['model'],q['parent'],q['K'],q['seed'])==('T1',p,12,seed))
            path=Path(row['raw_path'])
            if file_sha256(path)!=row['raw_sha256']:raise RuntimeError('Historical trajectory hash differs')
            with np.load(path,allow_pickle=False) as f:trajectory=f['path'];endpoint=f['output']
            if trajectory.shape!=(51,112,9) or trajectory.dtype!=np.float32 or not np.isfinite(trajectory).all():raise RuntimeError('Historical path invalid')
            if not np.array_equal(trajectory[-1],endpoint):raise RuntimeError('Historical endpoint differs')
            if not np.array_equal(trajectory[:,:12],np.broadcast_to(C,(51,12,9))):raise RuntimeError('Historical C not exact')
            if not np.array_equal(trajectory[0,12:],arrays[f'noise_seed{seed}'][0,12:]):raise RuntimeError('Historical Gaussian differs')
            paths[p,seed]=path;raw_records[p,seed]=copy.deepcopy(row)
            source_scores[p,seed]=next(q for q in old['scores'] if (q['model'],q['parent'],q['K'],q['seed'])==('T1',p,12,seed))['metrics']
    bank=dict(cases_by_key=cases,cases=list(cases.values()),noises={s:arrays[f'noise_seed{s}'] for s in DEVELOPMENT},
        archived_paths=paths,archived_records=raw_records,source_scores=source_scores,prior_results=old)
    critical=[SOURCE/'NATIVE_PATCH_AND_SEAM_REPORT.md',SOURCE/'results.json',inp,CHECKPOINT,CONFIG,*paths.values()]
    critical.extend([PROJECT/name for name in ('native_inpainting_model.py','stable_runtime.py','official_wrapper.py','native_patch_seams.py','native_patch_metrics.py','native_inpainting_metrics.py','evaluate.py','native_patch_run.py')])
    critical.extend((REPO/'models').rglob('*.py'))
    audit=dict(status='PASS',checkpoint=ck,checkpoint_state_sha256=EXPECTED_STATE_SHA,source_inputs=record(inp),
        parent_ids=list(UIDS),K=12,N_total=112,N_free=100,source_faces=FACE_IDS,
        conditions={str(p):dict(sha256=archive_array_hash(cases[p,12]['constraints']),L=cases[p,12]['bbox_diagonal']) for p in (0,1)},
        noise_hashes={str(s):archive_array_hash(arrays[f'noise_seed{s}']) for s in DEVELOPMENT},
        source_trajectories=[dict(parent=p,seed=s,**record(path)) for (p,s),path in paths.items()],
        source_score_rows=8,all_C51_exact=True,archived_initial_free_Gaussian_exact=True,
        critical_files={str(p):record(p) for p in sorted(set(critical))},
        L_scope='Previously known target bbox diagonal metadata only, disclosed diagnostic normalization; not unknown-target deployment',
        original_EMA_loaded=False,model_calls=0,optimizer_updates=0,seconds=time.perf_counter()-started)
    return bank,audit


def load_frozen_T1(counts):
    """Instantiate official architecture on meta; strict-load only full native T1."""
    from models.equidit import DiT
    from models.utils import get_embedder
    from .native_inpainting_model import NativeInpaintingModel
    started=time.perf_counter();config=yaml.safe_load(CONFIG.read_text())
    config_model=dict(config['model']);config_model.pop('model_type')
    required=dict(version=3,hidden_dim=768,num_layers=12,num_heads=12,max_length=800,face_bin=20,
        use_qknorm=True,use_rmsnorm=True,use_coord_encoding=True,use_dit_like_pe=False,face_cond=True)
    if any(config_model.get(k)!=v for k,v in required.items()) or config['transport']['prediction']!='velocity':raise RuntimeError('Official architecture changed')
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    if payload['arm']!='Native_correct' or payload['completed_updates']!=500 or payload['schema']!='native_connected_patch_v1':raise RuntimeError('Wrong T1 checkpoint metadata')
    with torch.device('meta'):
        backbone=DiT(**config_model)
        model=NativeInpaintingModel(backbone,trainable=False)
    embed_fn,_=get_embedder(config_model['pe_freq'],input_dims=3);model.backbone.x_embedder.embed_fn=embed_fn
    incompatible=model.load_state_dict(payload['model'],strict=True,assign=True)
    model.set_trainable(False).eval();del payload
    if state_sha256(model)!=EXPECTED_STATE_SHA:raise RuntimeError('Strict T1 state hash mismatch')
    model.to('cuda');assert_frozen(model)
    def count_forward(module,args):
        if counts['model_forward_attempts']>=2100:raise RuntimeError('Hard 42-rollout/2100-forward budget')
        assert_frozen(module);counts['model_forward_attempts']+=1
    def returned(module,args,value):
        if value.requires_grad:raise RuntimeError('Model output unexpectedly requires gradient')
        counts['model_forward_returns']+=1
    model.register_forward_pre_hook(count_forward);model.register_forward_hook(returned)
    return model,dict(status='PASS',strict=True,missing_keys=incompatible.missing_keys,unexpected_keys=incompatible.unexpected_keys,
        file_sha256=EXPECTED_CHECKPOINT_SHA,state_sha256=EXPECTED_STATE_SHA,parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameter_count=0,source='Only designated Native_correct full T1; original official EMA not loaded',seconds=time.perf_counter()-started)


def assert_frozen(model):
    if model.training or any(p.requires_grad or p.grad is not None or p.dtype!=torch.float32 for p in model.parameters()):
        raise RuntimeError('Frozen T1 parameter invariant violated')
    if model.experiment_trainable:raise RuntimeError('Native execution must be readonly')


def initial_counts():
    return dict(model_forward_attempts=0,model_forward_returns=0,model_backward_calls=0,optimizer_updates=0,model_parameter_updates=0,
        complete_generation_attempts=0,complete_generations=0,baseline_replays=0,development_guided_generations=0,
        confirmation_generations=0,euler_updates=0,archived_baseline_trajectories_reused=8,geometry={},
        offline_corrector_calls=0,trajectory_corrector_calls=0,model_cuda_seconds=0.,geometry_cpu_seconds=0.,transfer_seconds=0.)


def make_protocol():
    return dict(schema='guided_native_geometry_validation_v1',run_scope='Frozen mixed-K T1, P0/P1 K12, zero new training',
        authorization=record(AUTH),real_original_V_epsilon_fraction=.005,
        epsilon_authorization='Explicit user answer epsilon=5e-3 L; fixed fixtures retain epsilon=.1',
        checkpoint=EXPECTED_CHECKPOINT_SHA,development_seeds=list(DEVELOPMENT),confirmation_seeds=list(CONFIRMATION),
        new_seed_prior_scan=dict(query=r'(?<![\w.])930[1-4](?![\w.])|seed_?930[1-4](?![0-9])',
            scope='Prior experiment *.py/*.md/*.json, excluding this new guided_native_geometry* implementation',matches=0,exit_code=1),
        fixed_geometry=dict(eta_fraction=1e-4,delta_fraction=.005,rho_fraction=.001,beta=.25,
            backtracking=[1.,.5,.25,.125],max_candidate_axes_signs_per_face=4,max_FP32_candidate_checks_per_module=64,
            RS_shared_rho_reference='Original same Euler Xhat',S_boundary_MSE_tolerance_L_squared=1e-15,
            R_separation_proxy_min_progress_L=1e-12,no_new_pair_policy='Proper and coplanar pair identity sets separately; C/free and affected free/free',
            near_zero_axis_tolerance=1e-12,primary_area_tolerance_L_squared=1e-12,independent_near_degenerate_guard_L_squared=1e-11),
        schedules=dict(steps=50,dt=.02,time='k/50',R_start_k=15,S_start_k=40,final_step_k=49,extra_endpoint_model_calls=0,N=112,K=12,free_faces=100),
        budgets=dict(training_updates=0,model_backward_calls=0,offline_corrector_invocations=24,baseline_replay_max=2,
            development_guided_max=24,confirmation_max=16,new_rollouts_max=42,model_forwards_max=2100),
        gates=dict(analytic='All mandated analytic detector fixtures must pass',archive='Historical metrics and proper-pair identities must reconcile; independent sampled audit must agree',
            R='At least 2 of 8 raw endpoints have fewer proper pairs; no new illegal pair identities or degeneracy in any R endpoint',
            S='At least 2 of 8 raw endpoints improve full length-weighted Gamma RMS; no new illegal pair identities or degeneracy in any S endpoint',
            RS='Only if both R and S pass their independent endpoint gates',
            off='Two full 51-state replays P0/P1 seed9201 must be bitwise equal to archived T1'),
        promotion=dict(proper_affected_face_relative_reduction_min=.25,maximum_parent_mean_R12_or_boundary_RMS_ratio=1.05,
            no_systematic_free_free_transfer='For each parent, mean free/free illegal pair count and unresolved pair count must not increase; report each individual change too',
            no_added_nonfinite_or_degenerate=True,C_bitwise_required=True,
            selection=['lowest macro parent proper affected-face fraction','lowest macro R12 symmetric RMS','lowest geometry CPU time'],
            selection_before_confirmation=True,confirmation_candidate_count_max=1),
        coordinates=dict(model_autocast='bfloat16',parameters='float32',Euler_C_noise_states='float32',geometry_working='float64',acceptance='actual float32'),
        invariants=['C bitwise fixed','No face additions/deletions','No topology rewelding','No GT in corrector/sampler','No macro-attraction','No extra CFG','No parameter search'],
        limitations=dict(CONTINUOUS_COLLISION_SAFETY='NOT_RUN',watertightness_certificate='NOT_PROVEN',new_parent_generalization='NOT_RUN',
            scale='Uses the original disclosed fixed parent bbox scale metadata, not an unknown-target deployment experiment'),
        implementation_development_counts=dict(geometry_fixed_fixture_runs=1,geometry_autograd_calls=11,primary_pair_tests=50,independent_pair_tests=50,verified_pair_tests=50,
            corrector_fixture_runs=1,corrector_invocations=9,corrector_FP32_candidates=12,corrector_accepted=9,corrector_rejected=3,
            corrector_direct_gate_checks=5,corrector_primary_pair_tests=20,corrector_independent_pair_tests=20,
            audit_helper_tests_runs=2,audit_helper_tests_per_run=3,audit_integration_geometry_autograd_calls=1,
            audit_integration_primary_pair_tests=1,audit_integration_independent_pair_tests=1,audit_integration_verified_pair_tests=1,
            model_forward_calls=0,model_backward_calls=0,optimizer_updates=0,asset_preflight_runs=1,
            shell_read_parse_failures=1,scope='Additional development checks before formal staged ledger; never silently combined with formal data'))


def seal_protocol(results):
    encoded=json.dumps(results['protocol'],sort_keys=True,separators=(',',':')).encode()
    digest=hashlib.sha256(encoded).hexdigest()
    if results.get('protocol_sha256',digest)!=digest:raise RuntimeError('Preregistered protocol changed')
    results['protocol_sha256']=digest


def add_counter(destination,values):
    for key,value in values.items():
        if isinstance(value,(int,float)) and not isinstance(value,bool):destination[key]=destination.get(key,0)+value


def score_rows_binding(rows,seeds):
    from .native_patch_metrics import patch_binding_four_grid
    result=[]
    for mode in sorted({x['model'] for x in rows}):
        for seed in seeds:
            matches=[next((x for x in rows if x['model']==mode and x['parent']==p and x['seed']==seed),None) for p in (0,1)]
            if all(x is not None for x in matches):result.append(dict(model=mode,K=12,seed=seed,metrics=patch_binding_four_grid(matches[0]['metrics'],matches[1]['metrics'])))
    return result


def frozen_new_sources():
    names=['geometry','corrector','audit']
    return {str(PROJECT/f'guided_native_geometry_{name}.py'):record(PROJECT/f'guided_native_geometry_{name}.py') for name in names}


def verify_records(records):
    changes=[]
    for name,entry in records.items():
        path=Path(name)
        if not path.exists() or file_sha256(path)!=entry['sha256']:changes.append(name)
    return dict(status='PASS' if not changes else 'FAIL',files=len(records),changed=changes)


def metric_flags(metrics):
    g=metrics['guided_geometry_audit'];cf=g['C_free'];ff=g['free_free']
    return dict(proper_pairs=cf['proper_pair_count'],proper_faces=cf['proper_affected_face_count'],proper_face_fraction=cf['proper_affected_face_fraction'],
        coplanar_pairs=cf['coplanar_positive_area_pairs'],unresolved=cf['unresolved_or_degenerate_pairs'],
        free_free_illegal=ff['proper_pair_count']+ff['coplanar_positive_area_pairs'],free_free_unresolved=ff['unresolved_or_degenerate_pairs'],
        boundary_rms=g['boundary']['rms_bbox_pct'],boundary_coverage=g['boundary']['coverage_fraction'],
        R12_symmetric_rms=metrics['remainder']['own']['symmetric_rms_bbox_pct'],
        whole_symmetric_rms=metrics['whole']['surface']['symmetric_rms_bbox_pct'],
        degenerate=len(g['free_face_area']['degenerate_faces_at_fixed_bbox_threshold']),C_exact=g['context_bitwise_preserved'])


def run_cpu_stages(results,bank,out):
    from . import guided_native_geometry_geometry as geo
    from . import guided_native_geometry_corrector as corrector
    from . import guided_native_geometry_audit as audit
    ledger=results['geometry_ledger'];counts=results['counts']
    fixture,arrays=geo.run_fixed_fixtures(ledger=ledger)
    corrector_fixture,corrector_arrays=corrector.run_cpu_fixtures()
    arrays.update({'corrector_'+k:v for k,v in corrector_arrays.items()})
    np.savez_compressed(out/'analytic_fixtures.npz',**arrays)
    results['fixture_probe']=dict(geometry=fixture,corrector=corrector_fixture,audit_helpers=audit.run_cpu_tests(),arrays=record(out/'analytic_fixtures.npz'))
    add_counter(counts['geometry'],corrector_fixture['counts'])
    results['status']='STAGE_A_PASS';atomic_json(out/'results.json',results)
    print(json.dumps(dict(event='stage_A_pass',fixture_families=9,corrector_fixture_families=corrector_fixture['fixture_count'],model_calls=0)),flush=True)
    results['archive_audit']=audit.audit_archived_trajectories(bank,bank['prior_results'],out,epsilon_fraction=.005,ledger=ledger)
    results['status']='STAGE_B_PASS';atomic_json(out/'results.json',results)
    print(json.dumps(dict(event='stage_B_reconciled',baselines=results['archive_audit']['parent_baselines'],independent_agree=results['archive_audit']['independent_all_resolved_and_illegal_agree'])),flush=True)
    if not results['archive_audit']['independent_all_resolved_and_illegal_agree']:
        results['status']='STOP_INDEPENDENT_REAL_DETECTOR_DISAGREEMENT';return
    base={(x['parent'],x['seed']):x for x in results['archive_audit']['endpoints']}
    rows=[];out.joinpath('offline').mkdir()
    for mode in ('R','S','RS'):
        for parent in (0,1):
            case=bank['cases_by_key'][parent,12];C=case['constraints'];L=case['bbox_diagonal'];boundary=corrector.prepare_boundary(C,L)
            for seed in DEVELOPMENT:
                with np.load(bank['archived_paths'][parent,seed],allow_pickle=False) as archive:raw=archive['output'].copy()
                counts['offline_corrector_calls']+=1
                changed,detail=corrector.correct_geometry(raw[12:],C,L,mode,boundary)
                add_counter(counts['geometry'],detail['counts']);counts['geometry_cpu_seconds']+=detail['seconds']
                output=np.concatenate([C.reshape(12,9),changed.reshape(100,9)])
                path=out/'offline'/f'{mode}_P{parent}_seed{seed}.npz'
                if path.exists():raise FileExistsError(path)
                np.savez_compressed(path,output=output)
                metric=audit.score_endpoint(output,bank,parent,ledger=ledger)
                before=base[parent,seed]['metrics']['guided_geometry_audit'];after=metric['guided_geometry_audit']
                changes={part:audit.pair_set_change(before[part],after[part]) for part in ('C_free','free_free')}
                no_new=all(not value['added'] for part in changes.values() for value in part.values())
                no_degenerate=set(after['free_face_area']['degenerate_faces_at_fixed_bbox_threshold']).issubset(set(before['free_face_area']['degenerate_faces_at_fixed_bbox_threshold']))
                resolved_not_worse=all(after[part]['unresolved_or_degenerate_pairs']<=before[part]['unresolved_or_degenerate_pairs'] for part in ('C_free','free_free'))
                hard_gate=no_new and no_degenerate and resolved_not_worse and np.array_equal(output[:12],raw[:12]) and np.isfinite(output).all()
                flags=metric_flags(metric);old_flags=metric_flags(base[parent,seed]['metrics'])
                row=dict(mode=mode,parent=parent,seed=seed,raw_path=str(path),raw_sha256=file_sha256(path),correction=detail,metrics=metric,
                    metrics_role='Endpoint coordinates supplied directly to shared scorer; scorer performs no additional repair',
                    pair_set_changes=changes,hard_gate=bool(hard_gate),proper_pairs_reduced=flags['proper_pairs']<old_flags['proper_pairs'],
                    full_boundary_RMS_improved=flags['boundary_rms']<old_flags['boundary_rms'],before=old_flags,after=flags)
                rows.append(row);results['offline_gate']=dict(status='RUNNING',rows=rows)
                atomic_json(out/'results.json',results)
                print(json.dumps(dict(event='offline_endpoint',mode=mode,parent=parent,seed=seed,hard_gate=bool(hard_gate),before=old_flags,after=flags,accepted=sum(x['accepted'] for x in detail['modules']),seconds=detail['seconds'])),flush=True)
    decisions={}
    for mode in ('R','S'):
        own=[x for x in rows if x['mode']==mode];key='proper_pairs_reduced' if mode=='R' else 'full_boundary_RMS_improved'
        successes=sum(x[key] for x in own)
        decisions[mode]=dict(pass_gate=all(x['hard_gate'] for x in own) and successes>=2,improved_endpoints=successes,all_hard_gates_pass=all(x['hard_gate'] for x in own))
    eligible=[mode for mode in ('R','S') if decisions[mode]['pass_gate']]
    if eligible==['R','S'] and all(x['hard_gate'] for x in rows if x['mode']=='RS'):eligible.append('RS')
    results['offline_gate']=dict(status='PASS_EXECUTION',rows=rows,R_pass=decisions['R']['pass_gate'],S_pass=decisions['S']['pass_gate'],decisions=decisions,eligible_arms=eligible)
    baseline_scores=[dict(model='B',parent=x['parent'],K=12,seed=x['seed'],metrics=x['metrics'],raw_path=x['source_path'],source='Reused archived uncorrected mixed-K T1') for x in results['archive_audit']['endpoints']]
    results['development']=dict(status='PENDING_GATED_SAMPLING' if eligible else 'NOT_RUN_OFFLINE_GATE_FAILED',scores=baseline_scores,binding=score_rows_binding(baseline_scores,DEVELOPMENT),generated=[],promotion=dict(status='NOT_RUN'))
    results['confirmation']=dict(status='NOT_RUN',selected=None,scores=[],binding=[],generated=[])
    results['status']='CPU_STAGES_COMPLETE' if eligible else 'STOP_NO_CORRECTOR_PASSED_OFFLINE_GATE'
    results['new_geometry_sources_check']=verify_records(results['new_geometry_sources'])
    if results['new_geometry_sources_check']['status']!='PASS':raise RuntimeError('Geometry implementation changed during execution')
    seal_protocol(results);atomic_json(out/'results.json',results)
    print(json.dumps(dict(event='offline_gate_complete',eligible_arms=eligible,decisions=decisions,model_calls=counts['model_forward_attempts'])),flush=True)


def start_cpu(out):
    if out.exists():raise FileExistsError('New isolated output must not already exist: '+str(out))
    out.mkdir(parents=True)
    from .guided_native_geometry_geometry import initial_geometry_ledger
    results=dict(status='INITIALIZING',protocol=make_protocol(),counts=initial_counts(),geometry_ledger=initial_geometry_ledger(),
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),new_geometry_sources=frozen_new_sources(),
        protection=dict(scope='All prior output file sizes/mtime_ns; critical used inputs additionally byte-hashed',metadata_before=metadata_protection(out)))
    seal_protocol(results);atomic_json(out/'results.json',results)
    started=time.perf_counter()
    try:
        bank,results['assets']=prepare_assets();atomic_json(out/'results.json',results)
        run_cpu_stages(results,bank,out)
    except BaseException as exc:
        results['status']='STOP_ERROR_CPU_STAGES';results['failure']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        print(results['failure']['traceback'],flush=True)
    results['cpu_stage_seconds']=time.perf_counter()-started
    seal_protocol(results);atomic_json(out/'results.json',results)
    return results




def select_candidate(results):
    scores=results['development']['scores'];generated=results['development']['generated'];candidates=[]
    for mode in results['offline_gate']['eligible_arms']:
        parents=[]
        for parent in (0,1):
            base=[metric_flags(x['metrics']) for x in scores if x['model']=='B' and x['parent']==parent]
            changed=[metric_flags(x['metrics']) for x in scores if x['model']==mode and x['parent']==parent]
            if len(base)!=4 or len(changed)!=4:raise RuntimeError('Promotion needs all four fixed seeds for both parents')
            keys=('proper_face_fraction','boundary_rms','R12_symmetric_rms','free_free_illegal','free_free_unresolved','degenerate')
            b={key:float(np.mean([x[key] for x in base])) for key in keys}
            c={key:float(np.mean([x[key] for x in changed])) for key in keys}
            flags=dict(proper_reduction_at_least_25pct=c['proper_face_fraction']<=.75*b['proper_face_fraction'],
                R12_worsening_at_most_5pct=c['R12_symmetric_rms']<=1.05*b['R12_symmetric_rms'],
                boundary_worsening_at_most_5pct=c['boundary_rms']<=1.05*b['boundary_rms'],
                no_systematic_FF_transfer=c['free_free_illegal']<=b['free_free_illegal'],
                no_increased_FF_unresolved=c['free_free_unresolved']<=b['free_free_unresolved'],
                no_added_degenerate=all(c0['degenerate']<=b0['degenerate'] for c0,b0 in zip(changed,base)),
                C_all_exact=all(x['C_exact'] for x in changed))
            parents.append(dict(parent=parent,base=b,corrected=c,gates=flags,pass_gate=all(flags.values())))
        cost=sum(row['costs']['correction_cpu_seconds'] for row in generated if row['model']==mode)
        candidates.append(dict(mode=mode,parents=parents,pass_gate=all(p['pass_gate'] for p in parents),
            selection_key=[float(np.mean([p['corrected']['proper_face_fraction'] for p in parents])),
                float(np.mean([p['corrected']['R12_symmetric_rms'] for p in parents])),float(cost)]))
    eligible=sorted((c for c in candidates if c['pass_gate']),key=lambda c:tuple(c['selection_key']))
    return dict(status='PASS_CANDIDATE_SELECTED' if eligible else 'NO_CANDIDATE_MET_ALL_PREREGISTERED_GATES',
        candidates=candidates,selected=eligible[0]['mode'] if eligible else None,selected_before_confirmation=True,
        evaluated_seed_scope=list(DEVELOPMENT),not_generalization='Both parents were training parents; new seeds only test noise robustness')


def interpret_original_audit(results):
    illegal={'proper_transverse_penetration','coplanar_positive_area_overlap'}
    pairs=[pair for row in results.get('archive_audit',{}).get('states',[]) for pair in row.get('original_formula_audit',{}).get('proper_pair_details',[])]
    counts=dict(total_proper_pair_occurrences=len(pairs),blind_energy_pairs=0,nonzero_gradient_pairs=0,
        confidently_legal_after=0,proper_after=0,coplanar_after=0,unresolved_after=0,energy_decreased_but_illegal_remains=0)
    for row in pairs:
        after=row['after_actual_fp32'];label=after['classification'];resolved=after.get('resolved',True)
        counts['blind_energy_pairs']+=int(row['blind_original_vertex_energy']);counts['nonzero_gradient_pairs']+=int(row['local_gradient_nonzero'])
        counts['confidently_legal_after']+=int(resolved and label not in illegal and 'unresolved' not in label.lower() and 'degenerate' not in label.lower())
        counts['proper_after']+=int(label=='proper_transverse_penetration');counts['coplanar_after']+=int(label=='coplanar_positive_area_overlap')
        counts['unresolved_after']+=int(not resolved or 'unresolved' in label.lower() or 'degenerate' in label.lower())
        counts['energy_decreased_but_illegal_remains']+=int(row['energy_decreased'] and label in illegal)
    return dict(counts=counts,original_proper_removed_field='Only absence of old proper classification; confident clearance additionally requires resolved legal after class. Use independently derived counts here.',
        scope='Same original proper pair after a single global original-potential step; all-pair new-defect changes are separately retained; no CCD guarantee')


def run_sampling_stage(results,bank,out):
    from .guided_native_geometry_sampling import sample_frozen
    from .guided_native_geometry_audit import score_endpoint
    counts=results['counts'];ledger=results['geometry_ledger']
    if results['status']!='CPU_STAGES_COMPLETE':return
    if verify_records(results['new_geometry_sources'])['status']!='PASS':raise RuntimeError('Geometry source changed before model use')
    results['runtime_before_load']=configure_stable_runtime()
    results['sampling_source']=record(PROJECT/'guided_native_geometry_sampling.py')
    model,results['strict_load']=load_frozen_T1(counts)
    results['model_state_before']=EXPECTED_STATE_SHA
    results['runtime_environment']=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),device_total_memory=torch.cuda.get_device_properties(0).total_memory,
        deterministic=torch.are_deterministic_algorithms_enabled(),cpu_threads=torch.get_num_threads())
    replays=[];results['baseline_replay']=dict(status='RUNNING',rows=replays)
    try:
        for parent in (0,1):
            case=bank['cases_by_key'][parent,12];seed=9201
            row=sample_frozen(model,bank['noises'][seed],case['constraints'],case['bbox_diagonal'],'off',counts,
                out/'baseline_replay',f'B_P{parent}_seed{seed}','baseline_replays')
            with np.load(row['raw_path'],allow_pickle=False) as a:actual=a['path']
            with np.load(bank['archived_paths'][parent,seed],allow_pickle=False) as a:expected=a['path']
            equal=bool(np.array_equal(actual,expected));row.update(parent=parent,seed=seed,model='B',bitwise_51_equal=equal,
                maximum_absolute_difference=float(np.abs(actual.astype(np.float64)-expected.astype(np.float64)).max()))
            replays.append(row);atomic_json(out/'results.json',results)
            print(json.dumps(dict(event='baseline_replay',parent=parent,seed=seed,bitwise_equal=equal,model_forwards=counts['model_forward_attempts'])),flush=True)
            if not equal:raise RuntimeError('Correction-off full trajectory differs from immutable T1 archive; stop without retuning')
        results['baseline_replay']['status']='PASS_BITWISE_51_BOTH_PARENTS'
        for mode in results['offline_gate']['eligible_arms']:
            for parent in (0,1):
                case=bank['cases_by_key'][parent,12]
                for seed in DEVELOPMENT:
                    row=sample_frozen(model,bank['noises'][seed],case['constraints'],case['bbox_diagonal'],mode,counts,
                        out/'development',f'{mode}_P{parent}_seed{seed}','development_guided_generations')
                    row.update(model=mode,parent=parent,K=12,seed=seed)
                    results['development']['generated'].append(row)
                    with np.load(row['raw_path'],allow_pickle=False) as a:output=a['output'].copy()
                    metric=score_endpoint(output,bank,parent,ledger=ledger)
                    results['development']['scores'].append(dict(model=mode,parent=parent,K=12,seed=seed,metrics=metric,raw_path=row['raw_path']))
                    atomic_json(out/'results.json',results)
                    print(json.dumps(dict(event='development_complete',mode=mode,parent=parent,seed=seed,metrics=metric_flags(metric),model_forwards=counts['model_forward_attempts'])),flush=True)
        results['development']['binding']=score_rows_binding(results['development']['scores'],DEVELOPMENT)
        results['development']['status']='PASS_FIXED_DEVELOPMENT_COMPLETE'
        results['development']['promotion']=select_candidate(results)
        selected=results['development']['promotion']['selected']
        results['status']='DEVELOPMENT_COMPLETE'
        results['confirmation']=dict(status='PENDING' if selected else 'NOT_RUN_NO_CANDIDATE_PASSED',selected=selected,scores=[],binding=[],generated=[])
        seal_protocol(results);atomic_json(out/'results.json',results)
        print(json.dumps(dict(event='promotion_fixed_before_new_noise',promotion=results['development']['promotion'])),flush=True)
        if selected:
            noises={seed:torch.randn((1,112,9),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32).numpy() for seed in CONFIRMATION}
            np.savez_compressed(out/'confirmation_independent_gaussian.npz',**{f'seed{seed}':noise for seed,noise in noises.items()})
            results['confirmation']['noise']=dict(artifact=record(out/'confirmation_independent_gaussian.npz'),generation='Independent CPU torch.randn seeds9301-9304; no OT or GT assignment',hashes={str(seed):archive_array_hash(noise) for seed,noise in noises.items()})
            for mode in ('B',selected):
                for parent in (0,1):
                    case=bank['cases_by_key'][parent,12]
                    for seed in CONFIRMATION:
                        row=sample_frozen(model,noises[seed],case['constraints'],case['bbox_diagonal'],'off' if mode=='B' else mode,counts,
                            out/'confirmation',f'{mode}_P{parent}_seed{seed}','confirmation_generations')
                        row.update(model=mode,parent=parent,K=12,seed=seed);results['confirmation']['generated'].append(row)
                        with np.load(row['raw_path'],allow_pickle=False) as a:output=a['output'].copy()
                        metric=score_endpoint(output,bank,parent,ledger=ledger)
                        results['confirmation']['scores'].append(dict(model=mode,parent=parent,K=12,seed=seed,metrics=metric,raw_path=row['raw_path']))
                        atomic_json(out/'results.json',results)
                        print(json.dumps(dict(event='confirmation_complete',mode=mode,parent=parent,seed=seed,metrics=metric_flags(metric),model_forwards=counts['model_forward_attempts'])),flush=True)
            results['confirmation']['binding']=score_rows_binding(results['confirmation']['scores'],CONFIRMATION)
            results['confirmation']['status']='PASS_FIXED_CONFIRMATION_COMPLETE'
        results['status']='PASS_EXECUTION'
    finally:
        assert_frozen(model)
        results['model_state_after']=state_sha256(model)
        results['model_state_frozen']=results['model_state_after']==EXPECTED_STATE_SHA
        if not results['model_state_frozen']:raise RuntimeError('Frozen model state changed')
        del model;gc.collect();torch.cuda.empty_cache()


def finalize(results,bank,out,refresh_report=False):
    results['original_formula_interpretation']=interpret_original_audit(results)
    critical=verify_records(results.get('assets',{}).get('critical_files',{}))
    after=metadata_protection(out);before=results['protection']['metadata_before']
    changed=[name for name,item in before.items() if after.get(name)!=item]
    results['protection'].update(critical_byte_hash_check=critical,historical_metadata_count=len(before),
        historical_metadata_changed=changed,all_historical_metadata_unchanged=not changed,
        caution='Full byte hashes checked for used critical inputs; all other historical outputs/official EMA protected by metadata checks, not claimed byte-hashed')
    results['new_geometry_sources_check']=verify_records(results['new_geometry_sources'])
    if changed or critical['status']!='PASS' or results['new_geometry_sources_check']['status']!='PASS':
        results['status']='FAIL_PRESERVATION_CHECK'
    # Separate corrector's own broad-phase ledger from independent scorer/probe ledger.
    results['ledger_definition']=dict(formal_probe_and_score='geometry_ledger',corrector_and_sampling='counts.geometry (includes formal corrector fixtures; nested trajectory domains if sampling ran)',
        extra_development='protocol.implementation_development_counts',
        model_forward_budget_includes='Every attempted actual model invocation, including baseline replays and partial failed paths',
        geometric_vs_model_autograd='CPU potential autograd is allowed and separately counted; model backward and updates must remain zero')
    counts=results['counts']
    if counts['model_forward_attempts']>2100 or counts['complete_generation_attempts']>42 or counts['optimizer_updates'] or counts['model_backward_calls'] or counts['model_parameter_updates']:
        raise RuntimeError('Hard experimental budget/invariant violated')
    seal_protocol(results)
    new_sources=sorted(PROJECT.glob('guided_native_geometry_*.py'))
    results['code_sources']=[record(path) for path in new_sources]
    chunks=[]
    for path in new_sources:
        relative=path.relative_to(REPO)
        proc=subprocess.run(['git','diff','--no-index','--','/dev/null',str(relative)],cwd=REPO,capture_output=True)
        if proc.returncode not in (0,1):raise RuntimeError('Cannot create scoped code diff: '+proc.stderr.decode())
        chunks.append(proc.stdout)
    (out/'code.diff').write_bytes(b''.join(chunks))
    check=subprocess.run(['git','apply','--reverse','--check',str(out/'code.diff')],cwd=REPO,capture_output=True)
    results['code_diff_reverse_check']=dict(exit_code=check.returncode,stderr=check.stderr.decode())
    if check.returncode:raise RuntimeError('Code diff reverse check failed')
    results['code_diff']=record(out/'code.diff')
    from .guided_native_geometry_report import deliver_report
    results['figure']=deliver_report(results,out,bank,replace_existing=refresh_report)
    atomic_json(out/'results.json',results)
    print(json.dumps(dict(event='finalized',status=results['status'],report=str(out/'GUIDED_NATIVE_GEOMETRY_REPORT.md'),counts=counts)),flush=True)


def finish_run(out):
    started=time.perf_counter();results=json.loads((out/'results.json').read_text(encoding='utf-8-sig'));seal_protocol(results)
    bank,asset_recheck=prepare_assets()
    results['assets_recheck']=dict(status=asset_recheck['status'],checkpoint_file_sha=asset_recheck['checkpoint']['sha256'],seconds=asset_recheck['seconds'],model_calls=0)
    try:
        run_sampling_stage(results,bank,out)
    except BaseException as exc:
        results['status']='STOP_ERROR_SAMPLING';results['failure']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        print(results['failure']['traceback'],flush=True)
    results['sampling_and_finish_seconds_before_report']=time.perf_counter()-started
    results['elapsed']=dict(cpu_stages_seconds=results.get('cpu_stage_seconds'),sampling_seconds=results['sampling_and_finish_seconds_before_report'])
    atomic_json(out/'results.json',results)
    finalize(results,bank,out)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--stage',choices=['cpu','finish','report'],required=True)
    args=parser.parse_args();out=args.out.resolve()
    if args.stage=='cpu':value=start_cpu(out)
    elif args.stage=='finish':value=finish_run(out)
    else:
        value=json.loads((out/'results.json').read_text(encoding='utf-8-sig'));bank,_=prepare_assets();finalize(value,bank,out,refresh_report=True)
    print(json.dumps(dict(status=value['status'],output=str(out),counts=value['counts'])),flush=True)




