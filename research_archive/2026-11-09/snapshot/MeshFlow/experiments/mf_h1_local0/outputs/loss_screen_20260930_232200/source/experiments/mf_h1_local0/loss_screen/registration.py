"""CPU-only registration for the fixed 2500-to-3000 seven-arm loss screen.

The caller must invoke register exactly once on a nonexistent directory. This
module never initializes a model, executes noise OT, runs fixtures again, or
freezes an unfinished driver. Root orchestration supplies the final source pin.
"""
from .common import (REPO,PACKAGE,INITIAL,INITIAL_SHA,INITIAL_STATE,ORIGINAL_SOURCE_SHA,
    INPUTS,INPUT_SHA,BANK,BANK_SHA,HISTORY,HISTORY_SHA,ARMS,CANDIDATES,DEV_SEEDS,CONF_SEEDS,
    LOG_STEPS,PARAMETERS,record,seal,write)
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from native_t1.artifacts import atomic_json,file_sha256,state_sha256
from native_t1.data import load_archive,NativeTrainingStream
from native_t1.geometry_checkpoint import validate_geometry_checkpoint
from native_t1.checkpoint import DEFAULT_CONFIG
from native_t1.portable_checkpoint import config_sha256


def _git(*arguments):
    return subprocess.run(['git','-C',str(REPO),*arguments],check=True,capture_output=True,text=True).stdout.rstrip('\n')


def _agents():
    # Only ancestor AGENTS apply to the experiment package. Save their content
    # if present; never interpret a source file as an executable instruction.
    records=[]
    for directory in reversed((PACKAGE,*PACKAGE.parents)):
        path=directory/'AGENTS.md'
        if path.is_file():records.append(dict(**record(path),text=path.read_text(encoding='utf-8-sig')))
    return dict(status='FOUND' if records else 'NONE_FOUND_ON_APPLICABLE_ANCESTOR_CHAIN',files=records,
        searched=[str(x/'AGENTS.md') for x in reversed((PACKAGE,*PACKAGE.parents))])


def _checked_record(path,expected):
    result=record(path)
    if result['sha256']!=expected:raise ValueError('Asset SHA256 differs: '+str(path))
    return result


def _initial(history):
    asset=_checked_record(INITIAL,INITIAL_SHA)
    checkpoint=torch.load(INITIAL,map_location='cpu',weights_only=True,mmap=True)
    validate_geometry_checkpoint(checkpoint,config_sha256(DEFAULT_CONFIG))
    if checkpoint['cumulative_updates']!=2500 or checkpoint['model_state_sha256']!=INITIAL_STATE:
        raise ValueError('Expected pure-FM G0 cumulative2500, not an auxiliary endpoint.')
    export=checkpoint.get('export',{})
    if (export.get('source_arm')!='G0' or export.get('source_file_sha256')!=ORIGINAL_SOURCE_SHA
        or export.get('source_state_sha256')!=INITIAL_STATE or export.get('auxiliary_modules_present') is not False
        or export.get('tensor_arithmetic_performed') is not False or export.get('training_updates_added')!=0):
        raise ValueError('Public G0 tensor-only export provenance differs.')
    if checkpoint['optimizer'] or export.get('optimizer_state_present') is not False:
        raise ValueError('Public G0 must have stripped optimizer; all arms create fresh AdamW.')
    state=checkpoint['model']
    if any(v.device.type!='cpu' or v.dtype!=torch.float32 for v in state.values()):
        raise ValueError('Initial generator must contain only CPU FP32 tensors.')
    if sum(v.numel() for v in state.values())!=PARAMETERS:
        raise ValueError('Generator parameter count differs.')
    # Existing identity algorithm without constructing a torch.nn.Module.
    actual=state_sha256(SimpleNamespace(state_dict=lambda:state))
    if actual!=INITIAL_STATE:raise ValueError('Actual initial tensor hash differs.')
    if any(not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('Initial generator has nonfinite tensors.')
    stream=copy.deepcopy(checkpoint['stream_state'])
    if (stream['seed'],stream['arm'],stream['batch_index'],stream['sample_index'])!=(1010,'Native_correct',1000,8000):
        raise ValueError('Must continue saved G0 RNG after batch1000/sample8000.')
    old=history['arms']['G0'];oldcp=old['checkpoints']['500']
    if (history['status']!='PASS_EXECUTION' or old['status']!='PASS' or old['completed_updates']!=500
        or oldcp['sha256']!=ORIGINAL_SOURCE_SHA or oldcp['state_sha256']!=actual or oldcp['cumulative_updates']!=2500
        or seal(old['final_stream_state'])!=seal(stream)):
        raise ValueError('Original G0 endpoint/RNG metadata does not match its portable export.')
    archive=load_archive(INPUTS,INPUT_SHA)
    restored=NativeTrainingStream(archive,seed=1010)
    restored.load_state_dict(copy.deepcopy(stream))
    if seal(restored.state_dict())!=seal(stream):raise ValueError('Input RNG roundtrip differs.')
    asset.update(state_sha256=actual,schema=checkpoint['schema'],cumulative_updates=2500,
        parameters=PARAMETERS,state_tensors=len(state),optimizer_state_present=False,
        CPU_metadata_and_tensor_identity='PASS',strict_model_key_shape_load='DEFERRED_TO_ROOT_GPU_PREFLIGHT',
        stream_sha256=seal(stream),stream_batch_index=1000,stream_sample_index=8000,
        stream_seed=1010,stream_arm='Native_correct',stream_restored_without_sampling=True,
        export=copy.deepcopy(export),context=copy.deepcopy(checkpoint['context']))
    del state,checkpoint,restored,archive
    return asset,stream


def _fixtures():
    result={}
    for name in ('relation','distribution'):
        path=PACKAGE/('cpu_'+name+'_record.json')
        item=json.loads(path.read_text(encoding='utf-8'))
        if item.get('status')!='PASS' or item.get('CUDA_initialized') is not False:
            raise ValueError('Completed CPU-only formula fixture record missing: '+name)
        if name=='distribution':
            hashes=item['source_sha256']
        else:hashes={'relation_losses.py':item['source_sha256'],'test_relation_losses.py':item['test_sha256']}
        for source,digest in hashes.items():
            if file_sha256(PACKAGE/source)!=digest:raise ValueError('CPU fixture no longer matches source: '+source)
        result[name]=dict(artifact=record(path),result=item)
    return dict(status='PASS_REUSED_COMPLETED_CPU_FIXTURES',rerun_at_registration=False,
        records=result,registration_model_calls=0,registration_OT_calls=0,registration_autograd_calls=0,
        prior_CPU_coordinate_autograd_calls=result['relation']['result']['counts']['autograd_calls']+result['distribution']['result']['CPU_coordinate_autograd_calls'],
        prior_CPU_loss_and_solver_work_is_not_training=True,
        L4_fixed96_warning='Perturbed two-face fixture marginal L1 residual .02446472894375129; finite approximation, not converged within1e-3.')


def _loss_definitions():
    return {
      'L0':dict(name='masked free-coordinate flow matching',formula='sum_valid_free_xyz (v-u)^2 / sum_effective8_valid_free_coordinates',auxiliary=None),
      'L1':dict(name='GT adjacent local side consistency',formula='mean_applicable_interface_pairs_and7points relu(-s*((p-a) dot n)/L)^2',
        relations='Trusted GT source shared edges between known and free; fixed across predictions',
        sign='sign((GT free off-edge vertex - known edge reference) dot known unit normal)',
        sign_ambiguous_abs_le=1e-6,sign_ambiguity_units='full-object bbox L',
        points='Three vertices, three edge midpoints, centroid',positive_contact_gap=0.,
        inapplicable='Coplanar/ambiguous signed relationships counted, not assigned arbitrary side',
        limitations='Not global SDF, IPC, or nonadjacent triangle collision prevention'),
      'L2':dict(name='GT adjacent unoriented dihedral preservation',formula='GT_edge_length_weighted_mean ((n_pred_i dot n_pred_j)^2-(n_GT_i dot n_GT_j)^2)^2',
        relations='Every actual GT shared-edge incident-face pair with >=1 free face; known-known excluded',
        known_normals='Fixed C',prediction_normal_epsilon=1e-8,normal_epsilon_units='cross product after division of coordinates by L',
        weights='Only fixed GT edge length, never predicted proximity',
        limitations='Translation, topology separation, sign and concavity ambiguities remain'),
      'L3':dict(name='Along-interface-edge tangential relaxation',formula='GT_edge_length_weighted_mean_endpoints (||r_perp/L||^2+.05*||r_parallel/L||^2)',
        parallel='tau*tau^T*r; tau is fixed GT shared-edge unit tangent',tangential_weight=.05,
        relations='Trusted known-free source edge endpoints, with normal multiple-edge incidence weighting',
        limitations='No projection, welding, or exact-edge guarantee'),
      'L4':dict(name='Area-weighted surface Sinkhorn divergence',formula='OT_eps(pred,GT)-.5*OT_eps(pred,pred)-.5*OT_eps(GT,GT)',
        quadrature=[[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]],
        probability='clamp_min(A_normalized,1e-12)/3, then normalize each measure to unit mass; predicted area retains gradient',
        normalized_area_floor=1e-12,cost='||p/L-q/L||^2 (no factor1/2)',epsilon=.0025,
        regularized_objective='sum(pi*C)+epsilon*[sum pi*(log(pi)-log(a)-log(b))-sum(pi)+sum(a)*sum(b)]',
        solver='96 fixed simultaneous half-damped Jacobi log-domain updates from zero dual potentials; full unrolled differentiation',
        iterations=96,GTGT_cache='Only detached GT-GT constant may be reused; actual solve/cache counts separate',
        marginal_residuals='Each solve row/column L1, Linf and max relative residual',
        marginal_L1_diagnostic_tolerance=.001,finite_residual_automatically_stops_arm=False,
        nonfinite_objective_or_gradient_stops_arm=True,negative_values_clamped=False,
        reference='https://www.kernel-operations.io/geomloss/api/pytorch-api.html',
        limitations='Finite-iteration approximate balanced OT; independent probability normalization does not constrain total area'),
      'L5':dict(name='Multiscale area and unsigned-normal surface measure',
        formula='mean_sigma [sum wp*wp*k(pred,pred)+sum wg*wg*k(GT,GT)-2*sum wp*wg*k(pred,GT)]',
        quadrature='Same three symmetric barycentric points as L4',
        weights='w_pred=A_pred/(3*A_GT_total), w_GT=A_GT/(3*A_GT_total), with detached GT total area; prediction is not mass-renormalized',
        Q='n*n^T; normalized cross product with epsilon1e-8',normal_epsilon=1e-8,
        kernel='exp(-||p/L-q/L||^2/(2*sigma_x^2))*exp(-||Q-R||_F^2/(2*sigma_Q^2))',
        sigma_x=[.02,.05,.10],sigma_Q=.5,predicted_area_normal_position_gradients=True,
        GTGT_cache='Detached GT-GT kernels only; actual kernel counts separate',negative_values_clamped=False,
        limitations='Finite quadrature, unsigned normal and surface-measure approximation, not exact topology or self-intersection guarantee'),
      'L6':dict(name='Known-free boundary and unknown-side band coverage',formula='.5*GT_edge_length_weighted_mean_gamma dist(point,S(pred_free))^2/L^2 + .5*GT_edge_length_weighted_mean_band dist(point,S(pred_free))^2/L^2',
        relations='Only actual trusted GT known-free shared edges, not all open C edges',s=[0.,.25,.5,.75,1.],r=[.1,.2],
        band='(1-r)*((1-s)*a+s*b)+r*c_free_GT',weights=[.5,.5],
        surface='Differentiable finite predicted free triangles only; fixed C excluded',
        distance_squared_denominator_epsilon=1e-24,nearest_triangle='Piecewise differentiable choice; selected distance is not detached',
        limitations='Finite local samples, barycentric rather than fixed physical band; may sacrifice remote coverage')}


def register(out):
    """Validate CPU assets and register once; zero model/OT/autograd/updates."""
    tick=time.perf_counter();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Registration refuses an existing output: '+str(out))
    if not out.is_relative_to(REPO.resolve()) or not out.name.startswith('loss_screen_'):
        raise ValueError('Use a new loss_screen_* output directory inside this workspace.')
    if torch.cuda.is_initialized():raise RuntimeError('Registration must run before CUDA initialization.')
    missing=[str(p) for p in (INITIAL,INPUTS,BANK,HISTORY,Path(DEFAULT_CONFIG)) if not p.is_file()]
    if missing:raise FileNotFoundError('Missing required assets; no fallback: '+', '.join(missing))
    agents=_agents();git=dict(HEAD=_git('rev-parse','HEAD'),status_porcelain=_git('status','--porcelain=v1','--untracked-files=normal'),
        reference_public_commit='5ef35b421ad00ccd3a97867ce0e33e5f77f65b1c',mutations_performed=False)
    assets=dict(inputs=_checked_record(INPUTS,INPUT_SHA),evaluation_inputs=_checked_record(BANK,BANK_SHA),
        historical_G0_results=_checked_record(HISTORY,HISTORY_SHA),chair_config=record(DEFAULT_CONFIG))
    history=json.loads(HISTORY.read_text(encoding='utf-8'))
    assets['initial_checkpoint'],stream=_initial(history)
    fixtures=_fixtures()
    fixtures['additional_read_only_sanity']=dict(provenance='Root completed once before registration; original message record, no rerun',
        input_sha256=INPUT_SHA,prepare_relations_calls=6,loss_calls=0,model_calls=0,OT_calls=0,autograd_calls=0,CUDA_initialized=False,
        cases=[dict(parent=p,K=K,interface_relations=interfaces,L1_applicable_relations=applicable,GT_nonmanifold_edges=0)
            for p,K,interfaces,applicable in ((0,4,6,4),(0,8,8,4),(0,12,12,6),(1,4,6,2),(1,8,10,4),(1,12,14,8))])
    from .evaluation import audit_seeds,PROTOCOL as EVALUATION_PROTOCOL
    seed_audit=audit_seeds(CONF_SEEDS,out)
    if seed_audit['status']!='PASS':
        raise ValueError('Confirmation seeds previously recorded; no automatic seed replacement: '+json.dumps(seed_audit,allow_nan=False))
    initial_stream_sha=seal(stream)
    protocol=dict(method='Exploratory fixed-budget six direct geometric-loss candidates plus fresh pure-FM baseline',
        user_override='All seven arms start at the same pure G0 cumulative2500 and add500 updates; final cumulative3000. Prior2500 G0 is not the new L0.',
        initial=dict(checkpoint_sha256=INITIAL_SHA,state_sha256=INITIAL_STATE,cumulative_updates=2500,parameters=PARAMETERS),
        training=dict(arms=list(ARMS),candidates=list(CANDIDATES),additional_updates_each=500,total_update_cap=3500,
            initial_stream_sha256=initial_stream_sha,stream_seed=1010,stream_start_batch=1000,stream_start_sample=8000,
            stream_end_batch=1500,stream_end_sample=12000,effective_batch=8,microbatch=1,train_K=[4,8,12],parents=[0,1],num_faces=112,
            optimizer='AdamW',fresh_optimizer_each_arm=True,lr=1e-5,betas=[.9,.95],weight_decay=0.,gradient_clip=1.,
            full_generator_role_Geo_trainable=True,modules_eval=True,scheduler=False,new_EMA=False,
            target_noise='Original free-only nested Hungarian OT permutes only Z; actual target/source face and corner identities follow target slots',
            FM_denominator='Original total effective-batch valid free coordinates',
            geometry_denominator='Each sample defined mean; sum over all eight samples divided by8, with inactive samples contributing0',
            xt='(1-t)*Z+t*Y',u='Y-Z',predicted_endpoint='Xt+(1-t)*v, free faces only',
            auxiliary='ramp(s)*lambda_j*I[t>=.5]*L_j(Yhat,C,Y)',ramp='min(additional_step/50,1)',gate='FP32_t>=0.5',
            inverse_time_compensation=False,flow_time_distribution='Original logit-normal',
            coordinate_scale='Only loss coordinates divide by fixed full-GT bbox diagonal; model/input coordinates unchanged',
            precision='Original BF16 generator; FP32 parameters/integration/geometric computation with geometric autocast disabled',
            deterministic_algorithms=True,warn_only=False,CUBLAS_WORKSPACE_CONFIG=':4096:8',TF32=False,SDPA='math',
            logs_at_steps=list(LOG_STEPS),save_steps=[250,500],cumulative_checkpoint_steps=[2750,3000],endpoint_selection='Fixed step500, no best checkpoint'),
        losses=_loss_definitions(),
        calibration=dict(common_initial_cumulative_updates=2500,clone_samples=8,fixed_times=[.5,.75],model_forwards=16,
            saved_velocities='One generator velocity per registered state; then independent velocity leaves, no extra generator calls per loss',
            lambda_formula='.1*median(||dL_FM/dv||)/median_nonzero(||dL_j/dv||), on applicable finite states',
            target_gradient_fraction=.1,invalid='No finite nonzero denominator stops candidate; never invent fallback lambda',
            times_calibrated=1,weight_scan=False,formal_stream_advanced=False,velocity_gradient_call_cap=112,
            training_velocity_gradient_ratio_steps=[100,500],training_ratio_microbatch_index=0,
            interpretation='Initial output-space scale only, not whole-model or sustained10% gradient guarantee'),
        failure_policy=dict(candidate_local=True,numerical_nonfinite='Stop affected candidate and preserve exception/counts; continue unrelated candidates',
            all_relations_inapplicable='NOT_APPLICABLE, no forced training',finite_L4_residual='Report finite96-step approximation;1e-3 diagnostic does not automatically stop or trigger retries',
            existing_BF16_permutation_failure='Preserve historical FAIL and tolerance; authorized baseline-inheritance contract; no new historical diagnosis'),
        evaluation=dict(architecture='Original strict T1+Geo generator only, no losses/GT/heads/guidance/repair during sampling',
            development=dict(seeds=list(DEV_SEEDS),seen_noise=True,models=list(ARMS),max_outputs=168),
            confirmation=dict(seeds=list(CONF_SEEDS),models='L0 plus at most two frozen selected candidates',max_outputs=72),
            parents=[0,1],patches=[0,1,2],patch0_noises=8,patch1_noises=2,patch2_noises=2,outputs_per_model=24,
            N=112,K=12,free_faces=100,euler_steps=50,C_bitwise_states=51,shared_raw_free_Gaussian=True,initial_noise_GT_OT=False,
            per_patch_reference='Each patch own free GT complement, original fixed evaluation points and thresholds',
            original_patches='Previously observed parents and patches; confirmation introduces only new noise',
            historical_G0='Reference-only24 old outputs may be reused, never counted as fresh L0 or ranked'),
        selection=dict(**EVALUATION_PROTOCOL,max_candidates=2,
            primary_metrics=['remainder_symmetric_RMS_bbox_pct','boundary_RMS_bbox_pct'],
            eligibility='Exclude numerical failure, changed C or wrong face count',
            macro='Noise mean within each parent/patch, then equal weight for six parent/patch cells',
            select='Take independently best nonbaseline RMS and independently best nonbaseline Gamma; each must strictly beat L0; deduplicate same winner; no runner-up replacement',
            missing_improvement='Leave slot empty',freeze_before_confirmation=True,retune_after_confirmation=False),
        budgets=dict(optimizer_updates=3500,training_model_forwards=28000,training_model_backwards=28000,
            calibration_model_forwards=16,calibration_velocity_autograd_calls=112,wiring_model_forwards_max=7,wiring_model_backwards_max=7,
            real_GT_oracle_loss_calls_max=48,teacher_model_forwards=0,development_formal_rollouts_max=168,
            confirmation_formal_rollouts_max=72,formal_rollouts_max=240,replay_rollouts_max=3,
            sampling_model_forwards_max=12150,all_model_forwards_max=40173,
            original_noise_OT_training_pairs_max=28000,original_noise_OT_calibration_pairs=8,
            distribution_internal_solves='Actual per-call accounting, separate from original noise OT; GT-GT cache reuse distinguished'),
        limitations=['Exploratory selection over six candidates, not independent discovery','Two trained parents and already-observed patches',
            'No independent training-seed replication','No loss guarantees watertightness, shared topology or absence of intersections'],
        stop='No seventh candidate, combinations, changed lambda/time/steps, automatic extension, commit or push')
    protocol_sha=seal(protocol);disk=shutil.disk_usage(REPO)
    r=dict(schema='native_t1_geometry_loss_screen_results_v1',status='INITIALIZED',output_dir=str(out),
        created_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,protocol_sha256=protocol_sha,
        assets=assets,stream_start=stream,git=git,AGENTS=agents,CPU_fixtures=fixtures,confirmation_seed_audit=seed_audit,
        environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,torch_cuda_build=torch.version.cuda,
            numpy=np.__version__,CUDA_initialized=False,cpu_count=os.cpu_count(),disk_total_bytes=disk.total,disk_free_bytes=disk.free,
            GPU_inventory='DEFERRED_TO_ROOT_PREFLIGHT'),
        counts=dict(calibration={},wiring={},training={},sampling={}),loss_accounting={},
        arms={arm:dict(status='PENDING',completed_updates=0,checkpoints={},logs=[]) for arm in ARMS},
        generated=[],replays=[],timing={},source_freeze=dict(status='PENDING_ROOT_FINALIZATION'),
        registration=dict(status='PASS',seconds=time.perf_counter()-tick,model_calls=0,OT_calls=0,autograd_calls=0,optimizer_updates=0,
            official_checkpoint_read=False,initial_public_checkpoint_CPU_mmap=True,initial_stream_generated_samples=0,fixtures_reexecuted=False))
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'protocol.json',protocol)
    write(r,out)
    return r