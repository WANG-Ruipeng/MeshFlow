"""Fixed pure Native T1 confirmation: original sampler, three frozen models, zero updates."""
from .stable_runtime import configure_stable_runtime
import argparse,copy,gc,hashlib,json,platform,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from .guided_native_geometry_run import PROJECT,REPO,SOURCE,CONFIG,prepare_assets,record,verify_records
from .official_wrapper import state_sha256
from .native_inpainting_run import load_native,release
from .native_patch_run import clamped_sample
from .nxc112_run import EMA,EMA_SHA,BASE_SHA
from .learning_data import array_hash
AUTH=Path('/mnt/c/Users/warpwang/.codex/attachments/5a072421-a492-4ba3-93b2-12bd6a31edcf/Pasted text.txt')
SEEDS=tuple(range(9401,9409));DEV=(9201,9202,9203,9204)
ENDPOINTS={'T1':('Native_correct','7928d77507bf1fe610876f61b5b11e614263323b5fc0f595cc6740ffd24af634'),
    'T2':('Native_independent','d2033369165f85b7dd6e906e25d039172e5a9344460a2906dd6c637491fc14aa')}


def save(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)


def event(name,**kw):print(json.dumps(dict(event=name,**kw),allow_nan=False),flush=True)


def write(r,out,status=None):
    if status:r['status']=status
    save(out/'results.json',r)


def seal(protocol):return hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def protected_metadata(out):
    paths=[p for p in (PROJECT/'outputs').rglob('*') if p.is_file() and not p.is_relative_to(out)]
    paths.extend(p for p in PROJECT.glob('*.py') if not p.name.startswith('t1_native_confirmation_'))
    paths.extend((REPO/'models').rglob('*.py'));paths.extend([EMA,CONFIG])
    return {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(paths))}


def protocol():
    return dict(schema='pure_native_T1_confirmation_v1',authorization=record(AUTH),new_seeds=list(SEEDS),development_seeds=list(DEV),
        seed_status='9401..9408 preregistered previously, no generated model outputs, no selection or tuning; exact record/filename searches found no prior use',
        N=112,K=12,free_faces=100,y=112,roles='Original known/free, known time1, free timek/50',
        contexts=dict(P0=[30,33,21,17,18,20,109,110,19,57,51,56],P1=[24,27,30,23,17,18,19,22,59,60,95,109]),
        models=['Native0','T1','T2'],load_execution_order=['T1','Native0','T2'],checkpoints=ENDPOINTS,
        official_EMA_state_sha256=BASE_SHA,official_EMA_file_sha256=EMA_SHA,
        sampler=dict(call='native_patch_run.clamped_sample, imported unchanged',steps=50,dt=1/50,t='torch.float32 k/50',
            model_precision='BF16 autocast',parameters_C_noise_states='FP32',known_first12=True,CUDA_before_init_CUBLAS=':4096:8',
            deterministic=True,warn_only=False,TF32=False,SDPA='math',eval=True,frozen=True,compile=False,
            extra_velocity_terms=0,position_corrections=0,extra_CFG=0,GT_or_OT_initialization=False),
        noise=dict(generator='CPU torch.Generator.manual_seed(seed)',shape=[1,112,9],dtype='float32',distribution='torch.randn',
            shared='Identical free slots12:112 for all3models and2contexts, no output-based resampling'),
        budget=dict(formal_rollouts=48,T1_historical_replays=2,total_attempted_rollouts=50,total_model_forwards=2500,
            extra_model_warmup_or_test_forwards=0,optimizer_updates=0,model_parameter_updates=0,model_backwards=0,
            geometry_gradient_calls=0,postprocessing_repairs=0,teacher_forwards=0),
        gates=['Asset identity/decomposition','Input topology calibration performed without repair','T1 two historical all51-state bitwise replays'],
        topology=dict(source_indices='Original source GT and C subset compared with exactFP32 coordinate reindexing',
            winding='Current corner order versus solvable orientation parity constraints separately; no output reorientation',
            anomalies='Source C-only inherited link/edge issues separate from generated free incidence; C defects do not cancel confirmation'),
        evaluation=dict(original_metrics_unchanged=True,free_extraction='Once by known_mask; same100 faces for both target scores',
            surface_points=512,surface_seed=9911,fixed_R12_points=512,fixed_R12_seed='9973+parent',
            original_R12_coverage_threshold_fraction=.01,boundary='Original length-weighted whole C boundary; RMS,p95,coverage',
            all_CF_pairs=1200,all_FF_pairs=4950,static_CC_pairs=66,CCD='NOT_RUN',
            near_C_readonly=dict(points=512,seed=9951,distance_fraction=.001,boundary_band_fraction=.001,
                classification='One fixed area sample bank; nearC interior/boundary-band, never error verdict/penalty/removal'),
            per_parent=['mean','median','Q1','Q3','worst','paired T1 vs Native0/T2 improvement counts'],
            P0_reference_line_pct=2.,reference_line_role='Historical descriptive reference only',
            binding='E00 E01 E11 E10; Delta0=E01-E00 Delta1=E10-E11 B=(Delta0+Delta1)/2; both directions separately',
            net_share_suppression='abs(sumB)<=1e-8 OR abs(sumB)<=.1*sumabsB; always show perseed and leave-one-out',
            development_not_pooled=True,independent_parent_count=2),
        figure=dict(seed=SEEDS[0],parents=[0,1],models=['Native0','T1','T2'],views=['full','free_only'],common_camera_and_bounds=True),
        stop='Complete fixed48 plus2replays, no result-based promotion gate and no next-stage training or modules')


def initialize(out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'raw').mkdir();p=protocol()
    r=dict(status='INITIALIZING',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol=p,protocol_sha256=seal(p),
        counts=dict(model_forward_attempts=0,model_forward_returns=0,native_forward_attempts=0,native_forward_returns=0,
            official_forward_attempts=0,official_forward_returns=0,rollout_attempts=0,complete_rollouts=0,formal_completed=0,
            historical_replays_completed=0,euler_updates=0,model_backward_calls=0,optimizer_updates=0,model_parameter_updates=0,
            geometry_gradient_calls=0,postprocessing_repairs=0,forward_by_stage={},attempts_by_stage={}),
        active_stage='INITIALIZING',models={},generated=[],scores=[],bindings=[],gates={},evaluation_ledger={},
        protection=dict(metadata_before=protected_metadata(out)),initial_git_status=subprocess.run(['git','status','--short'],cwd=REPO,capture_output=True,text=True).stdout.strip(),
        AGENTS_search=dict(status='No applicable AGENTS.md found',scope='E:/AGENTS.md,repo ancestors,repo recursive rg --files AGENTS.md'),
        environment=dict(python=platform.python_version(),torch=torch.__version__,torch_CUDA=torch.version.cuda,numpy=np.__version__,platform=platform.platform()))
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    r['environment']['GPU_query']=dict(stdout=gpu.stdout.strip(),stderr=gpu.stderr.strip(),returncode=gpu.returncode,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(r,out)
    bank,assets=prepare_assets();old=bank['prior_results'];r['assets']=assets
    endpoints={}
    for model,(arm,expected) in ENDPOINTS.items():
        item=record(SOURCE/arm/'endpoint_step500.pt')
        if item['sha256']!=expected or old['arms'][arm]['checkpoint']['sha256']!=expected:raise RuntimeError('Wrong endpoint file '+model)
        endpoints[model]=dict(item,state_sha256=old['arms'][arm]['checkpoint']['state_sha256'],backbone_state_sha256=old['arms'][arm]['checkpoint']['backbone_state_sha256'],arm=arm)
        r['assets']['critical_files'][item['path']]=item
    official=record(EMA)
    if official['sha256']!=EMA_SHA:raise RuntimeError('Official checkpoint bytes differ')
    r['assets'].update(model_endpoints=endpoints,official_EMA=official,Native0_state_sha256=old['native_initial_state_sha256'])
    r['assets']['critical_files'][official['path']]=official
    for name in ('native_inpainting_run.py','native_patch_run.py','native_patch_data.py','guided_native_geometry_audit.py','gpu_guidance_protected_weld.py','gpu_guidance_evaluation.py'):
        item=record(PROJECT/name);r['assets']['critical_files'][item['path']]=item
    cases=[]
    for p in (0,1):
        case=bank['cases_by_key'][p,12];source=next(c for c in old['assets']['cases'] if (c['parent_index'],c['K'])==(p,12))
        C=case['constraints'];tri=case['full_target'].reshape(112,3,3).astype(np.float64)
        area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)
        area_fraction=float(area[case['source_face_ids']].sum()/area.sum())
        if abs(area_fraction-source['area_fraction'])>1e-12:raise RuntimeError('Context area audit differs')
        cases.append(dict(parent=p,parent_uid=case['parent_uid'],N=112,K=12,L=case['bbox_diagonal'],source_face_ids=case['source_face_ids'],
            context_area_fraction=area_fraction,context_xyz_min=C.reshape(-1,3).min(0).tolist(),context_xyz_max=C.reshape(-1,3).max(0).tolist(),
            full_xyz_min=tri.reshape(-1,3).min(0).tolist(),full_xyz_max=tri.reshape(-1,3).max(0).tolist(),face_fraction=12/112,
            semantic_annotation=False,condition_geometry_sha256=array_hash(C)))
    r['assets']['fixed_cases']=cases
    r['development']=dict(source_results=record(SOURCE/'results.json'),seeds=list(DEV),scope='Historical repeated development; never pooled withnewconfirmation',
        scores=[copy.deepcopy(x) for x in old['scores'] if x['K']==12 and x['model'] in ('Native0','T1','T2')],
        bindings=[copy.deepcopy(x) for x in old['bindings'] if x['K']==12 and x['model'] in ('Native0','T1','T2')])
    prior=PROJECT/'outputs/gpu_guidance_and_protected_weld_20260929_160351/results.json'
    q=json.loads(prior.read_text());prior_seeds=sorted(set(x['seed'] for x in q['generated']))
    if set(SEEDS)&set(prior_seeds) or q['confirmation']['status']!='NOT_RUN':raise RuntimeError('New seed usage requires preregistration review')
    r['seed_usage_audit']=dict(prior_result=record(prior),prior_confirmation=q['confirmation'],prior_generated_seeds=prior_seeds,
        prior_Gaussian_archive_exists=(prior.parent/'confirmation_independent_Gaussian.npz').exists(),
        exact_model_seed_record_search='No prior seed:9401..9408 or seed940x output references at17:07:14+08',
        exact_output_filename_search_matches=0,broad_context_false_hits='Hash substrings and scalar0.019401; not seed use',
        conclusion='Previously preregistered only; no model results examined, noselectionandtuning; retain9401..9408')
    write(r,out,'ASSETS_PASS');event('assets_pass',output=str(out),areas=cases,models={k:v['sha256'] for k,v in endpoints.items()})
    return r


class Ledger:
    def __init__(self,r):self.r=r
    def attach(self,model,kind):
        def pre(module,args):
            c=self.r['counts'];stage=self.r['active_stage']
            if c['model_forward_attempts']>=2500:raise RuntimeError('Hard2500forward budget')
            if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):raise RuntimeError('Frozen eval required')
            c['model_forward_attempts']+=1;c[kind+'_forward_attempts']+=1;c['forward_by_stage'][stage]=c['forward_by_stage'].get(stage,0)+1
        def post(module,args,value):
            if value.requires_grad:raise RuntimeError('Backbone graph forbidden')
            self.r['counts']['model_forward_returns']+=1;self.r['counts'][kind+'_forward_returns']+=1
        model.register_forward_pre_hook(pre);model.register_forward_hook(post)


def role_facts(model):
    role=model.role_embedding.detach();return dict(shape=list(role.shape),dtype=str(role.dtype),requires_grad=role.requires_grad,
        array_sha256=array_hash(role.cpu().numpy()),nonzero=int(torch.count_nonzero(role)),RMS=float(role.double().square().mean().sqrt()),max_abs=float(role.abs().max()))


def load_frozen(model_name,r):
    model,ema_load=load_native(Ledger(r));model.set_trainable(False).eval()
    record_load=dict(official_EMA_load=ema_load,original_Native0_initial_state_sha256=state_sha256(model),original_role=role_facts(model))
    if record_load['original_Native0_initial_state_sha256']!=r['assets']['Native0_state_sha256'] or record_load['original_role']['nonzero']!=0:raise RuntimeError('Native0 initialization changed')
    if model_name in ENDPOINTS:
        item=r['assets']['model_endpoints'][model_name];payload=torch.load(item['path'],map_location='cpu',weights_only=True,mmap=True)
        if (payload['arm'],payload['completed_updates'],payload['schema'])!=(item['arm'],500,'native_connected_patch_v1'):raise RuntimeError('Endpointmetadatawrong')
        keys=model.load_state_dict(payload['model'],strict=True);del payload
        expected=item['state_sha256'];expected_backbone=item['backbone_state_sha256']
        record_load['endpoint_strict_load']=dict(missing_keys=keys.missing_keys,unexpected_keys=keys.unexpected_keys,strict=True,file_sha256=item['sha256'])
    else:expected=r['assets']['Native0_state_sha256'];expected_backbone=BASE_SHA
    full=state_sha256(model);backbone=state_sha256(model.backbone)
    if full!=expected or backbone!=expected_backbone:raise RuntimeError('Loaded state hash failed')
    r['models'][model_name]=dict(status='LOADED_FROZEN',load=record_load,state_before=full,backbone_before=backbone,role_before=role_facts(model),
        parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    event('model_loaded',model=model_name,state=full,trainable=0);return model


def sample_one(model,model_name,parent,seed,noise,C,r,out,replay=False):
    c=r['counts']
    if c['rollout_attempts']>=50:raise RuntimeError('Hard50rolloutattempt budget')
    stage='T1_historical_replay' if replay else model_name+'_confirmation';r['active_stage']=stage
    c['rollout_attempts']+=1;c['attempts_by_stage'][stage]=c['attempts_by_stage'].get(stage,0)+1
    label=f'{model_name}_P{parent}_K12_seed{seed}'+('_historical_replay' if replay else '')
    target=out/'raw'/f'{label}.npz'
    if target.exists():raise FileExistsError(target)
    before=c['model_forward_attempts'];tick=time.perf_counter()
    z=torch.from_numpy(noise.copy()).cuda();context=torch.from_numpy(C.copy()).cuda()
    raw,path,stats=clamped_sample(model,z,context,c,native=True)
    wall=time.perf_counter()-tick
    if path.shape!=(51,112,9) or raw.dtype!=np.float32 or not np.array_equal(raw,path[-1]):raise RuntimeError('Rawtrajectorycontract')
    expected=np.broadcast_to(C.reshape(12,9),(51,12,9))
    if not np.array_equal(path[:,:12].view(np.uint32),expected.copy().view(np.uint32)):raise RuntimeError('Cbitwisechanged')
    if not np.array_equal(path[0,12:].view(np.uint32),noise[0,12:].view(np.uint32)):raise RuntimeError('FreeGaussianchanged')
    if not np.isfinite(path).all() or c['model_forward_attempts']-before!=50:raise RuntimeError('NonfiniteorNFE')
    disk=time.perf_counter();np.savez_compressed(target,output=raw,path=path);disk=time.perf_counter()-disk
    row=dict(phase='historical_replay' if replay else 'confirmation',model=model_name,parent=parent,seed=seed,K=12,variant='Raw',
        raw_path=str(target),raw_sha256=record(target)['sha256'],output_sha256=array_hash(raw),sampling=stats,
        costs=dict(complete_generation_wall_seconds=wall,sampler_seconds=stats['seconds'],peak_cuda_allocated=stats['peak_cuda_allocated'],NPZ_export_seconds=disk,
            scope='Original source sampler incl per-step state D2H; wall additionally input transfer; no CPU geometry'),
        C_all51_FP32_bits_exact=True,all_states_finite=True,initial_free_Gaussian_exact=True,no_postprocessing=True)
    if replay:c['historical_replays_completed']+=1
    else:c['formal_completed']+=1;r['generated'].append(row)
    event('sample_complete',model=model_name,parent=parent,seed=seed,replay=replay,forwards=c['model_forward_attempts'],seconds=wall)
    return row


def preflight(r,bank,out):
    from .t1_native_confirmation_topology import run_input_audit,run_cpu_fixtures as topology_tests
    from .t1_native_confirmation_evaluation import run_cpu_tests as metric_tests, audit_context
    r['helper_tests']=dict(topology=topology_tests(),evaluation=metric_tests())
    from .nxc112_data import _stage_b_parents
    parents,manifest_audit=_stage_b_parents()
    if [p['uid'] for p in parents]!=[bank['cases_by_key'][p,12]['parent_uid'] for p in (0,1)]:raise RuntimeError('Manifestparentsdiffer')
    manifest_path=PROJECT/'outputs/run_20260928_01/split_manifest.json';manifest=json.loads(manifest_path.read_text())
    r['assets']['data_manifest']=dict(manifest_audit,manifest_record=record(manifest_path),selected_parents=parents)
    paths=[manifest_path,Path(manifest['split_path']),Path(manifest['split_path']).with_name('test.npz'),*[Path(p['path']) for p in parents]]
    for path in paths:
        entry=record(path);r['assets']['critical_files'][entry['path']]=entry
    for parent in parents:
        if record(parent['path'])['sha256']!=parent['sha256']:raise RuntimeError('Manifestsourcehashdiffers')
    r['input_topology']=run_input_audit(bank,out)
    r['assets']['context_audits']={str(p):audit_context(bank['cases_by_key'][p,12]['constraints'],bank['cases_by_key'][p,12]['bbox_diagonal'],r['evaluation_ledger']) for p in (0,1)}
    r['implementation_development_counts']=dict(topology_fixture_runs=1,topology_fixture_families=5,indexed_topology_audits=8,exact_coordinate_indexings=1,
        evaluation_helper_runs=1,evaluation_fixture_families=7,model_forwards=0,geometry_gradient_calls=0,geometry_repairs=0,optimizer_updates=0,failed_executions=0)
    # The audits may find real C defects: only asset/coordinate identity errors raise.
    r['new_core_sources']={str(p):record(p) for p in PROJECT.glob('t1_native_confirmation_*.py') if p.name!='t1_native_confirmation_report.py'}
    r['noise_ledger']=dict(model_results_seen_before_noise_creation=False,arrays={},shared_across_models_and_contexts=True)
    arrays={}
    for seed in SEEDS:
        generator=torch.Generator(device='cpu').manual_seed(seed);a=torch.randn((1,112,9),generator=generator,dtype=torch.float32).numpy()
        arrays[f'noise_seed{seed}']=a;r['noise_ledger']['arrays'][str(seed)]=dict(shape=list(a.shape),dtype=str(a.dtype),sha256=array_hash(a),free_sha256=array_hash(a[:,12:]))
    np.savez_compressed(out/'fixed_confirmation_noise.npz',**arrays);r['noise_ledger']['file']=record(out/'fixed_confirmation_noise.npz')
    write(r,out,'PREFLIGHT_PASS');event('preflight_pass',model_forwards=0,noise_seeds=list(SEEDS))


def generate(r,bank,out):
    if verify_records(r['new_core_sources'])['status']!='PASS':raise RuntimeError('Newsourcechanged')
    with np.load(out/'fixed_confirmation_noise.npz') as f:noises={seed:f[f'noise_seed{seed}'].copy() for seed in SEEDS}
    for name in ('T1','Native0','T2'):
        model=load_frozen(name,r);write(r,out)
        try:
            if name=='T1':
                r['gates']['T1_replays']=[]
                for parent in (0,1):
                    case=bank['cases_by_key'][parent,12]
                    row=sample_one(model,name,parent,9201,bank['noises'][9201],case['constraints'],r,out,replay=True)
                    with np.load(row['raw_path']) as a,np.load(bank['archived_paths'][parent,9201]) as b:equal=np.array_equal(a['path'].view(np.uint32),b['path'].view(np.uint32))
                    row['all51_bitwise_equals_archive']=bool(equal);r['gates']['T1_replays'].append(row);write(r,out)
                    if not equal:raise RuntimeError('HistoricalT1replaydiffers;stopformalconfirmation')
            for parent in (0,1):
                for seed in SEEDS:
                    case=bank['cases_by_key'][parent,12]
                    sample_one(model,name,parent,seed,noises[seed],case['constraints'],r,out)
                    write(r,out,'CONFIRMATION_GENERATING')
            current=r['models'][name];current.update(state_after=state_sha256(model),backbone_after=state_sha256(model.backbone),role_after=role_facts(model))
            if current['state_before']!=current['state_after'] or current['backbone_before']!=current['backbone_after'] or current['role_before']!=current['role_after']:raise RuntimeError('Frozenmodelchanged')
            current['status']='PASS_FROZEN';write(r,out)
        finally:
            del model;release()
    if r['counts']['model_forward_attempts']!=2500 or len(r['generated'])!=48 or r['counts']['complete_rollouts']!=50:raise RuntimeError('Fixedmatrixcountfailed')
    r['all_raw_complete_before_evaluation']=True;write(r,out,'RAW_COMPLETE')


def score(r,bank,out):
    from . import t1_native_confirmation_evaluation as ev
    bank['input_topology']=r['input_topology'];ledger=r['evaluation_ledger']
    bank['context_audits']={int(p):value for p,value in r['assets']['context_audits'].items()}
    # All new raw geometry exists before the first diagnostic, no feedback loop.
    for row in r['generated']:
        with np.load(row['raw_path']) as f:raw=f['output'].copy()
        tick=time.perf_counter();metrics=ev.score_output(raw,bank,row['parent'],ledger=ledger)
        r['scores'].append(dict(phase='confirmation',model=row['model'],parent=row['parent'],seed=row['seed'],K=12,variant='Raw',raw_path=row['raw_path'],metrics=metrics,
            offline_scoring_seconds=time.perf_counter()-tick))
        write(r,out,'OFFLINE_SCORING');event('scored',model=row['model'],parent=row['parent'],seed=row['seed'],scores=len(r['scores']))
    r['summary']=ev.aggregate_confirmation(r['scores'],SEEDS)
    r['bindings']=r['summary']['bindings']
    r['development']['summary']=ev.aggregate_confirmation(r['development']['scores'],DEV,phase='development')
    r['assets']['context_audits']=bank.get('context_audits',{})
    r['protection']['critical_files_after']=verify_records(r['assets']['critical_files'])
    after=protected_metadata(out);old=r['protection']['metadata_before'];changed=[p for p,v in old.items() if after.get(p)!=v]
    r['protection']['historical_metadata_after']=dict(status='PASS' if not changed else 'FAIL',files=len(old),changed=changed)
    r['protection']['new_core_after']=verify_records(r['new_core_sources'])
    if changed or any(r['protection'][x]['status']!='PASS' for x in ('critical_files_after','new_core_after')):raise RuntimeError('Assetprotectionfailed')
    r['counts']['active_model']='NONE';r['active_stage']='FINISHED';r['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    write(r,out,'PASS_EXECUTION');event('experiment_complete',counts=r['counts'])


def execute(out,stage):
    if stage=='init':return initialize(out)
    r=json.loads((out/'results.json').read_text())
    try:
        if seal(r['protocol'])!=r['protocol_sha256']:raise RuntimeError('Frozenprotocolchanged')
        configure_stable_runtime();bank,_=prepare_assets()
        if stage=='preflight':
            if r['status']!='ASSETS_PASS':raise RuntimeError('Preflightalreadyrun/wrongstate')
            preflight(r,bank,out)
        elif stage=='run':
            if r['status']!='PREFLIGHT_PASS':raise RuntimeError('Preflightmustpass')
            generate(r,bank,out);score(r,bank,out)
        elif stage=='report':
            if r['status']!='PASS_EXECUTION':raise RuntimeError('Experimentincomplete')
            from .t1_native_confirmation_report import deliver_report
            r['delivery']=deliver_report(r,out,bank);write(r,out)
    except BaseException as error:
        r['failure']=dict(stage=stage,error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True)
        write(r,out,'STOP_ASSET_OR_NUMERICAL_OR_IMPLEMENTATION_FAILURE');event('stopped',error=repr(error),counts=r['counts']);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--stage',choices=['init','preflight','run','report'],required=True)
    args=parser.parse_args();execute(args.out.resolve(),args.stage)


