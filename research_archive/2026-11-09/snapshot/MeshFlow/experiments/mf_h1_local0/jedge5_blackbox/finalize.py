"""Final CPU identity/accounting audit; no models, losses, gradients or scoring.

Call only after all workers, report and figures have finished. The failed
50-forward recording is counted as an executed rollout but has no saved raw.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,tempfile,time
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path(__file__).resolve().parent
ACTIVE={'RUNNING','PREPARING','SCORING','TRAINING'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def rec(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def write(path,data):
    p=Path(path)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def arrhash(a,scheme='native_numpy'):
    a=np.ascontiguousarray(a)
    if scheme=='probe':prefix=str(a.dtype).encode()+str(a.shape).encode()
    elif scheme=='native_torch':prefix=str(('torch.'+str(a.dtype),tuple(a.shape))).encode()
    else:prefix=str((str(a.dtype),tuple(a.shape))).encode()
    return hashlib.sha256(prefix+a.tobytes()).hexdigest()


def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def git(*args,cwd=ROOT,allow=(0,)):
    r=subprocess.run(['git','--no-pager',*args],cwd=cwd,capture_output=True)
    if r.returncode not in allow:raise RuntimeError(r.stderr.decode(errors='replace'))
    return r.stdout


def finalize(out):
    out=Path(out).resolve();tick=time.perf_counter()
    if (out/'final_audit.json').exists():raise FileExistsError('Final artifact already sealed')
    checks=[];references={}
    def check(name,passed,actual=None,expected=None):
        checks.append(dict(name=name,passed=bool(passed),actual=actual,expected=expected))
    def load(relative,status='PASS'):
        path=out/relative
        if not path.is_file():raise FileNotFoundError('Final dependency missing: '+str(path))
        d=read(path)
        if d.get('status') in ACTIVE:raise RuntimeError('Do not seal while worker active: '+relative)
        if status:check(relative+'/status',d.get('status')==status,d.get('status'),status)
        references[relative]=rec(path);return d
    protocol=load('registration/protocol.json','REGISTERED')
    amended=load('registration/protocol_amendment1.json','REGISTERED')
    old=Path(protocol['previous_run']);oldprotocol=read(old/'registration/protocol.json')
    check('amendment_model_budget',protocol['total_model_forwards_max']==1440 and amended['total_model_forwards_max']==1490)
    check('amendment_only_one_failed_full',amended.get('failed_completed_rollouts')==1 and
          amended.get('complete_replays_including_failed_recording')==17 and amended.get('replay_forwards_including_failed_recording')==1250)
    for key in ('cases','models','precisions','nfe','source_hashes','assets','noise_archive_sha256','probes'):
        check('amendment_unchanged/'+key,amended.get(key)==protocol.get(key))
    check('modelset',protocol['models']==['B','J'])
    check('registered_precision_and_nfe',protocol['precisions']==['bf16'] and protocol['nfe']==[50,100])
    expected_cases={(model,0,patch,seed,nfe) for model in ('B','J') for patch in (0,4)
                    for seed in (10401,10402) for nfe in (50,100)}
    noise_path=old/'registration/noises.npz';noise=arrays(noise_path)
    check('protected_original_noise_file',sha(noise_path)==protocol['noise_archive_sha256'])
    protections=[]
    for name,want in protocol['source_hashes'].items():
        p=ROOT/name;actual=sha(p)
        check('source_unchanged/'+name,actual==want,actual,want);protections.append(dict(kind='source',**rec(p),registered_sha256=want))
    # Include all original three protected checkpoints and both original input files.
    for item in oldprotocol['checkpoint_identities']:
        p=Path(item['path']);want=item.get('checkpoint_sha256',item['sha256']);actual=sha(p)
        check('checkpoint_unchanged/'+item['label'],actual==want,actual,want)
        protections.append(dict(kind='checkpoint',label=item['label'],**rec(p),registered_sha256=want))
    for name,item in oldprotocol['inputs'].items():
        p=Path(item['path']);actual=sha(p);check('data_unchanged/'+name,actual==item['sha256'])
        protections.append(dict(kind='data',label=name,**rec(p),registered_sha256=item['sha256']))
    current_head=git('rev-parse','HEAD').decode().strip();check('git_HEAD_unchanged',current_head==protocol['git_head'])
    diffs=[]
    for name in git('ls-files','-z','--','native_t1').decode().split('\0'):
        if name.endswith('.py') and not any(x in Path(name).parts for x in ('tests','runs','maintenance')):
            for options in ([],['--cached']):
                if git('diff','--no-ext-diff',*options,'--',name):diffs.append(dict(path=name,index=bool(options)))
    check('tracked_public_production_no_diff',not diffs,diffs)
    execution=[];cases=[];raw_by_case={}
    for model in ('B','J'):
        r=load('trajectories/'+model+'/results.json');c=r['counts'];expectedstate=protocol['assets'][model]['state_sha256']
        check(model+'/trajectory_count',c['model_forward_attempts']==c['model_forward_returns']==600 and
              c['full_rollout_attempts']==c['complete_rollouts']==8 and len(r['rows'])==8)
        check(model+'/frozen_state',r['identity'].get('state_sha256')==expectedstate and r.get('final_state_sha256')==expectedstate)
        check(model+'/frozen_file',r['identity'].get('checkpoint_sha256')==protocol['assets'][model]['file_sha256'])
        check(model+'/no_backward_update_repair',all(c.get(k,0)==0 for k in ('model_backward_calls','optimizer_updates','geometry_gradient_calls','postprocessing_repairs')))
        execution.append(dict(unit='trajectories_'+model,forwards_attempted=c['model_forward_attempts'],forwards_returned=c['model_forward_returns'],
            full_attempted=c['full_rollout_attempts'],full_completed=c['complete_rollouts'],saved_raw=len(r['rows']),seconds=r.get('seconds')))
        for row in r['rows']:
            key=(model,row['parent'],row['patch'],row['seed'],row['nfe']);path=out/row['path'];z=arrays(path);nfe=row['nfe']
            check(str(key)+'/unique',key not in raw_by_case);raw_by_case[key]=z
            check(str(key)+'/filehash',sha(path)==row['sha256'])
            check(str(key)+'/dtype_shape',z['states'].dtype==np.float32 and z['states'].shape==(nfe+1,112,9) and
                z['velocities'].dtype==np.float32 and z['velocities'].shape==(nfe,112,9) and z['raw'].shape==(112,9))
            check(str(key)+'/finite',all(np.isfinite(z[k]).all() for k in ('states','raw','velocities','C','times')))
            check(str(key)+'/endpoint',bits(z['raw'],z['states'][-1]))
            check(str(key)+'/C_clamped',bits(z['states'][:,:12],np.broadcast_to(z['C'].reshape(1,12,9),(nfe+1,12,9))))
            check(str(key)+'/initial_noise_free',bits(z['states'][0,12:],noise['seed_'+str(row['seed'])].reshape(112,9)[12:]))
            check(str(key)+'/recorded_production_path',row.get('euler_reconstruction_bitwise') is True and row.get('known_coordinates_clamped_bitwise') is True)
            oldpath=Path(row['old_path']);oldz=arrays(oldpath)
            check(str(key)+'/old_filehash',sha(oldpath)==row['old_sha256'])
            inds=oldz.get('state_indices',np.arange(len(oldz['states'])))
            check(str(key)+'/all_archived_states_exact',bits(z['states'][inds],oldz['states']))
            check(str(key)+'/old_endpoint_exact',bits(z['raw'],oldz['raw']))
            check(str(key)+'/condition_hash',arrhash(z['C'].reshape(12,9),'native_torch')==row['audit']['condition_sha256'])
            check(str(key)+'/noise_hash',arrhash(noise['seed_'+str(row['seed'])],'native_torch')==row['audit']['noise_sha256'])
            cases.append(dict(model=model,model_state_sha256=expectedstate,checkpoint_file_sha256=protocol['assets'][model]['file_sha256'],
                parent=row['parent'],patch=row['patch'],seed=row['seed'],N=112,K=12,nfe=nfe,precision='bf16',
                parameter_dtype='torch.float32',integration_dtype='float32',actual_model_output_dtype=row['audit']['actual_dtypes']['velocity']['dtype'],
                file=rec(path),old_file=rec(oldpath),old_state_indices=inds.tolist(),
                condition_tensor_hash=row['audit']['condition_sha256'],noise_tensor_hash=row['audit']['noise_sha256'],
                array_hashes={k:dict(dtype=str(v.dtype),shape=list(v.shape),native_numpy_sha256=arrhash(v),probe_sha256=arrhash(v,'probe')) for k,v in z.items()},
                raw_known_velocity_zero=row.get('raw_known_velocity_zero'),known_coordinates_clamped_bitwise=row.get('known_coordinates_clamped_bitwise')))
    check('full_16_case_matrix',{(r['model'],r['parent'],r['patch'],r['seed'],r['nfe']) for r in cases}==expected_cases)
    reused_fp32=[]
    for model in ('B','J'):
        original_results=read(old/'A'/model/'results.json')
        for patch in (0,4):
            for seed in (10401,10402):
                matches=[x for x in original_results['rows'] if x['parent']==0 and x['patch']==patch and x['seed']==seed
                         and x['precision']=='fp32' and x['nfe']==100]
                check(f'reused_FP32/{model}/{patch}/{seed}/unique_manifest',len(matches)==1)
                if len(matches)!=1:continue
                row=matches[0];path=old/row['path'];z=arrays(path)
                check(f'reused_FP32/{model}/{patch}/{seed}/file_SHA',sha(path)==row['sha256'])
                check(f'reused_FP32/{model}/{patch}/{seed}/101_states',z['states'].dtype==np.float32 and z['states'].shape==(101,112,9)
                      and np.array_equal(z['state_indices'],np.arange(101)) and np.isfinite(z['states']).all())
                bf=raw_by_case[(model,0,patch,seed,100)]
                check(f'reused_FP32/{model}/{patch}/{seed}/same_initial_condition',bits(z['states'][0],bf['states'][0]) and
                      bits(z['states'][:,:12],np.broadcast_to(bf['C'].reshape(1,12,9),(101,12,9))))
                reused_fp32.append(dict(model=model,parent=0,patch=patch,seed=seed,precision='fp32',nfe=100,
                    model_state_sha256=protocol['assets'][model]['state_sha256'],file=rec(path),
                    original_results=rec(old/'A'/model/'results.json'),new_model_forwards=0,reused=True,
                    purpose='Archived FP32/100 state trajectory used by dynamics.py; no fresh FP32 generation',
                    condition_tensor_hash=row['audit']['condition_sha256'],noise_tensor_hash=row['audit']['noise_sha256'],
                    array_hashes={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=arrhash(v)) for k,v in z.items()}))
    check('eight_reused_FP32_complete_trajectories',len(reused_fp32)==8)
    failed=load('trajectories/B/failed_attempt1.json','STOPPED');fc=failed['counts']
    check('failed_recording_actual50F1full',fc['model_forward_attempts']==fc['model_forward_returns']==50 and fc['complete_rollouts']==fc['full_rollout_attempts']==1)
    check('failed_recording_no_raw_rows',failed.get('rows')==[])
    check('failed_recording_observation_error','Known velocity not exactly zero' in failed.get('failure',{}).get('error',''))
    check('failed_recording_identity',failed['identity']['state_sha256']==protocol['assets']['B']['state_sha256'])
    check('failed_recording_no_scientific_retry_change',fc.get('sampling_failures')==0 and all(fc.get(k,0)==0 for k in
          ('model_backward_calls','optimizer_updates','geometry_gradient_calls','postprocessing_repairs')))
    execution.append(dict(unit='failed_recording_B_1',forwards_attempted=50,forwards_returned=50,full_attempted=1,full_completed=1,
        saved_raw=0,seconds=failed.get('seconds'),note='Finished Euler rollout; observation assertion failed before raw saving. No fabricated raw artifact.'))
    matched=[]
    for model in ('B','J'):
        r=load('probes/'+model+'/results.json');selection=load('probes/'+model+'/selection.json',None);c=r['counts']
        check(model+'/probe120',c['model_forward_attempts']==c['model_forward_returns']==120 and len(r['states'])==12 and len(r['forward_log'])==120)
        check(model+'/probe_frozen',r['state_before']==r['state_after']==protocol['assets'][model]['state_sha256'])
        check(model+'/probe_rng_unchanged',all(r.get(k) is True for k in ('CPU_torch_RNG_unchanged','CUDA_torch_RNG_unchanged','NumPy_global_RNG_unchanged')))
        check(model+'/probe_no_backward',all(c.get(k,0)==0 for k in ('model_backward_calls','optimizer_updates','OT_calls','autograd_calls','full_rollouts','euler_updates')))
        check(model+'/probe_selection_unchanged',sha(out/'probes'/model/'selection.json')==r['selection_sha256'] and selection['selections']==r['selections'])
        logs={(x['case'],x['precision'],x['tag']):x for x in r['forward_log']}
        check(model+'/probe_log_unique',len(logs)==120 and all(x['status']=='RETURNED' and x['C_bitwise_equal'] is True for x in r['forward_log']))
        execution.append(dict(unit='probes_'+model,forwards_attempted=120,forwards_returned=120,full_attempted=0,full_completed=0,saved_raw=0,seconds=r.get('wall_seconds')))
        seen=set()
        for row in r['states']:
            key=(row['parent'],row['patch'],row['seed'],row['k']);check(model+str(key)+'/unique_probe',key not in seen);seen.add(key)
            z=raw_by_case[(model,row['parent'],row['patch'],row['seed'],50)];p=Path(row['path']);q=arrays(p);k=row['k']
            check(model+str(key)+'/probe_filehash',sha(p)==row['sha256'])
            check(model+str(key)+'/probe_matched_state',bits(q['state'],z['states'][k]) and arrhash(q['state'],'probe')==row['state_sha256'])
            check(model+str(key)+'/probe_matched_velocity',bits(q['recorded_bf16_velocity'],z['velocities'][k]) and bits(q['bf16_base'],z['velocities'][k]))
            check(model+str(key)+'/probe_matched_time_C',bits(q['time'],z['times'][k]) and bits(q['C'],z['C'].reshape(12,9)))
            for direction in (61401,61402):
                for sign in ('plus','minus'):
                    check(model+str(key)+f'/direction{direction}/{sign}/C_fixed',bits(q[f'direction_{direction}_{sign}_input'][:12],q['C']))
                check(model+str(key)+f'/direction{direction}/identity',arrhash(q[f'direction_{direction}_offset'],'probe')==r['directions'][str(direction)]['array_sha256'])
            tag=f"P0_patch{row['patch']}_seed{row['seed']}_k{k:02d}"
            # Worker uses unpadded k in the actual forward-log key.
            unpadded=f"P0_patch{row['patch']}_seed{row['seed']}_k{k}"
            for precision in ('bf16','fp32'):
                log=logs.get((unpadded,precision,'base'),logs.get((tag,precision,'base')))
                check(model+str(key)+'/'+precision+'/base_input_identity',log is not None and log['input_sha256']==row['state_sha256'])
                if log:check(model+str(key)+'/'+precision+'/output_identity',log['output_sha256']==arrhash(q[precision+'_base'],'probe'))
            matched.append(dict(model=model,parent=row['parent'],patch=row['patch'],seed=row['seed'],k=k,
                time=float(q['time']),source_nfe=50,query_precisions=['bf16','fp32'],model_state_sha256=r['state_before'],
                state_array_sha256=row['state_sha256'],state_hash_scheme='dtype bytes + shape bytes + array bytes (probe)',
                file=rec(p),source_trajectory_file_sha256=next(v['file']['sha256'] for v in cases if (v['model'],v['patch'],v['seed'],v['nfe'])==(model,row['patch'],row['seed'],50)),
                array_hashes={name:arrhash(value,'probe') for name,value in q.items()}))
    check('24_matched_probe_states',len(matched)==24)
    check('shared_two_probe_directions',read(out/'probes/B/results.json')['directions']==read(out/'probes/J/results.json')['directions'])
    geometry=load('geometry/endpoint_results.json');shapes=load('geometry/trajectory_results.json');dynamics=load('dynamics/results.json')
    rms_names=('symmetric_RMS_bbox_pct','forward_RMS_bbox_pct','backward_RMS_bbox_pct')
    rms_max_delta={name:max(abs(x['original_score_delta'][name]) for x in geometry['rows']) for name in rms_names}
    # Use the original scorer reconciliation contract; observed floating-point
    # rounding is not a bitwise identity test. Saved delta = actual - reference.
    check('geometry_16_endpoint_RMS_reconciliations',len(geometry['rows'])==16 and
        all(set(x['original_score_delta'])==set(rms_names) and
            all(np.isclose(x[name],x[name]-x['original_score_delta'][name],rtol=1e-12,atol=1e-12)
                for name in rms_names) for x in geometry['rows']),
        actual=dict(max_absolute_delta=rms_max_delta,compared_outputs=len(geometry['rows'])),
        expected=dict(rtol=1e-12,atol=1e-12,source='geometry.py original endpoint reconciliation'))
    check('geometry_CPU_fixtures',geometry.get('tests',{}).get('status')=='PASS')
    plot_manifest=load('geometry/directional_plot_manifest.json')
    known_case={Path(x['file']['path']).stem:x for x in cases}
    check('directional_plot_16_identities',len(plot_manifest['rows'])==16 and all(
        x['key'] in known_case and x['raw_sha256']==known_case[x['key']]['old_file']['sha256'] and
        x['model_state_sha256']==protocol['assets'][x['model']]['state_sha256'] for x in plot_manifest['rows']))
    check('shape_16_saved_trajectories',len(shapes['rows'])==16)
    case_files={Path(v['file']['path']).stem:v['file']['sha256'] for v in cases}
    check('shape_source_identity',all(case_files.get(x['key'])==x['source_sha256'] for x in shapes['rows']))
    for name,d in [('geometry',geometry),('shape',shapes),('dynamics',dynamics)]:
        check(name+'/no_scientific_model_update',d.get('model_forwards',0)==d.get('optimizer_updates',0)==0)
    recorder_test=load('registration/recorder_cpu_test.json')
    check('recorder_mock_only',recorder_test.get('real_model_forwards')==0 and recorder_test.get('all_states_bitwise') is True)
    recorder_history=load('registration/recorder_cpu_test_execution_history.json')
    invocations=recorder_history['invocations']
    mock_forward_total=sum(x['cpu_mock_model_forwards'] for x in invocations)
    mock_real_forward_total=sum(x['real_model_forwards'] for x in invocations)
    check('recorder_two_actual_mock_invocations',len(invocations)==2 and
          all(x['status']=='PASS' and x['cpu_mock_model_forwards']==100 and x['real_model_forwards']==0 for x in invocations)
          and mock_forward_total==recorder_history['cpu_mock_model_forwards_total']==200
          and mock_real_forward_total==recorder_history['real_model_forwards_total']==0,
          actual=dict(invocations=invocations,cpu_mock_model_forwards_total=mock_forward_total,real_model_forwards_total=mock_real_forward_total))
    fixture=load('edge/fixture_record.json');teacher=load('edge/teacher_summary.json');proxy=load('edge/trajectory_summary.json')
    check('Edge_geometry_autograd3',fixture['counts']['geometry_autograd_calls']==3 and teacher['counts']['geometry_autograd_calls']==proxy['counts']['geometry_autograd_calls']==0)
    check('Edge_real_teacher_reconciles',teacher['states']==20 and teacher['max_recorded_scalar_absolute_error']<=2e-9 and teacher['max_observational_reconstruction_error']<=2e-9)
    check('Edge_proxy16',proxy['trajectories']==16 and proxy['counts']['nearest_boundary_state_calls']==1216)
    check('Edge_proxy_sources',all(case_files.get(x['name'])==x['input_sha256'] for x in proxy['rows']))
    for name,d in [('fixture',fixture),('teacher',teacher),('proxy',proxy)]:
        check('Edge/'+name+'/zero_model_OT_updates',all(d['counts'][k]==0 for k in ('model_forwards','model_backwards','OT_calls','optimizer_updates','repairs')))
    # CPU record is persisted by its original executing worker; this audit never reruns tests.
    cpu_candidates=[out/'probes/cpu_test_record.json',out/'registration/probes_cpu_test.json',out/'registration/probe_cpu_test.json',out/'probes/cpu_tests.json']
    cpu=next((p for p in cpu_candidates if p.is_file()),None)
    if cpu is None:
        cpu=next((p for p in SOURCE.glob('*probe*test*record*.json')),None)
    if cpu is None:raise FileNotFoundError('Original probe CPU test record must be persisted before final audit')
    test=read(cpu);check('probe_CPU_tests_PASS',test.get('status')=='PASS');references['probe_CPU_test_record']=rec(cpu)
    combined=load('probes/combined_results.json')
    master=load('results.json',None);check('master_results_complete',master.get('status') in ('PASS','PASS_EXECUTION'));check('human_report_exists',(out/'BLACKBOX_REPORT.md').is_file())
    totals=dict(model_forward_attempts=sum(x['forwards_attempted'] for x in execution),
        model_forward_returns=sum(x['forwards_returned'] for x in execution),
        full_rollout_attempts=sum(x['full_attempted'] for x in execution),complete_rollouts=sum(x['full_completed'] for x in execution),
        saved_trajectories=len(cases),model_backward_calls=0,optimizer_updates=0,OT_calls=0,CPU_geometry_autograd_calls=3,
        CPU_public_Edge_forward_calls=fixture['counts']['public_edge_forward_calls']+teacher['counts']['public_edge_forward_calls'],
        CPU_finite_difference_loss_evaluations_subset=fixture['counts']['finite_difference_loss_evaluations'],
        cpu_mock_model_forwards=mock_forward_total)
    check('1490_actual_forwards',totals['model_forward_attempts']==totals['model_forward_returns']==1490)
    check('17_executed_full_16_saved',totals['full_rollout_attempts']==totals['complete_rollouts']==17 and len(cases)==16)
    check('budget_matches_amendment',totals['model_forward_returns']==amended['total_model_forwards_max'])
    manifest=dict(schema='jedge5_blackbox_per_case_identity_v1',status='PASS' if all(x['passed'] for x in checks) else 'FAIL',
        array_hash_schemes=dict(native_numpy='sha256(str((str(dtype),tuple(shape))) + rawbytes)',native_torch='same with torch.dtype string',
                               probe='sha256(str(dtype)+str(shape)+rawbytes)'),trajectories=cases,reused_FP32_trajectories=reused_fp32,matched_probe_states=matched,
        failed_attempt=dict(record=rec(out/'trajectories/B/failed_attempt1.json'),forwards=50,full_rollouts=1,raw_artifact='NOT_SAVED',
            failure_kind='Incorrect observation-only assertion on raw known velocity; known coordinates are clamped by unchanged Euler sampler'),
        noise_archive=rec(noise_path))
    write(out/'case_identity_manifest.json',manifest)
    # Only isolated new experiment Python goes into the patch. Existing baseline is not copied.
    sources=sorted(SOURCE.rglob('*.py'));patch=[];source_records=[]
    for p in sources:
        relative=p.relative_to(ROOT).as_posix();patch.append(git('diff','--no-index','--binary','--','/dev/null',relative,allow=(1,)))
        source_records.append(dict(path=relative,sha256=sha(p),bytes=p.stat().st_size))
    diff=out/'code.diff'
    if diff.exists():raise FileExistsError(diff)
    diff.write_bytes(b''.join(patch))
    with tempfile.TemporaryDirectory(prefix='jedge5_blackbox_diff_') as temporary:
        git('apply','--check',str(diff),cwd=temporary);git('apply',str(diff),cwd=temporary)
        for x in source_records:
            check('diff_reconstruction/'+x['path'],sha(Path(temporary)/x['path'])==x['sha256'])
    artifacts=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name not in ('artifact_manifest.json','final_audit.json') and not any(x in p.parts for x in ('__pycache__',)):
            artifacts.append(dict(relative_path=str(p.relative_to(out)),**rec(p)))
    artifact_manifest=dict(schema='jedge5_blackbox_artifacts_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        artifacts=artifacts,sources=source_records,source_patch=rec(diff),
        scope='Hash inventory only. No checkpoints, datasets, old source trees or model tensors copied. This manifest and final_audit are excluded from self-recursive hashes.')
    write(out/'artifact_manifest.json',artifact_manifest)
    failures=[x for x in checks if not x['passed']]
    final=dict(schema='jedge5_blackbox_final_audit_v1',status='PASS' if not failures else 'FAIL',
        created_utc=datetime.now(timezone.utc).isoformat(),counts=totals,execution=execution,checks=checks,failed_checks=failures,
        protected_assets=protections,registered_HEAD=protocol['git_head'],actual_HEAD=current_head,public_production_diffs=diffs,
        RMS_reconciliation=dict(max_absolute_delta=rms_max_delta,rtol=1e-12,atol=1e-12),
        references=references,case_identity_manifest=rec(out/'case_identity_manifest.json'),artifact_manifest=rec(out/'artifact_manifest.json'),code_diff=rec(diff),
        shared_GPU_lock=dict(status='NOT_USED',requirement_timing='User requirement arrived after all actual GPU work had completed',
            this_run='Resource snapshots and measured two-worker dispatch; no shared lock claim',
            future='Shared lock requirement applies to subsequent GPU executions; no new GPU work was launched to simulate compliance'),
        audit_execution=dict(seconds=time.perf_counter()-tick,model_constructions=0,model_forwards=0,model_backwards=0,
            geometry_loss_calls=0,geometry_autograd_calls=0,OT_calls=0,optimizer_updates=0,geometry_scoring_calls=0,
            checkpoint_deserializations=0,weight_access='Streaming file SHA only'),
        limitations=['The 50-forward failed recording is counted but no raw data are invented.',
            'CPU audit compares stored bytes/hashes and existing scalar records; it does not repeat geometry scoring or scientific probes.',
            'Worker durations may overlap under parallel execution and must not be summed as elapsed wall time.',
            'PASS concerns accounting and identity, not causal attribution, ODE stiffness, or unseen-object generalization.'])
    write(out/'final_audit.json',final);return final


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();r=finalize(a.out)
    print(json.dumps(dict(status=r['status'],counts=r['counts'],failed_checks=len(r['failed_checks']))),flush=True)
    raise SystemExit(0 if r['status']=='PASS' else 2)

if __name__=='__main__':main()
