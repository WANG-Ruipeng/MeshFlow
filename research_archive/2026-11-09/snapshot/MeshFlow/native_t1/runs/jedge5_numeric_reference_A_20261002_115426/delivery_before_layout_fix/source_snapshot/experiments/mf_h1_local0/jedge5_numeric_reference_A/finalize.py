"""Final CPU audit/export for window A; run only after its fixed matrix ends.

This module never calls a model, a solver, OT, an optimizer or git mutation.
It audits persisted facts and validates an added-file patch in a run-local tree.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import importlib.metadata
import json
import os
import re
import subprocess
import tempfile
import time
import numpy as np
from .assets import ROOT, ASSETS, read_json, write_json, file_sha256, array_hash, load_registered_cases, verify_protected


def require(ok, message):
    if not ok: raise ValueError(message)


def bits(left,right):
    left=np.ascontiguousarray(left);right=np.ascontiguousarray(right)
    return left.shape==right.shape and left.dtype==right.dtype and left.tobytes()==right.tobytes()


def own_run(value):
    out=Path(value).resolve()
    require(out.parent==ROOT/'native_t1/runs' and out.name.startswith('jedge5_numeric_reference_A_'), 'Wrong window A run')
    return out


def _dtype_values(value):
    if isinstance(value,dict):
        return ([value['dtype']] if 'dtype' in value else [])+sum((_dtype_values(v) for k,v in value.items() if k!='dtype'),[])
    if isinstance(value,list): return sum((_dtype_values(v) for v in value),[])
    return []


def check_runtime(audit,model):
    require(audit['state_sha256_before']==ASSETS[model]['state']==audit['state_sha256_after'], 'Frozen state changed')
    require(audit['GPU_workers_this_window']==1, 'GPU worker-count contract changed')
    for key in ('training_updates','backward_calls','OT_calls'):
        require(audit[key]==0, 'Unexpected '+key)
    require(audit['postprocessing'] is None, 'Unexpected postprocessing')
    runtime=audit['runtime']
    for key in ('deterministic_algorithms','sdpa_math_enabled','all_modules_eval'):
        require(runtime[key] is True, 'Runtime flag must be true: '+key)
    for key in ('deterministic_warn_only','tf32_matmul','tf32_cudnn','cuda_autocast_enabled','gradient_enabled','sdpa_flash_enabled','sdpa_mem_efficient_enabled','sdpa_cudnn_enabled'):
        require(runtime[key] is False, 'Runtime flag must be false: '+key)
    require(runtime['cublas_workspace_config']==':4096:8' and runtime['parameter_dtype']=='torch.float32', 'CUBLAS/parameter dtype differs')
    actual=_dtype_values(audit['actual_dtypes'])
    require(actual and all(v=='torch.float32' for v in actual), 'Actual floating dtype audit differs')


def trajectory_audit(out,folder,case):
    row=read_json(folder/'result.json');progress=read_json(folder/'progress.json')
    require(row['id']==folder.name==progress['id'], 'Trajectory identifier mismatch')
    require(row['case_id']==case['id'], 'Case identity mismatch')
    require(row['input_hashes']==case['array_hashes'], 'Input identity mismatch')
    check_runtime(row['audit'],row['model'])
    nfe=row['nfe'];audit=row['audit']
    require(nfe==progress['model_forward_attempts']==progress['model_forward_returns']==audit['model_forward_attempts']==audit['model_forward_returns'], 'NFE/actual forward accounting mismatch')
    solver=row['solver'];cap={'R1':400,'R2':800,'D1':1200,'D2':1200}[solver]
    require(0<=nfe<=cap and progress['cap']==cap, 'Per-trajectory NFE limit exceeded')
    functions=read_json(folder/'function_log.json');steps=read_json(folder/'solver_step_log.json')
    require(len(functions)==nfe and [r['nfe'] for r in functions]==list(range(1,nfe+1)), 'Actual function ledger differs')
    for f in functions:
        for name in ('controller_time','library_function_time','actual_function_time'):
            require(0<=f[name]<=1, 'Function time beyond original ODE domain: '+name)
        require(all(f[k]=='torch.float32' for k in ('function_time_dtype','state_dtype','velocity_dtype')), 'Non-FP32 function input/output')
        require(float(np.float32(f['library_function_time']))==f['actual_function_time'], 'Actual network time differs from FP32 conversion')
    require([s['step_index'] for s in steps]==list(range(len(steps))), 'Step ledger index differs')
    accepted=[s for s in steps if s['accepted'] is True];rejected=[s for s in steps if s['accepted'] is False]
    require(len(accepted)==row['accepted_steps'] and len(rejected)==row['rejected_steps'], 'Accepted/rejected counts differ')
    current=0.;previous_nfe=steps[0]['nfe_before'] if steps else nfe
    initial_nfe=previous_nfe
    for step in steps:
        require(abs(step['t0']-current)<1e-12 and 0<=step['t0']<=step['t1']<=1 and step['h']>0, 'Step time/width contract differs')
        require(step['nfe_before']==previous_nfe and step['nfe_after']-step['nfe_before']==step['nfe_in_attempt'], 'Step NFE accounting differs')
        previous_nfe=step['nfe_after']
        if step['accepted']:current=step['t1']
        if solver.startswith('R') and step['accepted']:
            require(step['nfe_in_attempt']==4, 'Classical RK4 must use four function calls')
        if solver.startswith('D') and step['accepted'] is not None:
            require(step['error_ratio'] is not None and np.isfinite(step['error_ratio']), 'Adaptive error estimate absent')
    require(previous_nfe==nfe and abs(current-row['last_accepted_time'])<1e-12, 'Last accepted state accounting differs')
    require(row['complete']==(row['status']=='COMPLETE' and row['last_accepted_time']==1.), 'Incomplete state mislabeled as endpoint')
    if solver.startswith('R'):
        expected_steps=100 if solver=='R1' else 200
        require(row['complete'] and nfe==4*expected_steps and len(accepted)==expected_steps, 'Fixed RK4 matrix incomplete')
    rawpath=out/row['path']
    require(out in rawpath.resolve().parents and file_sha256(rawpath)==row['sha256'], 'Raw path/hash differs')
    with np.load(rawpath,allow_pickle=False) as z:
        raw=z['raw'].copy();states=z['states'].copy();times=z['state_times'].copy();C=z['C'].copy()
        last=float(z['last_accepted_time'])
    require(raw.dtype==np.float32 and states.dtype==np.float32 and raw.shape==(112,9) and states.shape[1:]==(112,9), 'Raw state dtype/shape differs')
    require(np.isfinite(raw).all() and np.isfinite(states).all(), 'Nonfinite saved state')
    require(bits(C,case['C']) and bits(raw[:12],case['C'].reshape(12,9)), 'Known coordinates changed')
    require(bits(states[:,:12],np.broadcast_to(case['C'].reshape(1,12,9),(len(states),12,9))), 'Saved C state changed')
    require(bits(states[0],case['initial_full'][0]) and times[0]==0., 'Initial state differs from actual registered Gaussian/C')
    require(array_hash(raw)==row['raw_array_sha256'] and last==row['last_accepted_time'], 'Saved endpoint/time identity differs')
    require(all(0<=t<=last for t in times) and len(times)==row['saved_state_count'], 'Saved fixed times outside accepted state range')
    if row['complete']:require(times[-1]==1. and bits(states[-1],raw), 'Complete endpoint not the final saved state')
    require(row['saved_C_bitwise'] and row['total_faces']==112 and row['free_faces']==100 and row['state_shape']==[1,100,9], 'State/C/face contract differs')
    if not row['repeat'] and row['complete']:
        score=read_json(out/'scores'/folder.name/'result.json')
        require(score['status']=='PASS' and score['row']['sha256']==row['sha256'], 'Missing or wrong endpoint score')
        require(file_sha256(out/score['metric_path'])==score['metric_sha256'], 'Detailed metric hash differs')
        for key in ('model_forwards','backward_calls','OT_calls','repairs'):require(score[key]==0, 'Offline scoring made unexpected model/repair call')
    return dict(id=row['id'],model=row['model'],case_id=row['case_id'],solver=solver,repeat=row['repeat'],status=row['status'],complete=row['complete'],
                nfe=nfe,initial_function_evaluations_outside_step_ledger=initial_nfe,accepted=len(accepted),rejected=len(rejected),
                last_accepted_time=last,raw_sha256=row['sha256'],function_log_sha256=file_sha256(folder/'function_log.json'),
                solver_step_log_sha256=file_sha256(folder/'solver_step_log.json'),audit=audit)


def _parse_csv_numbers(snapshot):
    result=[]
    payload=snapshot.get('gpu',{})
    if payload.get('returncode')!=0:return result
    for row in csv.DictReader(io.StringIO(payload.get('stdout',''))):
        normalized={k.strip():v.strip() for k,v in row.items()}
        vals={}
        for token,name in (('memory.total','total_MiB'),('memory.used','used_MiB'),('memory.free','free_MiB'),('utilization.gpu','compute_percent')):
            found=next((v for k,v in normalized.items() if k.startswith(token)),None)
            match=re.search(r'[-+]?\d+(?:\.\d+)?',found or '')
            vals[name]=float(match.group()) if match else None
        vals['name']=next((v for k,v in normalized.items() if k=='name'),None)
        result.append(vals)
    return result


def resource_summary(out,worker_rows):
    sampled=[];references=[];process_na=False
    for row in worker_rows:
        relative=row['audit']['resource_snapshots_path'];path=out/relative
        snapshots=read_json(path);references.append(dict(path=relative,sha256=file_sha256(path),snapshots=len(snapshots)))
        for shot in snapshots:
            sampled.extend(_parse_csv_numbers(shot))
            text=shot.get('processes',{}).get('stdout','')
            process_na=process_na or 'N/A' in text or 'Not Supported' in text
    def maximum(name):return max((r[name] for r in sampled if r.get(name) is not None),default=None)
    return dict(GPU_names=sorted({r['name'] for r in sampled if r.get('name')}),
        global_memory_used_peak_MiB=maximum('used_MiB'),global_memory_free_min_MiB=min((r['free_MiB'] for r in sampled if r.get('free_MiB') is not None),default=None),
        global_GPU_compute_peak_percent=maximum('compute_percent'),global_GPU_total_MiB=maximum('total_MiB'),
        torch_peak_allocated_bytes=max((r['audit']['peak_cuda_allocated'] for r in worker_rows),default=None),
        torch_peak_reserved_bytes=max((r['audit']['peak_cuda_reserved'] for r in worker_rows),default=None),
        process_VRAM_peak_bytes=None,process_VRAM_status='NOT_AVAILABLE_WDDM_NA' if process_na else 'NOT_DERIVED_FROM_GLOBAL_MEMORY',
        note='Global snapshots include unrelated allocations. Allocated is within reserved; do not add them. WDDM process N/A is not a process peak. GPU lock makes this a shared-machine diagnostic, not isolated throughput.',
        GPU_workers_this_window=1,CPU_scoring_workers=2,resources=references)


def code_export(out):
    """Added-file diff, exact source snapshot, git-apply byte roundtrip; no git index."""
    source=Path(__file__).resolve().parent
    paths=sorted(p for p in source.rglob('*') if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts)
    require(paths and sum(p.stat().st_size for p in paths)<5*1024*1024, 'Unexpectedly large source export')
    snapshot=out/'source_snapshot';require(not snapshot.exists(), 'Source snapshot already exists')
    chunks=[];records=[]
    for p in paths:
        relative=p.relative_to(ROOT).as_posix();raw=p.read_bytes()
        require(b'\x00' not in raw, 'Unexpected binary source')
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        chunks.append(f'diff --git a/{relative} b/{relative}\nnew file mode 100644\nindex {"0"*40}..{blob}\n--- /dev/null\n+++ b/{relative}\n'.encode())
        lines=raw.splitlines(keepends=True)
        if lines:
            chunks.append(f'@@ -0,0 +1,{len(lines)} @@\n'.encode())
            for line in lines:
                chunks.append(b'+'+line)
                if not line.endswith(b'\n'):chunks.append(b'\n\\ No newline at end of file\n')
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        records.append(dict(path=relative,sha256=file_sha256(p),bytes=len(raw)))
    diff=out/'code.diff';require(not diff.exists(), 'code.diff already exists')
    diff.write_bytes(b''.join(chunks))
    # Stop repository discovery at the run boundary; git apply only writes the
    # fresh run-local temporary directory, never the real index or work tree.
    sandbox=Path(tempfile.mkdtemp(prefix='code_diff_roundtrip_',dir=out))
    env=dict(os.environ)
    for key in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE'):env.pop(key,None)
    env['GIT_CEILING_DIRECTORIES']=str(out)
    commands=[]
    for mode in ('check','apply'):
        cmd=['git','-c','core.autocrlf=false','apply']+(['--check'] if mode=='check' else [])+[str(diff)]
        run=subprocess.run(cmd,cwd=sandbox,env=env,text=True,capture_output=True)
        commands.append(dict(mode=mode,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
        require(run.returncode==0, 'git apply '+mode+' failed: '+run.stderr)
    for record in records:
        require(file_sha256(sandbox/record['path'])==record['sha256'], 'Code diff byte roundtrip mismatch: '+record['path'])
    require({p.relative_to(sandbox).as_posix() for p in sandbox.rglob('*') if p.is_file()}=={r['path'] for r in records}, 'Unexpected files in patch roundtrip')
    return dict(status='PASS',source_files=records,code_diff_sha256=file_sha256(diff),source_snapshot=str(snapshot.relative_to(out)),
                roundtrip_directory=str(sandbox.relative_to(out)),git_commands=commands,
                index_or_repository_mutation=False,weights_or_data_copied=False)


def finalize(out):
    out=own_run(out);tick=time.perf_counter()
    require(not (out/'final_audit.json').exists(), 'Final audit already exists')
    dispatch=read_json(out/'dispatch.json');require(dispatch['status']=='PASS', 'Fixed matrix has not finished')
    protocol=read_json(out/'registration/protocol.json');cases=load_registered_cases(out);case_by_id={r['id']:r for r in cases}
    expected={(m,c['id'],s) for m in ASSETS for c in cases for s in ('R1','R2','D1','D2')}
    workers=[];formal=[];repeats=[]
    for folder in sorted((out/'trajectories').iterdir()):
        if not folder.is_dir():continue
        row=read_json(folder/'result.json')
        audit=trajectory_audit(out,folder,case_by_id[row['case_id']]);workers.append(audit)
        (repeats if row['repeat'] else formal).append(audit)
    require(len(formal)==32 and {(r['model'],r['case_id'],r['solver']) for r in formal}==expected, 'Formal matrix differs')
    require(len(repeats)==1 and (repeats[0]['model'],repeats[0]['case_id'],repeats[0]['solver'])==('B','P0_patch0_seed10401','R1'), 'Repeat differs from preregistration')
    original=out/'trajectories/B_P0_patch0_seed10401_R1/raw.npz';repeat=out/'trajectories/B_P0_patch0_seed10401_R1_repeat/raw.npz'
    with np.load(original,allow_pickle=False) as a,np.load(repeat,allow_pickle=False) as b:
        require(set(a.files)==set(b.files) and all(bits(a[k],b[k]) for k in a.files), 'Separate-process repeat is not bitwise identical')
    for label in ASSETS:
        path=out/'prechecks'/(label+'_precheck')/'result.json';row=read_json(path);audit=row['audit']
        require(row['status']=='PASS' and row['old_FP32_entry_bitwise_equal'], 'FP32 entry precheck failed')
        require(len(row['endpoint_t1_checks'])==4 and all(r['actual_t']==1. and r['finite'] for r in row['endpoint_t1_checks']), 'True t=1 checks incomplete')
        require(audit['model_forward_attempts']==audit['model_forward_returns']==6, 'Precheck forward count differs')
        check_runtime(audit,label);workers.append(dict(id=label+'_precheck',audit=audit))
    progress_rows=[];unknown=[]
    for path in sorted(out.glob('**/progress.json')):
        p=read_json(path);result_path=path.parent/'result.json'
        record=dict(path=str(path.relative_to(out)),**p)
        if not result_path.exists():
            failure_path=path.parent/'failure.json';require(failure_path.exists(), 'Unfinished worker without preserved failure')
            failure=read_json(failure_path)
            record['failure']=failure
            record['returns_unknown']=p['model_forward_attempts']-p['model_forward_returns']
            unknown.append(record)
        progress_rows.append(record)
    attempts=sum(r['model_forward_attempts'] for r in progress_rows)
    confirmed=sum(r['model_forward_returns'] for r in progress_rows)
    unknown_returns=sum(r['returns_unknown'] for r in unknown)
    require(attempts<=30000 and confirmed+unknown_returns==attempts, 'Global hard budget or uncertain-return accounting differs')
    require(len(unknown)==1 and unknown[0]['path']=='prechecks/B_precheck_attempt1/progress.json' and unknown[0]['model_forward_attempts']==6 and unknown[0]['model_forward_returns']==5, 'Unexpected unfinished attempt')
    formal_F=sum(r['nfe'] for r in formal);repeat_F=sum(r['nfe'] for r in repeats)
    require(formal_F+repeat_F+12+6==attempts, 'Actual global count differs from independent decomposition')
    fixed_F=sum(r['nfe'] for r in formal if r['solver'].startswith('R'));adaptive_F=formal_F-fixed_F
    require(fixed_F==9600 and adaptive_F<=19200 and repeat_F==400, 'Solver budget differs')
    protection=verify_protected(out);require(protection['status']=='PASS', 'Read asset changed')
    score_identity=read_json(out/'registration/scoring_identity.json');extra=[]
    for path,expected_hash in score_identity['read_source_sha256'].items():
        actual=file_sha256(path);require(actual==expected_hash, 'Scoring read source changed')
        extra.append(dict(path=path,sha256=actual))
    later_sources=read_json(out/'registration/additional_read_sources.json')
    later_checks=[]
    for record in later_sources['files']:
        actual=file_sha256(record['path'])
        require(actual==record['sha256'], 'Additional actually imported source changed')
        later_checks.append(dict(path=record['path'],sha256=actual))
    addendum=read_json(out/'registration/precheck_accounting_addendum.json')
    require(addendum['total_charged_precheck_calls']==18 and addendum['revised_maximum_charged_calls']==29218, 'Precheck accounting addendum differs')
    score_rows=[read_json(p) for p in sorted((out/'scores').glob('*/result.json'))]
    require(len(score_rows)==sum(r['complete'] for r in formal), 'Score count differs from complete formal endpoints')
    analysis=read_json(out/'analysis.json')
    require(analysis['status']=='PASS' and analysis['counts']['formal_rows']==32 and analysis['model_forwards']==0, 'Analysis missing or model forward in CPU analysis')
    fixture=read_json(out/'registration/cpu_fixture.json')
    require(fixture['status']=='PASS' and fixture['counts']['cpu_fixture_function_evaluations']==1682, 'Unexpected solver fixture record')
    # Two previously executed 1682-evaluation fixture passes were reported by
    # the solver task; this finalizer only reads/copies their retained result.
    fixture_accounting=dict(actual_CPU_fixture_function_evaluations=3364,executions=2,per_execution=1682,
        source='Final retained JSON proves 1682 CPU function evaluations; another 1682 development verification evaluations are reported in the solver-agent execution/tool record, whose stdout was not retained as a separate file',
        final_file_evidence_function_evaluations=1682,additional_agent_execution_record_function_evaluations=1682,additional_raw_stdout_file=None,
        retained_final_fixture='registration/cpu_fixture.json',finalizer_reruns=0,actual_neural_model_forwards=0)
    test_path=Path(__file__).resolve().parent/'tests/test_analysis.json'
    analysis_test=read_json(test_path);require(analysis_test['status']=='PASS' and analysis_test['model_forwards']==0, 'CPU analysis fixture failed')
    write_json(out/'registration/analysis_CPU_test.json',analysis_test)
    resource=resource_summary(out,workers)
    timing=dict(recorded_lock_wait_seconds=sum(r['audit']['lock_wait_seconds'] for r in workers)+sum(r.get('lock_wait_seconds',0) for r in unknown),
        recorded_locked_wall_seconds=sum(r['audit']['locked_wall_seconds'] for r in workers),
        recorded_model_load_seconds=sum(r['audit']['model_load_seconds'] for r in workers),
        recorded_model_and_solver_wall_seconds=sum(r['audit']['model_and_solver_wall_seconds'] for r in workers),
        recorded_model_forward_CUDA_event_seconds=sum(r['audit']['model_forward_cuda_event_seconds'] for r in workers),
        CPU_geometry_scoring_seconds_sum=sum(r['seconds'] for r in score_rows),CPU_analysis_seconds=analysis['CPU_analysis_seconds'],
        unknown_failed_attempt_GPU_seconds=None,
        note='Timing columns overlap and must not be summed as total wall time. Failed initial precheck did not persist compute/resources; its GPU duration is unknown. CPU scoring ran concurrently in two workers. Lock waits are separated; no isolated throughput claim.')
    dependency=read_json(out/'registration/dependency_info.json')
    current_dependency_version=importlib.metadata.version('torchdiffeq')
    require(current_dependency_version==dependency['torchdiffeq_version'], 'Installed torchdiffeq version changed')
    dependency_sources=[]
    for path,expected_hash in dependency['sources_sha256'].items():
        actual_hash=file_sha256(path)
        require(actual_hash==expected_hash, 'Installed solver dependency source changed: '+path)
        dependency_sources.append(dict(path=path,expected_sha256=expected_hash,actual_sha256=actual_hash,unchanged=True))
    dependency_protection=dict(status='PASS',version_before=dependency['torchdiffeq_version'],
        version_after=current_dependency_version,sources=dependency_sources,
        environment_modified=False,model_forwards=0)
    code=code_export(out)
    result=dict(schema='numeric_reference_A_final_audit_v1',status='PASS',scientific_completeness='COMPLETE' if all(r['complete'] for r in formal) else 'PARTIAL',
        actual_counts=dict(model_forward_attempts=attempts,confirmed_model_forward_returns=confirmed,unconfirmed_return_count=unknown_returns,
            formal_trajectory_attempts=32,formal_complete=sum(r['complete'] for r in formal),formal_incomplete=sum(not r['complete'] for r in formal),
            extra_repeat_complete=1,formal_RK4_NFE=fixed_F,formal_adaptive_NFE=adaptive_F,repeat_NFE=repeat_F,successful_precheck_NFE=12,failed_precheck_attempted_NFE=6,
            CPU_strict_checkpoint_loads=4,optimizer_updates=0,backward_calls=0,OT_calls=0,CPU_fixture_function_evaluations=3364,finalizer_model_forwards=0),
        per_trajectory=formal+repeats,precheck_attempts=progress_rows,unknown_return_evidence=unknown,
        same_solver_separate_process_repeat='ALL_SAVED_ARRAYS_BITWISE_IDENTICAL',protected_assets=protection,
        additional_scoring_protected_sources=extra,
        additional_actual_import_protection=dict(scope=later_sources['scope'],files=later_checks),
        precheck_accounting_addendum=addendum,
        resources=resource,timing=timing,CPU_fixture_accounting=fixture_accounting,
        CPU_analysis_fixture=analysis_test,installed_solver_dependency_protection=dependency_protection,
        code_export=code,production_configuration_changed=False,committed_or_pushed=False,
        HEAD_start=protocol['HEAD'],HEAD_end=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        note='Artifact/runtime/accounting audit, not a numerical convergence or geometry-success claim.')
    result['finalizer_CPU_wall_seconds']=time.perf_counter()-tick
    write_json(out/'final_audit.json',result)
    print(json.dumps(dict(status=result['status'],scientific_completeness=result['scientific_completeness'],counts=result['actual_counts'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    finalize(p.parse_args().out)
