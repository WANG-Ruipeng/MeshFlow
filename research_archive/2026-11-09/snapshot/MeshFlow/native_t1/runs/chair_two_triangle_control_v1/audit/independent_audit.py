"""Independent CPU audit of saved data; no model/torch imports or generation."""
from pathlib import Path
import collections
import hashlib
import json
import math
import re
import sys
import numpy as np

OUT=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parents[4]
EXPECTED_STATE='166c71fef888f26494fc4a0af1776b77a484efa72a6f3ce06552c604fbecc972'
EXPECTED_FILE='d640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def torch_array_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(('torch.'+str(a.dtype),tuple(a.shape))).encode()+a.tobytes()).hexdigest()

def run():
    plan=read(OUT/'plan.json');result=read(OUT/'results.json');load=read(OUT/'audit/model_load.json')
    ledger=[json.loads(v) for v in (OUT/'ledger.jsonl').read_text().splitlines() if v.strip()]
    errors=[];sources=[];inputs=[];conditions=[];outputs=[];banks={};counts=collections.Counter()
    def checked(name,checks,**facts):
        failed=[k for k,v in checks.items() if not v]
        if failed:errors.append(dict(name=name,failed=failed))
        return dict(name=name,status='PASS' if not failed else 'BLOCKED',checks=checks,**facts)
    for name,expected in plan['frozen_sources'].items():
        actual=sha(ROOT/name);sources.append(checked(name,{'hash_unchanged':actual==expected},expected_sha256=expected,actual_sha256=actual))
    for name,expected in plan['input_file_hashes'].items():
        actual=sha(OUT/name);inputs.append(checked(name,{'hash_unchanged':actual==expected},expected_sha256=expected,actual_sha256=actual))
    for recorded in plan['conditions']:
        C=np.load(OUT/'inputs'/(recorded['condition']+'.npy'),allow_pickle=False);cross=np.cross(C[:,1].astype(np.float64)-C[:,0],C[:,2].astype(np.float64)-C[:,0])
        norm=np.linalg.norm(cross,axis=1);normal=cross/norm[:,None];vertices=np.unique(C.reshape(-1,3),axis=0)
        edges=collections.Counter(tuple(sorted((tuple(t[i]),tuple(t[(i+1)%3])))) for t in C for i in range(3))
        tests=dict(two_faces=C.shape==(2,3,3),FP32=C.dtype==np.float32,finite=bool(np.isfinite(C).all()),four_unique_vertices=len(vertices)==4,
            exactly_one_shared_edge=sum(n==2 for n in edges.values())==1,four_boundary_edges=sum(n==1 for n in edges.values())==4,
            no_nonmanifold_edges=max(edges.values())<=2,positive_nondegenerate_area=bool(np.all(norm*.5>1e-12)),
            coplanar=bool(np.all(C[:,:,1]==C[0,0,1])),normal_consistent_positive_Y=bool(np.array_equal(normal,np.array([[0.,1.,0.],[0.,1.,0.]]))),
            plan_width_matches=bool(np.isclose(np.ptp(vertices[:,0]),recorded['width'],rtol=1e-6)),
            plan_depth_matches=bool(np.isclose(np.ptp(vertices[:,2]),recorded['depth'],rtol=1e-6)),
            plan_height_matches=bool(np.isclose(vertices[0,1],recorded['seat_y'],rtol=1e-6)),
            planned_area_matches=bool(np.allclose(norm*.5,recorded['triangle_area'],rtol=1e-6)))
        conditions.append(checked(recorded['condition'],tests,triangle_area_FP64_from_saved_FP32=(norm*.5).tolist(),normals=normal.tolist(),connected_components=1))
    expected_runtime=dict(cublas_workspace_config=':4096:8',deterministic_algorithms=True,deterministic_warn_only=False,
        tf32_matmul=False,tf32_cudnn=False,cudnn_benchmark=False,external_flash_available=False,sdpa_math_enabled=True,
        sdpa_flash_enabled=False,sdpa_mem_efficient_enabled=False,sdpa_cudnn_enabled=False,cuda_autocast_enabled=True,
        cuda_autocast_dtype='torch.bfloat16',gradient_enabled=False,parameter_dtype='torch.float32',native_active_context_depth=1,
        native_precision='bf16',all_modules_eval=True,trainable_parameters=0,gradient_checkpointing=False)
    result_by_id={r['id']:r for r in result['requests']}
    expected_counts=dict(model_forward_attempts=50,model_forward_returns=50,euler_updates=50,complete_rollouts=1,rollout_attempts=1,
        model_backward_calls=0,optimizer_updates=0,geometry_gradient_calls=0,postprocessing_repairs=0)
    for job in plan['requests']:
        path=OUT/job['output_dir'];saved=read(path/'run.json');bank=arrays(path/'raw.npz');banks[job['id']]=bank
        C=np.load(OUT/job['C'],allow_pickle=False);z=arrays(OUT/job['input_npz'])['initial_gaussian'];raw=bank['output'];trajectory=bank['path']
        s=saved['sampling'];counts.update(saved['counts']);reference=np.broadcast_to(C.reshape(2,9),(51,2,9)).copy()
        checks=dict(result_matches_run=saved==result_by_id[job['id']],request_fields_frozen=all(saved[k]==v for k,v in job.items()),
            raw_file_hash=sha(path/'raw.npz')==saved['raw_sha256'],raw_FP32_shape=raw.dtype==np.float32 and raw.shape==(160,9),
            trajectory_FP32_shape=trajectory.dtype==np.float32 and trajectory.shape==(51,160,9),all_finite=bool(np.isfinite(trajectory).all()),
            raw_is_last_state=bits(raw,trajectory[-1]),saved_C_same=bits(C,bank['C']),C_all51_bitwise=bits(trajectory[:,:2].copy(),reference),
            initial_free_noise_exact=bits(trajectory[0,2:],z[0,2:]),saved_noise_exact=bits(z,bank['initial_gaussian']),known_gaussian_slots_zero=bool((z[:,:2]==0).all()),
            noise_registered_bytes_hash=hashlib.sha256(z.tobytes()).hexdigest()==job['noise_sha256'],
            sampler_noise_hash=torch_array_hash(z)==s['noise_sha256'],sampler_C_hash=torch_array_hash(C.reshape(2,9))==s['condition_sha256'],
            counts_exact=saved['counts']==expected_counts,nfe_50=s['nfe']==50,Geo_calls_50=s['actual_Geo_recomputations']==50,
            BF16_actual_output=s['observed_raw_output_dtypes']==['torch.bfloat16'],FP32_integration=s['integration_dtype']=='float32',
            runtime_all_expected=all(s['runtime'].get(k)==v for k,v in expected_runtime.items()),
            no_GT_input=s['GT_model_inputs'] is False,no_guidance=s['geometry_gradient_calls']==0,no_postprocessing=s['postprocessing'] is None,
            checkpoint_pin=saved['checkpoint_sha256']==EXPECTED_FILE)
        vertices=[];faces=[]
        for line in (path/'raw.obj').read_text().splitlines():
            if line.startswith('v '):vertices.append([float(x) for x in line.split()[1:]])
            elif line.startswith('f '):faces.append([int(x) for x in line.split()[1:]])
        checks['OBJ_raw_coordinates_roundtrip']=bits(np.asarray(vertices,np.float32).reshape(160,9),raw)
        checks['OBJ_triangle_soup_indices']=np.array_equal(np.asarray(faces),np.arange(1,481).reshape(160,3))
        outputs.append(checked(job['id'],checks,counts=saved['counts'],seconds=s['seconds'],raw_sha256=saved['raw_sha256'],
            peak_cuda_allocated_bytes=s['peak_cuda_allocated'],peak_cuda_reserved_bytes=s['peak_cuda_reserved'],peak_scope=s['peak_scope']))
    pairing=[]
    for seed in sorted({j['seed'] for j in plan['requests']}):
        jobs=[j for j in plan['requests'] if j['seed']==seed];first,second=[banks[j['id']] for j in jobs]
        C1,C2=first['C'],second['C'];scaled=C1.copy();scaled[:,:,0]*=np.float32(1.25)
        pairing.append(checked('seed_'+str(seed),dict(two_widths_same_full_gaussian=bits(first['initial_gaussian'],second['initial_gaussian']),
            two_widths_same_free_initial_state=bits(first['path'][0,2:],second['path'][0,2:]),only_C_x_scaled_125=bits(scaled,C2),
            same_N_K=jobs[0]['N']==jobs[1]['N']==160 and jobs[0]['K']==jobs[1]['K']==2,
            outputs_actually_differ=not bits(first['output'][2:],second['output'][2:])),requests=[j['id'] for j in jobs]))
    events=collections.Counter(r['event'] for r in ledger);ended=[r for r in ledger if r['event']=='SAMPLE_END'];begun=[r for r in ledger if r['event']=='SAMPLE_BEGIN']
    ledger_totals=collections.Counter()
    for row in ended:ledger_totals.update(row['counts'])
    budget=checked('fixed_budget',dict(exactly_four_registered=len(plan['requests'])==4,len_results_four=len(result['requests'])==4,
        model_load_event_once=events['MODEL_LOADED']==1,begins_four=len(begun)==4,ends_four=len(ended)==4,
        begin_ids_equal_plan=[x['request'] for x in begun]==[x['id'] for x in plan['requests']],end_ids_equal_plan=[x['request'] for x in ended]==[x['id'] for x in plan['requests']],
        ledger_counts_equal_outputs=dict(ledger_totals)==dict(counts),total_forwards_200=counts['model_forward_attempts']==counts['model_forward_returns']==result['total_model_forwards']==200,
        Euler_updates_200=counts['euler_updates']==200,rollouts_4=counts['complete_rollouts']==counts['rollout_attempts']==4,
        zero_training_backward_optimizer=all(counts[k]==0 for k in ('model_backward_calls','optimizer_updates','geometry_gradient_calls','postprocessing_repairs')) and all(result[k]==0 for k in ('training_updates','backwards','optimizer_updates')),
        no_failed_or_retry_events=not any('FAIL' in k or 'STOP' in k or 'RETRY' in k for k in events),no_failure_file=not (OUT/'failure.json').exists(),
        plan_hash_matches=sha(OUT/'plan.json')==result['plan_sha256']),actual_counts=dict(counts),event_counts=dict(events),new_model_calls_by_this_audit=0)
    checkpoint_actual=sha(ROOT/plan['checkpoint'])
    model=checked('model_identity',dict(checkpoint_file_actual_pin=checkpoint_actual==plan['checkpoint_sha256']==load['checkpoint_sha256']==EXPECTED_FILE,
        saved_before_after_state_pin=result['state_before']==result['state_after']==load['state_sha256']==load['model_state_sha256']==EXPECTED_STATE,
        loader_frozen=load['trainable_parameters']==0,loader_no_forward=load['model_forward_calls']==0,
        optimizer_not_restored=load['optimizer_restored'] is False,RNG_not_restored=load['rng_restored'] is False),
        actual_file_sha256=checkpoint_actual,recorded_state_sha256=EXPECTED_STATE,
        independent_tensor_rehash=False,evidence_limit='Independent audit matches saved state hashes and original file bytes; it does not reload tensors or recover a departed process.')
    telemetry=[json.loads(v) for v in (OUT/'audit/gpu_telemetry.jsonl').read_text().splitlines() if v.strip()]
    device=[];process_values=[];process_unavailable=0;empty=0;process_pids=set()
    for r in telemetry:
        if r['device']['returncode']==0:
            vals=[float(v.strip()) for v in r['device']['stdout'].split(',')];device.append(dict(time=r['time'],used_MiB=vals[0],free_MiB=vals[1],utilization_percent=vals[2]))
        text=r['processes']['stdout']
        if not text.strip():empty+=1
        for line in text.splitlines():
            cols=[v.strip() for v in line.split(',')]
            if len(cols)<3:continue
            process_pids.add(cols[0]);number=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:MiB)?',cols[-1])
            if number:process_values.append(dict(time=r['time'],pid=cols[0],used_MiB=float(number.group(1))))
            else:process_unavailable+=1
    gpu=dict(telemetry_rows=len(telemetry),valid_device_samples=len(device),global_device_peak_used_MiB=max(v['used_MiB'] for v in device),
        global_device_min_free_MiB=min(v['free_MiB'] for v in device),global_device_max_utilization_percent=max(v['utilization_percent'] for v in device),
        numeric_process_samples=len(process_values),empty_process_rows=empty,unavailable_process_values=process_unavailable,pids=sorted(process_pids),
        process_peak_used_MiB=max((v['used_MiB'] for v in process_values),default=None),process_memory_status='AVAILABLE' if process_values else 'NA',
        peak_interpretation='Device occupancy includes unrelated processes; sampled device peak is not process peak and not an exact continuous maximum.',
        rollout_peak_allocated_bytes=max(r['sampling']['peak_cuda_allocated'] for r in result['requests']),
        rollout_peak_reserved_bytes=max(r['sampling']['peak_cuda_reserved'] for r in result['requests']),
        load_peak_allocated_bytes=load['peak_allocated'],load_peak_reserved_bytes=load['peak_reserved'],
        torch_peak_interpretation='Separate load and per-rollout scopes; includes resident model. Allocated is contained in reserved; do not add them.')
    decision=read(OUT/'audit/dispatch_decision.json');first=result['requests'][0]
    after_first=read(OUT/'audit'/('resources_after_'+first['id']+'.json'));pre=read(OUT/'audit/resources_before_model.json')
    seen=[x['utilization_percent'] for x in device if x['time']<=after_first['time']]
    serial=checked('resource_dispatch',dict(one_worker=result['worker_count']==decision['actual_workers']==1,
        resources_before_model_exist=pre['time']<read(OUT/'audit'/('resources_before_'+first['id']+'.json'))['time'],
        first_task_duration_matches=decision['first_sampling_seconds']==first['sampling']['seconds'],
        first_reserved_matches=decision['first_peak_reserved']==first['sampling']['peak_cuda_reserved'],
        load_duration_matches=decision['load_seconds']==load['seconds'],remaining_jobs_three=decision['remaining_jobs']==3,
        first_observed_utilization_matches=max(seen)==decision['observed_global_GPU_utilization_max'],
        all_before_after_resource_snapshots=all((OUT/'audit'/('resources_'+prefix+r['id']+'.json')).exists() for r in result['requests'] for prefix in ('before_','after_')),
        headroom_maintained=min(v['free_MiB'] for v in device)>=decision['headroom_MiB']),recorded_decision=decision,
        interpretation='Serial justification is startup overhead versus only three remaining short jobs, supported by first actual measurement. It is not a GPU-saturation or VRAM-capacity claim, and parallel throughput was not benchmarked.')
    result_audit=dict(schema='chair_two_triangle_independent_audit_v1',status='PASS' if not errors else 'BLOCKED',errors=errors,
        budget=budget,conditions=conditions,outputs=outputs,same_seed_width_pairs=pairing,model_identity=model,source_hash_checks=sources,input_hash_checks=inputs,
        GPU_telemetry=gpu,serial_dispatch=serial,formal_source_count=len(sources),formal_sources_unchanged=all(r['status']=='PASS' for r in sources),
        evidence_file_hashes={str(p.relative_to(OUT)):sha(p) for p in [OUT/'plan.json',OUT/'results.json',OUT/'ledger.jsonl',OUT/'audit/model_load.json',OUT/'audit/gpu_telemetry.jsonl',OUT/'audit/dispatch_decision.json',OUT/'scripts/run_pilot.py']},
        scientific_comparison=dict(same_seed_width_intervention_valid=True,new_data_instances=2,noise_draws=2,width_settings=2,
            limits=['Only two fixed noises and one constructed rectangular patch family; no generalization or success-rate estimate.',
            'No target mesh exists for this constructed input; do not report GT completion accuracy or treat output change alone as correct conditional response.',
            'C preservation does not imply attachment, connectivity, visibility or watertightness.',
            'Changing width also changes C area/edge lengths; this is a width-defined geometric intervention, not a width-feature-isolated intervention.',
            'Sampling durations include first-call startup effects; these four sequential jobs are not a controlled timing benchmark.',
            'Strict runtime settings and saved-array checks do not establish GPU replay repeatability without a replay; none was added.']),
        independent_CPU_only=True,torch_imported='torch' in sys.modules,audit_model_calls=0,audit_new_samples=0,audit_script_sha256=sha(__file__))
    target=OUT/'audit/independent_audit.json';target.write_text(json.dumps(result_audit,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result_audit['status'],errors=errors,forwards=counts['model_forward_attempts'],formal_sources=len(sources),GPU=gpu)))

if __name__=='__main__':run()
