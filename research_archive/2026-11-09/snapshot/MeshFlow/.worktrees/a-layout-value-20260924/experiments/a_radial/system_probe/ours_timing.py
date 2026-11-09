"""Isolated fixed-index cold/warm CUDA-generation plus original CPU-outlet timing.

No source generation/export artifact is modified. Every repeat recomputes its
field from the same registered epsilon and compares it against production bytes.
This is synchronized single-sample latency, never throughput or OS-cache-cold.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from ..models import ShapeAE,ShapeFM
from ..exporter import MeshExporter,ASSETS
from ..legacy.common import face_quality
from ..legacy.exact_accel import audit as exact_audit
from ..mesh_ops import topology
from ..io_utils import array_hash,file_hash,new_dir,write_json
from .official import configure
from .ours_metrics import (SEEDS,ARMS,read_json,require,inside,verify_protocol,
    seed_inputs,load_sample,verify_outlet_binding,audit_inputs)

IDS=('00000','00016','00032','00048')
STAGES=('noise_H2D','FM32','latent_inverse_once','AE_decode','D2H_arrays',
        'CPU_export','face_quality','topology','exact_audit')


def cases():
    return [{'seed':seed,'sample_id':sid,'arm':arm} for seed in SEEDS for sid in IDS for arm in ARMS]


def case_id(case):return f"seed{case['seed']}/{case['sample_id']}/{case['arm']}"


def warm_schedule():
    base=cases();answer=[]
    for case in base:answer.append({**case,'phase':'warmup','repeat':0})
    for repeat,order in enumerate((base,list(reversed(base)),base),1):
        answer += [{**case,'phase':'measured','repeat':repeat} for case in order]
    return answer


def build_plan(run,out):
    run=Path(run).resolve();out=Path(out).resolve();protocol=read_json(run/'RUN_PROTOCOL.json')
    verify_protocol(run,protocol)
    preflight=audit_inputs(run,protocol);write_json(out/'INPUT_AUDIT.json',preflight)
    require(preflight['status']=='PASS','Production bindings failed; timing must not substitute inputs')
    require(protocol['deterministic_algorithms'] and protocol['tf32'] is False and protocol['threads']==1,
            'Production numerical settings changed')
    bound={str(run/'RUN_PROTOCOL.json'):file_hash(run/'RUN_PROTOCOL.json')};seed_details={}
    for seed in SEEDS:
        inputs=seed_inputs(run,seed,protocol);samples={}
        for rel in inputs['checkpoint_paths'].values():bound[str(run/rel)]=file_hash(run/rel)
        bound[str(inputs['sample_dir']/'SAMPLES.json')]=inputs['sample_manifest_sha256']
        for arm in ARMS:
            bound[str(inputs['export_dir']/f'{arm}_MANIFEST.json')]=inputs['outlet_manifest_sha256'][arm]
        for sid in IDS:
            arrays,identity=load_sample(inputs,sid);record=inputs['sample_records'][sid]
            path=inside(inputs['sample_dir'],record['path']);bound[str(path)]=identity['sample_npz_sha256']
            exports={}
            for arm in ARMS:
                row=verify_outlet_binding(inputs,arm,sid,record);mesh=inside(inputs['export_dir'],row['mesh_path'])
                require(row['status']=='ok' and file_hash(mesh)==row['mesh_sha256'],'Selected production output missing or changed')
                bound[str(mesh)]=row['mesh_sha256'];exports[arm]={'path':str(mesh),'sha256':row['mesh_sha256']}
            samples[sid]={'path':str(path),'file_sha256':identity['sample_npz_sha256'],
                'array_hash':record['array_hash'],'nodes_hash':record['nodes_hash'],'exports':exports}
        seed_details[str(seed)]={'checkpoint_paths':{k:str(run/v) for k,v in inputs['checkpoint_paths'].items()},
            'checkpoint_sha256':inputs['checkpoint_sha256'],'samples':samples}
    root=Path(__file__).resolve().parents[1]
    source_names=set(protocol['training_source_sha256'])|{'system_probe/ours_timing.py','system_probe/ours_metrics.py',
        'legacy/common.py','legacy/exact_accel.py','legacy/deform.py','legacy/chart.py','legacy/networks.py',
        'legacy/geometry_task.py','mesh_ops.py'}
    for name in source_names:bound[str(root/name)]=file_hash(root/name)
    for name in ('support.npz','original_A_10.pt','original_A_11.pt'):bound[str(ASSETS/name)]=file_hash(ASSETS/name)
    for name,digest in protocol['data_hashes'].items():bound[str(Path(protocol['data'])/name)]=digest
    plan={'scope':'FIXED_INDEX_SINGLE_SAMPLE_LATENCY_NOT_FULL128_OR_OS_CACHE_COLD',
        'run':str(run),'generation_protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),
        'seeds':list(SEEDS),'sample_ids':list(IDS),'case_count':24,'cases':cases(),'seed_details':seed_details,
        'bound_files_sha256':bound,'cuda_device':'cuda:0','threads':1,'FM_steps':32,'latent_inverse_calls':1,
        'cold_repeats_per_case':1,'cold_child_process_count':24,'warmup_per_case':1,'warm_repeats_per_case':3,
        'warm_schedule':warm_schedule(),'warm_order':'all24 warmup; measured round1 forward, round2 reverse, round3 forward',
        'cold_order':'seed10 then11; IDs00000/00016/00032/00048; fixed/a/qem; one new process per case',
        'cold_process_scope':'Parent wall includes spawn/import/CUDA startup/verified loading/pipeline/identity verification/child result writes and shutdown; parent log writing excluded',
        'cold_loaded_pipeline_scope':'Starts after checkpoint/sample/exporter loading; initially empty exporter template cache, VTK import lazy in CPU_export',
        'warm_cache_scope':'Both seeds AE/FM on CUDA, original A/support exporters on CPU; templates and VTK warmed; one preceding complete warmup per case',
        'mesh_total_scope':'saved epsilon already in host memory -> H2D -> FM32 -> one latent inverse -> AE -> D2H(noise,z_standardized,z,nodes) -> original CPU mesh',
        'checked_total_scope':'mesh_total plus original face_quality/topology/exact endpoint audit; geometry failure retained',
        'pipeline_disk_IO_included':False,'pipeline_hash_and_production_comparison_included':False,
        'cold_parent_wall_includes_child_disk_IO':True,'D2H_includes_registered_latents_for_identity':True,
        'synchronization':'torch.cuda.synchronize before and after each stage and at pipeline boundary; instrumented single-sample latency',
        'no_OS_cache_flush':True,'no_throughput_claim':True,'full128_population_claim':False,
        'no_shape_resampling_or_repair':True,'runtime_failure_is_not_geometry_invalidity':True,
        'timing_qualification_requires_every_record_exact_production_replay':True,
        'isolation_requirement':'No other GPU training or heavy CPU evaluation; parent schedules and confirms at invocation'}
    write_json(out/'TIMING_PROTOCOL.json',plan);return plan


def verify_bound_inputs(plan):
    for path,digest in plan['bound_files_sha256'].items():require(file_hash(path)==digest,'Bound timing input changed: '+path)


def load_arrays(record):
    path=Path(record['path']);require(file_hash(path)==record['file_sha256'],'Production sample file changed')
    with np.load(path,allow_pickle=False) as archive:arrays={k:np.array(archive[k]) for k in ('noise','z_standardized','z','nodes')}
    require(array_hash(*arrays.values())==record['array_hash'] and array_hash(arrays['nodes'])==record['nodes_hash'],
            'Production sample arrays changed')
    return arrays


class Runtime:
    def __init__(self,plan,seed,sample_ids=IDS,arms=ARMS):
        require(torch.cuda.is_available(),'Real CUDA required; no CPU timing fallback')
        configure(seed);self.seed=seed;self.device=torch.device('cuda:0');entry=plan['seed_details'][str(seed)]
        self.ae=ShapeAE().to(self.device);self.fm=ShapeFM().to(self.device)
        for name,model in (('ae',self.ae),('fm',self.fm)):
            path=Path(entry['checkpoint_paths'][name]);require(file_hash(path)==entry['checkpoint_sha256'][name],'Final checkpoint changed')
            state=torch.load(path,map_location='cpu',weights_only=True)
            if name=='fm':require(state['config']['ae_sha256']==entry['checkpoint_sha256']['ae'],'FM/AE pairing changed')
            model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
        self.exporter=MeshExporter(a_seed=seed);self.expected={sid:load_arrays(entry['samples'][sid]) for sid in sample_ids}
        self.expected_meshes={}
        for sid in sample_ids:
            for arm in arms:
                record=entry['samples'][sid]['exports'][arm];path=Path(record['path'])
                require(file_hash(path)==record['sha256'],'Production mesh file changed')
                with np.load(path,allow_pickle=False) as archive:self.expected_meshes[(sid,arm)]={k:np.array(archive[k]) for k in archive.files}
        torch.cuda.synchronize()
    def transfer(self,noise):return torch.from_numpy(noise[None].copy()).to(self.device)
    def fm32(self,eps):return self.fm.sample_standardized(eps,32)
    def inverse(self,zs):return self.fm.unstandardize(zs)
    def decode(self,z):return self.ae.decode_nodes(z)
    def host(self,eps,zs,z,nodes):
        return {k:t.detach().cpu().numpy()[0].copy() for k,t in [('noise',eps),('z_standardized',zs),('z',z),('nodes',nodes)]}
    def export(self,nodes,arm):return self.exporter.export(nodes,512,arm)
    def quality(self,mesh):return face_quality(mesh['vertices'],mesh['faces'])
    def topology(self,mesh):return topology(mesh['vertices'],mesh['faces'])
    def exact(self,mesh):return exact_audit(mesh['vertices'],mesh['faces'])


def pipeline(runtime,expected,arm,*,sync=torch.cuda.synchronize,clock=time.perf_counter):
    """Measured computation only. Injectable sync/clock permit meaningful CPU tests."""
    timings={};mesh=None;generated=None;record={'runtime_status':'failed'};start=None
    def stage(name,fn):
        sync();before=clock()
        try:return fn()
        finally:
            sync();timings[name]=clock()-before
    try:
        sync();start=clock()
        with torch.no_grad():
            eps=stage('noise_H2D',lambda:runtime.transfer(expected['noise']))
            zs=stage('FM32',lambda:runtime.fm32(eps))
            z=stage('latent_inverse_once',lambda:runtime.inverse(zs))
            nodes=stage('AE_decode',lambda:runtime.decode(z))
            generated=stage('D2H_arrays',lambda:runtime.host(eps,zs,z,nodes))
        require(all(np.isfinite(x).all() for x in generated.values()),'Nonfinite generated pipeline array')
        mesh,meta=stage('CPU_export',lambda:runtime.export(generated['nodes'],arm))
        record['mesh_total_seconds']=clock()-start;record['original_export_metadata']=meta
        record['actual_faces']=len(mesh['faces']);record['budget512_match']=len(mesh['faces'])==512
        record['face_quality']=stage('face_quality',lambda:runtime.quality(mesh))
        record['topology']=stage('topology',lambda:runtime.topology(mesh))
        record['exact']=stage('exact_audit',lambda:runtime.exact(mesh))
        sync();record['checked_total_seconds']=clock()-start
        record['geometry_valid']=bool(record['topology']['closed_oriented_g0'] and record['exact']['embedded_endpoint_pass'])
        record['geometry_and_budget_valid']=bool(record['geometry_valid'] and record['budget512_match'])
        record['runtime_status']='ok'
    except Exception as exc:
        record.update(runtime_error=repr(exc),runtime_traceback=traceback.format_exc())
        if start is not None:record['seconds_until_runtime_failure']=clock()-start
    record['stage_seconds']=timings
    record['mesh_and_check_include_transfers']=True
    record['pipeline_disk_IO_included']=False
    return record,mesh,generated


def replay_identity(expected,generated,expected_mesh,mesh):
    comparisons={}
    for name in ('noise','z_standardized','z','nodes'):
        a,b=expected[name],generated[name]
        same=a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
        comparisons[name]={'exact_equal':bool(same),'expected_hash':array_hash(a),'actual_hash':array_hash(b),
            'max_abs_error':float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.shape==b.shape else None}
    vertex_equal=expected_mesh['vertices'].dtype==mesh['vertices'].dtype and np.array_equal(expected_mesh['vertices'],mesh['vertices'])
    face_equal=expected_mesh['faces'].dtype==mesh['faces'].dtype and np.array_equal(expected_mesh['faces'],mesh['faces'])
    radial={}
    for key in ('u','r','alpha'):
        if key in expected_mesh or key in mesh:radial[key]=key in expected_mesh and key in mesh and expected_mesh[key].dtype==mesh[key].dtype and np.array_equal(expected_mesh[key],mesh[key])
    return {'exact_production_replay':all(x['exact_equal'] for x in comparisons.values()) and bool(vertex_equal and face_equal and all(radial.values())),
        'arrays':comparisons,'mesh_vertices_exact_equal':bool(vertex_equal),'mesh_faces_exact_equal':bool(face_equal),
        'radial_arrays_exact_equal':radial,
        'expected_mesh_typed_hash':array_hash(expected_mesh['vertices'],expected_mesh['faces']),
        'actual_mesh_typed_hash':array_hash(mesh['vertices'],mesh['faces']),
        'canonical_geometry_hash':array_hash(np.asarray(mesh['vertices'],np.float64),np.asarray(mesh['faces'],np.int64)),
        'canonical_hash_schema':'array_hash(vertices_float64,faces_int64); independent QEM connectivity',
        'mismatch_does_not_trigger_retry_or_substitution':True}


def save_record(folder,row,mesh=None,generated=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);before=time.perf_counter()
    if generated is not None:
        path=folder/'generated.npz';np.savez_compressed(path,**generated)
        row['generated_file']='generated.npz';row['generated_file_sha256']=file_hash(path)
    if mesh is not None:
        path=folder/'mesh.npz';np.savez_compressed(path,**mesh)
        row['mesh_file']='mesh.npz';row['mesh_file_sha256']=file_hash(path)
        row['mesh_typed_hash']=array_hash(mesh['vertices'],mesh['faces'])
    row['artifact_array_write_and_hash_seconds']=time.perf_counter()-before
    row['artifact_writing_outside_pipeline']=True;write_json(folder/'RESULT.json',row)
    return row


def run_case(runtime,case,folder,mode,phase,repeat):
    expected=runtime.expected[case['sample_id']]
    row,mesh,generated=pipeline(runtime,expected,case['arm'])
    row.update(case,mode=mode,phase=phase,repeat=repeat,case_id=case_id(case))
    before=time.perf_counter()
    try:
        if mesh is not None and generated is not None:
            row['replay']=replay_identity(expected,generated,runtime.expected_meshes[(case['sample_id'],case['arm'])],mesh)
        else:row['replay']={'exact_production_replay':False,'unavailable_due_to_runtime_error':True}
    except Exception as exc:row['replay']={'exact_production_replay':False,'error':repr(exc)}
    row['production_identity_verification_seconds_outside_pipeline']=time.perf_counter()-before
    return save_record(folder,row,mesh,generated)


def failed_record(case,mode,phase,repeat,error):
    return {**case,'case_id':case_id(case),'mode':mode,'phase':phase,'repeat':repeat,'runtime_status':'failed',
        'runtime_error':str(error),'stage_seconds':{},'replay':{'exact_production_replay':False},'no_retry':True}


def child(plan_path,seed,sid,arm,out):
    plan=read_json(plan_path);case={'seed':seed,'sample_id':sid,'arm':arm};require(case in cases(),'Unregistered cold case')
    started=time.perf_counter()
    try:
        verify_bound_inputs(plan);runtime=Runtime(plan,seed,sample_ids=(sid,),arms=(arm,))
        loading=time.perf_counter()-started
        row=run_case(runtime,case,out,'cold','measured',1)
        # Separate setup record avoids rewriting the immutable per-attempt result.
        write_json(Path(out)/'SETUP.json',{'verified_input_and_runtime_loading_seconds':loading,
            'post_import_worker_seconds':time.perf_counter()-started,'CUDA_device':torch.cuda.get_device_name(0),
            'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,'python':platform.python_version(),
            'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'tf32':torch.backends.cuda.matmul.allow_tf32,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cached_templates_before_pipeline':False,'CUDA_context_new_process':True})
        return row
    except Exception as exc:
        if not Path(out).exists():save_record(out,failed_record(case,'cold','measured',1,repr(exc)))
        write_json(Path(out)/'SETUP_FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        return None


def median_stats(values):
    if not values:return None
    values=np.asarray(values,np.float64)
    return {'n':len(values),'median':float(np.median(values)),'mean':float(values.mean()),
            'p05':float(np.quantile(values,.05)),'p95':float(np.quantile(values,.95)),
            'min':float(values.min()),'max':float(values.max())}


def summarize(rows,cold_parent):
    require(len(rows)==120,'Missing warmup/cold/measured slots')
    keys=[(r['mode'],r['phase'],r['seed'],r['sample_id'],r['arm'],r['repeat']) for r in rows]
    expected={('cold','measured',c['seed'],c['sample_id'],c['arm'],1) for c in cases()}
    expected|={('warm',x['phase'],x['seed'],x['sample_id'],x['arm'],x['repeat']) for x in warm_schedule()}
    require(len(keys)==len(set(keys)) and set(keys)==expected,'Timing slots duplicated or substituted')
    cold_keys=[case_id(x) for x in cold_parent]
    require(len(cold_keys)==24 and len(set(cold_keys))==24 and set(cold_keys)=={case_id(x) for x in cases()},
            'Cold parent process ledger missing or substituted')
    groups=[]
    for seed in ('pooled',10,11):
        for mode in ('cold','warm'):
            for arm in ARMS:
                chosen=[r for r in rows if r['mode']==mode and r['phase']=='measured' and r['arm']==arm and (seed=='pooled' or r['seed']==seed)]
                times={name:median_stats([r[name] for r in chosen if name in r]) for name in ('mesh_total_seconds','checked_total_seconds')}
                stages={name:median_stats([r['stage_seconds'][name] for r in chosen if name in r['stage_seconds']]) for name in STAGES}
                parent=[r['parent_process_wall_seconds'] for r in cold_parent if r['arm']==arm and (seed=='pooled' or r['seed']==seed)] if mode=='cold' else []
                groups.append({'seed':seed,'mode':mode,'arm':arm,'planned_records':len(chosen),
                    'runtime_success_count':sum(r['runtime_status']=='ok' for r in chosen),
                    'exact_production_replay_count':sum(r['replay']['exact_production_replay'] for r in chosen),
                    'geometry_valid_count':sum(r.get('geometry_valid') is True for r in chosen),
                    'actual512_count':sum(r.get('budget512_match') is True for r in chosen),
                    'times_seconds':times,'stages_seconds':stages,'cold_parent_wall_seconds':median_stats(parent),
                    'timings_include_geometry_invalid_and_replay_mismatch_if_pipeline_completed':True})
    return {'groups':groups,'expected_cases':24,'cold_records':24,'warmup_records':24,'warm_measured_records':72,
        'all_recorded_outputs_exactly_match_production':all(r['replay']['exact_production_replay'] for r in rows),
        'runtime_failures':sum(r['runtime_status']!='ok' for r in rows),
        'cold_process_failure_count':sum(r.get('returncode',0)!=0 for r in cold_parent),
        'all_cold_processes_completed_normally':all(r.get('returncode',0)==0 for r in cold_parent),
        'geometry_invalid_count_including_repeats':sum(r.get('geometry_valid') is False for r in rows),
        'only8_fixed_samples_not_full128':True,'cold_not_OS_cache_cold':True,'warm_repeats_not_independent_training_runs':True,
        'failure_times_not_dropped':True,'full_pipeline_time_has_synchronization_instrumentation':True}


def run(run,out=None,isolation_note=None):
    require(isolation_note is not None and isolation_note.strip(),'Parent must supply isolation note for actual timing')
    require(not torch.cuda.is_initialized(),'Parent CUDA context must not exist before cold workers')
    run=Path(run).resolve();out=new_dir(out or run/'timing').resolve();plan=build_plan(run,out)
    write_json(out/'ISOLATION.json',{'operator_note':isolation_note,'OS_cache_flushed':False,
        'isolation_is_operator_scheduled_not_inferred_from_GPU_presence':True})
    plan_path=out/'TIMING_PROTOCOL.json';rows=[];cold_parent=[];root=Path(__file__).resolve().parents[3]
    for index,case in enumerate(cases()):
        target=out/'cold'/case_id(case)
        command=[sys.executable,'-B','-m','experiments.a_radial.system_probe.ours_timing','worker',
            '--plan',str(plan_path),'--seed',str(case['seed']),'--sample-id',case['sample_id'],'--arm',case['arm'],'--out',str(target)]
        started=time.perf_counter();completed=subprocess.run(command,cwd=root,capture_output=True,text=True)
        elapsed=time.perf_counter()-started
        if (target/'RESULT.json').exists():row=read_json(target/'RESULT.json')
        else:
            row=failed_record(case,'cold','measured',1,'worker missing RESULT; returncode='+str(completed.returncode))
            if target.exists():write_json(target/'PARENT_FAILURE.json',row)
            else:save_record(target,row)
        envelope={**case,'parent_process_wall_seconds':elapsed,'returncode':completed.returncode,
            'includes_import_startup_load_child_disk_and_shutdown':True,'command':command,
            'stdout':completed.stdout,'stderr':completed.stderr}
        write_json(out/'cold_processes'/f'{index:02d}.json',envelope);cold_parent.append(envelope);rows.append(row)
        print(json.dumps({'event':'cold_complete','case':case,'runtime_status':row['runtime_status'],'replay':row['replay']['exact_production_replay']}),flush=True)
    verify_bound_inputs(plan);runtimes={};load_errors={};load_records={}
    for seed in SEEDS:
        started=time.perf_counter()
        try:runtimes[seed]=Runtime(plan,seed)
        except Exception as exc:load_errors[seed]=repr(exc)
        load_records[str(seed)]={'seconds':time.perf_counter()-started,'error':load_errors.get(seed)}
    write_json(out/'WARM_LOADING.json',{'models_loaded_outside_warm_pipeline':True,'seeds':load_records})
    for index,item in enumerate(warm_schedule()):
        case={k:item[k] for k in ('seed','sample_id','arm')};phase=item['phase'];repeat=item['repeat']
        target=out/'warm'/phase/f'repeat{repeat}'/case_id(case)
        if case['seed'] in load_errors:row=save_record(target,failed_record(case,'warm',phase,repeat,load_errors[case['seed']]))
        else:row=run_case(runtimes[case['seed']],case,target,'warm',phase,repeat)
        rows.append(row)
        if (index+1)%24==0:print(json.dumps({'event':'warm_progress','completed':index+1,'planned':96}),flush=True)
    verify_bound_inputs(plan)
    result={'rows':rows,'cold_parent_processes':cold_parent,'timing_protocol_sha256':file_hash(plan_path),
        'production_inputs_unchanged':True,'original_production_outputs_modified':False}
    write_json(out/'RESULTS.json',result);summary=summarize(rows,cold_parent);write_json(out/'SUMMARY.json',summary)
    lines=['三出口固定索引冷／热联合计时','',
        '仅每种子00000、00016、00032、00048，共8个固定样本、24出口案例；没有挑选最快样本。',
        '冷：每案例单独新进程一次，共24次；父wall包含启动/import/加载及子进程写盘，不代表OS文件缓存冷。',
        '热：两种子模型和CPU出口缓存常驻，每案例先预热一次，再3轮记录，顺序正向／反向／正向。',
        'mesh与checked均包含H2D/FM32/一次latent逆归一/AE/D2H/原CPU出口；checked另含质量、拓扑与精确审计。',
        '阶段使用CUDA同步边界，属于带同步仪器的单样本延迟。pipeline不含写盘与生产结果hash比较；父cold wall包含子进程这些工作。','',
        f"全部120记录（含24预热）精确重放生产结果：{summary['all_recorded_outputs_exactly_match_production']}；运行失败：{summary['runtime_failures']}。几何失败与运行失败分开记账。",'',
        '|种子|模式|出口|完整mesh中位秒|含审计中位秒|cold父wall中位秒|重放通过/计划|','|---|---|---|---:|---:|---:|---:|']
    for group in summary['groups']:
        def val(value):return '-' if value is None else f"{value['median']:.6f}"
        lines.append(f"|{group['seed']}|{group['mode']}|{group['arm']}|{val(group['times_seconds']['mesh_total_seconds'])}|{val(group['times_seconds']['checked_total_seconds'])}|{val(group['cold_parent_wall_seconds'])}|{group['exact_production_replay_count']}/{group['planned_records']}|")
    lines+=['','未提供外部MeshFlow的同条件计时，不能据此声称完整系统胜出。原始生成、采样与出口数据未修改。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'timing_dir':str(out),'runtime_failures':summary['runtime_failures'],
        'all_exact_replay':summary['all_recorded_outputs_exactly_match_production']}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run');p.add_argument('--run',required=True);p.add_argument('--out');p.add_argument('--isolation-note',required=True)
    w=sub.add_parser('worker');w.add_argument('--plan',required=True);w.add_argument('--seed',required=True,type=int)
    w.add_argument('--sample-id',required=True);w.add_argument('--arm',required=True,choices=ARMS);w.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='worker':child(args.plan,args.seed,args.sample_id,args.arm,args.out)
    else:run(args.run,args.out,args.isolation_note)


if __name__=='__main__':main()
