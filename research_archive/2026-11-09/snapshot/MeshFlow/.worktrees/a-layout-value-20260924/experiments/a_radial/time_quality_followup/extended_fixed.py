"""Four registered Fixed failures: one uninterrupted 64-step trajectory each.

Reuses frozen optimizer instrumentation and audits. No network or training, no
restart from the returned best iterate, no outcome-based stopping or selection.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

UIDS=('5eb44c2aa29844391f9f3ce366e030fb','736820a85063a4371e5b68ee10358ea3',
      '8c919e992db7cb081e5b68ee10358ea3','e97a846820b094b8f439510dd98d9326')
GRID=(0,1,2,4,8,16,32,64)
OLD_GRID=GRID[:6]
SNAPSHOTS=tuple(range(65))
np=None


def need(ok,message):
    if not ok: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''): h.update(chunk)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def resolve(root,path): return Path(path) if Path(path).is_absolute() else root/path


def array_hash(*values):
    h=hashlib.sha256()
    for a in values:
        a=np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(json.dumps(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def load_arrays(path):
    with np.load(path,allow_pickle=False) as z: return {k:np.array(z[k]) for k in z.files}


def equal_array(actual,expected,label):
    need(actual.dtype==expected.dtype and actual.shape==expected.shape
         and np.array_equal(actual,expected),'Typed array differs: '+label)


def equal_faces(actual,expected,label):
    need(np.issubdtype(actual.dtype,np.integer) and np.issubdtype(expected.dtype,np.integer)
         and actual.shape==expected.shape and np.array_equal(actual,expected),
         'Integer connectivity differs: '+label)


def without_clock(value,ignore_iteration=False):
    excluded={'elapsed_seconds','audit_seconds','timing'}
    if ignore_iteration: excluded.add('iteration')
    if isinstance(value,dict):
        return {k:without_clock(v,ignore_iteration) for k,v in value.items() if k not in excluded}
    if isinstance(value,list): return [without_clock(v,ignore_iteration) for v in value]
    return value


def prepare(source_run,out):
    source_run=Path(source_run).resolve(); out=Path(out).resolve()
    need(not out.exists(),'Output already exists; no overwrite')
    p=read(source_run/'RUN_PROTOCOL.json'); s=read(source_run/'SUMMARY.json')
    c=read(source_run/'COMPLETE.json'); verification=read(source_run/'FINAL_VERIFICATION.json')
    ph=sha(source_run/'RUN_PROTOCOL.json')
    need(c['status']=='ALL_PLANNED_ATTEMPTS_COMPLETE' and verification['status']=='PASS'
         and c['protocol_sha256']==s['protocol_sha256']==verification['protocol_sha256']==ph,
         'Source experiment incomplete or unverified')
    need(s['recorded_cases']==144 and not s['partial'] and not s['technical_issues'],
         'Source denominator or technical status differs')
    need(tuple(p['observation_steps'])==OLD_GRID and p['max_steps']==16
         and p['snapshot_steps']==list(range(17)) and p['max_seconds']==3600
         and p['threads']==4 and p['interop_threads']==1,'Source execution contract changed')
    failed=tuple(row['uid'] for row in s['ledger'] if row['arms']['fixed']['final_pass_count']==0)
    need(failed==UIDS,'Registered four failures differ; do not choose another subset')
    bound={str(source_run/n):sha(source_run/n) for n in
           ('RUN_PROTOCOL.json','SUMMARY.json','COMPLETE.json','FINAL_VERIFICATION.json')}
    for name,digest in p['bound_files_sha256'].items():
        path=resolve(source_run,name)
        if path.suffix=='.py' or path.name=='support.npz':
            need(sha(path)==digest,'Frozen dependency changed: '+str(path)); bound[str(path)]=digest
    bound[str(Path(__file__).resolve())]=sha(__file__)
    specs=[]
    for uid in UIDS:
        idx=p['all16_sorted_uids'].index(uid); b=p['data'][uid]
        data=resolve(source_run,b['path']); need(sha(data)==b['sha256'],'Data identity changed')
        folder=source_run/'cases'/uid/'fixed'/'repeat_0'; task=read(folder/'TASK.json')
        arrays=resolve(folder,task['arrays_file']).resolve()
        need(arrays.is_relative_to(folder.resolve()) and sha(arrays)==task['arrays_file_sha256'],'Old arrays identity')
        parent_record=next(r for r in c['cases'] if r['uid']==uid and r['arm']=='fixed' and r['repeat']==0)
        need(sha(folder/'TASK.json')==parent_record['task_sha256']
             ==verification['input_sha256'][str(folder/'TASK.json')]
             and task['arrays_file_sha256']==verification['input_sha256'][str(arrays)]
             and task['protocol_sha256']==ph and task['data_file_sha256']==b['sha256'],
             'Old task/data are not bound to completed independent verification')
        need(task['status']=='completed' and task['completed_full_budget']
             and task['optimizer_seed']==p['optimizer_seed_base']+idx
             and task['metric_seeds']==[base+idx for base in p['evaluation_seed_bases']],
             'Old successful execution/seeds required')
        need(task['observations'][-1]['report']['qualified'] is True
             and task['observations'][-1]['report']['target_pass'] is False,'Old failure meaning changed')
        for path,digest in ((data,b['sha256']),(folder/'TASK.json',sha(folder/'TASK.json')),
                            (arrays,task['arrays_file_sha256'])): bound[str(path)]=digest
        specs.append({'uid':uid,'original_sorted16_index':idx,'data':str(data),'data_sha256':b['sha256'],
            'mainfield_hash':b['mainfield_hash'],'reference_hash':b['reference_hash'],
            'source_task':str(folder/'TASK.json'),'source_arrays':str(arrays),
            'optimizer_seed':task['optimizer_seed'],'metric_seeds':task['metric_seeds'],
            'quality_floor':task['baseline']['report']['quality']})
    protocol={'study':'FIXED_EXTENDED4_CONTINUOUS64_v1','created_utc':utc(),
        'source_run':str(source_run),'source_protocol_sha256':ph,'uids':list(UIDS),'cases':specs,
        'max_steps':64,'max_seconds':3600,'observation_steps':list(GRID),
        'snapshot_steps':list(SNAPSHOTS),'threads':4,'interop_threads':1,'device':'cpu',
        'points_each_direction':8192,'expected_cases':4,'repeats_per_uid':1,
        'planned_full_audits_per_case':9,'bound_files_sha256':bound,
        'continuation':'Single call from original Fixed at step0; preserves Adam and current/best state through step64',
        'prefix_parity':'Compare0..16 against source repeat0, excluding clocks, never substitute saved endpoint',
        'reference':'Same frozen2050-node4096-face radial field, not raw asset GT',
        'selection':'Frozen optimizer accepted feasible minimum INTERNAL loss; no MC feedback',
        'scope':'Posthoc bounded four-failure mechanism diagnostic, not an independent validation set or new timing main experiment',
        'timing':'Actual per-case wall includes condition/support, baseline+8 synchronous uncached audits, optimizer and final query; prefix comparison and disk excluded',
        'no_training_no_network_no_retry_no_fallback':True,
        'stagnation_interpretation':'Only observed finite64-step trajectory; no claim about infinite steps or alternative optimizers'}
    out.mkdir(parents=True)
    write(out/'RUN_PROTOCOL.json',protocol)
    return protocol


def verify_bindings(p):
    for path,digest in p['bound_files_sha256'].items():
        need(sha(path)==digest,'Bound file changed: '+path)


def retain_arrays(storage,result,events):
    snapshots=[] if result is None else result['snapshots']
    if snapshots:
        storage['current_iterations']=np.array([s['iteration'] for s in snapshots],np.int32)
        storage['current_u']=np.stack([s['u'] for s in snapshots])
        storage['current_vertices']=np.stack([s['vertices'] for s in snapshots])
        storage['faces']=snapshots[0]['faces'].copy()
    if events:
        storage['observer_iterations']=np.array([e['iteration'] for e in events],np.int32)
        storage['observer_current_u']=np.stack([e['current_u'] for e in events])
        storage['observer_delivered_u']=np.stack([e['delivered_u'] for e in events])


def prefix_check(storage,task,old_storage,old_task):
    need(task['optimizer'] is not None and len(task['optimizer']['trace'])>=17,'Prefix was not completed')
    count=0
    for key in ('current_iterations','current_u','current_vertices','observer_iterations',
                'observer_current_u','observer_delivered_u','delivered_u','delivered_vertices'):
        equal_array(storage[key][:17],old_storage[key],key); count+=1
    equal_array(storage['faces'],old_storage['faces'],'faces'); count+=1
    for prefix in ('baseline','raw')+tuple('observed_'+str(i) for i in OLD_GRID):
        for suffix in ('u','r','vertices','faces'):
            key=prefix+'_'+suffix; equal_array(storage[key],old_storage[key],key); count+=1
    for suffix in ('u','r','vertices'):
        equal_array(storage['observed_16_'+suffix],old_storage['final_'+suffix],'step16 final '+suffix); count+=1
    for key in ('trace','attempts'):
        current=[r for r in task['optimizer'][key] if r['iteration']<=16]
        need(without_clock(current)==without_clock(old_task['optimizer'][key]),'Prefix semantic '+key+' differs')
    need(without_clock(task['baseline']['report'])==without_clock(old_task['baseline']['report']),
         'Baseline numerical report differs')
    for step in OLD_GRID:
        row=next(r for r in task['observations'] if r['iteration']==step)
        old=next(r for r in old_task['observations'] if r['iteration']==step)
        need(row['status']==old['status']=='success','Prefix audit not successful')
        need(without_clock(row['report'])==without_clock(old['report']),
             'Prefix quality/exact/MC report differs at '+str(step))
        need(row['best_feasible_iteration']==old['candidate_best_feasible_iteration'],
             'Prefix incumbent iteration differs')
    return {'status':'PASS','typed_array_comparisons':count,'current_steps':17,'observed_steps':list(OLD_GRID),
        'trace_and_all_attempts_equal_except_elapsed':True,'saved_step16_equals_old_returned_incumbent':True,
        'quality_exact_and_both_MC_equal_except_audit_timings':True,'source_repeat':0}


def finite_stall(storage,result):
    if result is None or not result['trace']: return None
    trace=result['trace']; last=trace[-1]['iteration']
    accepted=[r['iteration'] for r in trace if r['accepted']]
    last_accept=accepted[-1] if accepted else 0
    arrays={int(k):u for k,u in zip(storage.get('current_iterations',[]),storage.get('current_u',[]))}
    same=bool(last_accept in arrays and all(i in arrays and np.array_equal(arrays[i],arrays[last_accept])
                            for i in range(last_accept,last+1)))
    pairs=[]
    for end in (16,32,64):
        a=[r for r in result['attempts'] if r['iteration']==end-1]
        b=[r for r in result['attempts'] if r['iteration']==end]
        if not a or not b: continue
        pairs.append({'iterations':[end-1,end],'proposal_counts':[len(a),len(b)],
            'same_proposal_diagnostics_except_iteration_and_time':without_clock(a,True)==without_clock(b,True),
            'both_all_rejected':not any(r['accepted'] for r in a+b),
            'reasons_at_second_step':[r['reason'] for r in b]})
    return {'actual_last_iteration':last,'last_accepted_iteration':last_accept,
        'accepted_after16':sum(r['accepted'] for r in trace if r['iteration']>16),
        'all_saved_current_directions_equal_from_last_accepted_through_actual_end':same,
        'adjacent_attempt_comparisons':pairs,'finite_observed_trajectory_only':True}


def run_case(run,p,spec):
    from ..exporter import RadialSupport
    from ..layout_student.deployment import make_fixed
    from ..time_quality16.observer_adapter import optimize_layout_observed
    from ..time_quality16.timed_eval import evaluate_candidate
    from ..time_quality16.runner import pack_mesh,reconstruct_incumbents,partial_optimizer_state
    uid=spec['uid']; out=run/'cases'/uid; out.mkdir(parents=True,exist_ok=False)
    write(out/'START.json',{'uid':uid,'utc':utc(),'protocol_sha256':sha(run/'RUN_PROTOCOL.json')})
    task={'uid':uid,'status':'running','spec':spec,'source_repeat':0,'baseline':None,
        'optimizer':None,'prefix_parity':None,'failure':None,'completed_full_budget':False,
        'observations':[{'iteration':i,'status':'unavailable_not_reached'} for i in GRID],
        'observation_results_control_optimizer':False,'one_continuous_call':True}
    storage={}; events=[]; result=None; phase='condition'; start=time.perf_counter()
    try:
        data=load_arrays(spec['data']); support=RadialSupport()
        need(array_hash(data['logr'])==spec['mainfield_hash']
             and array_hash(data['ref_vertices'],data['ref_faces'])==spec['reference_hash'],'Field/reference identity')
        rv,rf=support.mesh(data['logr'])
        equal_array(rv,data['ref_vertices'],'reference vertices'); equal_array(rf,data['ref_faces'],'reference faces')
        def evaluate(u,floor=None,baseline_metrics=None):
            return evaluate_candidate(u,data['faces'],data['logr'],support,data['ref_vertices'],data['ref_faces'],
                floor=floor,baseline_metrics=baseline_metrics,seeds=spec['metric_seeds'],points=8192)
        phase='baseline_audit'; baseline,report=evaluate(data['base_u']); pack_mesh(storage,'baseline',baseline)
        need(report['audit']['valid'] is True and report['quality']==spec['quality_floor'],'Fixed floor/reference changed')
        task['baseline']={'report':report,'case_elapsed_seconds':time.perf_counter()-start}
        phase='Fixed_initialization'; raw,_=make_fixed(support,data['logr'],data['base_u'],data['faces'])
        pack_mesh(storage,'raw',raw)
        floor=report['quality']; optimizer_floor={'min_angle_deg':floor['min_triangle_angle_deg'],
                                               'angle_q05_deg':floor['triangle_min_angle_p05_deg']}
        def on_record(*,iteration,current_u,delivered_u,faces,row,best_feasible_iteration):
            need(iteration==len(events) and iteration in SNAPSHOTS,'Observer sequence')
            need(np.array_equal(faces,data['faces']),'Observer faces changed')
            event={'iteration':iteration,'current_u':current_u,'delivered_u':delivered_u,
                   'best_feasible_iteration':best_feasible_iteration,'original_record':row}
            if iteration in GRID:
                observed=task['observations'][GRID.index(iteration)]; before=time.perf_counter()
                try:
                    mesh,audit=evaluate(delivered_u,floor,report['metrics'])
                    pack_mesh(storage,'observed_'+str(iteration),mesh)
                    observed.update(status='success',report=audit,best_feasible_iteration=best_feasible_iteration,
                        geometry_hash=array_hash(mesh['u'],mesh['r'],mesh['vertices'],mesh['faces']))
                except Exception as exc:
                    observed.update(status='audit_error',error=repr(exc),traceback=traceback.format_exc())
                observed['audit_seconds']=time.perf_counter()-before
                observed['case_elapsed_seconds']=time.perf_counter()-start
            events.append(event)
        phase='continuous_optimizer64'; opt_start=time.perf_counter()
        result=optimize_layout_observed(data['logr'],raw['u'],data['faces'],optimizer_floor,device='cpu',
            max_steps=64,max_seconds=3600,seed=spec['optimizer_seed'],snapshot_steps=SNAPSHOTS,on_record=on_record)
        task['optimizer_wall_with_observers_seconds']=time.perf_counter()-opt_start
        task['optimizer']={k:result[k] for k in ('stats','trace','attempts')}
        phase='final_query_and_incumbent_identity'
        incumbents=reconstruct_incumbents(result)
        need(len(events)==len(incumbents),'Observer count changed')
        for event,chosen,snapshot in zip(events,incumbents,result['snapshots']):
            equal_array(event['current_u'],snapshot['u'],'current snapshot')
            equal_array(event['delivered_u'],chosen['u'],'delivered incumbent')
            need(event['best_feasible_iteration']==chosen['best_feasible_iteration'],'Incumbent index differs')
        storage['delivered_u']=np.stack([x['u'] for x in incumbents])
        storage['delivered_vertices']=np.stack([x['vertices'] for x in incumbents])
        radii=support.query(data['logr'],result['u']); vertices=result['u']*radii[:,None]
        equal_array(vertices,result['vertices'],'fresh final vertices')
        pack_mesh(storage,'final',{'u':result['u'],'r':radii,'vertices':vertices,'faces':result['faces']})
        task['completed_full_budget']=result['stats']['iterations']==64
        if task['completed_full_budget'] and task['observations'][-1]['status']=='success':
            for k,value in (('u',result['u']),('r',radii),('vertices',vertices)):
                equal_array(storage['observed_64_'+k],value,'observed64/final '+k)
            equal_faces(storage['observed_64_faces'],result['faces'],'observed64/final faces')
            task['observed64_final_connectivity']={'same_integer_values':True,
                'observed_dtype':str(storage['observed_64_faces'].dtype),
                'optimizer_return_dtype':str(result['faces'].dtype),'no_dtype_conversion_applied':True}
        task['status']='completed' if all(x['status']=='success' for x in task['observations']) else 'incomplete_or_audit_error'
    except Exception as exc:
        task['status']='error'; task['failure']={'phase':phase,'error':repr(exc),'traceback':traceback.format_exc()}
        if result is None: result=partial_optimizer_state(exc)
        if result is not None: task['optimizer']={k:result[k] for k in ('stats','trace','attempts')}
    finally:
        task['total_case_wall_seconds']=time.perf_counter()-start
    retain_arrays(storage,result,events)
    task['snapshot_slots']=[{'iteration':i,'available':i in set(storage.get('current_iterations',[]))} for i in SNAPSHOTS]
    task['observer_events']=[{k:v for k,v in e.items() if k not in ('current_u','delivered_u')} for e in events]
    try:
        task['prefix_parity']=prefix_check(storage,task,load_arrays(spec['source_arrays']),read(spec['source_task']))
    except Exception as exc:
        task['prefix_parity']={'status':'FAIL','error':repr(exc),'traceback':traceback.format_exc()}
        if task['failure'] is None:
            task['failure']={'phase':'postrun_prefix_readback','error':repr(exc),'traceback':traceback.format_exc()}
        task['status']='error'
    task['finite_stall_diagnostic']=finite_stall(storage,result)
    task['completed_full_audit_calls']=int(task['baseline'] is not None)+sum(x['status']=='success' for x in task['observations'])
    task['planned_full_audit_calls']=9
    task['no_resume_no_model_no_MC_selection']=True
    task['prefix_readback_and_artifact_writes_excluded_from_case_wall']=True
    np.savez_compressed(out/'ARRAYS.npz',**storage)
    task['arrays_file']='ARRAYS.npz'; task['arrays_sha256']=sha(out/'ARRAYS.npz')
    task['array_hashes']={k:array_hash(a) for k,a in storage.items()}
    write(out/'TASK.json',task)
    return task


def execute(run):
    os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ.setdefault('OMP_NUM_THREADS','4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    global np
    import numpy as np
    import torch
    from ..time_quality16.observer_adapter import source_identity
    run=Path(run).resolve(); p=read(run/'RUN_PROTOCOL.json')
    need(not (run/'STARTED.json').exists(),'No retry of an attempted diagnostic')
    need(p['uids']==list(UIDS) and p['max_steps']==64 and p['observation_steps']==list(GRID)
         and p['snapshot_steps']==list(SNAPSHOTS),'Frozen diagnostic contract changed')
    verify_bindings(p)
    torch.set_num_threads(4)
    if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    need(not torch.cuda.is_initialized(),'CPU only')
    write(run/'STARTED.json',{'utc':utc(),'protocol_sha256':sha(run/'RUN_PROTOCOL.json'),
        'adapter_identity':source_identity(),'torch':torch.__version__,'numpy':np.__version__,
        'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'timing_is_single_diagnostic_not_isolated_repeat_benchmark':True})
    tasks=[]; start=time.perf_counter()
    try:
        for spec in p['cases']:
            t=run_case(run,p,spec); tasks.append(t)
            print(json.dumps({'uid':t['uid'],'status':t['status'],'prefix':t['prefix_parity']['status'],
                              'actual_steps':(t['optimizer'] or {}).get('stats',{}).get('iterations')}),flush=True)
        verify_bindings(p)
        rows=[]
        for t in tasks:
            endpoints=[]
            for step in (16,32,64):
                o=next(x for x in t['observations'] if x['iteration']==step)
                endpoint={'iteration':step,'status':o['status'],'case_seconds':o.get('case_elapsed_seconds')}
                if o['status']=='success':
                    r=o['report']; endpoint.update(qualified=r['qualified'],target_pass=r['target_pass'],
                        actual_faces=r['actual_face_count'],exact_valid=r['audit']['valid'],
                        gain_percent=[None if g is None else 100*g for g in r['gain_fractions']],
                        min_angle_margin_deg=r['quality']['min_triangle_angle_deg']-r['quality_floor']['min_triangle_angle_deg'],
                        p05_angle_margin_deg=r['quality']['triangle_min_angle_p05_deg']-r['quality_floor']['triangle_min_angle_p05_deg'],
                        geometry_hash=o['geometry_hash'])
                endpoints.append(endpoint)
            rows.append({'uid':t['uid'],'status':t['status'],'prefix_parity':t['prefix_parity'],
                'completed_full_budget':t['completed_full_budget'],'total_case_seconds':t['total_case_wall_seconds'],
                'endpoints':endpoints,'finite_stall_diagnostic':t['finite_stall_diagnostic']})
        summary={'status':'ALL_REGISTERED_ATTEMPTS_RECORDED','planned_cases':4,'recorded_cases':len(tasks),
            'technical_or_runtime_error_cases':sum(t['status']!='completed' for t in tasks),
            'prefix_pass_count':sum(t['prefix_parity']['status']=='PASS' for t in tasks),'rows':rows,
            'driver_wall_seconds':time.perf_counter()-start,'source_inputs_unchanged':True,
            'cuda_initialized':torch.cuda.is_initialized(),'finite_trajectory_only_no_unbounded_claim':True,
            'single_diagnostic_not_new_timing_primary_experiment':True,'no_training':True}
        write(run/'SUMMARY.json',summary); write(run/'COMPLETE.json',summary)
        return summary
    except BaseException as exc:
        write(run/'FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),
            'completed_uids':[t['uid'] for t in tasks],'no_automatic_retry':True})
        raise


def self_test(source_run=None):
    global np
    import numpy as np
    a=np.array([[1.,2.,3.]],dtype=np.float64); equal_array(a,a.copy(),'copy')
    for changed in (a.astype(np.float32),a+1e-8,a.reshape(3)):
        try: equal_array(a,changed,'negative')
        except ValueError: pass
        else: raise AssertionError('Array mutation/dtype/shape was accepted')
    need(without_clock({'elapsed_seconds':1,'q':2})==without_clock({'elapsed_seconds':3,'q':2}),'Clock filter')
    need(without_clock({'q':2})!=without_clock({'q':3}),'Scientific values must not be removed')
    arr={'current_iterations':np.arange(4),'current_u':np.stack([a,a+1,a+1,a+1])}
    result={'trace':[{'iteration':i,'accepted':i==1} for i in range(4)],'attempts':[]}
    out=finite_stall(arr,result)
    need(out['last_accepted_iteration']==1 and out['all_saved_current_directions_equal_from_last_accepted_through_actual_end'],'Finite plateau detection')
    arr['current_u'][3,0,0]+=1e-8
    need(not finite_stall(arr,result)['all_saved_current_directions_equal_from_last_accepted_through_actual_end'],'Plateau negative control')
    faces=np.array([[0,1,2]],np.int32); equal_faces(faces,faces.astype(np.int64),'dtype semantics')
    for changed in (faces.astype(np.float64),faces+1,faces.reshape(3)):
        try: equal_faces(faces,changed,'negative')
        except ValueError: pass
        else: raise AssertionError('Noninteger/changed connectivity accepted')
    real_prefix_checks=[]
    if source_run is not None:
        for uid in UIDS:
            folder=Path(source_run)/'cases'/uid/'fixed'/'repeat_0'
            old=read(folder/'TASK.json'); arrays=load_arrays(resolve(folder,old['arrays_file']))
            task=copy.deepcopy(old)
            for o in task['observations']: o['best_feasible_iteration']=o['candidate_best_feasible_iteration']
            task['optimizer']['trace'][0]['elapsed_seconds']+=1
            task['baseline']['report']['audit']['audit_seconds']+=1
            prefix_check(arrays,task,arrays,old)
            altered=copy.deepcopy(task)
            altered['observations'][0]['report']['metrics'][0]['rms_bbox_percent']+=1e-8
            try: prefix_check(arrays,altered,arrays,old)
            except ValueError: pass
            else: raise AssertionError('MC mutation accepted by prefix check')
            altered_arrays=dict(arrays); altered_arrays['current_u']=arrays['current_u'].astype(np.float32)
            try: prefix_check(altered_arrays,task,arrays,old)
            except ValueError: pass
            else: raise AssertionError('Prefix dtype mutation accepted')
            real_prefix_checks.append(uid)
    return {'status':'PASS','real_saved_prefix_schema_checks':real_prefix_checks,
        'scope':'typed-array/connectivity, semantic-clock, saved-prefix MC/dtype and finite-stall controls; no optimizer/model/MC execution'}


def main():
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare'); p.add_argument('--source-run',required=True); p.add_argument('--out',required=True)
    p=subs.add_parser('run'); p.add_argument('--run',required=True)
    p=subs.add_parser('self-test'); p.add_argument('--source-run'); args=parser.parse_args()
    if args.command=='prepare': result=prepare(args.source_run,args.out); print(json.dumps({'status':'PREPARED','uids':result['uids']}))
    elif args.command=='run': result=execute(args.run); print(json.dumps({'status':result['status'],'recorded_cases':result['recorded_cases']}))
    else: print(json.dumps(self_test(args.source_run)))


if __name__=='__main__': main()