"""Independent post-COMPLETE integrity check. Stdlib + NumPy; no project imports.
No radial query, model, MC, optimizer or exact audit is executed; never overwrite.
"""
import argparse, datetime, hashlib, json, math, csv
from pathlib import Path
ARMS = ('fixed', 'early_v1_10', 'balanced_10')
GRID = (0, 1, 2, 4, 8, 16)
ANGLES = ('min_triangle_angle_deg', 'triangle_min_angle_p05_deg')
np = None

def need(ok, message):
    if not ok: raise ValueError(message)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def filehash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def ahash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a); h.update(str(a.dtype).encode()); h.update(json.dumps(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()
def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: np.array(z[k]) for k in z.files}
def sourcepath(root, name): return Path(name) if Path(name).is_absolute() else root / name

def mesh_check(z, prefix, faces, expected_hash):
    u, r, v = (z[prefix+'_'+k] for k in ('u', 'r', 'vertices'))
    f = z.get(prefix+'_faces', z.get('faces'))
    need(u.shape == v.shape == (258,3) and r.shape == (258,), 'mesh shape')
    need(u.dtype == r.dtype == v.dtype == np.float64, 'mesh dtype')
    need(all(np.isfinite(a).all() for a in (u,r,v)) and (r>0).all(), 'finite positive geometry')
    need(np.allclose(np.linalg.norm(u,axis=1),1,atol=1e-9,rtol=0), 'nonunit directions')
    need(np.issubdtype(f.dtype,np.integer) and np.array_equal(f,faces), 'connectivity differs')
    need(np.array_equal(v,u*r[:,None]), 'saved query radius times direction differs')
    need(ahash(u,r,v,f)==expected_hash, 'geometry_hash differs')
    return u,r,v,f

def report_check(report, baseline, seeds):
    q,a,m = report['quality'],report['audit'],report['metrics']; floor=q if baseline is None else baseline['quality']
    need(report['quality_floor']==floor and report['metric_seeds']==seeds and report['points_each_direction']==8192, 'floor/MC identity')
    need(len(m)==2 and all(x['seed']==s and x['points_per_direction']==8192 and math.isfinite(x['rms_bbox_percent']) and x['rms_bbox_percent']>=0 for x,s in zip(m,seeds)), 'MC values')
    qualified=bool(q['face_count']==report['actual_face_count']==512 and a.get('valid') is True and a.get('exact_executed') is True and q['degenerate_count']==0 and all(math.isfinite(q[k]) and q[k]+1e-6>=floor[k] for k in ANGLES))
    defined=[True,True] if baseline is None else [x['rms_bbox_percent']>0 for x in baseline['metrics']]
    gains=[0.,0.] if baseline is None else [1-x['rms_bbox_percent']/b['rms_bbox_percent'] if good else None for x,b,good in zip(m,baseline['metrics'],defined)]
    target=bool(baseline is not None and qualified and all(g is not None and g>=.02 for g in gains))
    need(report['qualified'] is qualified and report['target_pass'] is target, 'qualification/target boolean differs')
    need(report['gain_fractions']==gains and report['rms_denominator_defined']==defined and report['minimum_gain_fraction']==.02, 'gain values/threshold differs')
    need(report['MC_cache_used'] is False and report['MC_used_for_optimizer_selection_or_stopping'] is False and report['final_accepted_radius_freshly_queried'] is True, 'audit contract')
    return target

def ready(x,p):
    g=p['host_gate']
    return bool(x['compiler_process_count']==0 and x['free_physical_GiB']>=g['minimum_free_physical_GiB'] and x['free_commit_GiB']>=g['minimum_free_commit_GiB'] and x['cpu_percent']<g['maximum_host_cpu_percent'])

def check_case(run,p,ph,spec,record,data,summary,bound):
    uid,arm,rep=(spec[k] for k in ('uid','arm','repeat')); folder=run/'cases'/uid/arm/f'repeat_{rep}'
    path=folder/'TASK.json'; t=read(path); sh=filehash(path); zpath=(folder/t['arrays_file']).resolve()
    need(zpath.is_relative_to(folder.resolve()) and filehash(zpath)==t['arrays_file_sha256'], 'ARRAYS file hash/path')
    need(sourcepath(run,record['task_path']).resolve()==path.resolve() and record['task_sha256']==sh, 'COMPLETE task binding')
    identity={'uid':uid,'arm':arm,'repeat':rep,'protocol_sha256':ph,'data_file_sha256':p['data'][uid]['sha256'],'model_sha256':None if arm=='fixed' else p['model_bindings'][arm]['model_sha256'],'model_file_sha256':None if arm=='fixed' else p['model_bindings'][arm]['file_sha256']}
    need(all(t.get(k)==v for k,v in identity.items()) and all(read(folder/'START.json').get(k)==v for k,v in identity.items()), 'case identity')
    need(t['source_sha256']==p['bound_files_sha256'] and t['cuda_initialized'] is False and t['status']==record['status'], 'source/device/status')
    need(t['status'] in ('completed','error') and t['independent_MC_changes_optimizer_or_selection_or_stopping'] is False, 'case state')
    for f,sha in ((path,sh),(zpath,t['arrays_file_sha256'])): need(summary['task_files_sha256'][str(f.relative_to(run))]==sha,'report file binding')
    z=load(zpath); total=t['timings']['total_end_to_end_seconds']; need(math.isfinite(total) and total>=0,'case time')
    if t['status']=='error': need(t['failure'] and t['failure']['error'] and t['failure']['traceback'] and 0<=t['failure']['case_elapsed_seconds']<=total,'failure evidence')
    need([s['iteration'] for s in t['snapshot_slots']]==list(range(17)) and [o['iteration'] for o in t['observations']]==list(GRID),'planned slots')
    actual=z.get('current_iterations',np.array([],dtype=np.int32)).tolist()
    need(actual==list(range(len(actual))) and [x['iteration'] for x in t['snapshot_slots'] if x['status']=='available']==actual,'executed snapshots')
    if actual: need(z['current_u'].shape==z['current_vertices'].shape==(len(actual),258,3),'snapshot arrays')
    seeds=[s+p['all16_sorted_uids'].index(uid) for s in p['evaluation_seed_bases']]; prints={}; baseline=t.get('baseline')
    if 'mainfield_hash' in t: need(t['mainfield_hash']==ahash(data['logr']) and t['reference_hash']==ahash(data['ref_vertices'],data['ref_faces']), 'same field/reference')
    for name,prefix in (('baseline','baseline'),('raw_initialization','raw'),('final_return','final')):
        value=t.get(name)
        if value is None: continue
        mesh_check(z,prefix,data['faces'],value['geometry_hash']); prints[prefix]=(value['geometry_hash'],value.get('metrics_hash'))
        if name=='baseline':
            need(np.array_equal(z['baseline_u'],data['base_u']) and digest(value['report']['metrics'])==value['metrics_hash'],'baseline identity')
            report_check(value['report'],None,seeds)
    success=[]; previous=-1.
    for o in t['observations']:
        need(all(o.get(k)==v for k,v in identity.items()),'observation identity')
        if o['status']!='success':
            need(o['status'] in ('audit_error','unavailable_not_reached') and o['audit_completed_case_seconds'] is None,'false completed audit')
            need(t['status']=='error' or not t['completed_full_budget'],'missing audit in full success')
            continue
        need(baseline is not None,'candidate without baseline'); step=o['iteration']; when=o['audit_completed_case_seconds']
        need(math.isfinite(when) and previous<=when<=total and o['case_elapsed_seconds']==when,'audit chronology'); previous=when
        u,r,v,f=mesh_check(z,o['mesh_prefix'],data['faces'],o['geometry_hash'])
        need(digest(o['report']['metrics'])==o['metrics_hash'] and ahash(v,data['faces'])==o['candidate_hash'],'candidate/metrics hash')
        if 'delivered_u' in z: need(np.array_equal(u,z['delivered_u'][step]) and np.array_equal(v,z['delivered_vertices'][step]),'audited wrong saved incumbent')
        if step==0: need(all(np.array_equal(x,z['raw_'+k]) for x,k in zip((u,r,v,f),('u','r','vertices','faces'))),'grid0/raw differs')
        if step==16 and 'final_u' in z: need(all(np.array_equal(x,z['final_'+k] if k!='faces' else z['faces']) for x,k in zip((u,r,v,f),('u','r','vertices','faces'))),'grid16/final differs')
        passed=report_check(o['report'],baseline['report'],seeds); success.append((step,when,passed)); prints['obs'+str(step)]=(o['geometry_hash'],o['metrics_hash'])
    audits=int(baseline is not None)+len(success)
    need(t['timings']['planned_full_audit_calls']==7 and t['timings']['completed_full_audit_calls']==audits,'audit count')
    if t['status']=='completed' and t['completed_full_budget']: need(len(actual)==17 and audits==7,'full budget missing observations')
    resources=read(folder/'RESOURCES.json'); before,after=(ready(resources[k],p) for k in ('before','after'))
    need(resources['boundary_resource_gate_pass'] is (before and after) and record['boundary_resource_gate_pass'] is (before and after),'resource flag mismatch')
    for f in (path,zpath,folder/'START.json',folder/'RESOURCES.json'): bound[str(f)]=filehash(f)
    return {'uid':uid,'arm':arm,'repeat':rep,'status':t['status'],'task_sha256':sh,'arrays_sha256':t['arrays_file_sha256'],'full_budget':t['completed_full_budget'],'audits':audits,'snapshots':len(actual),'prints':prints,'array_hashes':{k:ahash(a) for k,a in z.items()},'resources':{'before_pass':before,'after_pass':after,'boundary_pass':before and after,'before':resources['before'],'after':resources['after']},'total':total,'failure_time':(t.get('failure') or {}).get('case_elapsed_seconds'),'success':success}

def main():
    a=argparse.ArgumentParser(description=__doc__); a.add_argument('--run',required=True); args=a.parse_args(); run=Path(args.run).resolve(); out=run/'FINAL_VERIFICATION.json'
    need((run/'COMPLETE.json').exists(),'Wait for COMPLETE; never inspect the live benchmark'); need(not out.exists(),'No overwrite')
    global np
    import numpy as np
    p=read(run/'RUN_PROTOCOL.json'); c=read(run/'COMPLETE.json'); s=read(run/'SUMMARY.json'); ph=filehash(run/'RUN_PROTOCOL.json')
    need(c['status']=='ALL_PLANNED_ATTEMPTS_COMPLETE' and c['protocol_sha256']==s['protocol_sha256']==ph,'completion identity')
    need(c['source_inputs_unchanged'] is True and c['cuda_initialized'] is False,'completion source/device')
    need(p['arms']==list(ARMS) and p['repeats']==3 and len(p['uids'])==16 and len(c['cases'])==144,'fixed denominator')
    need([{k:r[k] for k in ('uid','arm','repeat')} for r in c['cases']]==p['case_order'],'case order')
    need(len(list((run/'cases').glob('*/*/repeat_*/TASK.json')))==144,'TASK file count')
    need(s['recorded_cases']==s['expected_cases']==144 and s['pending_cases']==0 and s['ledger_rows']==16 and s['partial'] is False,'report denominator')
    need([r['uid'] for r in s['ledger']]==p['uids'],'report UID rows')
    with (run/'LEDGER.csv').open(encoding='utf-8-sig',newline='') as f: csvrows=list(csv.DictReader(f))
    need(len(csvrows)==16 and [r['uid'] for r in csvrows]==p['uids'],'CSV denominator')
    bound={str(run/n):filehash(run/n) for n in ('RUN_PROTOCOL.json','COMPLETE.json','SUMMARY.json','same_time_curve.json','LEDGER.csv','LEDGER.md')}; data={}; results=[]; errors=[]
    for uid,b in p['data'].items():
        path=sourcepath(run,b['path']); need(filehash(path)==b['sha256'],'data hash'); bound[str(path)]=b['sha256']; data[uid]=load(path)
    for b in p['model_bindings'].values():
        path=sourcepath(run,b['path']); need(filehash(path)==b['file_sha256'],'model-only hash'); bound[str(path)]=b['file_sha256']
    for path,sha in p['bound_files_sha256'].items():
        if Path(path).suffix=='.py': need(filehash(sourcepath(run,path))==sha,'frozen source hash'); bound[str(sourcepath(run,path))]=sha
    for spec,record in zip(p['case_order'],c['cases']):
        try: results.append(check_case(run,p,ph,spec,record,data[spec['uid']],s,bound))
        except Exception as exc: errors.append({'case':spec,'error':repr(exc)})
    lookup={(r['uid'],r['arm'],r['repeat']):r for r in results}; repeat_checks=[]
    for row,csvrow in zip(s['ledger'],csvrows):
        for arm in ARMS:
            reps=[lookup.get((row['uid'],arm,i)) for i in range(3)]; group=row['arms'][arm]
            if not all(reps): continue
            try:
                need(json.loads(csvrow[arm+'_repeats'])==group['repeats'],'CSV/JSON repeat rows differ')
                for rep,record in zip(reps,group['repeats']):
                    need(record['repeat']==rep['repeat'] and record['source_status']==rep['status'] and record['total_seconds']==rep['total'] and record['failure_seconds']==rep['failure_time'],'report case identity/timing')
                    eligible=[x for x in rep['success'] if rep['failure_time'] is None or x[1]<rep['failure_time']]; hits=[x[1] for x in eligible if x[2]]
                    final=bool(rep['status']=='completed' and rep['full_budget'] and any(x[0]==16 and x[2] for x in rep['success']))
                    need(record['first_hit_seconds']==(hits[0] if hits else None) and record['final_pass'] is final,'report first-hit/final differs')
                for key in set().union(*(r['prints'] for r in reps)):
                    need(len({r['prints'][key] for r in reps if key in r['prints']})==1,'repeat geometry/MC differs: '+key)
                if all(r['status']=='completed' and r['full_budget'] for r in reps): need(reps[0]['array_hashes']==reps[1]['array_hashes']==reps[2]['array_hashes'],'repeat snapshot arrays differ')
                repeat_checks.append({'uid':row['uid'],'arm':arm,'status':'PASS'})
            except Exception as exc: errors.append({'group':[row['uid'],arm],'error':repr(exc)})
    failed=sum(r['status']=='error' for r in results); resource_rows=[{'uid':r['uid'],'arm':r['arm'],'repeat':r['repeat'],**r['resources']} for r in results]
    if len(results)==144 and s['runtime_failure_cases']!=failed: errors.append({'error':'report runtime failure count'})
    all_boundaries=len(results)==144 and all(r['boundary_pass'] for r in resource_rows)
    if len(results)==144 and c['all_boundary_resource_gates_pass'] is not all_boundaries: errors.append({'error':'aggregate resource flag'})
    changed=[path for path,sha in bound.items() if filehash(path)!=sha]
    if changed: errors.append({'error':'inputs changed during readback','paths':changed})
    value={'status':'PASS' if not errors and len(results)==144 else 'FAIL','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verifier_sha256':filehash(__file__),'protocol_sha256':ph,'expected_cases':144,'verified_cases':len(results),'ledger_rows':16,'runtime_failure_cases':failed,'science_failure_is_not_integrity_failure':True,'completed_audits':sum(r['audits'] for r in results),'planned_audits':1008,'executed_snapshots':sum(r['snapshots'] for r in results),'planned_snapshot_slots':2448,'cases':results,'repeat_checks':repeat_checks,'resources':{'records':resource_rows,'planned_count':144,'recorded_count':len(resource_rows),'before_failed':sum(not r['before_pass'] for r in resource_rows),'after_failed':sum(not r['after_pass'] for r in resource_rows),'either_failed':sum(not r['boundary_pass'] for r in resource_rows),'all_boundary_gates_pass':all_boundaries,'continuous_isolation_not_certified':True},'source_inputs_unchanged':not changed,'input_sha256':bound,'errors':errors,'scope':'storage/identity and recorded quality/validity/MC decision recheck; no fresh field query, geometry audit, MC, model or optimization; saved r*u is checked bitwise, not independently re-queried; historical full checkpoints rely on driver pre/post verification'}
    with out.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({k:value[k] for k in ('status','verified_cases','runtime_failure_cases','completed_audits')}))
    if value['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()