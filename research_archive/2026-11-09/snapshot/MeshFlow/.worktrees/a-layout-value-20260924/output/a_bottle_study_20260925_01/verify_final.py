"""Read-only independent recomputation of the 14x3x1 raw attempt ledger."""
import json, hashlib, math, statistics, sys
from pathlib import Path
import numpy as np
from experiments.a_radial.exporter import RadialSupport

ROOT=Path('output/a_bottle_condition14_once_20260925_01').resolve()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def check(ok,msg):
    if not ok: raise AssertionError(msg)
def save(n,d): (ROOT/n).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
p=read(ROOT/'RUN_PROTOCOL.json'); ps=sha(ROOT/'RUN_PROTOCOL.json'); s=read(ROOT/'SUMMARY.json')
arms=['fixed','correct_3k','mismatched_3k']; uids=p['uids']; grid=[0,1,2,4,8,16]
check(len(uids)==14 and p['repeats']==1 and p['max_steps']==16 and p['expected_cases']==42,'protocol count')
check(p['study_protocol_sha256']=='350fcc259a7760fe0b200eea9dba23fc3a8c4f7aa611eaa49cca99f97d84a6e0','study')
paths=list((ROOT/'cases').glob('*/*/repeat_*/TASK.json')); check(len(paths)==42,'all42 TASK files')
seen=set(); rows={u:{'uid':u,'donor_uid':p['donor_mapping'][u],'arms':{}} for u in uids}; raw={}; files={}; audits=0; snapshots=0; flagged=[]; support=RadialSupport()
for path in paths:
    t=read(path); u,a,n=t['uid'],t['arm'],t['repeat']; key=(u,a,n)
    check(key not in seen and u in uids and a in arms and n==0,'unique authorized attempt'); seen.add(key)
    check(path.parent==ROOT/'cases'/u/a/'repeat_0','path identity'); check(t['protocol_sha256']==ps,'protocol hash')
    check(t['data_file_sha256']==p['data'][u]['sha256'],'target file hash')
    expected_donor=None if a=='fixed' else u if a=='correct_3k' else p['donor_mapping'][u]
    check(t['condition_uid']==expected_donor,'condition routing'); check(t['target_mainfield_hash']==p['data'][u]['mainfield_hash'],'target field')
    if a!='fixed':
        check(t['condition_mainfield_hash']==p['data'][expected_donor]['mainfield_hash'],'donor field')
        check(t['model_sha256']=='7c02c7926d11c76f22d520a024fa7fac5d4c08c69050cb3745c226a4e6c1b760','frozen model')
    idx=uids.index(u)
    if 'optimizer_seed' in t: check(t['optimizer_seed']==1493501+idx and t['metric_seeds']==[1593501+idx,1693501+idx],'target seeds')
    files[str(path.relative_to(ROOT))]=sha(path)
    arrpath=path.parent/t['arrays_file']; check(sha(arrpath)==t['arrays_file_sha256'],'arrays hash'); files[str(arrpath.relative_to(ROOT))]=sha(arrpath)
    with np.load(p['data'][u]['path']) as data: field=data['logr'].copy(); faces=data['faces'].copy()
    with np.load(arrpath) as z:
        for prefix in ['baseline','raw','observed_0','observed_1','observed_2','observed_4','observed_8','observed_16','final']:
            if prefix+'_u' not in z: continue
            rr=support.query(field,z[prefix+'_u'])
            check(np.array_equal(rr,z[prefix+'_r']),'fresh target radius '+u+' '+a+' '+prefix)
            check(np.array_equal(z[prefix+'_vertices'],z[prefix+'_u']*rr[:,None]),'target vertices')
        if 'faces' in z: check(np.array_equal(z['faces'],faces) and z['faces'].shape==(512,3),'512 connectivity')
    total=t['timings']['total_end_to_end_seconds']; failure=t.get('failure'); failtime=(failure or {}).get('case_elapsed_seconds',total) if t['status']=='error' else None
    check(math.isfinite(total) and total>=0,'elapsed'); obs=[]; last=-1
    check([x['iteration'] for x in t['observations']]==grid,'audit grid')
    for x in t['observations']:
        if x['status']!='success': continue
        r=x['report']; when=x['case_elapsed_seconds']; check(last<=when<=total,'chronology');last=when; audits+=1
        check(r['actual_face_count']==512 and r['points_each_direction']==8192 and r['metric_seeds']==[1593501+idx,1693501+idx],'audit fixed contract')
        check(r['audit']['exact_executed'] is True and r['final_accepted_radius_freshly_queried'],'real audit/query')
        check(not r['MC_used_for_optimizer_selection_or_stopping'] and not r['MC_cache_used'],'MC independence')
        gains=[]
        for m,b in zip(r['metrics'],t['baseline']['report']['metrics']):
            gains.append(1-m['rms_bbox_percent']/b['rms_bbox_percent'] if b['rms_bbox_percent'] else None)
        check(gains==r['gain_fractions'],'RMS gains arithmetic')
        good=bool(r['qualified'] and all(g is not None and g>=.02 for g in gains))
        check(r['target_pass']==good,'joint decision')
        obs.append((when,x['iteration'],good,r))
    audits+=int(t.get('baseline') is not None);snapshots+=sum(x['status']=='available' for x in t.get('snapshot_slots',[]))
    def state(seconds):
        if failtime is not None and failtime<=seconds:return {'pass':False,'iteration':None,'state':'runtime_failure'}
        done=[x for x in obs if x[0]<=seconds and (failtime is None or x[0]<failtime)]
        return {'pass':bool(done and done[-1][2]),'iteration':done[-1][1] if done else None,'state':'audited' if done else 'not_yet_audited'}
    final=next((x for x in obs if x[1]==16),None)
    complete=t['status']=='completed' and t.get('completed_full_budget') and final is not None
    hits=[x for x in obs if x[2] and (failtime is None or x[0]<failtime)]
    r=final[3] if final else None
    margin=None if r is None else {'rms_gain_margin_percentage_points':[(g-.02)*100 if g is not None else None for g in r['gain_fractions']], 'angle_margin_degrees':{k:r['quality'][k]-r['quality_floor'][k] for k in ['min_triangle_angle_deg','triangle_min_angle_p05_deg']}}
    resources=read(path.parent/'RESOURCES.json')
    if not resources['boundary_resource_gate_pass']: flagged.append({'uid':u,'arm':a,'resources':resources})
    rows[u]['arms'][a]={'status':t['status'],'complete16':bool(complete),'final_pass':bool(complete and final[2]),'TTQ_seconds':hits[0][0] if hits else None,'total_seconds':total,'at_seconds':{str(b):state(b) for b in [2,4,8]},'final_margins':margin,'final_gain_percent':[g*100 if g is not None else None for g in r['gain_fractions']] if r else None,'final_RMS_bbox_percent':[m['rms_bbox_percent'] for m in r['metrics']] if r else None,'final_quality':r['quality'] if r else None,'fixed_quality_floor':r['quality_floor'] if r else None,'actual_face_count':r['actual_face_count'] if r else None,'failure':failure,'boundary_resource_gate_pass':resources['boundary_resource_gate_pass']}
    raw[key]=t
check(len(seen)==42,'full denominator')
for u in uids:
    for field in ['geometry_hash','metrics_hash']:
        hashes=[(raw[(u,a,0)].get('baseline') or {}).get(field) for a in arms]
        check(len({v for v in hashes if v is not None})<=1,'common Fixed reference '+field)
summary={}
for a in arms:
    vals=[rows[u]['arms'][a] for u in uids]
    summary[a]={'denominator':14,'final_pass':sum(v['final_pass'] for v in vals),'current_pass':{str(b):sum(v['at_seconds'][str(b)]['pass'] for v in vals) for b in [2,4,8]},'runtime_errors':sum(v['status']=='error' for v in vals),'total_seconds_median':statistics.median(v['total_seconds'] for v in vals),'total_seconds_min':min(v['total_seconds'] for v in vals),'total_seconds_max':max(v['total_seconds'] for v in vals),'sum_case_seconds':sum(v['total_seconds'] for v in vals)}
    check(summary[a]['final_pass']==s['groups'][a]['final_pass_count'],'independent final count')
    for b in [2,4,8]:
        reported=next(x for x in s['primary_current_audited']+s['auxiliary_current_audited'] if x['arm']==a and x['seconds']==b)
        check(summary[a]['current_pass'][str(b)]==reported['current_deliverable_pass_count'],'independent budget count')
pairs={}
for left,right in [('correct_3k','fixed'),('correct_3k','mismatched_3k')]:
    key=left+'_vs_'+right; groups={'both_pass':[],'only_left_pass':[],'only_right_pass':[],'neither_pass':[]}
    ttq={'both_hit_left_earlier':[],'both_hit_right_earlier':[],'both_hit_equal':[],'only_left_hit':[],'only_right_hit':[],'neither_hit':[]}
    for u in uids:
        l,r=rows[u]['arms'][left],rows[u]['arms'][right]
        group='both_pass' if l['final_pass'] and r['final_pass'] else 'only_left_pass' if l['final_pass'] else 'only_right_pass' if r['final_pass'] else 'neither_pass';groups[group].append(u)
        lt,rt=l['TTQ_seconds'],r['TTQ_seconds']
        group=('both_hit_left_earlier' if lt<rt else 'both_hit_right_earlier' if rt<lt else 'both_hit_equal') if lt is not None and rt is not None else 'only_left_hit' if lt is not None else 'only_right_hit' if rt is not None else 'neither_hit';ttq[group].append(u)
    pairs[key]={'final':groups,'observed_TTQ':ttq,'paired_total_difference_median_seconds':statistics.median(rows[u]['arms'][left]['total_seconds']-rows[u]['arms'][right]['total_seconds'] for u in uids)}
result={'status':'PASS','scope':'Independent raw TASK recomputation and saved-array target-radius readback; no model/optimizer/MC rerun','parents':14,'attempts':42,'timing_repeats':1,'audit_calls':audits,'available_snapshot_slots':snapshots,'resource_flag_count':len(flagged),'resource_flags':flagged,'summary':summary,'paired':pairs,'rows':[rows[u] for u in uids],'protocol_sha256':ps,'task_files_sha256':files,'verifier_sha256':sha(__file__),'single_wall_observation_no_stability_claim':True}
save('FINAL_VERIFICATION.json',result)
print(json.dumps({k:result[k] for k in ['status','parents','attempts','audit_calls','available_snapshot_slots','resource_flag_count','summary','paired']},ensure_ascii=False))