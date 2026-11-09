"""Independent post-run single-state audit. No extraction, fitting or sampling."""
from pathlib import Path
import argparse,collections,hashlib,itertools,json,shutil,time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from audit_fm_outputs import independent_topology, valid, failure_tags


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def equal(a,b):return np.array_equal(a,b,equal_nan=True)
def assign(a,b):return linear_sum_assignment(((a[:,None,:]-b[None,:,:])**2).sum(-1))[1]
def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()
def canon(f):return np.array(sorted(tuple(sorted(map(int,t))) for t in f),dtype=np.int64)
def decode(state):
    x=state[:,:3].astype(np.float64);F=state[:,6:].astype(np.float64).reshape(-1,4,4)
    T=np.broadcast_to(np.eye(4),(len(x),4,4)).copy();T[:,:3,3]=-x
    Q=T.transpose(0,2,1)@(F@F.transpose(0,2,1))@T
    return dict(v=x,p=x.copy(),n=state[:,3:6].copy(),Q=Q,eigs=np.linalg.svd(Q[:,:3,:3],compute_uv=False))
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def semantic(s,t,diag):
    if not np.isfinite(s).all():return dict(finite=False)
    p=s[:,:3];pt=t[:,:3];n=s[:,3:6];nt=t[:,3:6];F=s[:,6:].reshape(-1,4,4);Ft=t[:,6:].reshape(-1,4,4)
    C=F@F.transpose(0,2,1);Ct=Ft@Ft.transpose(0,2,1);ln=np.linalg.norm(n,axis=1);lt=np.linalg.norm(nt,axis=1);good=(ln>0)&(lt>0)
    angles=np.degrees(np.arccos(np.clip(np.sum(n[good]*nt[good],axis=1)/(ln[good]*lt[good]),-1,1)))
    rms=float(np.sqrt(np.mean(np.sum((p-pt)**2,axis=1))))
    result=dict(finite=True,P_rms=rms,P_rms_pct_bbox=100*rms/diag,n_mse=float(np.mean((n-nt)**2)),n_magnitude_rms=float(np.sqrt(np.mean((ln-lt)**2))),n_oriented_angle_mean_deg=float(np.mean(angles)) if len(angles) else None,n_oriented_angle_p95_deg=float(np.quantile(angles,.95)) if len(angles) else None,n_zero_count=int((~good).sum()),F_mse=float(np.mean((F-Ft)**2)),F_relative=rel(F,Ft),C_relative=rel(C,Ct))
    e=np.linalg.svd(C[:,:3,:3],compute_uv=False)[:,0];et=np.linalg.svd(Ct[:,:3,:3],compute_uv=False)[:,0]
    result['zero_spatial_spectrum_count']=int(((e<=0)|(et<=0)).sum())
    if np.all(e>0) and np.all(et>0):result['C_spectral_relative']=rel(C/e[:,None,None],Ct/et[:,None,None])
    if np.all(good) and np.all(e>0) and np.all(et>0):
        a,b=decode(s),decode(t);result.update(world_Q_relative=rel(a['Q'],b['Q']),world_Q_spectral_relative=rel(a['Q']/a['eigs'][:,0,None,None],b['Q']/b['eigs'][:,0,None,None]))
    return result
def near_dict(a,b):
    return a.keys()==b.keys() and all((a[k] is None and b[k] is None) or (a[k] is not None and b[k] is not None and np.isclose(a[k],b[k],rtol=2e-8,atol=2e-10)) for k in a)


def main(root):
    started=time.perf_counter();root=Path(root).resolve();out=root/'post_analysis'/'independent_audit'
    if out.exists():raise RuntimeError('Refuse existing independent audit')
    checks=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    protocol=read(root/'RUN_PROTOCOL.json');execution=read(root/'EXECUTION.json');previous=Path(protocol['previous_run']);target_run=Path(protocol['target_run'])
    check('execution_complete',execution['status']=='COMPLETE')
    manifest=read(root/'MANIFEST.json');bad=[]
    for r in manifest['files']:
        p=root/r['path']
        if not p.is_file() or sha(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:bad.append(r['path'])
    check('original_manifest',not bad,dict(count=len(manifest['files']),bad=bad))
    check('protocol_hash',sha(root/'RUN_PROTOCOL.json')==read(root/'RUN_PROTOCOL_SHA256.json')['sha256'])
    check('source_snapshot',all(sha(root/r['path'])==r['sha256'] for r in read(root/'SOURCE_SNAPSHOT.json')['files']))
    check('external_source_bindings',all(sha(r['path'])==r['sha256'] for r in protocol['source_files']))
    data=arrays(root/'data.npz');target=arrays(root/'target/physical_state.npz')['state'];old=arrays(previous/'data.npz')
    check('one_q_only_target',data['states'].shape==(1,256,22) and equal(data['states'][0],target) and str(data['object_names'][0])=='thin_u_frame06__q_only')
    check('old_chart_reused',equal(data['mean'],old['mean']) and equal(data['scale'],old['scale']))
    check('source_target_binding',sha(root/'target/original_parameters.npz')==sha(target_run/'parameters.npz') and sha(root/'target/canonical_mesh.obj')==sha(target_run/'mesh.obj'))
    original=arrays(root/'target/original_parameters.npz');decoded=decode(target)
    check('target_state_semantics',equal(target[:,:3],original['v']) and equal(target[:,3:6],original['n']) and np.allclose(decoded['Q'],original['Q'],rtol=1e-10,atol=1e-13))
    noise=arrays(root/'sample_noise.npz')['noise'];probe=arrays(root/'probe_noise.npz');times=np.array([0,.25,.5,.75,.9,.99],np.float32)
    check('sampling_noise_old_bytes',sha(root/'sample_noise.npz')==sha(previous/'sample_noise.npz'))
    check('probe_seed_and_times',equal(probe['noise'],np.random.default_rng(260929).standard_normal((32,256,22)).astype(np.float32)) and equal(probe['times'],times))
    check('sample_probe_distinct',not ({array_sha(x) for x in noise}&{array_sha(x) for x in probe['noise']}))
    train=root/'training/direct';tr=read(train/'RESULT.json');trace=[json.loads(l) for l in (train/'TRAIN_TRACE.jsonl').read_text().splitlines()];ck=torch.load(train/'final.pt',map_location='cpu',weights_only=False)
    check('3000_updates',tr['status']=='COMPLETED' and tr['completed_steps']==tr['attempted_batches']==3000 and [x['step'] for x in trace]==list(range(1,3001)) and ck['completed_steps']==3000 and all(int(x['step'])==3000 for x in ck['optimizer_state_dict']['state'].values()))
    check('training_hash_bindings',sha(train/'final.pt')==tr['final_checkpoint_sha256'] and sha(train/'PROTOCOL.json')==tr['protocol_sha256']==ck['protocol_sha256'] and sha(train/'initial_model.pt')==tr['initial_model_sha256'])
    check('training_data_binding',read(train/'PROTOCOL.json')['data_sha256']==sha(root/'data.npz') and tr['exposure_counts']==[24000])
    ini=torch.load(train/'initial_model.pt',map_location='cpu',weights_only=True);oldini=torch.load(previous/'training/direct/initial_model.pt',map_location='cpu',weights_only=True)
    check('same_model_initial_tensors',ini.keys()==oldini.keys() and all(torch.equal(ini[k],oldini[k]) for k in ini))
    digest=bytes(32)
    for row in trace:digest=hashlib.sha256(digest+bytes.fromhex(row['batch_sha256'])).digest()
    check('stream_digest',digest.hex()==tr['attempted_stream_sha256']==ck['attempted_stream_sha256'])
    # Replay training noise sequence; check all 24000 whole-noise samples for exact
    # reuse by sample/probe, and four independent whole-element Hungarian batches.
    generator=torch.Generator();generator.manual_seed(260927);forbidden={array_sha(x) for x in noise}|{array_sha(x) for x in probe['noise']};collisions=[];batch_replay=[]
    for step in range(3000):
        n=torch.randn(8,256,22,generator=generator,dtype=torch.float32);tt=torch.rand(8,1,1,generator=generator,dtype=torch.float32)
        for b,x in enumerate(n.numpy()):
            if array_sha(x) in forbidden:collisions.append([step+1,b])
        if step in [0,999,1999,2999]:
            x=n.numpy()[...,:3]*data['scale'][:3]+data['mean'][:3];targets=np.stack([((target[assign(p,target[:,:3])]-data['mean'])/data['scale']).astype(np.float32) for p in x])
            h=hashlib.sha256(n.numpy().tobytes()+tt.numpy().tobytes()+targets.tobytes()+np.zeros(8,np.int64).tobytes()).hexdigest()
            batch_replay.append(dict(step=step+1,sha256=h,matched=h==trace[step]['batch_sha256']))
    check('independent_training_batch_replays',all(x['matched'] for x in batch_replay),batch_replay)
    check('sampling_probe_no_exact_training_noise_reuse',not collisions,dict(training_noise_count=24000,collisions=collisions,caution='Distinct pseudorandom streams and no exact reuse, not a mathematical independence proof.'))
    canonical=arrays(root/'target/canonical_mesh.npz');canonical_faces=canon(canonical['input_vertex_ids'][canonical['faces']])
    # The corner order is frozen in the actual target fixed-domain artifact.
    signs=arrays(root/'oracle/float64_fixed/corners.npz')['corners'];check('fixed_actual_plus_minus_one',signs.shape==(8,3) and set(map(tuple,signs))==set(itertools.product([-1.,1.],repeat=3)))
    geometry_records=[]
    def geometry_check(row,directory,expected,perm=None):
        label=str(directory.relative_to(root));stored=arrays(directory/'state.npz')['physical_state'];check(label+'_selected_chart',equal(stored,expected))
        check(label+'_row_binding',read(directory/'result.json')==row)
        pp=directory/'parameters.npz'
        if pp.is_file():
            pars=arrays(pp);want=decode(expected)
            check(label+'_world_Q_redecode',all(equal(pars[k],want[k]) for k in want) and sha(pp)==row['parameters_sha256'])
            x=want['v'];lo=x.min(0);hi=x.max(0);extent=hi-lo
            if row['domain']=='fixed':corners=signs
            else:
                margin=np.where(extent>0,.1*extent,1e-6*extent.max());corners=np.where(signs>0,(hi+margin)[None],(lo-margin)[None])
            check(label+'_domain_corners',equal(arrays(directory/'corners.npz')['corners'],corners))
            tp=directory/'extractor_trace.npz'
            if tp.is_file():
                z=arrays(tp)
                if 'delaunay_points' in z:check(label+'_trace_sites',equal(z['delaunay_points'][:256],x) and equal(z['delaunay_points'][256:],corners))
        mesh=directory/'mesh.npz';obs=dict(location=label,status=row['status'],domain=row['domain'],valid=valid(row),declared_valid=row['valid'],failure_tags=failure_tags(row),faces=row.get('faces'),rms_pct_bboxdiag=row.get('rms_pct_bboxdiag'))
        check(label+'_valid_contract',valid(row)==row['valid'])
        if mesh.is_file():
            z=arrays(mesh);top=independent_topology(z['vertices'],z['faces']);obs['independent_topology']=top
            check(label+'_topology',top['faces']==row['faces'] and len(top['bad_vertex_links'])==row['strict_noncycle_vertex_count'] and top['boundary_edges']==row['boundary_edges'] and top['nonmanifold_edges']==row['nonmanifold_edges'] and (top['betti_if_closed_oriented'] is None or top['betti_if_closed_oriented']==[row['beta'+str(k)] for k in range(3)]))
            check(label+'_mesh_hash',sha(directory/'mesh.obj')==row['mesh_sha256'])
            if perm is not None:
                mapping=np.r_[perm,np.arange(256,264)];obs['same_original_target_id_faces']=equal(canon(mapping[z['input_vertex_ids'][z['faces']]]),canonical_faces)
        geometry_records.append(obs);return obs
    targetcheck=read(root/'TARGET_CHECK.json');oracles=[]
    for row in targetcheck['rows']:
        s=target if row['precision']=='float64' else ((((target-data['mean'])/data['scale']).astype(np.float32)*data['scale'].astype(np.float32))+data['mean'].astype(np.float32)).astype(np.float64)
        oracles.append(geometry_check(row,root/'oracle'/(row['precision']+'_'+row['domain']),s,np.arange(256)))
    check('four_oracle_slots',{(r['precision'],r['domain']) for r in targetcheck['rows']}==set(itertools.product(['float64','float32_chart'],['fixed','enclosing'])) and len(targetcheck['rows'])==4)
    samples={};mix_summaries={};diag=float(np.linalg.norm(np.ptp(target[:,:3],axis=0)))
    def audit_rollout(nfe):
        sd=root/f'sampling_{nfe}';sample=arrays(sd/'endpoints.npz');ro=read(sd/'ROLLOUT.json');check(f'rollout{nfe}_32',len(ro['rows'])==32 and {r['sample_id'] for r in ro['rows']}==set(range(32)) and sample['states'].shape==(32,256,22))
        check(f'rollout{nfe}_bindings',ro['noise_sha256']==sha(root/'sample_noise.npz') and ro['checkpoint_sha256']==sha(train/'final.pt'))
        for row in ro['rows']:
            i=row['sample_id'];pred=sample['states'][i];perm=sample['target_permutations'][i]
            if row['status']=='COMPLETE':
                check(f'rollout{nfe}_{i}_post_assignment',equal(perm,assign(pred[:,:3],target[:,:3])) and row['nfe']==nfe)
                check(f'rollout{nfe}_{i}_endpoint_semantic',near_dict(semantic(pred,target[perm],diag),row['endpoint_semantic']))
            initial=(noise[i]*data['scale'].astype(np.float32)+data['mean'].astype(np.float32)).astype(np.float64)
            check(f'rollout{nfe}_{i}_initial_noise_state',np.allclose(sample['snapshots'][i,0],initial,rtol=2e-6,atol=2e-7))
            check(f'rollout{nfe}_{i}_checkpoint',row['checkpoint_sha256']==sha(train/'final.pt'))
        ledger=read(root/f'interventions_{nfe}/LEDGER.json');combs=[''.join(c) for c in itertools.product('PO',repeat=3)] if nfe==64 else ['PPP']
        slots=set(itertools.product(range(32),combs,['fixed','enclosing']));check(f'mix{nfe}_complete_slots',len(ledger)==len(slots) and {(r['sample_id'],r['combination'],r['domain']) for r in ledger}==slots)
        for row in ledger:
            i=row['sample_id'];combo=row['combination'];perm=sample['target_permutations'][i];pred=sample['states'][i];available=np.all(perm>=0) and len(np.unique(perm))==256
            state=np.concatenate([pred[:,a:b] if c=='P' else target[perm,a:b] for c,(a,b) in zip(combo,[(0,3),(3,6),(6,22)])],axis=1) if available else np.full((256,22),np.nan)
            check(f'mix{nfe}_{i}_{combo}_{row["domain"]}_metadata',row['nfe']==nfe and row['generated']==(combo=='PPP') and row['oracle_assisted']==(combo!='PPP') and equal(row['target_permutation'],perm) and row['checkpoint_sha256']==sha(train/'final.pt'))
            geometry_check(row,root/f'interventions_{nfe}'/f'{i:03d}_{combo}_{row["domain"]}',state,perm if available else None)
        summary={d:{c:dict(denominator=32,valid=sum(valid(r) for r in ledger if r['domain']==d and r['combination']==c),status_counts=dict(collections.Counter(r['status'] for r in ledger if r['domain']==d and r['combination']==c)),failure_tags=dict(collections.Counter(t for r in ledger if r['domain']==d and r['combination']==c for t in failure_tags(r)))) for c in combs} for d in ['fixed','enclosing']}
        stored=read(root/f'interventions_{nfe}/SUMMARY.json');check(f'mix{nfe}_summary',all(summary[d][c]['valid']==stored['by_domain'][d][c]['valid'] for d in summary for c in combs))
        samples[nfe]=dict(rollout=ro,states_shape=list(sample['states'].shape));mix_summaries[nfe]=summary
    audit_rollout(64)
    ps=arrays(root/'probes/probe_states.npz');prows=read(root/'probes/PROBE_ROWS.json');grows=read(root/'probes/GEOMETRY_ROWS.json');psum=read(root/'probes/SUMMARY.json')
    check('probe_192_slots',len(prows)==192 and {(r['probe_id'],r['time']) for r in prows}==set(itertools.product(range(32),map(float,times))))
    check('probe64_geometry_slots',len(grows)==64 and {(r['probe_id'],r['domain']) for r in grows}==set(itertools.product(range(32),['fixed','enclosing'])))
    check('probe_binding',psum['noise_sha256']==sha(root/'probe_noise.npz') and psum['checkpoint_sha256']==sha(train/'final.pt'))
    for row in prows:
        i=row['probe_id'];ti=int(np.where(times==row['time'])[0][0]);t=float(times[ti]);perm=assign(probe['noise'][i,:,:3].astype(np.float64)*data['scale'][:3]+data['mean'][:3],target[:,:3]);assigned=target[perm]
        check(f'probe_{i}_{ti}_matching',equal(ps['target_permutations'][i],perm))
        zt=((assigned-data['mean'])/data['scale']).astype(np.float32);z0=probe['noise'][i];ztpath=(1-t)*z0+t*zt;path=(ztpath*data['scale'].astype(np.float32)+data['mean'].astype(np.float32)).astype(np.float64)
        check(f'probe_{i}_{ti}_path',np.allclose(ps['path_states'][i,ti],path,rtol=2e-6,atol=2e-7))
        check(f'probe_{i}_{ti}_semantic',near_dict(semantic(ps['endpoint_predictions'][i,ti],assigned,diag),row['endpoint']) and near_dict(semantic(ps['path_states'][i,ti],assigned,diag),row['zero_velocity_endpoint']))
    ix=int(np.argmin(abs(times-.9)))
    for row in grows:
        i=row['probe_id'];geometry_check(row,root/'probes'/f'endpoint09_{i:03d}_{row["domain"]}',ps['endpoint_predictions'][i,ix],ps['target_permutations'][i])
    probevalid={d:sum(valid(r) for r in grows if r['domain']==d) for d in ['fixed','enclosing']};check('probe_summary_counts',probevalid==psum['valid_at09_by_domain'])
    decision=read(root/'STEP_TEST_DECISION.json');trigger=[d for d in ['fixed','enclosing'] if probevalid[d]>=24 and mix_summaries[64][d]['PPP']['valid']<=8]
    check('preregistered_step_test_decision',decision['trigger_domains']==trigger and decision['triggered']==bool(trigger))
    if trigger:audit_rollout(256)
    else:check('no_untriggered_step_test',not (root/'sampling_256').exists() and not (root/'interventions_256').exists())
    ooo=[r for r in geometry_records if '_OOO_' in r['location']]
    result=dict(passed=all(c['passed'] for c in checks),check_count=len(checks),failed_checks=[c for c in checks if not c['passed']],checks=checks,training=dict(completed_steps=tr['completed_steps'],seconds=tr['training_elapsed_seconds'],including_save_seconds=tr['total_wall_seconds_including_save'],exposure=tr['exposure_counts'],batch_replay=batch_replay),mix_by_nfe=mix_summaries,probe09_valid=probevalid,step_test_decision=decision,target_oracles=oracles,OOO_identity_by_domain={d:dict(count=sum(r['domain']==d for r in ooo),valid=sum(r['valid'] for r in ooo if r['domain']==d),same_canonical_target_faces=sum(r.get('same_original_target_id_faces',False) for r in ooo if r['domain']==d)) for d in ['fixed','enclosing']},geometry_records=geometry_records,sampling={str(k):dict(seconds=v['rollout']['seconds'],scope='Free integration plus endpoint Hungarian matching and snapshot semantic diagnostics, not pure ODE wall time.') for k,v in samples.items()},execution=execution,source_manifest_sha256=sha(root/'MANIFEST.json'),audit_seconds=time.perf_counter()-started,scope='Independent saved-state/chart/decode/domain/assignment/topology checks; exact intersection findings are hash-bound existing audit artifacts, not re-executed. No training or sampling rerun.',limitations=['Only PPP is generation; oracle substitutions and t=.9 probe are target-assisted.','One target and one training seed; OOO repeats one identity, not independent objects.','Snapshots compared with initial assigned endpoint are not true off-path ODE velocity errors.','Both domains reported separately; no union or best-domain success metric.'])
    out.mkdir(parents=True);write(out/'SINGLE_STATE_AUDIT.json',result);shutil.copy2(Path(__file__),out/'audit_single_state.py');dependency=Path(__file__).with_name('audit_fm_outputs.py');shutil.copy2(dependency,out/dependency.name)
    write(out/'SUPPLEMENTAL_MANIFEST.json',dict(original_manifest_sha256=sha(root/'MANIFEST.json'),files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
    print(json.dumps({k:result[k] for k in ['passed','check_count','failed_checks','training','mix_by_nfe','probe09_valid','step_test_decision','OOO_identity_by_domain','sampling','audit_seconds']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);main(p.parse_args().root)
