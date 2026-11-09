"""File/array/ledger audit only: no torch, model, integration or scoring imports."""
import argparse,difflib,hashlib,json,math,os,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'native_t1/runs/jedge5_solver_pilot_20261002_121000'
METHODS={'E_quad50':(50,50),'Midpoint25':(25,50),'Heun25':(25,50),'Heun50':(50,100)}
NULL_KEYS={'exact_C_free_shared_edge_count','new_free_related_nonmanifold_vertex_count','exact_C_free_shared_vertex_count','C_inherited_nonmanifold_vertex_count','C_current_winding_conflict_count','free_related_current_winding_conflict_count','shared_edge_component_count','exact_index_boundary_edge_count','exact_index_overincident_edge_count'}

def pathof(value,base=ROOT):
    s=str(value).replace('\\','/')
    if os.name!='nt' and len(s)>2 and s[1:3]==':/':s='/mnt/'+s[0].lower()+'/'+s[3:]
    if os.name=='nt' and s.startswith('/mnt/'):s=s[5].upper()+':/'+s[7:]
    p=Path(s);return (p if p.is_absolute() else base/p).resolve()

def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)

def ah(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.tobytes());return h.hexdigest()

def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def sealed_out(value):
    out=pathof(value)
    if out.parent!=ROOT/'native_t1/runs' or not out.name.startswith('jedge5_solver_pilot_'):raise ValueError('Isolated solver run required')
    return out

def refresh_manifest(out):
    out=sealed_out(out)
    if not (out/'final_audit.json').exists():raise ValueError('Final audit required before artifact manifest')
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='manifest.json' and p.suffix not in ('.tmp','.lock'):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            files.append(dict(path=p.relative_to(out).as_posix(),sha256=h.hexdigest(),bytes=p.stat().st_size))
    record=dict(status='PASS',timestamp=time.time(),files=files,excluded_self='manifest.json',pending_root_deliverables=[n for n in ('SOLVER_PILOT_REPORT.md','results.json') if not (out/n).exists()],scientific_calls=0)
    save(out/'manifest.json',record);return record

def run(value):
    out=sealed_out(value);tick=time.perf_counter();files={};checks=0;npz_reads=0
    for n in ('final_audit.json','case_identity_manifest.json','code.diff'):
        if (out/n).exists():raise FileExistsError(out/n)
    def check(ok,label):
        nonlocal checks
        checks+=1
        if not bool(ok):raise AssertionError(label)
    def sha(value):
        p=pathof(value)
        if str(p) not in files:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            files[str(p)]=dict(sha256=h.hexdigest(),bytes=p.stat().st_size)
        return files[str(p)]['sha256']
    def read(p):sha(p);return json.loads(pathof(p).read_text(encoding='utf-8-sig'))
    def arrays(p):
        nonlocal npz_reads
        npz_reads+=1;sha(p)
        with np.load(pathof(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    # Fail before creating audit artifacts if the final matrix is not ready.
    workers=[read(out/f'raw/case{i}/results.json') for i in range(4)]
    scored=[read(out/f'scores/case{i}/results.json') for i in range(4)]
    evaluation=read(out/'evaluation/results.json');plot=read(out/'evaluation/plot_manifest.json')
    check(all(r['status']=='PASS' for r in workers+scored+[evaluation,plot]),'All scientific writers completed PASS')
    report=dict(status='RUNNING',started_timestamp=time.time())
    try:
        protocol=read(out/'registration/protocol.json');fixture=read(out/'registration/sampler_cpu_tests.json');scorer=read(out/'registration/scoring_identity.json')
        check(protocol['status']=='FROZEN' and protocol['expected_actual_model_forwards']==1000 and protocol['max_new_full_trajectories']==16,'Registered budget')
        check([(c['case_id'],c['parent'],c['patch'],c['seed']) for c in protocol['cases']]==[(0,0,0,10401),(1,0,0,10402),(2,0,4,10401),(3,0,4,10402)],'Registered cases')
        check(sha(protocol['checkpoint'])==protocol['checkpoint_file_sha256'],'J checkpoint file unchanged; no tensor reload')
        for p,h in protocol['protected'].items():check(sha(ROOT/p)==h,'Public source protected '+p)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();check(head==protocol['source_HEAD'],'HEAD unchanged')
        check(fixture['status']=='PASS' and fixture['tests_run']==6 and fixture['CUDA_initialized'] is False,'CPU fixtures passed')
        fc=fixture['counts'];check(fc['CPU_mock_forward_attempts']==752 and fc['CPU_mock_forward_returns']==751 and fc['CPU_mock_complete_rollouts']==12 and fc['expected_rejections']==4,'Mock attempts/returns including intentional failure')
        check(all(fc[k]==0 for k in ('real_model_forwards','model_backwards','optimizer_updates','OT_calls','geometry_autograd_calls')),'Fixtures added zero real model calls')
        for p,h in fixture['sources'].items():check(sha(p)==h,'Fixture/sampler source unchanged')
        totals={};lockrows=[];identities=[];references={};newrows={}
        expected=dict(model_forward_attempts=250,model_forward_returns=250,accepted_steps=150,full_rollout_attempts=4,complete_rollouts=4,sampling_failures=0,model_backward_calls=0,optimizer_updates=0,geometry_gradient_calls=0,OT_calls=0,postprocessing_repairs=0)
        for cid,w in enumerate(workers):
            case=protocol['cases'][cid];check(w['case_id']==cid and len(w['rows'])==4 and {r['method'] for r in w['rows']}==set(METHODS),'Four methods per case')
            check(w['counts']==expected,'Exact per-worker budget')
            for k,v in w['counts'].items():totals[k]=totals.get(k,0)+v
            ident=w['identity'];check(ident['strict'] is True and ident['missing_keys']==ident['unexpected_keys']==[] and ident['state_sha256']==w['final_model_state_sha256']==protocol['model_state_sha256'],'Strict unchanged model state')
            check(ident['checkpoint_sha256']==protocol['checkpoint_file_sha256'] and ident['cumulative_updates']==3000 and ident['trainable_parameters']==0 and ident['training_objective']['name']=='edge5','Frozen J3000 identity')
            check(w['lock_path']==protocol['gpu_dispatch']['lock'] and w['lock_inode']>0 and w['lock_owner_pid']>0 and w['CUDA_initialized_before_locked_runtime'] is False and w['lock_held_until_process_exit'] is True,'Lock-before-CUDA runtime facts')
            lockrows.append(dict(case_id=cid,inode=w['lock_inode'],pid=w['lock_owner_pid'],acquired=w['lock_acquired_timestamp'],last_snapshot=max(s['timestamp'] for s in w['resource_snapshots']),wait_seconds=w['lock_wait_seconds']))
            check(sha(case['first_velocity_source'])==case['first_velocity_source_sha256'],'Blackbox reference file unchanged');ref=arrays(case['first_velocity_source']);references[cid]=ref
            for row in w['rows']:
                method=row['method'];steps,nfe=METHODS[method];p=pathof(row['path'],out);check(out in p.parents and sha(p)==row['sha256'],'New raw file identity');z=arrays(p);a=row['audit']
                check(row['case_id']==cid and row['parent']==0 and row['patch']==case['patch'] and row['seed']==case['seed'] and row['model']=='J' and row['precision']=='bf16' and row['nfe']==nfe,'Row registered identity')
                check(row['model_state_sha256']==protocol['model_state_sha256'] and row['model_file_sha256']==protocol['checkpoint_file_sha256'],'Row model hashes')
                for k,shape in dict(raw=(112,9),states=(steps+1,112,9),C=(12,3,3),stage_states=(nfe,112,9),stage_velocities=(nfe,112,9)).items():check(z[k].shape==shape and z[k].dtype==np.float32 and np.isfinite(z[k]).all(),'Finite FP32 '+k)
                C=z['C'].reshape(12,9);check(bits(z['raw'],z['states'][-1]) and bits(z['C'],ref['C'].reshape(12,3,3)) and bits(z['states'][0],ref['states'][0].reshape(112,9)),'Endpoint/C/initial identity')
                for k in ('states','stage_states'):check(bits(z[k][:,:12].copy(),np.broadcast_to(C,z[k][:,:12].shape).copy()),'C bitwise in every accepted/stage state')
                check(bits(z['stage_velocities'][0],ref['velocities'][0].reshape(112,9)) and ah(z['stage_velocities'][0])==case['first_velocity_sha256']==a['first_stage_velocity_sha256'] and row['first_velocity_matches_archive'] is True,'Independent old t0 velocity bitwise')
                check(ah(z['C'])==case['C_sha256']==row['C_sha256'] and ah(z['states'][0])==case['initial_state_sha256']==row['initial_state_sha256'] and a['noise_sha256']==row['noise_sha256']==case['noise_sha256'],'Array and recorded noise hashes')
                check(a['model_forward_attempts']==a['model_forward_returns']==nfe and a['known_accepted_all_bitwise'] and a['known_stages_all_bitwise'] and not a['known_velocity_applied'] and a['inputs_unchanged'] and not a['cpu_mock'],'Recorded sampler gate')
                rt=a['runtime'];check(rt['deterministic_algorithms'] and not rt['deterministic_warn_only'] and rt['cublas_workspace_config']==':4096:8' and not rt['tf32_matmul'] and not rt['tf32_cudnn'] and rt['sdpa_math_enabled'] and rt['cuda_autocast_enabled'] and rt['cuda_autocast_dtype']=='torch.bfloat16' and rt['parameter_dtype']=='torch.float32' and not rt['gradient_enabled'] and a['integration_dtype']=='torch.float32' and a['actual_velocity_dtypes']==['torch.bfloat16'],'Actual stable BF16/FP32 runtime')
                ideal=[(k/steps)**2 if method=='E_quad50' else k/steps for k in range(steps+1)];times=[];kinds=[]
                for k in range(steps):
                    times.append(ideal[k]);kinds.append('start')
                    if method!='E_quad50':times.append((ideal[k]+ideal[k+1])*.5 if method=='Midpoint25' else ideal[k+1]);kinds.append('midpoint' if method=='Midpoint25' else 'Heun_predictor')
                check(bits(z['accepted_times_FP32'],np.asarray(ideal,np.float32)) and bits(z['stage_times_FP32'],np.asarray(times,np.float32)) and bits(z['step_sizes_FP32'],np.asarray(np.diff(ideal),np.float32)) and z['stage_kind'].tolist()==kinds,'Actual registered grid/stages')
                check(a['actual_endpoint_t1_forward_calls']==int(method.startswith('Heun')),'Heun actual final t=1 call')
                check(bits(z['stage_states'][z['stage_kind']=='start'],z['states'][:-1]),'First stage equals accepted starting state')
                identities.append(dict(case_id=cid,method=method,nfe=nfe,path=row['path'],sha256=row['sha256'],C_sha256=row['C_sha256'],noise_sha256=row['noise_sha256'],initial_state_sha256=row['initial_state_sha256'],first_velocity_sha256=case['first_velocity_sha256'],accepted_states=steps+1,stage_states=nfe,C_all_bitwise=True));newrows[cid,method]=row
        lockrows.sort(key=lambda r:r['acquired']);check(len({r['inode'] for r in lockrows})==1 and len({r['pid'] for r in lockrows})==4,'Four PIDs shared one lock inode')
        check(all(b['acquired']>=a['last_snapshot'] for a,b in zip(lockrows,lockrows[1:])),'Observed lock intervals nonoverlapping')
        check(len(list((out/'raw').rglob('*.npz')))==16,'No extra new replay/partial NPZ')
        reference=ROOT/'native_t1/runs/edge_factorial_20261001_123000/source_snapshot/experiments/mf_h1_local0/edge_factorial_pilot/evaluation.py'
        check(scorer['status']=='PASS' and sha(reference)==scorer['scorer_sha256'] and sha(ROOT/'experiments/mf_h1_local0/jedge5_diagnosis/scoring.py')==scorer['original_scorer_module_sha256'],'Original scorer sources unchanged')
        check(scorer['new_CC_audits']==0 and sha(scorer['CC_source'])==sha(out/'registration/static_context_audit.json')==scorer['CC_sha256'],'Exact reused CC cache')
        check(len(evaluation['rows'])==24 and evaluation['new_scored_outputs']==16 and evaluation['reused_baselines']==8 and evaluation['model_forwards']==evaluation['optimizer_updates']==0,'24-row CPU evaluation')
        check(evaluation['scoring_version_sha256']==scorer['scorer_sha256'] and evaluation['evaluation_inputs_sha256']==scorer['evaluation_inputs_sha256'],'Scorer/input version identity')
        by={};scored_by={};ledger={};oldroot=pathof(protocol['baseline_archive']);old=read(oldroot/'scores/A_J/results.json');reused=[]
        for cid,s in enumerate(scored):
            check(s['completed']==4 and len(s['rows'])==4 and s['source_sha256']==sha(out/f'raw/case{cid}/results.json'),'Scoring completed and bound to raw record')
            check(all(s[k]==0 for k in ('model_forwards','optimizer_updates','OT_calls','repairs')),'No model/OT/repair in CPU scoring')
            for k,v in s['ledger'].items():ledger[k]=ledger.get(k,0)+v
            for row in s['rows']:scored_by[cid,row['method']]=row
        keys=set(evaluation['rows'][0]['scalars']);check(len(keys)==54,'54 scalar fields')
        for row in evaluation['rows']:
            cid,method=row['case_id'],row['method'];check((cid,method) not in by,'Unique evaluation identity');by[cid,method]=row
            check(set(row['scalars'])==keys and row['topology']=='NOT_RUN','Consistent metric schema and explicit topology NOT_RUN')
            for k,v in row['scalars'].items():
                if k in NULL_KEYS:check(v is None or v=='NOT_RUN','Unrun topology field '+k)
                else:check(isinstance(v,(int,float)) and math.isfinite(v) and v>=-1e-12 and (not k.endswith('_fraction') or v<=1+1e-12),'Finite/range scalar '+k)
            check(sha(pathof(row['path'],out))==row['sha256'] and sha(pathof(row['metric_path'],out))==row['metric_sha256'],'Evaluation raw/detail SHA')
            if method in METHODS:check(row==scored_by[cid,method] and all(row[k]==v for k,v in newrows[cid,method].items()),'New scoring exactly copies runtime identity')
            else:
                nfe=50 if method=='Euler50' else 100;check(method in ('Euler50','Euler100') and row['reused'] and row['nfe']==nfe,'Only Euler50/100 baseline reused')
                matches=[x for x in old['rows'] if x['parent']==0 and x['patch']==row['patch'] and x['seed']==row['seed'] and x['nfe']==nfe and x['precision']=='bf16'];check(len(matches)==1 and matches[0]['scalars']==row['scalars'] and matches[0]['sha256']==row['sha256'],'Old scalars/raw reused unchanged')
                z=arrays(pathof(row['path'],out));states=z['states'].reshape(-1,112,9);indices=z['state_indices'];check(len(indices)==len(states) and int(indices[0])==0 and int(indices[-1])==nfe and np.all(np.diff(indices)>0),'Archived baseline saved-state indices; sparse archive allowed');check(bits(states[0],references[cid]['states'][0].reshape(112,9)) and bits(states[-1],z['raw'].reshape(112,9)) and np.isfinite(states).all(),'Reused baseline initial/final state')
                check(bits(states[:,:12].copy(),np.broadcast_to(z['C'].reshape(12,9),states[:,:12].shape).copy()),'Reused baseline C bits');reused.append(dict(case_id=cid,method=method,path=row['path'],sha256=row['sha256']))
        check(set(by)=={(i,m) for i in range(4) for m in (*METHODS,'Euler50','Euler100')} and ledger==evaluation['ledger'],'Complete fixed evaluation matrix and summed ledger')
        comp=evaluation['comparisons'];check(len(comp)==16 and len({(x['case_id'],x['candidate']) for x in comp})==16,'16 unique paired comparisons')
        for c in comp:
            baseline='Euler100' if c['candidate']=='Heun50' else 'Euler50';a,b=by[c['case_id'],c['candidate']],by[c['case_id'],baseline]
            check(c['baseline']==baseline and c['nfe']==a['nfe']==b['nfe'],'Equal NFE including Heun50=100')
            check(c['differences']=={k:a['scalars'][k]-b['scalars'][k] if k not in NULL_KEYS else None for k in keys},'Per-case signed differences exact')
        for group,ids in (('normal_patch0',[0,1]),('sensitive_patch4',[2,3]),('all_selected',[0,1,2,3])):
            for method in METHODS:
                rr=[r for r in comp if r['case_id'] in ids and r['candidate']==method];want={}
                for k in keys-NULL_KEYS:
                    v=[r['differences'][k] for r in rr];want[k]=dict(mean_delta=float(np.mean(v)),median_paired_delta=float(np.median(v)),negative_count=sum(x<0 for x in v),positive_count=sum(x>0 for x in v),count=len(v))
                check(evaluation['macro_differences'][group][method]==want,'Paired macro mean/median/count exact')
        check(plot['extra_model_forwards']==0 and plot['directional_surface_query_reconstructions']==24 and plot['points_each_direction']==512 and len(plot['rows'])==24,'Plot-only 24 CPU query reconstructions separately recorded')
        save(out/'case_identity_manifest.json',dict(status='PASS',registered=protocol['cases'],fresh=identities,reused=reused,checkpoint_file_sha256=protocol['checkpoint_file_sha256'],model_state_sha256=protocol['model_state_sha256']))
        chunks=[]
        for p in sorted(Path(__file__).parent.glob('*.py')):
            s=''.join(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+p.relative_to(ROOT).as_posix()));chunks.append(s if s.endswith('\n') else s+'\n')
        (out/'code.diff').write_text(''.join(chunks),encoding='utf-8',newline='')
        report.update(status='PASS',HEAD=head,actual_sampling_counts=dict(totals,replays=0),CPU_fixture_counts=fc,CPU_fixture_reruns=0,GPU_dispatch=dict(actual_processes=4,simultaneous_workers=1,lock_intervals=lockrows,note='Recorded intervals and same inode support serial GPU execution; exact process exit timestamps were not recorded. This audit never opens the lock.'),evaluation=dict(new_scores=16,reused_scores=8,scalar_fields=54,NOT_RUN_fields=sorted(NULL_KEYS),scoring_ledger=ledger,plot_query_reconstructions=24,equal_NFE_pairs=16),state_hash_evidence='Worker strict-load and before/after hashes; audit hashes checkpoint file without torch.load.',code_diff_sha256=sha(out/'code.diff'))
    except BaseException as exc:report.update(status='FAIL',error=repr(exc));raise
    finally:
        report.update(checks=checks,seconds=time.perf_counter()-tick,verified_files=files,audit_counts=dict(model_constructions=0,model_forwards=0,backwards=0,optimizer_updates=0,OT_calls=0,geometry_scoring_calls=0,autograd_calls=0,NPZ_reads=npz_reads,unique_hashed_files=len(files),hashed_bytes=sum(v['bytes'] for v in files.values())));save(out/'final_audit.json',report)
    refresh_manifest(out);return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--refresh-manifest',action='store_true');a=p.parse_args();r=refresh_manifest(a.out) if a.refresh_manifest else run(a.out);print(json.dumps({k:r[k] for k in ('status','checks','seconds') if k in r}),flush=True)
if __name__=='__main__':main()
