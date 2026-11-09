"""Independent saved-artifact audit for SINGLE_STATE_METRIC0; no extraction."""
import argparse,collections,hashlib,itertools,json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from audit_interaction import read,sha,arrays,write,equal,deep_equal,noise_hash,summary,contains,extra_semantic
from audit_fm_outputs import independent_topology,valid,failure_tags
from audit_single_state import decode,assign,semantic

TIMES=[0.,.25,.5,.75,.9,.99]
FIELDS=['x','n','F'];OLD=np.array([3.,3.,16.])/22
SHARED=['encoder.0.weight','encoder.0.bias','encoder.2.weight','encoder.2.bias','decoder.0.weight','decoder.0.bias','decoder.2.weight','decoder.2.bias']

def independent_grams(model,noise,times,target,mean,scale,ids=(0,8,16,24)):
    """CPU float32 model/autograd; float64 Gram accumulation; no updates."""
    params=dict(model.named_parameters());selected=[params[n] for n in SHARED]
    records=[]
    for i in ids:
        z0=noise[i].astype(np.float32)
        perm=assign(z0[:,:3].astype(np.float64)*scale[:3]+mean[:3],target[:,:3])
        zt=((target[perm]-mean)/scale).astype(np.float32)
        for t in times:
            z=(1-float(t))*torch.from_numpy(z0[None])+float(t)*torch.from_numpy(zt[None])
            truth=torch.from_numpy((zt-z0)[None]);prediction=model(z,torch.tensor([float(t)],dtype=torch.float32))
            squared=(prediction-truth).square();losses=[squared[...,a:b].mean() for a,b in [(0,3),(3,6),(6,22)]]
            grads=[torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,selected,retain_graph=j<2)]).double() for j,loss in enumerate(losses)]
            G=torch.stack(grads);gram=(G@G.T).detach().numpy()
            records.append(dict(probe_id=int(i),time=float(t),gram=gram.tolist(),losses=[float(x.detach()) for x in losses]))
    return records

def manifest_check(root,check,name='original_manifest'):
    manifest=read(root/'MANIFEST.json');bad=[]
    for r in manifest['files']:
        p=root/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:bad.append(r['path'])
    check(name,not bad,dict(count=len(manifest['files']),bad=bad))

def paired_results(ledger):
    a={r['sample_id']:r for r in ledger['B0']};b={r['sample_id']:r for r in ledger['M0']}
    pairs=[]
    for i in range(32):
        x,y=a[i],b[i];rx=x.get('rms_pct_bboxdiag');ry=y.get('rms_pct_bboxdiag')
        pairs.append(dict(sample_id=i,B0_valid=valid(x),M0_valid=valid(y),B0_rms_pct_bboxdiag=rx,M0_rms_pct_bboxdiag=ry,delta_M0_minus_B0=None if rx is None or ry is None else ry-rx))
    delta=[r['delta_M0_minus_B0'] for r in pairs if r['delta_M0_minus_B0'] is not None]
    levels=sorted(set([0.,.20,.5,1.,2.,3.,4.,5.]+[r['rms_pct_bboxdiag'] for rr in ledger.values() for r in rr if r.get('rms_pct_bboxdiag') is not None]))
    curves=[dict(threshold_pct_bboxdiag=t,**{arm:sum(valid(r) and r.get('rms_pct_bboxdiag',float('inf'))<=t for r in rr) for arm,rr in ledger.items()}) for t in levels]
    return dict(denominator=32,pairs=pairs,delta_summary=summary(delta),M0_lower=sum(x<0 for x in delta),M0_higher=sum(x>0 for x in delta),equal=sum(x==0 for x in delta),valid_patterns=dict(collections.Counter(str(int(r['B0_valid']))+str(int(r['M0_valid'])) for r in pairs)),descriptive_threshold_curves=curves,warning='Threshold curves are descriptive; the registered 0.20% and >=24/32 gate is unchanged.')


def main(root):
    started=time.perf_counter();torch.set_num_threads(1);root=Path(root).resolve()
    out=root/'post_analysis/independent_audit_v2'
    if out.exists():raise RuntimeError('Refuse existing independent audit')
    checks=[]
    def check(name,ok,detail=None):checks.append(dict(name=name,passed=bool(ok),detail=detail))
    protocol=read(root/'RUN_PROTOCOL.json');baseline=Path(protocol['previous_run']);parent_run=Path(protocol['parent_run'])
    execution=read(root/'EXECUTION.json')
    check('execution_finished',execution['status'] in ['COMPLETE','CALIBRATION_REJECTED'])
    manifest_check(root,check)
    check('run_protocol_hash',sha(root/'RUN_PROTOCOL.json')==read(root/'RUN_PROTOCOL_SHA256.json')['sha256'])
    check('source_snapshot',all(sha(root/r['path'])==r['sha256'] for r in read(root/'SOURCE_SNAPSHOT.json')['files']))
    check('frozen_hashes',all(sha(root/k)==v for k,v in protocol['hashes'].items()))
    check('parent_sources',all(sha(r['path'])==r['sha256'] for r in protocol['sources']))
    for name in ['final.pt','initial.pt','TRAIN_TRACE.jsonl','PROTOCOL.json','RESULT.json']:
        check('historical_control_'+name,sha(root/'control'/name)==sha(baseline/'training/B0'/name))
    check('parent_checkpoint_exact',sha(root/'parent/final.pt')==sha(parent_run/'training/direct/final.pt'))
    for name in ['fm_model.py','fm_state.py','interaction_model.py','train_fm_pilot.py','train_interaction.py','single_state_common.py','interaction_semantics.py','sample_single_state.py','common.py','safe_extract.py','safe_extract_domain.py','exact_audit.py']:
        check('unchanged_core_'+name,sha(root/'code'/name)==sha(baseline/'code'/name))
    check('data_chart_and_probes_exact',sha(root/'data.npz')==sha(baseline/'data.npz') and sha(root/'probe_noise.npz')==sha(baseline/'probe_noise.npz'))
    preflight=read(root/'preflight/RESULT.json');check('preflight_passed',preflight['passed'] and all(x['passed'] for x in preflight['checks']))
    replay=arrays(root/'preflight/B0_ARCHIVED_REPLAY.npz')
    check('saved_archived_rollout_replay',np.isfinite(replay['actual']).all() and np.max(abs(replay['actual']-replay['reference']))<=5e-6 and equal(replay['reference'],arrays(root/'control/archived_endpoints.npz')['states'][0]))
    data=arrays(root/'data.npz');target=data['states'][0];diag=float(np.linalg.norm(np.ptp(target[:,:3],axis=0)))
    noise=arrays(root/'sample_noise.npz')['noise'];probe=arrays(root/'probe_noise.npz')
    check('fresh32_seed260931',noise.shape==(32,256,22) and equal(noise,np.random.default_rng(260931).standard_normal((32,256,22)).astype(np.float32)))
    forbidden={noise_hash(x) for x in arrays(baseline/'sample_noise.npz')['noise']}|{noise_hash(x) for x in probe['noise']}
    check('fresh_noise_disjoint_previous_and_probe',not ({noise_hash(x) for x in noise}&forbidden))
    parent=torch.load(root/'parent/final.pt',map_location='cpu',weights_only=False)
    sys.path.insert(0,str(root/'code'))
    for key in ['fm_model','interaction_model']:sys.modules.pop(key,None)
    from fm_model import DeepSetsVelocity
    model=DeepSetsVelocity(128);model.load_state_dict(parent['model_state_dict']);model.eval()
    check('shared_eight_parameters85376',sum(dict(model.named_parameters())[k].numel() for k in SHARED)==85376 and sum(p.numel() for p in model.parameters())==88214)
    wd=read(root/'calibration/WEIGHTS.json')
    if not wd['passed']:
        decision=read(root/'DECISION.json')
        check('rejected_calibration_no_updates_or_outputs',execution['status']=='CALIBRATION_REJECTED' and decision['status']=='STOP_CALIBRATION' and decision['train_updates']==0 and decision['free_outputs']==0 and not (root/'training/M0').exists() and not list((root/'evaluation').glob('*/FREE_LEDGER.json')))
        result=dict(passed=all(x['passed'] for x in checks),checks=checks,check_count=len(checks),failed_checks=[x for x in checks if not x['passed']],calibration=wd,engineering_decision='STOP_CALIBRATION',scope='Manifest/provenance/guarded refusal audit; no training or free outputs were requested after refusal.')
        out.mkdir(parents=True);write(out/'METRIC_AUDIT.json',result)
        shutil.copy2(Path(__file__),out/'audit_metric.py')
        write(out/'SUPPLEMENTAL_MANIFEST.json',dict(original_manifest_sha256=sha(root/'MANIFEST.json'),files=[dict(path=str(x.relative_to(root)),bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(out.iterdir()) if x.is_file()]))
        print(json.dumps(result,indent=2));return
    w=np.array([wd['weights'][k] for k in FIELDS])
    check('calibration_pass',wd['passed'] and wd['status']=='PASS' and wd['checkpoint_sha256']==sha(root/'parent/final.pt') and wd['data_sha256']==sha(root/'data.npz') and wd['probe_noise_sha256']==sha(root/'probe_noise.npz'))
    check('calibration_frozen_guards',wd['epsilon']==1e-12 and wd['min_gradient_rms_exclusive']==1e-8 and wd['max_weight_ratio']==1e6 and wd['common_scale_limits']==[1e-6,1e6] and wd['clipping_allowed'] is False)
    check('shared_names_exact',wd['shared_parameter_names']==SHARED and wd['shared_parameter_count']==85376)
    calrows=read(root/'calibration/GRADIENT_ROWS.json');calnp=arrays(root/'calibration/GRAMS.npz')
    check('probe_bank_time_storage',equal(probe['times'],np.asarray(TIMES,dtype=probe['times'].dtype)))
    check('calibration24_slots',len(calrows)==24 and {(r['probe_id'],r['time']) for r in calrows}==set(itertools.product([0,8,16,24],TIMES)))
    saved_gram=np.array([r['gram'] for r in calrows]);G=saved_gram.mean(0);g=np.sqrt(np.diag(G));rel=1/(g+1e-12);c=float(np.sqrt((OLD@G@OLD)/(rel@G@rel)));want=c*rel
    check('saved_Gram_arrays',equal(saved_gram,calnp['gram']) and np.allclose(G,wd['mean_gram'],rtol=1e-13,atol=1e-15))
    check('positive_frozen_weight_equation',np.allclose(w,want,rtol=1e-13,atol=1e-15) and np.all(w>0) and np.all(g>1e-8) and rel.max()/rel.min()<=1e6 and 1e-6<=c<=1e6)
    check('calibration_norm_match',np.isclose(w@G@w,OLD@G@OLD,rtol=1e-12,atol=1e-15))
    cpu=independent_grams(model,probe['noise'],TIMES,target,data['mean'],data['scale'])
    cpu_G=np.array([r['gram'] for r in cpu]);relative_norm=np.linalg.norm(cpu_G-saved_gram)/np.linalg.norm(saved_gram)
    per_probe_relative=np.linalg.norm((cpu_G-saved_gram).reshape(24,-1),axis=1)/np.linalg.norm(saved_gram.reshape(24,-1),axis=1)
    check('independent_CPU24_gradient_Grams',relative_norm<=2e-4 and float(per_probe_relative.max())<=2e-4 and all((a['probe_id'],a['time'])==(b['probe_id'],b['time']) for a,b in zip(cpu,calrows)),dict(relative_frobenius=relative_norm,max_per_probe_relative=float(per_probe_relative.max()),tolerance=2e-4,CPU_G_mean=cpu_G.mean(0).tolist()))
    for i,row in enumerate(calrows):
        gram=np.array(row['gram'])
        for kind,a in [('old_weighted',OLD),('new_weighted',w)]:
            effect=row[kind];want_norm=np.sqrt(a@gram@a)
            check(f'cal_{i}_{kind}',row['status']=='COMPLETE' and effect['identity_passed'] and np.isclose(effect['gram_predicted_gradient_norm'],want_norm,rtol=1e-12,atol=1e-14) and effect['gradient_sum_relative_error']<=1e-5)
    check('calibration_no_updates',wd['optimizer_updates']==0 and wd['mesh_extractions']==0 and wd['parameter_grad_buffers_remain_none'] and wd['model_tensor_digest_before']==wd['model_tensor_digest_after'])
    # Full original state restoration and all6000 coupled batches.
    d=root/'training/M0';r=read(d/'RESULT.json');tp=read(d/'PROTOCOL.json')
    initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);final=torch.load(d/'final.pt',map_location='cpu',weights_only=False)
    trace=[json.loads(s) for s in (d/'TRAIN_TRACE.jsonl').read_text().splitlines()]
    oldtrace=[json.loads(s) for s in (root/'control/TRAIN_TRACE.jsonl').read_text().splitlines()]
    check('updates3001_through9000',r['status']=='COMPLETED' and r['new_updates']==6000 and r['verified_batch_count']==6000 and [x['step'] for x in trace]==list(range(3001,9001)))
    check('6000_batches_paired_to_B0',len(trace)==len(oldtrace)==6000 and [x['batch_sha256'] for x in trace]==[x['batch_sha256'] for x in oldtrace] and all(x['batch_match'] and x['batch_sha256']==x['expected_batch_sha256'] for x in trace))
    check('initial_model_and_rng_exact',deep_equal(initial['model_state_dict'],parent['model_state_dict']) and deep_equal(initial['rng'],parent['rng']))
    check('initial_optimizer_exact',deep_equal(initial['optimizer_state_dict'],parent['optimizer_state_dict']))
    check('initial_model_file',deep_equal(initial['model_state_dict'],torch.load(d/'initial_model.pt',map_location='cpu',weights_only=True)))
    check('parameter_set_exact',initial['optimizer_param_names']==list(parent['model_state_dict'])==final['optimizer_param_names'])
    check('final_optimizer_steps9000',all(float(s['step'])==9000 for s in final['optimizer_state_dict']['state'].values()))
    check('final_weights_finite',all(bool(torch.isfinite(v).all()) for v in final['model_state_dict'].values()))
    check('exposure_and_progress',initial['completed_steps']==3000 and initial['exposure_counts']==[24000] and final['completed_steps']==9000 and final['exposure_counts']==[72000] and final['new_exposure_counts']==[48000])
    check('training_weights_hash_and_constants',tp['weights']==wd['weights']==initial['weights']==final['weights']==r['weights'] and all(x['weights_sha256']==sha(root/'calibration/WEIGHTS.json') for x in [tp,initial,final,r]))
    check('training_checkpoint_hash',sha(d/'final.pt')==r['final_checkpoint_sha256'] and final['protocol_sha256']==sha(d/'PROTOCOL.json'))
    check('checkpoint_chart_unchanged',all(deep_equal(x[k],parent[k]) for x in [initial,final] for k in ['mean','scale']))
    added=bytes(32);cumulative=bytes.fromhex(parent['attempted_stream_sha256'])
    for i,row in enumerate(trace):
        bh=bytes.fromhex(row['batch_sha256']);added=hashlib.sha256(added+bh).digest();cumulative=hashlib.sha256(cumulative+bh).digest()
        check(f'update_{3001+i}_fixed_loss',np.isclose(row['loss'],sum(w[j]*row[k+'_mse'] for j,k in enumerate(FIELDS)),rtol=3e-6,atol=3e-7) and np.isclose(row['old22_loss'],sum(OLD[j]*row[k+'_mse'] for j,k in enumerate(FIELDS)),rtol=3e-6,atol=3e-7))
    check('stream_hashes',cumulative.hex()==r['attempted_stream_sha256']==final['attempted_stream_sha256'] and added.hex()==r['new_attempted_stream_sha256']==final['new_attempted_stream_sha256'])
    for step in [5000,7000]:
        snap=torch.load(d/f'step_{step}.pt',map_location='cpu',weights_only=False)
        check(f'diagnostic_snapshot_{step}',snap['completed_steps']==step and snap['status']=='DIAGNOSTIC_MODEL_SNAPSHOT' and snap['resumable'] is False and 'optimizer_state_dict' not in snap and snap['weights']==wd['weights'] and snap['weights_sha256']==sha(root/'calibration/WEIGHTS.json'))
    check('two_snapshot_hashes',len(r['saved_model_snapshots'])==2 and all(sha(x['path'])==x['sha256'] for x in r['saved_model_snapshots']))
    generator=torch.Generator();generator.set_state(parent['rng']['batch_generator_cpu']);batch_replay=[];collisions=[]
    forbidden|={noise_hash(x) for x in noise}
    for i in range(6000):
        n=torch.randn(8,256,22,generator=generator);tt=torch.rand(8,1,1,generator=generator)
        if any(noise_hash(x) in forbidden for x in n.numpy()):collisions.append(i+3001)
        if i in [0,999,2999,5999]:
            sx=n.numpy()[...,:3]*data['scale'][:3]+data['mean'][:3]
            targets=np.stack([((target[assign(x,target[:,:3])]-data['mean'])/data['scale']).astype(np.float32) for x in sx])
            h=hashlib.sha256(n.numpy().tobytes()+tt.numpy().tobytes()+targets.tobytes()+np.zeros(8,np.int64).tobytes()).hexdigest()
            batch_replay.append(dict(step=i+3001,matched=h==trace[i]['batch_sha256'],sha256=h))
    check('independent_four_batch_replay',all(x['matched'] for x in batch_replay),batch_replay)
    check('no_sample_probe_noise_reuse_in_added48000',not collisions,collisions)
    check('final_batch_rng_replay',torch.equal(generator.get_state(),final['rng']['batch_generator_cpu']))
    check('schedule72000',final['rng']['balanced_schedule']['cycles']==72000 and final['rng']['balanced_schedule']['order']==[0] and final['rng']['balanced_schedule']['cursor']==1)
    # Fresh paired free outputs, independent decode/topology/semantic checks.
    signset=set(itertools.product([-1.,1.],repeat=3));ledger={};results={};geometry_records=[]
    for arm in ['B0','M0']:
        ev=root/'evaluation'/arm;ep=arrays(ev/'endpoints.npz');roll=read(ev/'ROLLOUT.json');free=read(ev/'FREE_LEDGER.json');saved=read(ev/'SUMMARY.json');ledger[arm]=free
        ck=sha(root/'control/final.pt') if arm=='B0' else sha(d/'final.pt')
        check(arm+'_32_slots',len(free)==32 and {x['sample_id'] for x in free}==set(range(32)) and ep['states'].shape==(32,256,22) and len(roll['rows'])==32)
        check(arm+'_sampling_binding',roll['noise_sha256']==sha(root/'sample_noise.npz') and roll['checkpoint_sha256']==ck and roll['training_weights_applied_to_velocity'] is False)
        for row in roll['rows']:
            i=row['sample_id'];pred=ep['states'][i];perm=ep['target_permutations'][i];label=f'{arm}_{i:03d}'
            check(label+'_fresh_rollout_metadata',row['split']=='fresh' and row['checkpoint_sha256']==ck)
            if row['status']=='COMPLETE':
                check(label+'_Hungarian',equal(perm,assign(pred[:,:3],target[:,:3])) and row['nfe']==64)
                check(label+'_semantic',contains(row['endpoint_semantic'],semantic(pred,target[perm],diag)) and contains(row['endpoint_semantic'],extra_semantic(pred,target[perm],diag)))
        for row in free:
            i=row['sample_id'];label=f'{arm}_{i:03d}';directory=ev/f'{i:03d}_PPP_fixed';pred=ep['states'][i];perm=ep['target_permutations'][i]
            check(label+'_free_only',row['combination']=='PPP' and row['generated'] and not row['oracle_assisted'] and row['nfe']==64 and row['domain']=='fixed' and row['split']=='fresh')
            check(label+'_metadata',row['checkpoint_sha256']==ck and row['noise_sha256']==sha(root/'sample_noise.npz') and equal(row['target_permutation'],perm))
            check(label+'_state_row',equal(arrays(directory/'state.npz')['physical_state'],pred) and read(directory/'result.json')==row)
            if (directory/'parameters.npz').is_file():
                params=arrays(directory/'parameters.npz');want=decode(pred)
                check(label+'_decode',all(equal(params[k],want[k]) for k in want) and sha(directory/'parameters.npz')==row['parameters_sha256'])
                corners=arrays(directory/'corners.npz')['corners'];check(label+'_domain',corners.shape==(8,3) and set(map(tuple,corners))==signset)
                if (directory/'extractor_trace.npz').is_file():
                    tr=arrays(directory/'extractor_trace.npz')
                    if 'delaunay_points' in tr:check(label+'_trace_sites',equal(tr['delaunay_points'][:256],params['v']) and equal(tr['delaunay_points'][256:],corners))
            check(label+'_valid',valid(row)==row['valid'])
            check(label+'_quality',row['quality_pass']==bool(valid(row) and row.get('rms_pct_bboxdiag',float('inf'))<=.20))
            if (directory/'mesh.npz').is_file():
                mesh=arrays(directory/'mesh.npz');top=independent_topology(mesh['vertices'],mesh['faces'])
                check(label+'_topology',top['faces']==row['faces'] and len(top['bad_vertex_links'])==row['strict_noncycle_vertex_count'] and top['nonmanifold_edges']==row['nonmanifold_edges'] and top['boundary_edges']==row['boundary_edges'] and (top['betti_if_closed_oriented'] is None or top['betti_if_closed_oriented']==[row['beta'+str(k)] for k in range(3)]))
                check(label+'_mesh_sha',sha(directory/'mesh.obj')==row['mesh_sha256'])
            geometry_records.append(dict(arm=arm,sample_id=i,valid=valid(row),quality_pass=row['quality_pass'],rms_pct_bboxdiag=row.get('rms_pct_bboxdiag'),faces=row.get('faces'),failure_tags=failure_tags(row)))
        results[arm]=dict(denominator=32,valid=sum(valid(x) for x in free),quality_pass=sum(x['quality_pass'] for x in free),all_output_rms=summary([x['rms_pct_bboxdiag'] for x in free if x.get('rms_pct_bboxdiag') is not None]),valid_output_rms=summary([x['rms_pct_bboxdiag'] for x in free if valid(x)]),faces=summary([x['faces'] for x in free if 'faces' in x]),invalid=sum(not valid(x) for x in free),valid_but_precision_failed=sum(valid(x) and not x['quality_pass'] for x in free),failure_tags=dict(collections.Counter(t for x in free for t in failure_tags(x))),integration_seconds=roll['integration_seconds'],posthoc_seconds=roll['posthoc_seconds'],geometry_seconds=saved['extraction_evaluation_seconds'])
        results[arm]['advance']=results[arm]['quality_pass']>=24
        check(arm+'_summary',all(results[arm][k]==saved[k] for k in ['valid','quality_pass','advance']))
        check(arm+'_time_sum',np.isclose(sum(x['integration_seconds'] for x in roll['rows']),roll['integration_seconds']) and np.isclose(sum(x['posthoc_seconds'] for x in roll['rows']),roll['posthoc_seconds']))
    check('64_free_rows',len(geometry_records)==64)
    paired=paired_results(ledger)
    decision=read(root/'DECISION.json')
    promote=results['M0']['advance'] and results['M0']['quality_pass']>results['B0']['quality_pass']
    check('registered_decision',decision['status']==('RETAIN_FOR_REPEAT_CONFIRMATION' if promote else 'PAUSE_CURRENT_JOINT_RECIPE') and not decision['automatic_additional_training'])
    # Fixed-checkpoint probe Gram identities, hash bindings and denominator.
    probes={}
    expected_probes={'M0_5000':root/'training/M0/step_5000.pt','M0_7000':root/'training/M0/step_7000.pt','M0_9000':root/'training/M0/final.pt','B0_9000':root/'control/final.pt'}
    check('four_fixed_probe_checkpoints',{x.parent.name for x in (root/'probes').glob('*/RESULT.json')}==set(expected_probes))
    for result_path in sorted((root/'probes').glob('*/RESULT.json')):
        directory=result_path.parent;pr=read(result_path);rows=read(directory/'GRADIENT_ROWS.json');ga=arrays(directory/'GRAMS.npz');key=directory.name
        pp=read(directory/'PROBE_PROTOCOL.json')
        check('probe_'+key+'_checkpoint_weight_binding',pr['checkpoint_sha256']==sha(expected_probes[key]) and pr['weights_sha256']==sha(root/'calibration/WEIGHTS.json') and pr['frozen_weights']==wd['weights'] and not pr['recalibrated'] and sha(directory/'PROBE_PROTOCOL.json')==read(directory/'PROBE_PROTOCOL_SHA256.json')['sha256'] and pp['weights']==wd['weights'])
        check('probe_'+key+'_24',pr['passed'] and len(rows)==24 and all(x['status']=='COMPLETE' for x in rows) and equal(ga['gram'],np.array([x['gram'] for x in rows])))
        check('probe_'+key+'_no_updates',pr['optimizer_updates']==0 and pr['mesh_extractions']==0 and pr['model_unchanged'] and pr['parameter_grad_buffers_remain_none'])
        check('probe_'+key+'_inputs',pr['data_sha256']==sha(root/'data.npz') and pr['probe_noise_sha256']==sha(root/'probe_noise.npz') and pr['shared_parameter_names']==SHARED)
        GG=ga['gram'].mean(0);gg=np.sqrt(np.diag(GG))
        check('probe_'+key+'_RMS',np.allclose([pr['summary']['unweighted_gradient_rms'][k] for k in FIELDS],gg,rtol=1e-12,atol=1e-14))
        for i,row in enumerate(rows):
            for kind,coeff in [('old_weighted',OLD),('new_weighted',w)]:
                effect=row[kind];check(f'probe_{key}_{i}_{kind}',effect['identity_passed'] and np.isclose(effect['gram_predicted_gradient_norm'],np.sqrt(coeff@np.array(row['gram'])@coeff),rtol=1e-12,atol=1e-14))
        probes[key]=dict(checkpoint_sha256=pr['checkpoint_sha256'],completed_steps=pr['checkpoint_completed_steps'],summary=pr['summary'])
    result=dict(passed=all(x['passed'] for x in checks),check_count=len(checks),failed_checks=[x for x in checks if not x['passed']],checks=checks,calibration=dict(weights=wd['weights'],g=g.tolist(),common_scale=c,CPU_Gram_relative_error=relative_norm,old_field_gradient_RMS=(OLD*g).tolist(),new_field_gradient_RMS=(w*g).tolist()),training=dict(new_updates=6000,total_updates=9000,training_seconds=r['training_elapsed_seconds'],final_checkpoint_sha256=sha(d/'final.pt'),stream_sha256=cumulative.hex()),results=results,paired=paired,probes=probes,geometry_records=geometry_records,audit_seconds=time.perf_counter()-started,original_manifest_sha256=sha(root/'MANIFEST.json'),engineering_decision='RETAIN_FOR_REPEAT_CONFIRMATION' if promote else 'PAUSE_CURRENT_JOINT_RECIPE_NO_FURTHER_TUNING',scope='Independent CPU calibration-gradient replay, full state/batch/noise/constant-weight audit, complete saved geometry denominator with independent topology/decode and semantics. Exact intersections and area-sampled surface distances are manifest-bound existing measurements, not rerun.',limitations=['Fresh noises confirm one known target, not new-object generalization.','Historical B0 weights and batch stream are exactly reused; full historical driver/runtime identity was not recorded.','Gradient RMS matching is not AdamW update matching.','No optimizer updates, GPU computation, extraction or new sample generation in this audit.'])
    result['prior_audit_note']='First audit retained at sibling independent_audit. Two audit checks compared float32 NPZ time labels to canonical decimal labels; corrected to registered TIMES. No experiment or tolerance changed.'
    out.mkdir(parents=True);write(out/'METRIC_AUDIT.json',result)
    for name in ['audit_metric.py','audit_interaction.py','audit_single_state.py','audit_fm_outputs.py']:shutil.copy2(Path(__file__).with_name(name),out/name)
    write(out/'SUPPLEMENTAL_MANIFEST.json',dict(original_manifest_sha256=sha(root/'MANIFEST.json'),files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))
    print(json.dumps({k:result[k] for k in ['passed','check_count','failed_checks','calibration','training','results','engineering_decision','audit_seconds']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);main(p.parse_args().root)
