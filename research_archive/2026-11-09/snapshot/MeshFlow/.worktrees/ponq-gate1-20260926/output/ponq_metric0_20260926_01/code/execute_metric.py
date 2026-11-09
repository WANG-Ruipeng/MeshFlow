"""SINGLE_STATE_METRIC0, one calibrated loss arm and one archived frozen control."""
import os,sys,time,json,shutil,argparse,subprocess,datetime,hashlib
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser()
    for k in ['out','previous','parent']:ap.add_argument('--'+k,required=True,type=Path)
    a=ap.parse_args();out=a.out.resolve();previous=a.previous.resolve();parent=a.parent.resolve();module=Path(__file__).resolve().parents[1]
    assert read(previous/'EXECUTION.json')['status']=='COMPLETE'
    assert read(previous/'post_analysis/independent_audit/INTERACTION_AUDIT.json')['passed']
    required=['metric_calibration.py','probe_metric.py','train_metric.py','preflight_metric.py','evaluate_metric.py','summarize_metric.py']
    for name in required:assert (module/'code'/name).is_file(),name
    out.mkdir(parents=True,exist_ok=False)
    for folder in ['logs','parent','control','training','evaluation','probes']:(out/folder).mkdir()
    for folder in ['code','vendor']:shutil.copytree(module/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','audit_metric.py','render_metric.py'))
    shutil.copy2(module/'SOURCE_PROVENANCE.json',out/'SOURCE_PROVENANCE.json')
    for name in ['data.npz','probe_noise.npz']:shutil.copy2(previous/name,out/name)
    shutil.copytree(previous/'target',out/'target')
    shutil.copy2(parent/'training/direct/final.pt',out/'parent/final.pt')
    for name in ['final.pt','initial.pt','TRAIN_TRACE.jsonl','PROTOCOL.json','RESULT.json']:
        shutil.copy2(previous/'training/B0'/name,out/'control'/name)
    shutil.copy2(previous/'sample_noise.npz',out/'control/archived_sample_noise.npz')
    shutil.copy2(previous/'evaluation/B0/endpoints.npz',out/'control/archived_endpoints.npz')
    noise=np.random.default_rng(260931).standard_normal((32,256,22)).astype(np.float32)
    np.savez_compressed(out/'sample_noise.npz',noise=noise)
    current={sha0 for sha0 in [hashlib.sha256(x.tobytes()).hexdigest() for x in noise]}
    old=[]
    for p in [out/'control/archived_sample_noise.npz',out/'probe_noise.npz']:
        with np.load(p,allow_pickle=False) as z:old.extend(hashlib.sha256(x.tobytes()).hexdigest() for x in z['noise'])
    assert len(current)==32 and not current&set(old)
    snapshots=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for d in ['code','vendor'] for p in sorted((out/d).rglob('*')) if p.is_file()]
    dump(out/'SOURCE_SNAPSHOT.json',dict(files=snapshots))
    protocol=dict(version='SINGLE_STATE_METRIC0',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_run=str(previous),parent_run=str(parent),parent_checkpoint='parent/final.pt',control_checkpoint='control/final.pt',
        target='unchanged valid thin_u_frame06__q_only;1state,256elements,canonical508-face mesh',
        model='original128-wide DeepSets B0,88214parameters; no new network/module; shared same3000checkpoint and restoredAdamW/RNG',
        training=dict(candidate='M0',parent_steps=3000,new_updates=6000,final_steps=9000,batch=8,additional_exposure=48000,
            control='existing archivedB0 final9000; no control retraining',
            optimizer='same AdamWlr3e-4,weight_decay1e-4,globalclip1,no scheduler; restore old moments',
            noise_time_coupling='resume parent dedicatedCPUgenerator and BalancedSchedule; verify every batch hash against archived B0',
            matching_cost='sum physicalP_xyz squared distances only; n and F contribute0; whole22target permutation',
            changed_variable='constant positive block weights on the three field-mean velocity losses only',
            unchanged=['chart mean/scale','Gaussian prior','time distribution','raw normals','principal target F','physical-P Hungarian','direct velocity','optimizer settings'],
            checkpoint_selection='final9000 only; no generation/loss-based early stop or selection',
            diagnostic_weight_checkpoints=[3000,5000,7000,9000],fault_wall_limit_seconds=3600),
        calibration=dict(noise='unchanged probe_noise.npz',probe_ids=[0,8,16,24],times=[0,.25,.5,.75,.9,.99],M=24,
            shared_names=['encoder.0.weight','encoder.0.bias','encoder.2.weight','encoder.2.bias','decoder.0.weight','decoder.0.bias','decoder.2.weight','decoder.2.bias'],
            shared_parameter_count=85376,excluded='all decoder.4 weight/bias rows',
            fields=['x','n','F'],unweighted_field_losses='mean velocity error squared within each block,3/3/16 channels',
            g='sqrt(mean_m ||grad_theta_shared L_b(m)||^2)',
            epsilon=1e-12,nearzero_gradient_rms_floor=1e-8,
            relative_weights='tilde_a_b=1/(g_b+epsilon)',
            common_scale='c=old_total_shared_gradient_RMS/tilde_total_shared_gradient_RMS; use complete per-probe3x3Gram cross terms',
            old_weights=[3/22,3/22,16/22],
            abnormal='Any nonfinite or g<=1e-8; new max/min>1e6; c outside[1e-6,1e6]; no clipping or weight search',
            frozen='WEIGHTS.json once before any candidate updates; positive weights required throughout',
            limitations='Initial shared parameter gradient scale matched, not AdamW update vector; not onlineGradNorm or perpetual balance'),
        sampling=dict(models=['B0','M0'],count_per_model=32,total_free_outputs=64,
            noise='new PCG64 seed260931 float64 standard_normal then float32; frozen before calibration/train; never pooled with prior64',
            solver='64Euler direct total_velocity; NO loss weights applied to velocities',domain='actual fixed +/-1corners',
            oracle_field_extractions=0,no_position_repair=True,no_raw_normal_normalization=True,no_refit=True),
        metrics=dict(geometry='unchanged complete validity contract',RMS='symmetric area-sampled surface distance8192/side to canonical target,bbox normalized',
            reference_seed=304206,output_seed='408000+sample_id,shared across arms',actual_faces='all retained,not forced508',
            geometry_gate='closed oriented2manifold/simple-cyclelinks/Betti[1,0,1]/zero exactintersections+degeneracy+duplicates/finite nonnegative capacities',
            gate_minimum=24,gate_denominator=32,RMS_threshold_pct_bbox=.20),
        monitoring=dict(checkpoints=['calibration_parent3000','M0_5000','M0_7000','M0_9000','B0_9000'],
            probe_ids=[0,8,16,24],times=[0,.25,.5,.75,.9,.99],parameter_gradients=True,
            evaluate_saved_weights_after_training=True,updates=0,selection=False),
        baseline_verification='core source/data/checkpoint hashes, recordedPython/Torch/NumPy, currentenvironment; one archived B0 noise0 endpoint replay maxabs<=5e-6, no extraction',
        historical_environment_limit='full historical driver/CUDA/scipy/cuDNN runtime not fully recorded; cannot certify every historical environment field; old training walltime is context only',
        stopping=dict(calibration_abnormal='record and stop without candidate training or clipped weights',
            candidate_passes='if M0 achieves24/32joint and improvesB0, retain one-seed evidence; repeat confirmation before multiple states',
            candidate_fails='archive local improvements if any; pause current Gaussian->joint22channel->PoNQ recipe; no automatic weights/network/Euler/budget search',
            scope='This fixed recipe and budget,not allFM; valid-initial-state deformation/Q-update evidence remains separate'),
        hashes={n:sha(out/n) for n in ['data.npz','probe_noise.npz','sample_noise.npz','parent/final.pt','control/final.pt','control/TRAIN_TRACE.jsonl','SOURCE_SNAPSHOT.json']},
        sources=[dict(path=str(p),sha256=sha(p)) for p in [previous/'MANIFEST.json',previous/'RUN_PROTOCOL.json',parent/'MANIFEST.json',previous/'post_analysis/independent_audit/INTERACTION_AUDIT.json']])
    dump(out/'RUN_PROTOCOL.json',protocol);dump(out/'RUN_PROTOCOL_SHA256.json',dict(sha256=sha(out/'RUN_PROTOCOL.json')))
    execution=dict(status='RUNNING',stages=[]);env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def stage(args,allow_rejection=False):
        num=len(execution['stages']);cmd=[sys.executable,'-B',str(out/'code'/args[0]),*args[1:]];logname=f'{num:02d}_{Path(args[0]).stem}.log'
        execution['active']=dict(command=cmd,log=logname);dump(out/'EXECUTION.json',execution)
        print('STAGE_START',num,*args[:3],flush=True);start=time.perf_counter()
        with (out/'logs'/logname).open('x') as log:r=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        execution['stages'].append(dict(command=cmd,returncode=r.returncode,seconds=time.perf_counter()-start,log=logname));execution.pop('active',None);dump(out/'EXECUTION.json',execution)
        print('STAGE_END',num,args[0],r.returncode,flush=True)
        if r.returncode and not allow_rejection:raise RuntimeError('Stage failed; output preserved: '+str(args))
    try:
        stage(['preflight_metric.py']);stage(['train_metric.py','--self-test'])
        stage(['metric_calibration.py','--checkpoint',str(out/'parent/final.pt'),'--out',str(out/'calibration')],allow_rejection=True)
        w=read(out/'calibration/WEIGHTS.json')
        if not w['passed']:
            execution['status']='CALIBRATION_REJECTED'
            dump(out/'DECISION.json',dict(status='STOP_CALIBRATION',train_updates=0,free_outputs=0,reason='Frozen calibration guards rejected; no clipping or retry'))
        else:
            weights_hash=sha(out/'calibration/WEIGHTS.json')
            stage(['train_metric.py','--data',str(out/'data.npz'),'--checkpoint',str(out/'parent/final.pt'),
                '--weights',str(out/'calibration/WEIGHTS.json'),'--baseline-trace',str(out/'control/TRAIN_TRACE.jsonl'),
                '--out',str(out/'training/M0'),'--steps','6000','--device','cuda'])
            r=read(out/'training/M0/RESULT.json');assert r['status']=='COMPLETED' and r['completed_steps']==9000 and r['new_updates']==6000
            assert sha(out/'calibration/WEIGHTS.json')==weights_hash,'Frozen weights modified'
            assert r['attempted_stream_sha256']==read(out/'control/RESULT.json')['attempted_stream_sha256'],'Train streams differ'
            dump(out/'TRAINING_STREAM_EQUALITY.json',dict(passed=True,candidate=r['attempted_stream_sha256'],control=read(out/'control/RESULT.json')['attempted_stream_sha256']))
            for arm in ['B0','M0']:stage(['evaluate_metric.py','--arm',arm])
            for label,ck in [('M0_5000',out/'training/M0/step_5000.pt'),('M0_7000',out/'training/M0/step_7000.pt'),('M0_9000',out/'training/M0/final.pt'),('B0_9000',out/'control/final.pt')]:
                stage(['probe_metric.py','--checkpoint',str(ck),'--label',label])
            scores={arm:read(out/'evaluation'/arm/'SUMMARY.json') for arm in ['B0','M0']}
            promote=scores['M0']['advance'] and scores['M0']['quality_pass']>scores['B0']['quality_pass']
            dump(out/'DECISION.json',dict(status='RETAIN_FOR_REPEAT_CONFIRMATION' if promote else 'PAUSE_CURRENT_JOINT_RECIPE',
                engineering_gate_passed=scores['M0']['advance'],candidate_improves_control=scores['M0']['quality_pass']>scores['B0']['quality_pass'],
                scores=scores,automatic_additional_training=False,scope=protocol['stopping']['scope']))
            execution['status']='COMPLETE'
    except Exception as exc:
        execution['status']='STAGE_FAILED';execution['error']=repr(exc)
    dump(out/'EXECUTION.json',execution)
    dump(out/'MANIFEST.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']))
    print(execution['status'],out,flush=True)
    if execution['status']=='STAGE_FAILED':raise SystemExit(1)
if __name__=='__main__':main()