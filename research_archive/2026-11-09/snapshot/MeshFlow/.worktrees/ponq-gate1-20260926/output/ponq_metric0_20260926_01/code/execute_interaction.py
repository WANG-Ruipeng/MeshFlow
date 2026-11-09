"""SINGLE_STATE_INTERACTION0: freeze once, train all arms, then evaluate all."""
import os,sys,time,json,shutil,argparse,subprocess,datetime,platform,hashlib
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def source_rows(out,folders):
    return [dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for folder in folders for p in sorted((out/folder).rglob('*')) if p.is_file()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True)
    a=ap.parse_args();out=a.out.resolve();previous=a.previous.resolve();module=Path(__file__).resolve().parents[1]
    assert read(previous/'EXECUTION.json')['status']=='COMPLETE'
    assert read(previous/'training/direct/RESULT.json')['completed_steps']==3000
    required=['interaction_model.py','train_interaction.py','preflight_interaction.py','interaction_semantics.py','probe_interaction.py','evaluate_interaction.py']
    for name in required:assert (module/'code'/name).is_file(),name
    out.mkdir(parents=True,exist_ok=False)
    for folder in ['logs','training','evaluation','probes','parent']:(out/folder).mkdir()
    for folder in ['code','vendor']:shutil.copytree(module/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['fm_model.py','fm_state.py','train_fm_pilot.py','safe_extract.py','safe_extract_domain.py','common.py','exact_audit.py','single_state_common.py','sample_single_state.py','run_clay.py']:
        assert sha(out/'code'/name)==sha(previous/'code'/name),('baseline_source_changed',name)
    shutil.copy2(module/'SOURCE_PROVENANCE.json',out/'SOURCE_PROVENANCE.json')
    for name in ['data.npz','probe_noise.npz']:shutil.copy2(previous/name,out/name)
    shutil.copytree(previous/'target',out/'target')
    shutil.copy2(previous/'sample_noise.npz',out/'sample_noise_old.npz')
    shutil.copy2(previous/'training/direct/final.pt',out/'parent/final.pt')
    with np.load(out/'sample_noise_old.npz',allow_pickle=False) as z:old=z['noise'].copy()
    fresh=np.random.default_rng(260930).standard_normal((32,256,22)).astype(np.float32)
    assert old.shape==fresh.shape and old.dtype==np.float32 and not np.array_equal(old,fresh)
    np.savez_compressed(out/'sample_noise_new.npz',noise=fresh)
    np.savez_compressed(out/'sample_noise.npz',noise=np.concatenate([old,fresh]),split=np.array(['old']*32+['new']*32),split_id=np.tile(np.arange(32),2))
    old_set={hashlib.sha256(x.tobytes()).hexdigest() for x in old}
    with np.load(out/'probe_noise.npz',allow_pickle=False) as z:probe_set={hashlib.sha256(x.tobytes()).hexdigest() for x in z['noise']}
    fresh_set={hashlib.sha256(x.tobytes()).hexdigest() for x in fresh}
    assert len(fresh_set)==32 and not fresh_set&old_set and not fresh_set&probe_set
    snapshot=source_rows(out,['code','vendor']);dump(out/'SOURCE_SNAPSHOT.json',dict(files=snapshot))
    sourcefiles=[previous/'MANIFEST.json',previous/'training/direct/final.pt',previous/'data.npz',previous/'sample_noise.npz',previous/'probe_noise.npz',previous/'post_analysis/independent_audit/SINGLE_STATE_AUDIT.json']
    protocol=dict(version='SINGLE_STATE_INTERACTION0',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_run=str(previous),parent_checkpoint_sha256=sha(out/'parent/final.pt'),
        target='single valid analytic thin_u_frame06__q_only,256 elements,508-face target; copied unchanged from single-state run',
        chart='unchanged22 physical extractor v3/raw n3/centered principal PSD factor F16; reuse old24-target mean/scale',
        arms=dict(B0='unchanged width128 DeepSets continuation',
            B1='after encoder before mean/max pool: H+W2 SiLU(W1 LayerNorm(H)),128-256-128,bias,affineLN; W2 weight+bias zero',
            B2='same insertion: H+4-head128-channel full self-attention(LayerNorm(H)),head32,no dropout,no extra FFN; output projection weight+bias zero'),
        expected_added_parameters=dict(B0=0,B1=66176,B2=66304),
        common_start='old model tensors and named optimizer states exact; new AdamW step/moments zero; branch initialization RNG independent of resumed train RNG',
        training=dict(arms=['B0','B1','B2'],order=['B0','B1','B2'],parent_updates=3000,additional_updates_per_arm=6000,total_updates_per_arm=9000,
            total_additional_updates=18000,batch=8,additional_exposure_per_arm=48000,
            optimizer='same AdamW lr3e-4 weight_decay1e-4 clip_global_norm1; no scheduler,AMP,TF32 or loss reweight',
            stream='restore same parent batch_generator_cpu and BalancedSchedule; same noise/time/physical-P Hungarian whole22 target permutation per update; hashes recorded',
            loss='mean22 total direct velocity MSE = (3 Lv + 3 Ln +16 LF)/22',
            criterion='final update only; no metric-based checkpoint selection or stopping',failure_wall_limit_seconds_per_arm=3600),
        preflight='same initial functions,permutation equivariance,parameter counts,old model+optimizer+RNG restoration,B0 continuation update equivalence; no training counted',
        sampling=dict(count_per_arm=64,nfe=64,domain='actual official fixed +/-1 eight anchors only',old_count=32,new_count=32,
            old='byte-identical previous NumPy public noise file',new='NumPy default_rng PCG64 seed260930 standard_normal float64 then cast float32; frozen before training',
            free_extractions=192,selection='all samples; old/new separate; no rejection retry,clipping,domain switch,unit normal repair or refit',
            timing='GPU synchronized integration per sample plus separate posthoc diagnostic and extraction/audit/metric costs; unchanged L diagnostics remain in direct total_velocity'),
        engineering_gate=dict(split='new',minimum_count=24,denominator=32,condition='original complete geometry validity AND sampled symmetric surface RMS <=0.20% target bbox diagonal',
            RMS_threshold_pct_bbox=.20,reference='fixed canonical target mesh',definition='new single-object engineering progression line; not retroactive old validity change'),
        geometry=dict(samples_per_direction=8192,reference_sample_seed=304206,output_sample_seed='406000+global sample_id',validity='same original closed oriented2manifold+simple-cycle links+Betti[1,0,1]+zero exact intersections/degeneracy/duplicates+finite nonnegative graph capacities',actual_faces='record all,not forced508'),
        field_interventions=dict(old_ids=[0,4,8,12,16,20,24,28],combinations=['POO','OPO','OOP'],extractions=72,
            matching='one post-rollout predictedP to targetP Hungarian; whole22 permutation reused for all combinations',
            fields='actual extractor P/raw n/centered F; re-decode worldQ at selected P',generated=False),
        probes=dict(noise='byte-identical previous probe_noise.npz',times=[0,.25,.5,.75,.9,.99],count=32,
            geometry_extractions=0,gradient_ids=[0,8,16,24],
            gradients='unweighted block-mean and actual3/22,3/22,16/22 weighted losses wrt shared hidden after branch before pool; diagnostics only,no updates',
            meanings='conditional-sample errors descriptive; early conditional target velocity need not equal marginal velocity; free rollout decisive',
            normal_nearzero=1e-8,near_duplicate_pct_bbox=.1,coverage_pct_bbox=[.1,.5,1.0],
            coverage_scope='target point support coverage,not generated distribution coverage'),
        forbidden_changes=['raw normal representation','Hungarian cost','target gauge','Q transport in velocity','loss weights','target ID or slot ID','target adjacency/normals as network input','new domain or solver comparison'],
        hashes={name:sha(out/name) for name in ['data.npz','sample_noise_old.npz','sample_noise_new.npz','sample_noise.npz','probe_noise.npz','parent/final.pt','SOURCE_SNAPSHOT.json']},
        parent_sources=[dict(path=str(p),sha256=sha(p)) for p in sourcefiles],
        environment=dict(python=sys.version,platform=platform.platform()))
    dump(out/'RUN_PROTOCOL.json',protocol);dump(out/'RUN_PROTOCOL_SHA256.json',dict(sha256=sha(out/'RUN_PROTOCOL.json')))
    execution=dict(status='RUNNING',stages=[]);env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def stage(arguments):
        num=len(execution['stages']);cmd=[sys.executable,'-B',str(out/'code'/arguments[0]),*arguments[1:]]
        logname=f'{num:02d}_{Path(arguments[0]).stem}.log'
        execution['active']=dict(command=cmd,log=logname);dump(out/'EXECUTION.json',execution)
        print('STAGE_START',num,*arguments[:3],flush=True);start=time.perf_counter()
        with (out/'logs'/logname).open('x') as log:r=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        execution['stages'].append(dict(command=cmd,returncode=r.returncode,seconds=time.perf_counter()-start,log=logname));execution.pop('active',None);dump(out/'EXECUTION.json',execution)
        print('STAGE_END',num,arguments[0],r.returncode,flush=True)
        if r.returncode:raise RuntimeError('Stage failed; outputs preserved: '+str(arguments))
    try:
        stage(['preflight_interaction.py','--checkpoint',str(out/'parent/final.pt'),'--data',str(out/'data.npz'),'--out',str(out/'preflight'),'--device','cuda'])
        for arm in ['B0','B1','B2']:
            stage(['train_interaction.py','--data',str(out/'data.npz'),'--checkpoint',str(out/'parent/final.pt'),'--out',str(out/'training'/arm),'--arm',arm,'--steps','6000','--device','cuda'])
            r=read(out/'training'/arm/'RESULT.json')
            assert r['status']=='COMPLETED' and r['completed_steps']==9000 and r['new_updates']==6000,r
        streams=[read(out/'training'/arm/'RESULT.json')['attempted_stream_sha256'] for arm in ['B0','B1','B2']]
        assert len(set(streams))==1,'Training streams disagree; preserve final models but do not compare mismatched arms'
        dump(out/'TRAINING_STREAM_EQUALITY.json',dict(passed=True,streams=dict(zip(['B0','B1','B2'],streams))))
        for arm in ['B0','B1','B2']:
            stage(['evaluate_interaction.py','--arm',arm]);stage(['probe_interaction.py','--arm',arm])
        execution['status']='COMPLETE'
    except Exception as exc:
        execution['status']='STAGE_FAILED';execution['error']=repr(exc)
    dump(out/'EXECUTION.json',execution)
    dump(out/'MANIFEST.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']))
    print(execution['status'],out,flush=True)
    if execution['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()