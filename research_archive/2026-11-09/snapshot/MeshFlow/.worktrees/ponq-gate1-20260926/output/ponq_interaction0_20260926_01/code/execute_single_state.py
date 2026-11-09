"""Fixed single-snapshot FM experiment with bounded, preregistered diagnostics."""
import os,sys,time,json,shutil,argparse,subprocess,datetime,platform
from pathlib import Path
from execute_study import dump,sha

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser()
    for key in ['out','previous','target','dynamics','domain']:ap.add_argument('--'+key,required=True,type=Path)
    a=ap.parse_args();out=a.out.resolve();module=Path(__file__).resolve().parents[1]
    assert read(a.dynamics/'RESULT.json')['passed'],'Dynamics gate failed'
    assert read(a.domain/'audit/INTEGRITY.json')['passed'],'Domain experiment recording gate failed'
    assert read(a.domain/'SUMMARY.json')['groups']['fixed_identity']['valid']==1,'Fixed identity did not pass'
    out.mkdir(parents=True,exist_ok=False)
    for folder in ['logs','training','oracle','audit','reports']:(out/folder).mkdir()
    for folder in ['code','vendor']:shutil.copytree(module/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(module/'SOURCE_PROVENANCE.json',out/'SOURCE_PROVENANCE.json')
    unchanged=['fm_model.py','fm_state.py','train_fm_pilot.py','safe_extract.py','common.py','exact_audit.py']
    for name in unchanged:assert sha(out/'code'/name)==sha(a.previous/'code'/name),('baseline_source_changed',name)
    assert sha(out/'code/safe_extract_domain.py')==sha(a.domain/'code/safe_extract_domain.py')
    snapshot=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for folder in ['code','vendor'] for p in sorted((out/folder).rglob('*')) if p.is_file()]
    dump(out/'SOURCE_SNAPSHOT.json',dict(files=snapshot))
    protocol=dict(version='SINGLE_SNAPSHOT_FM_DIAG_V1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_run=str(a.previous.resolve()),target_run=str(a.target.resolve()),dynamics_run=str(a.dynamics.resolve()),domain_run=str(a.domain.resolve()),
        target='thin_u_frame06__q_only analytic transport, one fixed valid snapshot, not freeze',
        train_arm='direct',training_steps=3000,batch=8,width=128,seed=260926,training_noise_stream='Torch CPU Generator seed260927',
        architecture_and_optimizer='unchanged prior trainer; AdamW3e-4, weight_decay1e-4, clip1, no scheduler; final checkpoint only; wall600s',
        chart='Physical [extractor v(3),raw mean normal(3),principal centered PSD square-root factor(16)]; world Q decoded at selected v',
        chart_statistics='mean/scale reused byte-identically from previous 24-target data.npz, not recalculated for single target',
        sample_count=32,sample_noise='Exact previous sample_noise.npz file (NumPy PCG64 seed260927); a different generator stream from training',
        primary_solver='64 explicit Euler steps; direct total_velocity; no oracle/reference/model conditioning in rollout',
        probe=dict(count=32,noise_generator='NumPy PCG64 seed260929',times=[0,.25,.5,.75,.9,.99],
            target_matching='one Hungarian assignment of initial physical noise xyz to target xyz, applied to all22 channels',
            endpoint='s_t+(1-t)*actual_total_velocity; target-assisted diagnostic, not generation',zero_velocity_baseline=True,
            geometry_times=[.9],geometry_extractions=64),
        intervention=dict(fields=['P=extractor_v','raw_n','centered_F16'],combinations=['PPP','PPO','POP','POO','OPP','OPO','OOP','OOO'],
            configurations=256,domains=['fixed','enclosing'],extractions=512,
            matching='one Hungarian predicted endpoint xyz to target xyz after free rollout; same whole22 target permutation for all8 combinations',
            Q='Concatenate selected P,n,F then decode C=FF^T and world Q at selected P; never transfer old worldQ or relocate vertices',
            generated='Only PPP (32 unique samples), other configurations oracle-assisted; OOO repeats identity, not32 independent objects'),
        domains=dict(fixed='actual official eight SIGNS corners +/-1',enclosing='site AABB with .1 axis extent margin; zero axis margin max_extent*1e-6; no site changes',selection='always report separately, never union successes'),
        target_oracles=dict(precisions=['float64','float32_chart'],domains=['fixed','enclosing'],extractions=4,required_for_training='Both fixed-domain oracles valid; enclosing validity is separately reported'),
        optional_step_test=dict(trigger='Any one domain has probe09 valid>=24/32 AND free64 valid<=8/32 in that SAME domain',
            action='one 256Euler replay with exact same final weights/noise;32 endpoints evaluated in both domains=64 extractions; no field mixtures, no further steps or training',
            caution='t=.9 already contains90%target. Trigger is engineering allocation rule, not causal proof of solver error.'),
        no_position_clipping=True,no_normal_normalization=True,no_quadric_refit=True,
        source_files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in [a.previous/'MANIFEST.json',a.previous/'sample_noise.npz',a.previous/'data.npz',a.target/'parameters.npz',a.target/'mesh.obj',a.dynamics/'RESULT.json',a.domain/'FROZEN_PROTOCOL.json',a.domain/'SUMMARY.json']],
        environment=dict(python=sys.version,platform=platform.platform()))
    dump(out/'RUN_PROTOCOL.json',protocol);dump(out/'RUN_PROTOCOL_SHA256.json',dict(sha256=sha(out/'RUN_PROTOCOL.json')))
    execution=dict(status='RUNNING',stages=[]);env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def stage(args):
        num=len(execution['stages']);cmd=[sys.executable,'-B',str(out/'code'/args[0]),*args[1:]];start=time.perf_counter()
        print('STAGE_START',num,args[0],flush=True)
        with (out/'logs'/f'{num:02d}_{Path(args[0]).stem}.log').open('w') as log:r=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        execution['stages'].append(dict(command=cmd,returncode=r.returncode,seconds=time.perf_counter()-start));dump(out/'EXECUTION.json',execution)
        print('STAGE_END',num,args[0],r.returncode,flush=True)
        if r.returncode:raise RuntimeError('Stage failed; original output preserved: '+args[0])
    try:
        stage(['prepare_single_state.py',str(a.previous.resolve()),str(a.target.resolve())])
        stage(['train_fm_pilot.py','--data',str(out/'data.npz'),'--out',str(out/'training/direct'),'--arm','direct','--steps','3000','--device','cuda'])
        assert read(out/'training/direct/RESULT.json')['finished_requested_budget'],'Fixed training budget incomplete'
        stage(['sample_single_state.py','--nfe','64']);stage(['probe_single_state.py']);stage(['mix_single_state.py','--nfe','64'])
        probes=read(out/'probes/SUMMARY.json')['valid_at09_by_domain'];free=read(out/'interventions_64/SUMMARY.json')['by_domain']
        trigger=[d for d in ['fixed','enclosing'] if probes[d]>=24 and free[d]['PPP']['valid']<=8]
        dump(out/'STEP_TEST_DECISION.json',dict(triggered=bool(trigger),trigger_domains=trigger,probe09=probes,
            free64={d:free[d]['PPP']['valid'] for d in ['fixed','enclosing']},rule=protocol['optional_step_test']))
        if trigger:
            stage(['sample_single_state.py','--nfe','256']);stage(['mix_single_state.py','--nfe','256'])
        execution['status']='COMPLETE'
    except Exception as exc:
        execution['status']='STAGE_FAILED';execution['error']=repr(exc)
    dump(out/'EXECUTION.json',execution)
    dump(out/'MANIFEST.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']))
    print(execution['status'],out,flush=True)
    if execution['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()