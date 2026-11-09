"""Bounded matched FM pilot, gated by cumulative geometry and target encoding."""
import os,sys,json,time,shutil,subprocess,platform,datetime,argparse,importlib.metadata
from pathlib import Path
from execute_study import sha,dump

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser()
    for key in ['out','parent','gate','state-validation','model-validation']:ap.add_argument('--'+key,required=True,type=Path)
    args=ap.parse_args();out=args.out.resolve();module=Path(__file__).resolve().parents[1]
    out.mkdir(parents=True,exist_ok=False)
    for folder in ['targets','samples','training','logs','audit','reports']:(out/folder).mkdir()
    for folder in ['code','vendor']:shutil.copytree(module/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(module/'SOURCE_PROVENANCE.json',out/'SOURCE_PROVENANCE.json')
    for label,validation in [('state',args.state_validation),('model',args.model_validation)]:
        doc=read(validation);assert doc['passed'],doc
        for item in doc['source_files']:
            assert sha(out/'code'/Path(item['path']).name)==item['sha256'],('untested_source',item['path'])
        shutil.copy2(validation,out/'audit'/(label.upper()+'_VALIDATION.json'))
    source_snapshot=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for folder in ['code','vendor'] for p in sorted((out/folder).rglob('*')) if p.is_file()]
    dump(out/'SOURCE_SNAPSHOT.json',dict(files=source_snapshot))
    protocol=dict(version='FM_PILOT_CLOSED24_V1',seed=260926,sampling_seed=260927,
        parent_run=str(args.parent.resolve()),cumulative_gate=str(args.gate.resolve()),
        target_source='All 24 reference-assisted refit states of 3 known parent objects, no target filtering',
        arms=['direct','transport_residual'],training_updates_per_arm=3000,maximum_training_wall_seconds_per_arm=600,
        training_batch=8,width=128,prior='Standard Gaussian in full standardized x3/n3/F16 chart',
        normal_normalization=False,representation='centered C=FF^T; target principal square root with audited machine-roundoff PSD clipping only',
        sample_count_per_arm=32,sampling_steps=64,solver='explicit Euler',sampling_batch=1,
        checkpoint_selection='final only; no best checkpoint or result-based extension',
        endpoint_rules='No position/Q repair. Frozen extractor guard. Preserve actual face count and every failed decode/extraction/audit.',
        gates='45 cumulative geometries valid plus thin-U >=90% net gain; independent algebra pass; 24 target chart roundtrips valid with unchanged input-ID surface',
        target_metric='Nearest of all 24 target element sets by symmetric squared point distance; mesh surface distance to that selected target is closed-set diagnostic, not generalization.',
        scope='One paired seed, 24 correlated states, only 3 parent shapes: small feasibility experiment, no SIGGRAPH contribution claim.',
        environment=dict(python=sys.version,platform=platform.platform(),versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','networkx']}),
        source_files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in [args.gate/'MANIFEST.json',args.parent/'MANIFEST.json',args.state_validation,args.model_validation]])
    dump(out/'RUN_PROTOCOL.json',protocol);dump(out/'RUN_PROTOCOL_SHA256.json',dict(sha256=sha(out/'RUN_PROTOCOL.json')))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    execution=dict(status='RUNNING',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stages=[])
    stages=[['prepare_fm_pilot.py',str(args.parent.resolve()),str(args.gate.resolve())],['audit_fm_precision.py']]
    stages += [['train_fm_pilot.py','--data',str(out/'data.npz'),'--out',str(out/'training'/arm),'--arm',arm,'--steps','3000','--device','cuda'] for arm in protocol['arms']]
    stages += [['sample_fm_pilot.py']]
    for index,stage in enumerate(stages):
        if stage[0]=='sample_fm_pilot.py':
            complete=all(read(out/'training'/arm/'RESULT.json')['finished_requested_budget'] for arm in protocol['arms'])
            if not complete:
                execution['status']='TRAINING_BUDGET_INCOMPLETE';break
        cmd=[sys.executable,'-B',str(out/'code'/stage[0]),*stage[1:]];t=time.perf_counter()
        print('STAGE_START',index,stage[0],flush=True)
        with (out/'logs'/f'{index:02d}_{Path(stage[0]).stem}.log').open('w') as log:
            r=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        execution['stages'].append(dict(command=cmd,returncode=r.returncode,wall_seconds=time.perf_counter()-t))
        dump(out/'EXECUTION.json',execution);print('STAGE_END',index,stage[0],r.returncode,flush=True)
        if r.returncode:
            execution['status']='STAGE_FAILED';break
    else:execution['status']='COMPLETE'
    dump(out/'EXECUTION.json',execution)
    dump(out/'MANIFEST.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']))
    print(execution['status'],out,flush=True)
    if execution['status']=='STAGE_FAILED':raise SystemExit(1)
if __name__=='__main__':main()