"""Execute a fresh isolated clay study; retain failures and source snapshots."""
import os,sys,json,time,shutil,subprocess,platform,datetime,importlib.metadata,argparse
from pathlib import Path
from execute_study import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);ap.add_argument('--parent',required=True,type=Path)
    a=ap.parse_args();out=a.out.resolve();parent=a.parent.resolve();module=Path(__file__).resolve().parents[1]
    out.mkdir(parents=True,exist_ok=False)
    for folder in ['inputs','runs','audit','logs','reports','figures']:(out/folder).mkdir()
    for folder in ['code','vendor']:shutil.copytree(module/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(module/'SOURCE_PROVENANCE.json',out/'SOURCE_PROVENANCE.json')
    dump(out/'SOURCE_SNAPSHOT.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for folder in ['code','vendor'] for p in sorted((out/folder).rglob('*')) if p.is_file()]))
    dump(out/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','torch','networkx','trimesh','matplotlib','shapely']},device='CPU',numerical_threads=1))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    execution=dict(status='RUNNING',network_training_updates=0,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stages=[])
    for stage in [['run_clay.py',str(parent)],['exact_audit.py'],['finalize_audit.py'],['verify_clay_trace.py','--root',str(out)],['summarize_clay.py','--root',str(out)]]:
        cmd=[sys.executable,'-B',str(out/'code'/stage[0]),*stage[1:]];t=time.perf_counter()
        print('STAGE_START',stage[0],flush=True)
        with (out/'logs'/(Path(stage[0]).stem+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        execution['stages'].append(dict(command=cmd,returncode=r.returncode,wall_seconds=time.perf_counter()-t));dump(out/'EXECUTION.json',execution)
        print('STAGE_END',stage[0],r.returncode,flush=True)
        if r.returncode:
            execution['status']='STAGE_FAILED';dump(out/'EXECUTION.json',execution);raise RuntimeError(f'Failed output preserved: {out}')
    execution['status']='COMPLETE';dump(out/'EXECUTION.json',execution)
    dump(out/'MANIFEST.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']))
    print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
