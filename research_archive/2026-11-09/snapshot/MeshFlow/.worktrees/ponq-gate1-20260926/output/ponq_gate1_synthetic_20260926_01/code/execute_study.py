"""Fresh source-snapshotted CPU study, no overwrites and no training."""
import argparse,datetime,hashlib,importlib.metadata,json,os,platform,shutil,subprocess,sys,time
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['synthetic','real','isolation'])
    p.add_argument('--out',type=Path,required=True)
    for name in ['original','selection','baseline']:p.add_argument('--'+name,type=Path)
    a=p.parse_args();out=a.out.resolve();module=Path(__file__).resolve().parents[1]
    if out.exists():raise FileExistsError(out)
    if a.mode=='real' and not a.selection:raise ValueError('Missing selection')
    if a.mode=='isolation' and not a.baseline:raise ValueError('Missing baseline')
    out.mkdir(parents=True)
    for name in ['inputs','runs','logs','audit','reports','figures']:(out/name).mkdir()
    for name in ['code','vendor']:shutil.copytree(module/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(module/'SOURCE_PROVENANCE.json',out/'SOURCE_PROVENANCE.json')
    versions={}
    for name in ['numpy','scipy','torch','trimesh','networkx','shapely','pandas','matplotlib']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    dump(out/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
         versions=versions,device='CPU',numerical_threads=1))
    source=[x for folder in ['code','vendor'] for x in sorted((out/folder).rglob('*')) if x.is_file()]
    dump(out/'SOURCE_SNAPSHOT.json',dict(files=[dict(path=str(x.relative_to(out)),bytes=x.stat().st_size,sha256=sha(x)) for x in source]))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    entry={'synthetic':['run_gate1.py'],'real':['run_selected.py',str(a.selection)],
           'isolation':['run_isolation.py',str(a.baseline)]}[a.mode]
    state=dict(mode=a.mode,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               stages=[],status='RUNNING',network_training_updates=0)
    dump(out/'EXECUTION.json',state)
    for stage in [entry,['exact_audit.py'],['finalize_audit.py']]:
        cmd=[sys.executable,'-B',str(out/'code'/stage[0]),*stage[1:]]
        print('STAGE_START',a.mode,stage[0],flush=True);t=time.perf_counter()
        with (out/'logs'/(Path(stage[0]).stem+'.log')).open('w') as log:
            r=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['stages'].append(dict(command=cmd,returncode=r.returncode,wall_seconds=time.perf_counter()-t))
        dump(out/'EXECUTION.json',state);print('STAGE_END',stage[0],r.returncode,flush=True)
        if r.returncode:
            state['status']='STAGE_FAILED';dump(out/'EXECUTION.json',state)
            raise RuntimeError('Preserved failed stage: '+str(out))
    if a.original and a.mode=='synthetic':
        old={(r['object'],r['state']):r for r in json.loads((a.original/'LEDGER.json').read_text())}
        comparisons=[]
        for row in json.loads((out/'LEDGER.json').read_text()):
            if row['status']!='EXTRACTED':continue
            key=row['object'],row['state'];rel=Path('runs')/('__'.join(key))/'mesh.obj'
            fields=['vertices','faces','beta0','beta1','beta2','nonmanifold_vertices',
                    'exact_intersection_pairs','exact_degenerate_faces','rms_pct_bboxdiag']
            comparisons.append(dict(object=key[0],state=key[1],mesh_byte_equal=sha(out/rel)==sha(a.original/rel),
                fields={f:dict(original=old[key].get(f),local=row.get(f),equal=old[key].get(f)==row.get(f)) for f in fields}))
        dump(out/'audit/ORIGINAL_COMPARISON.json',comparisons)
    state['status']='COMPLETE';dump(out/'EXECUTION.json',state)
    files=[x for x in sorted(out.rglob('*')) if x.is_file() and x.name!='MANIFEST.json']
    dump(out/'MANIFEST.json',dict(files=[dict(path=str(x.relative_to(out)),bytes=x.stat().st_size,sha256=sha(x)) for x in files]))
    print('COMPLETE',out,flush=True)
if __name__=='__main__':main()