"""New frozen C panel through original HINGE_OPP015 -> ALPHA -> HOLE.

No generation, D0, new noise, source Y, radius sweep or repeat passes. Original
RAW/BASE_POST remain read-only; every CPU intermediate and rejection is saved.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,concurrent.futures,json,multiprocessing,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from . import evaluate as V
from r2.tools.hinge_opposite_v1 import engine as H
from r2.tools.hinge_v2_cpu_v1 import alpha,holes

C,E=V.C,V.E
STAGES=('HINGE_OPP015','ALPHA','FINAL_POST')
VERSION='chair_back_original_opposite_hinge_alpha_hole_v1'


def save_npz(path,values):
    path=C.resolve(path)
    if path.exists():raise FileExistsError('Refusing overwrite '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.npz.partial')
    with tmp.open('xb') as f:np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def freeze(root=V.ROOT):
    root=C.resolve(root);dest=root/'cpu_protocol.json'
    if dest.exists():
        r=C.read(dest);E.verify(r['protected']);E.verify(r['code']);return r
    qq=V.prepare(root)['requests'];protected={str(root/'eval_manifest.json'):C.sha(root/'eval_manifest.json')}
    for q in qq:
        base,_=V.artifact(root,q,'BASE_POST');raw,_=V.artifact(root,q,'RAW')
        paths=(base,base.parent/'pre_geo.npz',raw,V.score_dir(root,q,'BASE_POST')/'pairs.json',V.score_dir(root,q,'BASE_POST')/'audit_receipt.json')
        for p in paths:protected[str(p)]=C.sha(p)
        prior=C.read(paths[-1]);E.verify(prior['files'])
    code=H.Q.definition_hashes()
    for mod in (H,alpha,holes,V):code[str(Path(mod.__file__))]=C.sha(mod.__file__)
    code[str(Path(__file__))]=C.sha(__file__)
    r=dict(status='FROZEN',version=VERSION,requests=qq,stages=STAGES,delta_fraction=.015,
        hinge='Original opposite guard on every new two-face edge; no additional radius',
        alpha='Original midpoint then frozen 0,.25,.5,.75,1 finite targets; one queue',
        hole='Existing exact indexed 3-open-edge cycles only; all adjacent free and >=1 SEAM; original guards',
        source_geometry='None beyond current C; BRIDGE has no RAW/source-parent identity',
        post_order='BASE_POST -> HINGE_OPP015 -> ALPHA -> FINAL_POST (HOLE)',
        protected=protected,code=code,model_calls=0,D0_calls=0,new_noise=0,new_queries=0)
    C.write(dest,r);return r


def work(job):
    root,q,protocol_sha=job;root=C.resolve(root)
    basepath,_=V.artifact(root,q,'BASE_POST');rawpath,_=V.artifact(root,q,'RAW')
    prepath=basepath.parent/'pre_geo.npz'
    pre=C.arrays(prepath)['output'];raw=C.arrays(rawpath)['output'];K=int(q['K']);N=int(q['N'])
    if pre.shape!=raw.shape or pre.shape!=(N,3,3):raise ValueError('RAW/PRE_GEO slot domains differ')
    baseline_C=C.arrays(basepath)['C'];results=[]
    for stage in STAGES:
        dest=V.mesh_path(root,q,stage).parent;done=dest/'complete.json'
        if stage=='HINGE_OPP015':
            src=basepath;pairpath=V.score_dir(root,q,'BASE_POST')/'pairs.json'
        else:
            previous='HINGE_OPP015' if stage=='ALPHA' else 'ALPHA'
            src=V.mesh_path(root,q,previous);pairpath=src.parent/'pairs.json'
        identity=dict(protocol_sha256=protocol_sha,request_id=q['request_id'],stage=stage,
            source_mesh_sha256=C.sha(src),source_pairs_sha256=C.sha(pairpath),pre_geo_sha256=C.sha(prepath),raw_sha256=C.sha(rawpath))
        if done.exists():
            r=C.read(done)
            if r['identity']!=identity:raise ValueError('CPU saved producer changed')
            E.verify(r['files']);results.append(r);continue
        if dest.exists():raise RuntimeError('Partial CPU attempt preserved; explicit recovery required: '+str(dest))
        tick=time.perf_counter();C.write(dest/'started.json',dict(identity=identity,utc=C.utc(),pid=os.getpid()))
        try:
            current=C.arrays(src);pairs=C.read(pairpath)
            if stage=='HINGE_OPP015':
                out,records,summary,before,after,newpairs=H.decode(current,pre,raw,pairs,.015,require_opposite_winding=True)
            elif stage=='ALPHA':
                out,records,summary,before,after,newpairs=alpha.decode(current,pre,raw,pairs,.015,alphas=True)
            else:
                out,newpairs,edges,records,summary=holes.decode(current,pairs);before=after=None
            if out['C'].tobytes()!=baseline_C.tobytes():raise AssertionError('CPU stage changed original C')
            if stage!='FINAL_POST' and len(out['faces'])!=len(current['faces']):raise AssertionError('Unexpected old-face deletion')
            if stage=='FINAL_POST' and not np.array_equal(out['output'][:len(current['output'])].view(np.uint32),current['output'].view(np.uint32)):
                raise AssertionError('HOLE changed preexisting faces')
            save_npz(dest/'post.npz',out);C.write(dest/'pairs.json',newpairs);C.write(dest/'summary.json',summary)
            C.write(dest/'candidate_records.json',records);C.csvwrite(dest/'candidates.csv',records)
            if before is not None:
                C.write(dest/'baseline_connectivity.json',before);C.write(dest/'connectivity.json',after)
                paired=H.Q.compare_ledgers(before,after,pairs,newpairs,summary['affected_source_raw_slots'])
                C.write(dest/'paired_ledger.json',paired)
                for key in ('CF_lost_legal_edges','FF_lost_legal_edges','CF_new_illegal_pairs','FF_new_illegal_pairs','deleted_source_faces'):
                    if paired['summary'][key]!=0:raise AssertionError('Frozen CPU guard violated '+key)
            if 'torch' in sys.modules:raise RuntimeError('CPU stage imported torch')
            r=dict(status='COMPLETE',version=VERSION,identity=identity,request_id=q['request_id'],stage=stage,
                row=dict(request_id=q['request_id'],condition_id=q['condition_id'],parent_id=q['parent_id'],split=q['split'],N=N,K=K,stage=stage),
                summary=summary,seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                post_sha256=C.sha(dest/'post.npz'),model_calls=0,GPU_calls=0,D0_calls=0,
                files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
            C.write(done,r);results.append(r);print('CPU_POST',stage,q['request_id'],summary.get('accepted'),flush=True)
        except BaseException as exc:
            C.write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),model_calls=0));raise
    return dict(rows=results,peak_RSS_KiB=max(r['peak_RSS_KiB'] for r in results))


def run(root=V.ROOT,workers=2):
    root=C.resolve(root);reg=freeze(root);done=root/'cpu_post_completion.json'
    if done.exists():
        r=C.read(done);E.verify(r['files']);return r
    identity=C.sha(root/'cpu_protocol.json');jobs=[(str(root),q,identity) for q in reg['requests']]
    before=root/'cpu_resources_before.json'
    if not before.exists():C.write(before,C.resources())
    tick=time.perf_counter();first=work(jobs[0]);snapshot=C.resources();count=V.worker_count(snapshot,first,workers)
    C.write(root/'cpu_dispatch.json',dict(workers=count,threads_per_worker=1,resources=snapshot,
        observed_first_actual_peak_RSS_KiB=first['peak_RSS_KiB'],GPU_workers=0),replace=(root/'cpu_dispatch.json').exists())
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        allresults=[first,*list(pool.map(work,jobs[1:]))]
    rows=[r for batch in allresults for r in batch['rows']]
    summary=[dict(request_id=r['request_id'],stage=r['stage'],seconds=r['seconds'],**r['summary']) for r in rows]
    C.csvwrite(root/'cpu_geometry_summary.csv',summary)
    files={str(V.mesh_path(root,q,s).parent/'complete.json'):C.sha(V.mesh_path(root,q,s).parent/'complete.json') for q in reg['requests'] for s in STAGES}
    files[str(root/'cpu_geometry_summary.csv')]=C.sha(root/'cpu_geometry_summary.csv')
    E.verify(reg['protected']);E.verify(reg['code'])
    r=dict(status='COMPLETE',requests=len(reg['requests']),cells=len(rows),stages=STAGES,workers=count,
        seconds=time.perf_counter()-tick,files=files,model_calls=0,GPU_calls=0,D0_calls=0,new_noise=0)
    C.write(done,r);return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(V.ROOT));p.add_argument('--workers',type=int,choices=(1,2,3,4),default=2)
    a=p.parse_args();r=run(a.root,a.workers);print(json.dumps({k:v for k,v in r.items() if k!='files'}),flush=True)
