"""Run immutable per-object attempts for the preregistered layout diagnostic."""
from __future__ import annotations
import argparse, json, time, traceback, datetime
from pathlib import Path
import numpy as np
import torch
from ..io_utils import file_hash,array_hash,write_json,save_npz,read_mesh
from ..exporter import MeshExporter
from .geometry import mesh_quality,evaluate_mesh,audit_mesh

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,Path):return str(x)
    return x

def emit(x): print(json.dumps(clean(x),ensure_ascii=False),flush=True)

def quality_pass(q,floor):
    return q['degenerate_count']==0 and all(q[k]+1e-6>=floor[k] for k in ['min_angle_deg','angle_q05_deg'])

def assess(v,f,rv,rf,floor,seed,points=8192):
    q=mesh_quality(v,f)
    t=time.perf_counter();metrics=evaluate_mesh(v,f,rv,rf,points=points,seed=seed);eval_s=time.perf_counter()-t
    audit=audit_mesh(v,f,exact=True)
    bp=len(f)==512;qp=quality_pass(q,floor)
    return {'quality':q,'metrics':metrics,'audit':audit,'evaluation_seconds':eval_s,
            'audit_seconds':audit['audit_seconds'],'budget_match':bp,'quality_pass':qp,
            'qualified':bool(bp and qp and audit['valid'])}

def aggregate(run):
    rows=[json.loads(p.read_text()) for p in sorted((run/'attempts').glob('*/*/result.json'))]
    write_json(run/'results.json',{'rows':rows,'attempts':len(rows),'scope':'closed-set layout diagnostic'},replace=(run/'results.json').exists())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--selection',choices=['smoke','remaining','all'],default='smoke')
    p.add_argument('--arms',nargs='+',default=None)
    a=p.parse_args();run=Path(a.run);protocol=json.loads((run/'RUN_PROTOCOL.json').read_text())
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    t=time.perf_counter();exporter=MeshExporter(a_seed=10)
    setup_s=time.perf_counter()-t
    arms=a.arms or protocol['arms']
    if any(x not in protocol['arms'] for x in arms):raise ValueError('unregistered arm')
    uids=protocol['smoke_uids'] if a.selection=='smoke' else protocol['uids']
    if a.selection=='remaining':uids=[u for u in uids if u not in protocol['smoke_uids']]
    code={str(p.relative_to(Path(__file__).parents[1])):file_hash(p) for p in Path(__file__).parents[1].rglob('*.py')}
    source_id=array_hash(np.frombuffer(json.dumps(code,sort_keys=True).encode(),dtype=np.uint8))[:16]
    invocations=run/'invocations';invocations.mkdir(exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    write_json(invocations/(stamp+'.json'),{'args':vars(a),'source_id':source_id,'source_hashes':code,'exporter_startup_seconds':setup_s})
    for uid in uids:
        index=protocol['uids'].index(uid);seed=protocol['evaluation']['seed_base']+index
        with np.load(run/'inputs'/f'{uid}.npz',allow_pickle=False) as z:y=z['logr'];rv=z['vertices'];rf=z['faces']
        expected=protocol['records'][index]
        assert array_hash(y)==expected['mainfield_hash'] and array_hash(rv,rf)==expected['ref_mesh_hash']
        fixed,fixed_meta=exporter.export(y,512,'fixed')
        floor={k:mesh_quality(fixed['vertices'],fixed['faces'])[k] for k in ['min_angle_deg','angle_q05_deg']}
        representation_path=run/'representation'/f'{uid}.json'
        if not representation_path.exists():
            data=Path(protocol['data_dir']);rawv,rawf=read_mesh(data/'raw'/f'{uid}.npz')
            with np.load(data/'lineage'/f'{uid}.npz',allow_pickle=False) as z:
                rawv=(rawv-z['centre'])/float(z['model_to_input_scale'])
            err=evaluate_mesh(rv,rf,rawv,rawf,points=8192,seed=seed+100000)
            write_json(representation_path,{'uid':uid,'mainfield_to_original':err,'not_used_by_layout_methods':True})
        for arm in arms:
            dest=run/'attempts'/uid/arm
            if (dest/'result.json').exists():
                emit({'event':'skip_completed','uid':uid,'arm':arm});continue
            dest.mkdir(parents=True,exist_ok=False)
            row={'uid':uid,'arm':arm,'mainfield_hash':expected['mainfield_hash'],'quality_floor':floor,
                 'source_id':source_id,'status':'started','requested_faces':512}
            write_json(dest/'START.json',row)
            emit({'event':'start','uid':uid,'arm':arm})
            try:
                started=time.perf_counter()
                trajectory=[]
                if arm in ['fixed','a','vtk_qem']:
                    mesh,meta=exporter.export(y,512,{'vtk_qem':'qem'}.get(arm,arm))
                    elapsed=time.perf_counter()-started
                    v,f=mesh['vertices'],mesh['faces']
                    extras={k:val for k,val in mesh.items() if k not in ['vertices','faces']}
                elif arm=='cgal_constrained':
                    from .cgal_runner import run_cgal
                    v,f,meta=run_cgal(rv,rf,512,floor['min_angle_deg'],floor['angle_q05_deg'],dest/'backend')
                    elapsed=time.perf_counter()-started;extras={}
                else:
                    from .optimizer import optimize_layout
                    if arm=='opt_a':
                        initial,initial_meta=exporter.export(y,512,'a')
                    else:
                        initial,initial_meta=exporter.export(y,512,'fixed')
                    init_seconds=time.perf_counter()-started
                    optimized=optimize_layout(y,initial['u'],initial['faces'],floor,device='cuda',
                         max_steps=protocol['optimizer']['max_steps'],max_seconds=protocol['optimizer']['max_seconds'],
                         seed=protocol['optimizer']['seed_base']+index,
                         snapshot_steps=tuple(protocol['optimizer']['snapshot_steps']))
                    elapsed=time.perf_counter()-started
                    v,f=optimized['vertices'],optimized['faces'];extras={'u':optimized['u']}
                    meta=optimized['stats'];meta['initialization_seconds']=init_seconds;meta['initialization_metadata']=initial_meta
                    write_json(dest/'optimization.json',clean({k:v for k,v in optimized.items() if k not in ['vertices','faces','u','snapshots']}))
                    for j,snap in enumerate(optimized['snapshots']):
                        sv=np.asarray(snap['vertices']);sf=np.asarray(snap.get('faces',f))
                        snap_path=dest/'snapshots'/f'{j:03d}.npz'
                        save_npz(snap_path,vertices=sv,faces=sf,u=snap['u'])
                        metrics=assess(sv,sf,rv,rf,floor,seed)
                        trajectory.append({'step':snap['iteration'],'elapsed_seconds':init_seconds+snap['elapsed_seconds'],
                            'internal_loss':snap.get('internal_loss'),**metrics,'mesh_file':str(snap_path.relative_to(run))})
                    row['trajectory']=trajectory
                mesh_file=dest/'mesh.npz';save_npz(mesh_file,vertices=v,faces=f,**extras)
                row.update(status='success',algorithm_seconds=elapsed,actual_faces=int(len(f)),
                    mesh_file=str(mesh_file.relative_to(run)),mesh_hash=array_hash(v,f),metadata=clean(meta))
                row.update(assess(v,f,rv,rf,floor,seed))
                write_json(dest/'result.json',clean(row));aggregate(run)
                emit({'event':'complete','uid':uid,'arm':arm,'seconds':round(elapsed,4),'rms':row['metrics']['rms_bbox_percent'],
                       'q05':row['quality']['angle_q05_deg'],'qualified':row['qualified'],'faces':len(f)})
            except Exception as exc:
                row.update(status='error',error=repr(exc),traceback=traceback.format_exc(),
                           algorithm_seconds=time.perf_counter()-started,qualified=False)
                write_json(dest/'result.json',clean(row));aggregate(run);emit(row)
    aggregate(run)
if __name__=='__main__':main()


