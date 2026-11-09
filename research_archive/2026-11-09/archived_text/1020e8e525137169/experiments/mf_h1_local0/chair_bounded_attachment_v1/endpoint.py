"""Apply the frozen pure corrector to all 16 cached outputs, before evaluation."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import time
import numpy as np
from .common import *


def _project_job(C,F,config):
    from .project import project_local_attachment
    C=np.array(C,copy=True);F=np.array(F,copy=True)
    c_before=C.tobytes();f_before=F.tobytes()
    corrected,log=project_local_attachment(C,F,fixed_config=config)
    if C.tobytes()!=c_before or F.tobytes()!=f_before:raise RuntimeError('Pure corrector mutated its input')
    return corrected,log


def persist(out,row,raw,bank,corrected,log,wall_seconds):
    out=Path(out);K=int(row['K']);F=raw[K:].reshape(-1,3,3)
    corrected=np.asarray(corrected)
    assert corrected.dtype==np.float32 and corrected.shape==F.shape
    delta=.01*float(np.linalg.norm(np.ptp(bank['C'].reshape(-1,3).astype(np.float64),axis=0)))
    disp=np.linalg.norm(corrected.astype(np.float64)-F.astype(np.float64),axis=2)
    output=raw.copy();output[K:]=corrected.reshape(-1,9)
    equation=True
    for constraint in log['accepted_constraints']:
        for group in constraint['groups']:
            target=np.asarray(group['target_point'],np.float32)
            for face,corner in group['face_corners']:
                equation=equation and np.array_equal(corrected[face,corner],target)
    unchanged=output.tobytes()==raw.tobytes()
    assert np.array_equal(disp,np.asarray(log['per_corner_displacement'],np.float64))
    checks=dict(C=output[:K].tobytes()==raw[:K].tobytes(),N=output.shape==raw.shape,
        finite=bool(np.isfinite(output).all()),budget=bool((disp<=delta).all()),
        accepted_equations=bool(equation),no_new_degenerate=unchanged or not log['final_checks']['reasons'],
        no_new_illegal_events=unchanged or not log['geometry_events']['added'])
    if log['status']=='IMPLEMENTATION_ERROR' or not all(checks.values()):raise RuntimeError('Endpoint final contract failure')
    if not log['accepted_constraints']:assert output.tobytes()==raw.tobytes()
    directory=out/'outputs'/row['request_id'];directory.mkdir(parents=True,exist_ok=True)
    log_path=directory/'intervention.json';atomic_json(log_path,log)
    path=directory/'raw.npz'
    save_npz(path,raw=output,C=bank['C'],noise=bank['z'],known_mask=bank['known_mask'],valid_mask=bank['valid_mask'])
    geometry_path=directory/'geometry_arrays.npz'
    save_npz(geometry_path,corner_displacement=disp,displacement_vector=corrected.astype(np.float64)-F.astype(np.float64),
        changed_corners=np.any(corrected!=F,axis=2),changed_faces=np.any(corrected!=F,axis=(1,2)))
    ply=directory/'mesh.ply';write_mesh(ply,output)
    source=read_json(result_path(out,row,'RAW'))
    value=dict(row,status='PROJECTED',raw_path=str(path),raw_sha256=digest(path),raw_array_sha256=array_hash(output),
        mesh_path=str(ply),mesh_sha256=digest(ply),intervention_log_path=str(log_path),log_path=str(log_path),
        intervention_log_sha256=digest(log_path),geometry_arrays_path=str(geometry_path),geometry_arrays_sha256=digest(geometry_path),
        geometry_contracts=checks,correction_status=log['status'],accepted_count=len(log['accepted_constraints']),
        changed=output.tobytes()!=raw.tobytes(),delta=delta,L_C=delta/.01,
        maximum_corner_displacement=float(disp.max(initial=0)),RMS_corner_displacement=float(np.sqrt(np.mean(disp**2))),
        modified_corner_fraction=float(np.mean(disp>0)),modified_face_fraction=float(np.mean((disp>0).any(1))),
        source_raw_path=source['raw_path'],source_raw_sha256=source['raw_sha256'],
        checkpoint_sha256=START_SHA,model_calls=0,metadata=source['metadata'],
        timing=dict(project_wall_seconds=wall_seconds,**log['seconds']),
        source_generation_seconds=source.get('source_generation_seconds',source.get('metadata',{}).get('seconds')))
    atomic_json(directory/'result.json',value)
    return value


def run(out,budget,workers=2):
    out=Path(out)
    from .project import FIXED_CONFIG
    jobs=[r for r in read_json(out/'requests.json') if r['label']=='ENDPOINT']
    assert len(jobs)==16
    remaining=[];rows=[]
    for row in jobs:
        p=result_path(out,row)
        if p.exists():
            value=read_json(p)
            if value['status']!='PROJECTED':raise RuntimeError('Unresolved or failed endpoint cannot be silently retried')
            assert digest(resolve(value['raw_path']))==value['raw_sha256']
            assert digest(resolve(value['intervention_log_path']))==value['intervention_log_sha256']
            rows.append(value)
        else:remaining.append(row)
    queue=iter(remaining);pending={}
    tick=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def dispatch():
            try:row=next(queue)
            except StopIteration:return False
            source=read_json(result_path(out,row,'RAW'))
            assert digest(resolve(source['raw_path']))==source['raw_sha256']
            raw=arrays(source['raw_path'])['raw']
            assert digest(resolve(row['input_npz']))==row['input_sha256']
            bank=arrays(row['input_npz']);K=int(row['K'])
            assert raw.dtype==np.float32 and raw.shape==(int(row['N']),9)
            assert raw[:K].tobytes()==bank['C'].reshape(K,9).tobytes()
            token=budget.begin('projection_endpoint',dict(request_id=row['request_id']))
            atomic_json(result_path(out,row),dict(row,status='RUNNING',projection_call_id=token))
            fut=pool.submit(_project_job,bank['C'].reshape(-1,3,3),raw[K:].reshape(-1,3,3),dict(FIXED_CONFIG))
            pending[fut]=(row,raw,bank,token,time.perf_counter())
            return True
        for _ in range(workers):dispatch()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                row,raw,bank,token,started=pending.pop(future)
                try:
                    corrected,log=future.result()
                    value=persist(out,row,raw,bank,corrected,log,time.perf_counter()-started)
                except BaseException as exc:
                    budget.end(token,exc);budget.snapshot();raise
                budget.end(token);rows.append(value);budget.snapshot()
                print(dict(event='ENDPOINT_SAVED',completed=len(rows),request_id=row['request_id'],status=value['correction_status'],accepted=value['accepted_count']),flush=True)
                dispatch()
    ordered={r['request_id']:r for r in rows}
    result=dict(status='COMPLETE',outputs=16,modified=sum(r['changed'] for r in rows),
        accepted_edges=sum(r['accepted_count'] for r in rows),CPU_workers=workers,threads_per_worker=2,
        seconds=time.perf_counter()-tick,all_results_saved_before_GT_evaluation=True,
        requests=[ordered[r['request_id']] for r in jobs],model_calls=0)
    atomic_json(out/'endpoint_summary.json',result)
    return {k:v for k,v in result.items() if k!='requests'}
