"""Incremental CPU evaluation against immutable input-only references."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os,time,traceback
import numpy as np
from .io import *
from . import control,legacy_quality,quality,registry
from .surface import sample_mesh_surface,surface_distance_squared
from .geometry import scan_geometry,topology
VERSION='direct_context_readout_evaluation_v1'
def sources():return {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
def _data(request):
    if digest(request['input_path'])!=request['input_sha256'] or digest(request['reference_path'])!=request['reference_sha256']:raise ValueError('Frozen input/reference identity differs')
    bank=arrays(request['input_path']);meta=read_json(request['reference_path'])
    if digest(meta['condition_path'])!=meta['condition_sha256']:raise ValueError('Condition changed')
    c=arrays(meta['condition_path']);c.update(condition_id=request['condition_id'],gamma_status=meta.get('gamma_status'))
    if int(bank['N'])!=request['N'] or int(bank['K'])!=request['K'] or int(c['N'])!=request['N'] or int(c['K'])!=request['K']:raise ValueError('Registered N/K differs from assets')
    if array_sha(bank['C'])!=array_sha(c['C']) or bank['z'].shape!=(1,request['N'],9):raise ValueError('C or Gaussian shape differs')
    if any(k in bank for k in ('GT_full','GT_free','gamma_edges','L0','source_face_ids')):raise ValueError('Inference bank contains evaluation-only data')
    return bank,meta,c
def open_reference(c,cid,folder=None):
    """Original raw I point sampling only, with no target or visibility construction."""
    q,face,w,bary,s=control.sample_C(c['C'],cid,2048);normal,good,_=control.normals(c['C'],float(c['L0']))
    border,audit=control.open_edges(c['C'])
    from .seams import nearest_segments
    d=np.sqrt(nearest_segments(q,border)[0]) if len(border) else np.full(len(q),np.inf)
    ref=dict(arrays=dict(C_points=q,C_point_faces=face,C_weights=w,C_barycentric=bary,C_point_normals=normal[face],
        C_point_normal_valid=good[face],C_open_edges=border,C_border_distance=d),identity=dict(condition_id=cid,C_seed=s,C_sha256=array_sha(c['C']),L0=float(c['L0'])))
    if folder:
        p=resolve(folder)/'I_reference.npz';m=p.with_suffix('.json')
        save_npz(p,ref['arrays']);write_json(m,dict(identity=ref['identity'],npz_sha256=digest(p),arrays_sha256={k:array_sha(v) for k,v in ref['arrays'].items()}),freeze=True)
    return ref
def _control_ref(meta):
    assets=meta['assets']
    for name in ('control.json','control.npz','quality.json','quality.npz'):
        if digest(assets[name]['path'])!=assets[name]['sha256']:raise ValueError('Scoring reference changed')
    r=read_json(assets['control.json']['path'])['reference'];r['arrays']=arrays(assets['control.npz']['path'])
    return r,arrays(assets['quality.npz']['path'])
def _contracts(r,raw,bank,result):
    N,K=r['N'],r['K'];arr=raw.get('raw',raw.get('output'));tri=np.asarray(arr).reshape(-1,3,3)
    path=raw.get('trajectory',raw.get('path'));C=bank['C']
    z=raw.get('z',raw.get('initial_gaussian'))
    values=dict(C_output_bitwise=tri.dtype==C.dtype and tri[:K].tobytes()==C.tobytes(),N_preserved=len(tri)==N,
        finite=bool(np.isfinite(tri).all()),C_all51_bitwise=None if path is None else bool(np.asarray(path).shape==(51,N,9) and all(np.asarray(x)[:K].tobytes()==C.reshape(K,9).tobytes() for x in path)),
        actual_Gaussian_bitwise=None if z is None else array_sha(np.asarray(z))==array_sha(bank['z']))
    for name in ('NPZ_C_bitwise','PLY_C_bitwise'):
        values[name]=result.get(name,result.get('contracts',{}).get(name,result.get('export',{}).get(name,result.get('export_npz_ply_bitwise'))))
    return tri,values
def score_request(request,out,recompute=False):
    tick=time.perf_counter();r=dict(request);label=r['label'];dest=resolve(out)/'scoring'/label/r['request_id'];dest.mkdir(parents=True,exist_ok=True)
    bank,meta,c=_data(r);L=float(c['L0']);N,K=int(r['N']),int(r['K'])
    rawpath=resolve(r['raw_path']);rh=digest(rawpath)
    if r.get('raw_sha256',r.get('raw_hash',rh))!=rh:raise ValueError('RAW hash changed')
    identity=dict(version=VERSION,request_id=r['request_id'],label=label,raw_sha256=rh,input_sha256=r['input_sha256'],reference_sha256=r['reference_sha256'],sources=sources())
    jp=dest/'score.json'
    if jp.exists() and not recompute:
        saved=read_json(jp)
        if saved['identity']!=identity:raise ValueError('Completed scoring identity changed')
        for p,h in saved.get('files',{}).items():
            if digest(p)!=h:raise ValueError('Scoring vector changed')
        return saved
    result=read_json(r['result_path']) if r.get('result_path') else {};raw=arrays(rawpath);tri,facts=_contracts(r,raw,bank,result);F=tri[K:]
    row={k:r[k] for k in ('request_id','label','panel','cohort','condition_id','parent_id','candidate','N','K')}
    row.update(identity=identity,L0=L,raw_path=str(rawpath),status='PASS',contracts=facts,metric_status={},files={},metrics={})
    if any(v is False for v in facts.values()) or not len(F):
        row.update(status='INVALID_OUTPUT',reason='Hard contract failed or empty free set');write_json(jp,row);return row
    # Trusted historical START is copied, never re-scored by default.
    h=r.get('historical_start')
    if label=='START' and h and not recompute:
        if digest(h['score_path'])!=h['score_sha256'] or digest(h['raw_path'])!=h['raw_sha256']:raise ValueError('Historical START cache changed')
        old_raw=arrays(h['raw_path'])
        current_arrays=dict(raw=tri.reshape(N,9),trajectory=np.asarray(raw.get('trajectory',raw.get('path'))),C=np.asarray(raw['C']).reshape(K,3,3),z=np.asarray(raw['z']))
        historic_arrays=dict(raw=np.asarray(old_raw.get('raw',old_raw.get('output'))).reshape(N,9),trajectory=np.asarray(old_raw.get('trajectory',old_raw.get('path'))),C=np.asarray(old_raw['C']).reshape(K,3,3),z=np.asarray(old_raw['z']))
        equal={k:array_sha(v)==array_sha(historic_arrays[k]) for k,v in current_arrays.items()}
        if not all(equal.values()):raise ValueError('START repack changed raw/trajectory/C/Gaussian arrays')
        row['historical_repack_audit']=dict(status='PASS',historical_file_sha256=h['raw_sha256'],current_file_sha256=rh,arrays_bitwise_equal=equal,canonical_raw_shape=[N,9],canonical_C_shape=[K,3,3],scientific_values_recomputed=0)
        old=read_json(h['score_path']);m=dict(old['metrics'])
        exclude={'K','N','L0','candidate','noise_seed','model_calls','generation_seconds','adapter_seconds','scoring_seconds','context_level','attempted','known_area_ratio','known_face_ratio','known_span_L0'}
        row['metrics']={k:v for k,v in m.items() if k not in exclude and (isinstance(v,(int,float,bool)) or v is None)}
        for name in ('SeamCov','ConflictCov_C','ExposedRetain'):row['metric_status'][name]=m.get(name+'_status','PASS')
        row['cache']='VERIFIED_HISTORICAL_START';row['scientific_scores_recomputed']=0
        vec=arrays(h['quality_vectors_path']);cv=arrays(h['control_vectors_path'])
    else:
        m={};vec={};cv={}
        if r['panel']=='P1':
            ref,qb=_control_ref(meta)
            legacyref=legacy_quality.prepare_reference(c['C'],c['GT_free'],L,gamma_points=c['legacy_gamma_points'],gamma_weights=c['legacy_gamma_weights'])
            for source,key in [('legacy_query_points','coverage_points'),('legacy_pairwise_points','pairwise_target_points')]:
                legacyref['arrays'][key]=c[source].copy();legacyref['audit']['array_sha256'][key]=array_sha(c[source])
            old=legacy_quality.score_free(F,c['C'],c['GT_free'],L,reference=legacyref)
            if old['status']!='PASS':raise ValueError('Original scorer rejected free surface: '+str(old.get('failure_reason')))
            m.update(old['metrics']);cf=old['detail']['CF']
            ds=np.asarray(old['detail']['fixed_GT_coverage']['per_point_distances'])
            m.update(fixed_query_MSE=float(np.mean((ds/L)**2)),coverage_fraction=float(np.mean(ds<=.01*L)),fixed_query_p95_L=float(np.quantile(ds/L,.95)))
            qf=sample_mesh_surface(F,2048,np.random.RandomState(quality.seed('Q_F',r['condition_id'],r['candidate'],2048)))
            dy=surface_distance_squared(qb['Q_Y2048'],F);df=surface_distance_squared(qf,c['GT_free'])
            m.update(quality.pr_metrics(dy,df,L));m.update(quality.angle_reference(c['GT_free'],L))
            m.update({k.replace('GT_free','predicted_free'):v for k,v in quality.angle_reference(F,L).items()})
            gm,gv=quality.interface_metrics(F,c['gamma_edges'],L);m.update(gm)
            value=control.score_control(F,c,ref,cf_scan=cf);m.update(value['metrics']);cv=value['arrays']
            vec.update(Q_F2048=qf,GT_to_F_d2_2048=dy,F_to_GT_d2_2048=df,fixed_query_distance512=ds)
            vec.update({'Gamma32_'+k:v for k,v in gv.items()})
            write_json(dest/'legacy_detail.json',old)
            for field,prefix in [('SeamCov','SeamCov'),('ConflictCov_C','ConflictCov_C'),('ExposedRetain','ExposedRetain')]:
                row['metric_status'][field]=value['statuses'][prefix]
        else:
            m,cf,ff=scan_geometry(c['C'],F,L)
            m.update({k.replace('GT_free','predicted_free'):v for k,v in quality.angle_reference(F,L).items()})
            ref=open_reference(c,r['condition_id'],resolve(r['reference_path']).parent)
            v,cv=control.conflict_metrics(F,c,ref,control.trusted_cf(c['C'],F,L,cf_scan=cf));m.update(v)
            row['metric_status']['ConflictCov_C']=m['ConflictCov_C_status']
            qf=sample_mesh_surface(F,2048,np.random.RandomState(quality.seed('Q_F',r['condition_id'],r['candidate'],2048)))
            if qf is None:raise ValueError('No positive finite free surface')
            vec['Q_F2048']=qf
            for field in ('SeamCov','SeamCov_upper','ExposedRetain','ExposedRetain_upper','ExposedRetain_global_lower','ExposedRetain_global_upper',
                          'GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0','D_sym2048','fixed_query_MSE','fixed_query_p95_L','coverage_fraction'):
                m[field]=None;row['metric_status'][field]='NA_NOT_APPLICABLE'
            for field in legacy_quality.METRIC_SPECS:
                if registry.describe(field,'P2',N,K,L)['requires_GT']:m.setdefault(field,None);row['metric_status'][field]='NA_NOT_APPLICABLE'
            for name in ('Precision','Recall','Fscore'):
                for tau in quality.TAUS:m[f'{name}2048_{quality.tkey(tau)}']=None
            write_json(dest/'pair_audit.json',dict(CF=cf,FF=ff))
        row['metrics']=m;row['scientific_scores_recomputed']=1
    row['metrics'].update(topology(tri,K));row['metrics'].update({k:v for k,v in facts.items() if v is not None})
    for name,value in [('quality_vectors',vec),('control_vectors',cv)]:
        p=dest/(name+'.npz');save_npz(p,value);row['files'][str(p)]=digest(p);row[name+'_path']=str(p)
    row.update(scoring_seconds=time.perf_counter()-tick,threads=int(os.environ.get('OMP_NUM_THREADS','2')),
        source_result_path=r.get('result_path'),resource_timing=result.get('metadata',result.get('timing',{})))
    write_json(jp,row);return row
def _worker(args):
    r,out=args
    os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    try:return score_request(r,out)
    except Exception as exc:
        p=resolve(out)/'errors'/r['label']/(r['request_id']+'.json')
        value={k:r.get(k) for k in ('request_id','label','panel','cohort','condition_id','parent_id','candidate','N','K')}
        value.update(status='SCORING_BLOCKED',reason=str(exc),traceback=traceback.format_exc(),metrics={},metric_status={},L0=None)
        write_json(p,value);return value
def evaluate(manifest,out,workers=2):
    rows=load_manifest(manifest);workers=max(1,min(int(workers),4));tick=time.perf_counter()
    if any('raw_path' not in r or 'label' not in r for r in rows):raise ValueError('Completed result manifest requires label and raw_path')
    if workers==1:results=[_worker((r,out)) for r in rows]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(_worker,[(r,out) for r in rows]))
    write_json(resolve(out)/'per_output.json',dict(schema=VERSION,rows=results))
    good=[r for r in results if 'metrics' in r]
    flat=[{**{k:r[k] for k in ('request_id','label','panel','cohort','condition_id','parent_id','candidate','N','K','status')},**r['metrics']} for r in good]
    write_csv(resolve(out)/'per_output_metrics.csv',flat);write_csv(resolve(out)/'metric_registry_results.csv',[v for r in good for v in registry.records(r)])
    from .aggregate import aggregate
    aggregation=aggregate(results,out)
    from .diversity import diversity
    diversity(good,out)
    summary=dict(status='COMPLETE' if len(good)==len(rows) and all(r['status']=='PASS' for r in good) else 'PARTIAL',
        requested=len(rows),scored=sum(r.get('status')=='PASS' for r in good),blocked=sum(r.get('status')=='SCORING_BLOCKED' for r in results),invalid=sum(r.get('status')=='INVALID_OUTPUT' for r in results),workers=workers,wall_seconds=time.perf_counter()-tick,aggregation=aggregation)
    write_json(resolve(out)/'evaluation_summary.json',summary);return summary
