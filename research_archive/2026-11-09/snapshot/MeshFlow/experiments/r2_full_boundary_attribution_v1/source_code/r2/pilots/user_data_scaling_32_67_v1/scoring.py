"""New panel adapter for unchanged original quality and boundary mathematics.

No network calls. RAW and source-oriented POST retain separate score records.
All source files and original scoring results remain read-only.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, concurrent.futures, multiprocessing, resource, sys, time, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from r2.pilots.user_data_scaling_32_67_v1 import evaluation as E

VERSION='r2_user_scaling_32_67_original_metrics_adapter_v1'
ILLEGAL={'proper_transverse_penetration','coplanar_positive_area_overlap'}

def helpers():
    A=E.module('_scaling_boundary_original',E.ROOT/'r2/tools/avmr_pmr_v1/run.py')
    G=E.module('_scaling_graph_original',E.ROOT/'r2/pilots/anchor_aware_denoiser_v1/evaluation.py')
    H=E.module('_scaling_graph_opposed_original',E.ROOT/'r2/tools/c_restore_coupling_v1/evaluation.py')
    Q=E.module('_scaling_quality_original',E.ROOT/'r2/tools/postprocess_avmr_pmr_v1/quality.py')
    return A,G,H,Q

def score_one(job):
    E.deadline();q,r,post,query,stage,registration_sha=job
    dest=E.OUT/'evaluation/scores'/r['label']/q['request_id']/stage;rp=dest/'score.json'
    path=r['raw_path'] if stage=='RAW' else post['post_path']
    identity=dict(version=VERSION,label=r['label'],request_id=q['request_id'],stage=stage,mesh_path=path,mesh_sha256=E.sha(path),
        input_sha256=q['input_sha256'],query_sha256=query['sha256'],registration_sha256=registration_sha)
    if rp.exists():
        v=E.read(rp)
        if v['identity']!=identity:raise ValueError('Scoring identity differs')
        E.verify(v['files']);return dict(path=str(rp),reused=True,seconds=0,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if dest.exists():raise RuntimeError('Incomplete scoring requires explicit engineering recovery: '+str(dest))
    E.write(dest/'started.json',identity);tick=time.perf_counter()
    try:
        A,G,H,Q=helpers();mod=E.old().cpu_math();b=E.old().read_input(q);C=b['C'];K=int(q['K']);a=E.arrays(path)
        tri=np.asarray(a['raw'] if stage=='RAW' else a['output']).reshape(-1,3,3);src,source=E.source(q,C);F=tri[K:]
        if tri.dtype!=np.float32 or not np.isfinite(tri).all():raise ValueError('Nonfinite/non-FP32 mesh')
        if not E.same(tri[:K],C if stage=='RAW' else source['C_original']):raise ValueError('C coordinate/order contract fails')
        if len(F):metrics,cf,ff=mod['geometry'].scan_geometry(C,F,E.L)
        else:
            cf,ff=Q._empty_scans(K)
            metrics={name+'_'+field:value for name in ('CF','FF') for field,value in (
                ('proper_pair_count',0),('coplanar_pair_count',0),('proper_affected_free_fraction',None),('unresolved_or_degenerate_pair_count',0))}
            metrics.update(FF_illegal_pair_count=0,FF_illegal_affected_free_fraction=None)
        reference=E.old().load_queries(query);trusted=mod['control'].trusted_cf(C,F,E.L,cf_scan=cf)
        im,iv=mod['control'].conflict_metrics(F,{'C':C,'L0':E.L},reference,trusted);metrics.update(im)
        if abs(im['ConflictCov_C']-im['ConflictCov_C_cross']-im['ConflictCov_C_dup']+im['ConflictCov_C_cross_and_dup'])>1e-12:raise ValueError('I union identity fails')
        cf_free={row['pair'][1] for row in cf['contacts'] if row['classification'] in ILLEGAL}
        metrics.update(CF_illegal_pair_count=sum(row['classification'] in ILLEGAL for row in cf['contacts']),
            CF_illegal_affected_free_fraction=len(cf_free)/len(F) if len(F) else None)
        shape={name:mod['shape'].shape_quality(mesh,E.L) for name,mesh in (('C',C),('free',F),('whole',tri))}
        fq=shape['free'];metrics.update(free_minimum_angle_lt5_fraction=fq['min_angle_lt5_fraction'],
            free_q_gt10_fraction=fq['q_actual_gt10_fraction'],free_near_degenerate_count=fq['near_degenerate_area_count'],
            free_exact_zero_area_count=fq['exact_zero_area_count'],free_total_surface_area=fq['total_area'],
            actual_free_faces=len(F),deleted_free_faces=int(q['N'])-len(tri),C_total_surface_area=shape['C']['total_area'],
            free_exact_duplicate_excess_fraction=mod['geometry'].topology(tri,K)['free_exact_duplicate_excess_fraction'])
        metrics.update(mod['geometry'].topology(tri,K))
        metrics.update(whole_minimum_angle_lt5_fraction=shape['whole']['min_angle_lt5_fraction'],whole_near_degenerate_count=shape['whole']['near_degenerate_area_count'],whole_exact_zero_area_count=shape['whole']['exact_zero_area_count'],whole_total_surface_area=shape['whole']['total_area'],C_near_degenerate_count=shape['C']['near_degenerate_area_count'],C_exact_zero_area_count=shape['C']['exact_zero_area_count'])
        pairs={'CF':cf,'FF':ff};E.write(dest/'pairs.json',pairs);E.save_npz(dest/'vectors.npz',iv);E.write(dest/'shape_per_face.json',shape)
        # Scores use canonical C only to preserve the old query owner/normal
        # identity. Its triangle point set is exactly the final source C set.
        if stage=='RAW':
            flat=np.ascontiguousarray(tri).reshape(-1,3)
            _,first,inverse=np.unique(flat.view(np.uint32).reshape(-1,3),axis=0,return_index=True,return_inverse=True)
            vertices=flat[first];faces=inverse.reshape(-1,3);canonical_graph,_,_=G.actual_valid_graph(vertices,faces,K,pairs)
            faces=faces.copy();faces[:K]=np.take_along_axis(faces[:K],source['P_inverse'],axis=1)
            E.package();from meshflow_control.winding_solver import orient_fixed_C
            orientation=orient_fixed_C(vertices,faces,K);faces=orientation['faces'];oriented_tri=vertices[faces]
            E.save_npz(dest/'RAW_source_direction.npz',dict(vertices=vertices,faces=faces,C=oriented_tri[:K],output=oriented_tri,flips=orientation['flips']))
            from meshflow_control.source_winding import _C_vertex_ID_topology
            source_topology=_C_vertex_ID_topology(source['C_original_source_vertex_ids'],faces[:K],source['has_real_source_vertex_ids'])
            if source_topology['status']=='CHANGED_BY_GEO_C':source_topology['status']='CHANGED_BY_EXACT_COORDINATE_INDEX'
            source_status=source_topology['status']
            source_topology['domain']='RAW_EXACT_UINT32_COORDINATE_INDEX; no tolerance welding or GEO_C'
            E.write(dest/'orientation.json',{k:v for k,v in orientation.items() if not isinstance(v,np.ndarray)})
        else:
            vertices,faces=a['vertices'],a['faces'];oriented_tri=tri;orientation=E.read(E.resolve(post['post_path']).parent/'orientation.json')
            source_topology=post['source_restoration']['source_C_vertex_ID_topology'];source_status=source_topology['status'];canonical_graph=None
        graph,edges,reached=G.actual_valid_graph(vertices,faces,K,pairs);graph.update(H.opposite_reach(oriented_tri,K,edges))
        graph['C_F_has_opposite_winding_edge']=int(graph['C_F_valid_opposite_winding_edge_count']>0)
        graph['domain']='SOURCE_C_RESTORED_FREE_ORIENTED_'+('EXACT_COORDINATE_RAW_INDEX' if stage=='RAW' else 'ACTUAL_POST_INDEX')
        graph['direction_solver_summary']=orientation['summary'];E.write(dest/'topology.json',graph);E.csvwrite(dest/'valid_edges.csv',edges)
        ids,points,ba=A.topology(C,source['C_source_vertex_ids']);dist,nearest=A.nearest(points,F)
        vrows=[dict(source_vertex_id=int(v),x=float(p[0]),y=float(p[1]),z=float(p[2]),distance_model=float(d) if np.isfinite(d) else None,
            distance_status='FINITE' if np.isfinite(d) else 'POSITIVE_INFINITY_EMPTY_FREE',distance_L_fraction=float(d/E.L) if np.isfinite(d) else None,
            nearest_free_face=int(n)//3 if n>=0 else None,nearest_free_corner=int(n)%3 if n>=0 else None) for v,p,d,n in zip(ids,points,dist,nearest)]
        vm=[dict(setting=n,threshold_model=t,scale=s,tau=v,domain='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY',**A.metric(dist,t)) for n,t,s,v in A.SETTINGS]
        E.csvwrite(dest/'boundary_vertices.csv',vrows);E.csvwrite(dest/'AVMR_PMR_all8.csv',vm)
        distance={key:(float(fn(dist)) if len(dist) and np.isfinite(dist).all() else None) for key,fn in (
            ('boundary_distance_mean_model',np.mean),('boundary_distance_median_model',np.median),('boundary_distance_p95_model',lambda x:np.quantile(x,.95)),('boundary_distance_max_model',np.max))}
        distance['boundary_distance_status']='NA_EMPTY_BOUNDARY' if not len(dist) else 'POSITIVE_INFINITY_EMPTY_FREE' if not len(F) else 'PASS'
        result={k:q[k] for k in E.META};result.update(label=r['label'],stage=stage,status='PASS',identity=identity,metrics=metrics,topology=graph,
            canonical_RAW_topology=canonical_graph,boundary=vm,boundary_audit=ba,distance=distance,source_C_connectivity_status=source_status,source_C_connectivity_audit=source_topology,
            C_preserved_exact=True,C_query_count=2048,C_query_weight_denominator=float(reference['arrays']['C_weights'].sum()),free_face_denominator=len(F),
            CF_tested_pair_denominator=K*len(F),FF_tested_pair_denominator=len(F)*(len(F)-1)//2,source_hashes=E.old().scorer_sources(),
            quality_C_orientation='canonical C metadata retained only for exact old query contract; same unordered final source C geometry',
            status_by_metric=dict(free_fractions='PASS' if len(F) else 'NA_EMPTY_FREE',I=im['ConflictCov_C_status'],S='NA_NO_REQUIRED_INTERFACE',
                E='NA_NO_EXPOSURE',GT='NA_NOT_UNIQUE_GENERATIVE_TARGET',semantic_use='AI_PROXY_REVIEW_SEPARATE',support_damage='AI_PROXY_REVIEW_SEPARATE'),
            full_quality_scans=1,new_queries=0,model_calls=0,seconds=time.perf_counter()-tick,
            files={str(p):E.sha(p) for p in dest.iterdir() if p.is_file()})
        E.write(rp,result)
        if 'torch' in sys.modules:raise RuntimeError('CPU score imported model framework')
        return dict(path=str(rp),reused=False,seconds=result['seconds'],peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException as exc:
        E.write(dest/'failure.json',dict(status='STOPPED',identity=identity,error=repr(exc),traceback=traceback.format_exc()));raise

def run():
    reg=E.load_registration();raw=E.collect_raw()['rows'];post=E.read(E.OUT/'evaluation/post_index.json')['rows'];posts={(r['label'],r['request_id']):r for r in post}
    qs={q['request_id']:q for q in reg['requests']};queries={q['condition_id']:q for q in reg['queries']};rh=E.sha(E.OUT/'evaluation/registration.json')
    jobs=[(qs[r['request_id']],r,posts[(r['label'],r['request_id'])],queries[r['condition_id']],stage,rh) for r in raw for stage in ('RAW','POST')]
    if len(jobs)>840:raise RuntimeError('Stage scoring budget exceeded')
    E.package();from meshflow_control.denoise import resources
    E.write(E.OUT/'evaluation/score_resources_before.json',resources());tick=time.perf_counter()
    first=score_one(jobs[0]);res=resources();mem={x.split(':')[0]:int(x.split()[1])*1024 for x in res['memory'].splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    workers=2 if (os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and (res['loadavg'] or [999])[0]<(os.cpu_count() or 1)-2 else 1
    E.write(E.OUT/'evaluation/score_dispatch.json',dict(workers=workers,threads_per_worker=2,first_authorized_task=first,resources=res,reason='Measured first actual score, RAM/CPU guard; no extra benchmark'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=[first,*pool.map(score_one,jobs[1:],chunksize=1)]
    E.write(E.OUT/'evaluation/score_completion.json',dict(status='COMPLETE',stages=len(jobs),receipts=receipts,seconds=time.perf_counter()-tick,workers=workers))
    return receipts

def aggregate():
    reg=E.load_registration();complete=E.read(E.OUT/'evaluation/score_completion.json');outputs=[];vm=[];vertices=[];qualities=[];topologies=[];qs={q['request_id']:q for q in reg['requests']}
    for rec in complete['receipts']:
        s=E.read(rec['path']);m={k:s[k] for k in (*E.META,'label','stage')};q=qs[s['request_id']]
        qualities.append(dict(m,**s['metrics'],I_status=s['status_by_metric']['I'],free_face_denominator=s['free_face_denominator'],C_query_weight_denominator=s['C_query_weight_denominator']))
        topologies.append(dict(m,**{k:v for k,v in s['topology'].items() if not isinstance(v,(dict,list))}))
        vals={**s['metrics'],**{k:v for k,v in s['topology'].items() if isinstance(v,(int,float)) or v is None},**s['distance']}
        for b in s['boundary']:
            vm.append(dict(m,**b));vals['A_VMR__'+b['setting']]=b['VMR'];vals['PMR__'+b['setting']]=b['perfect']
        vals.update(C_preserved_exact=s['C_preserved_exact'],source_C_connectivity_status=s['source_C_connectivity_status'])
        outputs.append(dict(m,**vals,structure_tags=q.get('structure_tags','UNKNOWN'),source_quality=q.get('source_quality','UNKNOWN'),
            N_bin='128_191' if q['N']<192 else '192_256',K_ratio_bin='le0.1' if q['ratio']<=.1 else 'le0.25' if q['ratio']<=.25 else 'gt0.25'))
        with (Path(rec['path']).parent/'boundary_vertices.csv').open(encoding='utf-8-sig',newline='') as f:
            import csv
            vertices.extend(dict(m,**x) for x in csv.DictReader(f))
    for name,rows in [('quality_per_output',qualities),('topology_per_output',topologies),('AVMR_PMR_all8',vm),('boundary_vertices',vertices),('per_output',outputs)]:E.csvwrite(E.OUT/'evaluation'/f'{name}.csv',rows)
    metrics=sorted({k for r in outputs for k,v in r.items() if (isinstance(v,(int,float)) or v is None) and k not in E.META})
    paired=[];small={(r['request_id'],r['stage']):r for r in outputs if r['label']=='USER_SMALL'}
    for r in outputs:
        if r['label']=='USER_SMALL':continue
        base=small[(r['request_id'],r['stage'])]
        for metric in metrics:
            a,b=r.get(metric),base.get(metric);valid=isinstance(a,(int,float)) and isinstance(b,(int,float))
            paired.append(dict(**{k:r[k] for k in (*E.META,'stage')},comparison=r['label']+'-USER_SMALL',metric=metric,
                baseline_value=b,candidate_value=a,delta=a-b if valid else None,status='PASS' if valid else 'NA_OR_MISSING',
                structure_tags=r['structure_tags'],source_quality=r['source_quality'],N_bin=r['N_bin'],K_ratio_bin=r['K_ratio_bin']))
    E.csvwrite(E.OUT/'evaluation/paired_per_request.csv',paired)
    def reduce(rows,keys):
        groups=collections.defaultdict(list)
        for row in rows:groups[tuple(row[k] for k in keys)].append(row)
        result=[]
        for key,rr in sorted(groups.items()):
            good=[r for r in rr if r['status']=='PASS'];row=dict(zip(keys,key));row.update(n_registered=len(rr),n_valid=len(good),status='PASS' if len(good)==len(rr) else 'PARTIAL_OR_NA')
            for field in ('baseline_value','candidate_value','delta'):row[field]=float(np.mean([r[field] for r in good])) if good else None
            row.update(delta_min=min((r['delta'] for r in good),default=None),delta_max=max((r['delta'] for r in good),default=None));result.append(row)
        return result
    conditions=reduce(paired,('comparison','panel','stage','parent_id','condition_id','region_type','tier','metric'))
    parents=reduce(conditions,('comparison','panel','stage','parent_id','metric'))
    E.csvwrite(E.OUT/'evaluation/paired_per_condition.csv',conditions);E.csvwrite(E.OUT/'evaluation/paired_per_parent.csv',parents)
    summary=[]
    for (comparison,panel,stage,metric),rr in __import__('itertools').groupby(sorted(parents,key=lambda r:tuple(r[k] for k in ('comparison','panel','stage','metric'))),key=lambda r:tuple(r[k] for k in ('comparison','panel','stage','metric'))):
        rr=list(rr);good=[r for r in rr if r['delta'] is not None];ds=np.asarray([r['delta'] for r in good]);
        summary.append(dict(comparison=comparison,panel=panel,stage=stage,metric=metric,parents=len(rr),valid_parents=len(good),complete_parents=sum(r['status']=='PASS' for r in rr),
            mean_delta=float(ds.mean()) if len(ds) else None,median_delta=float(np.median(ds)) if len(ds) else None,
            negative_delta_parents=int((ds<0).sum()),equal_delta_parents=int((ds==0).sum()),positive_delta_parents=int((ds>0).sum()),
            minimum_delta=float(ds.min()) if len(ds) else None,maximum_delta=float(ds.max()) if len(ds) else None,
            caveat='Numerical directions only; favorable direction belongs to metric contract; one training stream'))
    E.csvwrite(E.OUT/'evaluation/paired_summary.csv',summary)
    # Type/tier and fixed N/K strata preserve parent equal weighting within each slice.
    stratified=[]
    for field in ('region_type','tier','N_bin','K_ratio_bin','source_quality','structure_tags'):
        groups=collections.defaultdict(list)
        for row in paired:
            values=row[field] if isinstance(row[field],list) else [row[field]]
            for value in values:groups[str(value)].append(row)
        for value,rr in groups.items():
            cc=reduce(rr,('comparison','panel','stage','parent_id','condition_id','metric'));pp=reduce(cc,('comparison','panel','stage','parent_id','metric'))
            ss=reduce(pp,('comparison','panel','stage','metric'))
            stratified.extend(dict(stratum=field,stratum_value=value,**row) for row in ss)
    E.csvwrite(E.OUT/'evaluation/paired_strata.csv',stratified)
    E.write(E.OUT/'evaluation/aggregate_completion.json',dict(status='COMPLETE',stage_records=len(outputs),parent_pairs=len(parents),metrics=len(metrics),
        hierarchy='noise within C, C within parent, parent equal; OLD6 separate',visual_review_not_inferred=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['score','aggregate'],required=True);a=p.parse_args()
    if a.stage=='score':run()
    else:aggregate()
if __name__=='__main__':main()
