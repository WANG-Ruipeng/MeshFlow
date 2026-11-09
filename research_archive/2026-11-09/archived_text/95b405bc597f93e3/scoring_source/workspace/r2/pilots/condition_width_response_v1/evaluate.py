"""CPU adapter for the existing metrics on 48 registered width requests.

Scientific functions are inherited unchanged. Only new input/mesh/query
identities, the 96-record budget and paired aggregation are adapted here.
Run this module as a separate CPU process, never in a generator process.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, concurrent.futures, csv, itertools, json
import multiprocessing, resource, sys, time, traceback
from fractions import Fraction
from pathlib import Path
import numpy as np
from . import common as E

VERSION='r2_width_response_original_metrics_adapter_v1'
LABEL='EXP67_FROZEN'
# Names inherited from the archived 32/67 metric dictionary; no result ranking read.
ORIGINAL_SCALARS=('A_VMR__L_tau_0.0001', 'A_VMR__L_tau_0.001', 'A_VMR__L_tau_0.002', 'A_VMR__L_tau_0.005', 'A_VMR__L_tau_0.01', 'A_VMR__L_tau_0.02', 'A_VMR__L_tau_1e-06', 'A_VMR__raw_tau_1e-6', 'CF_coplanar_pair_count', 'CF_illegal_affected_free_fraction', 'CF_illegal_pair_count', 'CF_proper_affected_free_fraction', 'CF_proper_pair_count', 'CF_unresolved_or_degenerate_pair_count', 'C_F_has_opposite_winding_edge', 'C_F_has_valid_edge', 'C_F_nonmanifold_edge_count', 'C_F_rejected_duplicate_edge_count', 'C_F_rejected_illegal_edge_count', 'C_F_rejected_unknown_edge_count', 'C_F_shared_edge_count', 'C_F_shared_edge_total_length', 'C_F_valid_edge_count', 'C_F_valid_edge_total_length', 'C_F_valid_opposite_winding_edge_count', 'C_exact_zero_area_count', 'C_near_degenerate_count', 'C_preserved_exact', 'C_total_surface_area', 'ConflictCov_C', 'ConflictCov_C_coplanar_illegal_pairs', 'ConflictCov_C_cross', 'ConflictCov_C_cross_and_dup', 'ConflictCov_C_dup', 'ConflictCov_C_interior_samples', 'ConflictCov_C_interior_weight_fraction', 'ConflictCov_C_proper_pairs', 'ConflictCov_C_segment_extraction_failures', 'ConflictCov_C_uncertain_fraction', 'ConflictCov_C_unresolved_pairs', 'ConflictCov_C_upper', 'EXACT_COORD_CF_shared_edges', 'EXACT_COORD_boundary_edges', 'EXACT_COORD_closed_two_faces_per_edge', 'EXACT_COORD_edges', 'EXACT_COORD_face_components', 'EXACT_COORD_nonmanifold_edges', 'EXACT_COORD_same_direction_two_face_edges', 'EXACT_COORD_vertices', 'FF_coplanar_pair_count', 'FF_illegal_affected_free_fraction', 'FF_illegal_pair_count', 'FF_proper_affected_free_fraction', 'FF_proper_pair_count', 'FF_unresolved_or_degenerate_pair_count', 'PMR__L_tau_0.0001', 'PMR__L_tau_0.001', 'PMR__L_tau_0.002', 'PMR__L_tau_0.005', 'PMR__L_tau_0.01', 'PMR__L_tau_0.02', 'PMR__L_tau_1e-06', 'PMR__raw_tau_1e-6', 'RAW_SOUP_CF_shared_edges', 'RAW_SOUP_face_components', 'actual_free_faces', 'boundary_distance_max_model', 'boundary_distance_mean_model', 'boundary_distance_median_model', 'boundary_distance_p95_model', 'boundary_edges', 'deleted_free_faces', 'exact_duplicate_faces', 'free_angle_p05_degrees', 'free_area', 'free_exact_duplicate_excess_fraction', 'free_exact_zero_area_count', 'free_exact_zero_area_fraction', 'free_minimum_angle_lt5_fraction', 'free_near_degenerate_count', 'free_near_degenerate_fraction', 'free_q_gt10_fraction', 'free_q_mean', 'free_total_surface_area', 'graph_valid_edges', 'nonmanifold_edges', 'opposite_winding_reached_free_area', 'opposite_winding_reached_free_area_fraction', 'opposite_winding_reached_free_faces', 'reached_free_area', 'reached_free_area_fraction', 'reached_free_faces', 'whole_exact_zero_area_count', 'whole_minimum_angle_lt5_fraction', 'whole_near_degenerate_count', 'whole_total_surface_area')
META=('request_id','parent_id','condition_id','panel','candidate','region_type','tier','N','K','ratio',
      'base_request_id','base_condition_id','variant','alpha')
REVIEW_FIELDS=('C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer',
              'whole_body_complete','support_missing','fragments_or_folded_layers','visible_join')
ILLEGAL={'proper_transverse_penetration','coplanar_positive_area_overlap'}

def old():
    return E.module('_width_original_cpu_evaluation',E.ROOT/'r2/pilots/user_condition_finetune_v1/evaluation.py')

def verify(mapping):
    for path,h in mapping.items():
        if E.sha(path)!=h:raise ValueError('Frozen file changed: '+str(path))

def load_source(row,C):
    verify({row['condition_source']:row['condition_source_sha256']})
    E.package()
    from meshflow_control.source_winding import validate_source
    a=E.arrays(row['condition_source'])
    return a,validate_source(a,C)

def definition_hashes():
    files=[Path(__file__),E.CODE/'common.py',E.ROOT/'r2/pilots/user_data_scaling_32_67_v1/scoring.py',
        E.ROOT/'r2/tools/avmr_pmr_v1/run.py',E.ROOT/'r2/pilots/anchor_aware_denoiser_v1/evaluation.py',
        E.ROOT/'r2/tools/c_restore_coupling_v1/evaluation.py',E.ROOT/'r2/tools/postprocess_avmr_pmr_v1/quality.py',
        E.ROOT/'r2/pilots/user_condition_finetune_v1/evaluation.py',
        E.ROOT/'r2/src/meshflow_control/source_winding.py',E.ROOT/'r2/src/meshflow_control/winding_solver.py']
    return {**old().scorer_sources(),**{str(p):E.sha(p) for p in files}}

def registration():
    reg=E.read(E.OUT/'registration.json')
    if reg['status']!='FROZEN' or reg['labels']!=[LABEL]:raise ValueError('Wrong/unfrozen experiment')
    rows=reg['requests']
    if len(rows)!=48 or len({q['request_id'] for q in rows})!=48:raise ValueError('Expected 48 unique requests')
    groups=collections.defaultdict(list)
    for q in rows:groups[q['base_request_id']].append(q)
    if len(groups)!=16 or len({q['parent_id'] for q in rows})!=8:raise ValueError('Wrong fixed panel')
    for rid,rr in groups.items():
        if len(rr)!=3 or {q['variant'] for q in rr}!={'m095','base','p105'}:raise ValueError('Variant panel changed')
        for key in ('base_condition_id','parent_id','candidate','N','K','Gaussian_sha256'):
            if len({q[key] for q in rr})!=1:raise ValueError('Pair identity differs: '+key)
        for q in rr:
            expected={'m095':np.float32(.95),'base':np.float32(1),'p105':np.float32(1.05)}[q['variant']]
            if q['alpha']!=float(expected):raise ValueError('Alpha definition differs')
    if len(reg['queries'])!=24 or {q['condition_id'] for q in reg['queries']}!={q['condition_id'] for q in rows}:
        raise ValueError('Query identities differ')
    verify(reg['preparation_source_hashes'])
    return reg

def indices(reg):
    expected={q['request_id'] for q in reg['requests']};result=[]
    for filename,key in (('raw_index.json','raw'),('post_index.json','post')):
        value=E.read(E.OUT/'evaluation'/filename);rows=value['rows']
        if value['status']!='COMPLETE' or len(rows)!=48 or {r['request_id'] for r in rows}!=expected:
            raise ValueError('All 48 RAW and all 48 POST must complete first')
        if any(r['label']!=LABEL for r in rows):raise ValueError('Wrong model label')
        if len({r['request_id'] for r in rows})!=48:raise ValueError('Duplicate mesh identity')
        for row in rows:verify({row[key+'_path']:row[key+'_sha256']})
        result.append({r['request_id']:r for r in rows})
    return result

def helpers():
    A=E.module('_width_boundary_original',E.ROOT/'r2/tools/avmr_pmr_v1/run.py')
    G=E.module('_width_graph_original',E.ROOT/'r2/pilots/anchor_aware_denoiser_v1/evaluation.py')
    H=E.module('_width_graph_opposed_original',E.ROOT/'r2/tools/c_restore_coupling_v1/evaluation.py')
    Q=E.module('_width_quality_original',E.ROOT/'r2/tools/postprocess_avmr_pmr_v1/quality.py')
    return A,G,H,Q

def score_one(job):
    q,r,post,query,stage,registration_sha=job
    dest=E.OUT/'evaluation/scores'/r['label']/q['request_id']/stage;rp=dest/'score.json'
    path=r['raw_path'] if stage=='RAW' else post['post_path']
    identity=dict(version=VERSION,label=r['label'],request_id=q['request_id'],stage=stage,mesh_path=path,mesh_sha256=E.sha(path),
        input_sha256=q['input_sha256'],query_sha256=query['sha256'],registration_sha256=registration_sha,condition_source_sha256=q['condition_source_sha256'],
        C_sha256=q['C_sha256'],Gaussian_sha256=q['Gaussian_sha256'],definition_hashes=definition_hashes())
    if rp.exists():
        v=E.read(rp)
        if v['identity']!=identity:raise ValueError('Scoring identity differs')
        verify(v['files']);return dict(path=str(rp),reused=True,seconds=0,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if dest.exists():raise RuntimeError('Incomplete scoring requires explicit engineering recovery: '+str(dest))
    E.write(dest/'started.json',identity);tick=time.perf_counter()
    try:
        if query['condition_id']!=q['condition_id'] or query['identity']['C_sha256']!=q['C_sha256']:
            raise ValueError('Transformed query/C identity mismatch; old-condition caches forbidden')
        A,G,H,Q=helpers();mod=old().cpu_math();b=old().read_input(q);C=b['C'];K=int(q['K']);a=E.arrays(path)
        tri=np.asarray(a['raw'] if stage=='RAW' else a['output']).reshape(-1,3,3);src,source=load_source(q,C);F=tri[K:]
        if tri.dtype!=np.float32 or not np.isfinite(tri).all():raise ValueError('Nonfinite/non-FP32 mesh')
        if not E.same(tri[:K],C if stage=='RAW' else source['C_original']):raise ValueError('C coordinate/order contract fails')
        if len(F):metrics,cf,ff=mod['geometry'].scan_geometry(C,F,E.L)
        else:
            cf,ff=Q._empty_scans(K)
            metrics={name+'_'+field:value for name in ('CF','FF') for field,value in (
                ('proper_pair_count',0),('coplanar_pair_count',0),('proper_affected_free_fraction',None),('unresolved_or_degenerate_pair_count',0))}
            metrics.update(FF_illegal_pair_count=0,FF_illegal_affected_free_fraction=None)
        reference=old().load_queries(query);trusted=mod['control'].trusted_cf(C,F,E.L,cf_scan=cf)
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
        # Scores use this variant's canonical C and transformed query metadata.
        # Its unordered triangles equal the source-oriented final C geometry.
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
        descriptive,descriptive_points=descriptive_distances(mod,points,F)
        E.write(dest/'descriptive_boundary_distance.json',descriptive)
        E.csvwrite(dest/'descriptive_boundary_distance_points.csv',descriptive_points)
        result={k:q[k] for k in META};result.update(label=r['label'],stage=stage,status='PASS',identity=identity,metrics=metrics,topology=graph,
            descriptive_boundary_distance=descriptive,
            canonical_RAW_topology=canonical_graph,boundary=vm,boundary_audit=ba,distance=distance,source_C_connectivity_status=source_status,source_C_connectivity_audit=source_topology,
            C_preserved_exact=True,C_query_count=2048,C_query_weight_denominator=float(reference['arrays']['C_weights'].sum()),free_face_denominator=len(F),
            CF_tested_pair_denominator=K*len(F),FF_tested_pair_denominator=len(F)*(len(F)-1)//2,source_hashes=identity['definition_hashes'],query_transform_audit=query['transform_audit'],
            query_measure='BASE_ORIGINAL_IID_AREA; variants fixed affine samples with self-normalized per-face area importance weights',
            quality_C_orientation='Variant-specific canonical C and paired transformed query; same unordered final source C geometry',
            status_by_metric=dict(free_fractions='PASS' if len(F) else 'NA_EMPTY_FREE',I=im['ConflictCov_C_status'],S='NA_NO_REQUIRED_INTERFACE',
                E='NA_NO_EXPOSURE',GT='NA_NOT_UNIQUE_GENERATIVE_TARGET',semantic_use='AI_PROXY_REVIEW_SEPARATE',support_damage='AI_PROXY_REVIEW_SEPARATE'),
            full_quality_scans=1,new_queries=0,model_calls=0,seconds=time.perf_counter()-tick,
            files={str(p):E.sha(p) for p in dest.iterdir() if p.is_file()})
        E.write(rp,result)
        if 'torch' in sys.modules:raise RuntimeError('CPU score imported model framework')
        return dict(path=str(rp),reused=False,seconds=result['seconds'],peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException as exc:
        E.write(dest/'failure.json',dict(status='STOPPED',identity=identity,error=repr(exc),traceback=traceback.format_exc()));raise


def run(max_workers=2):
    reg=registration();raw,post=indices(reg)
    queries={q['condition_id']:q for q in reg['queries']};rh=E.sha(E.OUT/'registration.json')
    jobs=[(q,raw[q['request_id']],post[q['request_id']],queries[q['condition_id']],stage,rh)
          for q in reg['requests'] for stage in ('RAW','POST')]
    if len(jobs)!=96:raise ValueError('Expected exactly 96 authorized stage scores')
    final=E.OUT/'evaluation/score_completion.json'
    if final.exists():
        prior=E.read(final)
        if prior['status']!='COMPLETE' or prior['stages']!=96 or prior['definition_hashes']!=definition_hashes():
            raise ValueError('Completed scoring definition differs')
        for rec in prior['receipts']:
            s=E.read(rec['path']);verify(s['files'])
            if s['identity']['registration_sha256']!=rh:raise ValueError('Registration changed')
            if E.sha(rec['path'])!=rec['sha256']:raise ValueError('Score record changed')
        return prior
    before=E.OUT/'evaluation/score_resources_before.json'
    if before.exists():before=E.OUT/'evaluation'/('score_resources_resume_'+str(time.time_ns())+'.json')
    E.resources(before);tick=time.perf_counter()
    first=score_one(jobs[0]);snapshot=E.resources()
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (snapshot.get('memory') or '').splitlines()
         if len(x.split())>1 and x.split()[1].isdigit()}
    can_parallel=(os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024)
    can_parallel=can_parallel and (snapshot['loadavg'] or [999])[0]<(os.cpu_count() or 1)-2
    workers=2 if max_workers==2 and can_parallel else 1
    dispatch=E.OUT/'evaluation/score_dispatch.json'
    if dispatch.exists():dispatch=E.OUT/'evaluation'/('score_dispatch_resume_'+str(time.time_ns())+'.json')
    E.write(dispatch,dict(workers=workers,threads_per_worker=2,
        first_authorized_task=first,resources=snapshot,reason='First actual score peak, RAM and CPU guards; no extra benchmark'))
    receipts=[first]
    def record_progress():
        E.write(E.OUT/'evaluation/score_progress.json',dict(status='RUNNING',completed=len(receipts),registered=96,
            workers=workers,seconds=time.perf_counter()-tick),replace=True)
    record_progress()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(score_one,job):i for i,job in enumerate(jobs[1:],1)}
        results={}
        try:
            for f in concurrent.futures.as_completed(futures):
                results[futures[f]]=f.result();receipts.append(results[futures[f]]);record_progress()
                print(json.dumps(dict(stage='SCORE',completed=len(receipts),registered=96)),flush=True)
        except BaseException:
            for f in futures:f.cancel()
            raise
    receipts=[first,*[results[i] for i in sorted(results)]]
    for rec in receipts:rec['sha256']=E.sha(rec['path'])
    final_value=dict(status='COMPLETE',stages=96,receipts=receipts,seconds=time.perf_counter()-tick,
        workers=workers,definition_hashes=definition_hashes(),new_queries=0,model_calls=0,
        scientific_definition='Existing geometry, quality, AVMR/PMR, valid graph and source-direction functions unchanged',
        transformed_query_measure='Fixed inherited sample allocation; affine points and normalized face-area importance weights')
    E.write(final,final_value)
    E.write(E.OUT/'evaluation/score_progress.json',dict(status='COMPLETE',completed=96,registered=96,workers=workers),replace=True)
    return final_value

def read_csv(path):
    with E.resolve(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def reviewed(reg):
    receipt_path=E.OUT/'evaluation/gallery/output_review_receipt.json';receipt=E.read(receipt_path)
    if receipt.get('status')!='COMPLETE':raise RuntimeError('Actual image review must complete before aggregate')
    path=receipt.get('review_csv') or receipt.get('review_file') or receipt.get('path')
    digest=receipt.get('review_csv_sha256') or receipt.get('review_file_sha256') or receipt.get('sha256')
    if not path or not digest:raise ValueError('Review receipt must bind CSV path and sha256')
    verify({path:digest});rows=read_csv(path)
    expected={(q['request_id'],stage):q for q in reg['requests'] for stage in ('RAW','POST')}
    if len(rows)!=96 or {(r['request_id'],r['stage']) for r in rows}!=set(expected):raise ValueError('Incomplete 96-row review')
    for r in rows:
        q=expected[(r['request_id'],r['stage'])]
        if r['label']!=LABEL or r['variant']!=q['variant'] or r['review_status']!='REVIEWED_AI_PROXY' or r['reviewer']!='AI_PROXY':
            raise ValueError('Review identity/status differs')
        if any(r[f] not in ('YES','NO','UNCERTAIN') for f in REVIEW_FIELDS) or len(r.get('notes','').strip())<8:
            raise ValueError('Actual observations with retained uncertainty required')
        page=E.OUT/'evaluation/gallery'/r['stage']/(q['base_request_id']+'.jpg')
        page_receipt=E.read(page.with_suffix('.json'))
        if r['reviewed_image_sha256']!=page_receipt['image_sha256'] or E.sha(page)!=r['reviewed_image_sha256']:
            raise ValueError('Reviewed page identity differs')
    return receipt,rows

def no_response_positive_control():
    """Analytic check only, not a generated control or a new quality metric."""
    alpha=np.asarray([np.float32(.95),np.float32(1),np.float32(1.05)],dtype=np.float64)
    cases=[]
    for constant in (0.,1.,3.75):
        width=np.full(3,constant,dtype=np.float64)
        slope=float((width[2]-width[0])/(alpha[2]-alpha[0]))
        amplitude=float(np.ptp(width))
        if slope!=0. or amplitude!=0.:raise AssertionError('No-response analytic check failed')
        cases.append(dict(alpha=alpha.tolist(),output_width=width.tolist(),d_width_d_alpha=slope,
            output_width_range=amplitude,relative_width_range=amplitude/constant if constant else None,
            relative_status='PASS' if constant else 'NA_ZERO_BASE_WIDTH'))
    return dict(status='PASS',kind='ANALYTIC_NO_RESPONSE_POSITIVE_CONTROL',cases=cases,model_calls=0,
        interpretation='An output width identical at all three C scales has zero response. No target-affine response or successful adoption is asserted.',
        scope='This checks arithmetic here; root response-analysis definitions must carry their own checks.')

def numeric(value):
    return isinstance(value,(int,float,bool,np.integer,np.floating,np.bool_)) and np.isfinite(value)

def average(rows,keys):
    grouped=collections.defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,rr in sorted(grouped.items()):
        good=[r for r in rr if r['status']=='PASS'];r=dict(zip(keys,key))
        r.update(n_registered=len(rr),n_valid=len(good),status='PASS' if len(good)==len(rr) else 'PARTIAL_OR_NA')
        for field in ('baseline_value','candidate_value','delta'):
            r[field]=float(np.mean([x[field] for x in good])) if good else None
        if good and all(x.get('delta_fraction') not in (None,'') for x in good):
            exact=sum((Fraction(x['delta_fraction']) for x in good),Fraction())/len(good)
            r.update(delta_fraction=str(exact),delta_sign_exact=(exact>0)-(exact<0))
        else:r.update(delta_fraction=None,delta_sign_exact=None)
        result.append(r)
    return result


def aggregate():
    reg=registration();review_receipt,review_rows=reviewed(reg)
    completion=E.read(E.OUT/'evaluation/score_completion.json')
    if completion['status']!='COMPLETE' or completion['stages']!=96:raise ValueError('Incomplete stage scoring')
    if completion['definition_hashes']!=definition_hashes():raise ValueError('Scoring definition changed')
    if (E.OUT/'evaluation/aggregate_completion.json').exists():
        previous=E.read(E.OUT/'evaluation/aggregate_completion.json');verify(previous['files'])
        if previous['review_receipt_sha256']!=E.sha(E.OUT/'evaluation/gallery/output_review_receipt.json'):
            raise ValueError('Review changed after aggregation')
        return previous
    outputs=[];quality=[];topology=[];boundary=[];vertices=[];desc=[];exact={};scalar_names=set();desc_names=set()
    qs={q['request_id']:q for q in reg['requests']}
    for rec in completion['receipts']:
        verify({rec['path']:rec['sha256']});s=E.read(rec['path']);verify(s['files'])
        m={k:s[k] for k in (*META,'label','stage')};key=(s['request_id'],s['stage'])
        quality.append(dict(m,**s['metrics'],I_status=s['status_by_metric']['I'],free_face_denominator=s['free_face_denominator'],
            C_query_weight_denominator=s['C_query_weight_denominator'],CF_tested_pair_denominator=s['CF_tested_pair_denominator'],
            FF_tested_pair_denominator=s['FF_tested_pair_denominator']))
        topology.append(dict(m,**{k:v for k,v in s['topology'].items() if not isinstance(v,(dict,list))}))
        vals={**s['metrics'],**{k:v for k,v in s['topology'].items() if numeric(v) or v is None},**s['distance']}
        for b in s['boundary']:
            boundary.append(dict(m,**b));vm='A_VMR__'+b['setting'];pm='PMR__'+b['setting']
            vals[vm]=b['VMR'];vals[pm]=b['perfect']
            if b['status']=='PASS' and b['boundary_vertex_count']:
                exact[key,vm]=Fraction(b['matched_count'],b['boundary_vertex_count']);exact[key,pm]=Fraction(b['perfect'])
        scalar_names.update(k for k,v in vals.items() if numeric(v) or v is None)
        scalar_names.add('C_preserved_exact')
        diagnostic=s['descriptive_boundary_distance']
        desc.append(dict(m,**diagnostic['metrics'],status=diagnostic['status'],boundary_vertex_denominator=diagnostic['boundary_vertex_denominator'],
            target_free_face_count=diagnostic['target_free_face_count'],definition_version=diagnostic['version']))
        for name,value in diagnostic['metrics'].items():
            vals['DESC__'+name]=value;desc_names.add('DESC__'+name)
        vals.update(C_preserved_exact=s['C_preserved_exact'],source_C_connectivity_status=s['source_C_connectivity_status'],
            I_status=s['status_by_metric']['I'],C_query_weight_denominator=s['C_query_weight_denominator'],
            C_query_count=s['C_query_count'],free_face_denominator=s['free_face_denominator'],
            CF_tested_pair_denominator=s['CF_tested_pair_denominator'],FF_tested_pair_denominator=s['FF_tested_pair_denominator'],
            query_measure=s['query_measure'],source_quality=qs[s['request_id']].get('source_quality','UNKNOWN'),
            score_path=rec['path'],score_sha256=rec['sha256'])
        outputs.append(dict(m,**vals))
        vertices.extend(dict(m,**x) for x in read_csv(E.resolve(rec['path']).parent/'boundary_vertices.csv'))
    if len(outputs)!=96 or len({(r['request_id'],r['stage']) for r in outputs})!=96:raise ValueError('Duplicate/missing stage')
    if scalar_names!=set(ORIGINAL_SCALARS):
        raise ValueError('Inherited metric schema differs: '+repr(dict(missing=sorted(set(ORIGINAL_SCALARS)-scalar_names),extra=sorted(scalar_names-set(ORIGINAL_SCALARS)))))
    tables={'per_output':outputs,'quality_per_output':quality,'topology_per_output':topology,
            'AVMR_PMR_all8':boundary,'boundary_vertices':vertices,'descriptive_distance_per_output':desc}
    baseline={(r['base_request_id'],r['stage']):r for r in outputs if r['variant']=='base'};paired=[]
    all_names=sorted(scalar_names|desc_names)
    for r in outputs:
        if r['variant']=='base':continue
        b=baseline[r['base_request_id'],r['stage']]
        for metric in all_names:
            v0,v1=b.get(metric),r.get(metric);valid=numeric(v0) and numeric(v1)
            ek1=((r['request_id'],r['stage']),metric);ek0=((b['request_id'],b['stage']),metric)
            exact_delta=exact[ek1]-exact[ek0] if ek1 in exact and ek0 in exact else None
            paired.append(dict(**{k:r[k] for k in (*META,'stage')},comparison=r['variant']+'-base',metric=metric,
                metric_origin='ADDED_DESCRIPTIVE_FINITE_DISTANCE_V1' if metric in desc_names else 'INHERITED_ORIGINAL',
                baseline_value=v0,candidate_value=v1,delta=v1-v0 if valid else None,status='PASS' if valid else 'NA_OR_MISSING',
                delta_fraction=str(exact_delta) if exact_delta is not None else None,
                delta_sign_exact=(exact_delta>0)-(exact_delta<0) if exact_delta is not None else None))
    parents=average(paired,('comparison','stage','parent_id','base_condition_id','region_type','tier','metric','metric_origin'))
    summary=[]
    for keys,rr in itertools.groupby(sorted(parents,key=lambda r:(r['comparison'],r['stage'],r['metric'])),
                                      key=lambda r:(r['comparison'],r['stage'],r['metric'])):
        rr=list(rr);good=[r for r in rr if r['status']=='PASS'];ds=[r['delta'] for r in good]
        signs=[r['delta_sign_exact'] if r['delta_sign_exact'] is not None else int(r['delta']>0)-int(r['delta']<0) for r in good]
        summary.append(dict(comparison=keys[0],stage=keys[1],metric=keys[2],metric_origin=rr[0]['metric_origin'],
            parents=len(rr),valid_parents=len(good),
            baseline_mean=float(np.mean([r['baseline_value'] for r in good])) if good else None,
            candidate_mean=float(np.mean([r['candidate_value'] for r in good])) if good else None,
            mean_delta=float(np.mean(ds)) if ds else None,median_delta=float(np.median(ds)) if ds else None,
            negative_delta_parents=signs.count(-1),equal_delta_parents=signs.count(0),positive_delta_parents=signs.count(1),
            sign_policy='AVMR/PMR exact integer-count fractions; other registered scalars strict original floating signs',
            direction='Numeric delta only; control response is not universally higher-is-better'))
    tables.update(paired_per_request=paired,paired_per_parent=parents,paired_summary=summary)
    strata=[]
    for field in ('region_type','tier'):
        for value in sorted({r[field] for r in parents}):
            pp=[r for r in parents if r[field]==value]
            strata.extend(dict(stratum=field,stratum_value=value,**r) for r in average(pp,('comparison','stage','metric','metric_origin')))
    tables['paired_strata']=strata
    absolute=[]
    for r in outputs:
        for name in all_names:
            value=r.get(name)
            absolute.append(dict(variant=r['variant'],stage=r['stage'],parent_id=r['parent_id'],metric=name,
                status='PASS' if numeric(value) else 'NA_OR_MISSING',baseline_value=value,candidate_value=value,delta=0. if numeric(value) else None))
    absolute_parents=average(absolute,('variant','stage','parent_id','metric'))
    absolute_summary=average(absolute_parents,('variant','stage','metric'))
    for rows in (absolute_parents,absolute_summary):
        for r in rows:
            r['value']=r.pop('candidate_value')
            for field in ('baseline_value','delta','delta_fraction','delta_sign_exact'):r.pop(field,None)
            r['metric_origin']='ADDED_DESCRIPTIVE_FINITE_DISTANCE_V1' if r['metric'] in desc_names else 'INHERITED_ORIGINAL'
    tables.update(absolute_per_parent=absolute_parents,absolute_summary=absolute_summary)
    observations=[]
    for stage in ('RAW','POST'):
        for variant in ('m095','base','p105'):
            rr=[r for r in review_rows if r['stage']==stage and r['variant']==variant]
            for field in REVIEW_FIELDS:
                count=collections.Counter(r[field] for r in rr)
                observations.append(dict(stage=stage,variant=variant,judgment=field,YES=count['YES'],NO=count['NO'],UNCERTAIN=count['UNCERTAIN'],
                    rows=len(rr),review_kind='NONBLIND_AI_PROXY',interpretation='Observation counts; UNCERTAIN is neither success nor failure'))
    tables['visual_review_counts']=observations
    paths=[]
    for name,rows in tables.items():
        path=E.OUT/'evaluation'/(name+'.csv');E.csvwrite(path,rows);paths.append(path)
    pc=E.OUT/'evaluation/no_response_positive_control.json';E.write(pc,no_response_positive_control());paths.append(pc)
    result=dict(status='COMPLETE',stage_records=96,inherited_scalar_metrics=len(scalar_names),descriptive_scalar_metrics=len(desc_names),parent_pairs=len(parents),
        inherited_metric_names=sorted(scalar_names),descriptive_metric_names=sorted(desc_names),
        hierarchy='Two original noises averaged within the single registered C per parent, then eight parents equally; region/tier slices retain parent weighting',
        review_receipt_sha256=E.sha(E.OUT/'evaluation/gallery/output_review_receipt.json'),
        visual_review_not_inferred=True,model_calls=0,geometry_rescored=0,files={str(p):E.sha(p) for p in paths})
    E.write(E.OUT/'evaluation/aggregate_completion.json',result)
    return result

def descriptive_distances(mod,points,F):
    """Fixed source-C boundary vertices to finite F surfaces/all finite F edges."""
    if not len(points):
        ds=np.empty(0);de=np.empty(0);status='NA_EMPTY_C_BOUNDARY'
    elif not len(F):
        ds=np.full(len(points),np.inf);de=ds.copy();status='POSITIVE_INFINITY_EMPTY_FREE'
    else:
        ds=np.sqrt(mod['surface'].surface_distance_squared(points,F))
        segments=np.concatenate([F[:,[a,b]] for a,b in ((0,1),(1,2),(2,0))])
        de=np.sqrt(mod['seams'].nearest_segments(points,segments)[0])
        status='PASS' if np.isfinite(ds).all() and np.isfinite(de).all() else 'NONFINITE_DISTANCE'
    metrics={}
    for domain,d in (('surface',ds),('all_finite_edges',de)):
        good=len(d)>0 and np.isfinite(d).all()
        for statistic,fn in (('mean',np.mean),('p95',lambda x:np.quantile(x,.95))):
            metrics['C_boundary_to_F_'+domain+'_'+statistic+'_model']=float(fn(d)) if good else None
    result=dict(version='fixed_C_boundary_finite_F_distances_v1',status=status,metrics=metrics,
        boundary_vertex_denominator=len(points),target_free_face_count=len(F),query_draws=0,
        measure='Uniform over the same unique source-ID C open-boundary vertices used by A-VMR; not edge-length weighted',
        target='F ONLY: finite triangles or all three finite segments of every free triangle; no C self-matching',
        units='model coordinates; L fixed '+repr(E.L),meaning='Descriptive proximity only; does not certify attachment or favorable quality')
    vectors=[dict(boundary_index=i,C_boundary_to_F_surface_model=float(a) if np.isfinite(a) else None,
        C_boundary_to_F_all_finite_edges_model=float(b) if np.isfinite(b) else None,status='PASS' if np.isfinite(a) and np.isfinite(b) else status)
        for i,(a,b) in enumerate(zip(ds,de))]
    return result,vectors

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('score','aggregate','check'),required=True)
    parser.add_argument('--max-workers',type=int,choices=(1,2),default=2);args=parser.parse_args()
    if args.stage=='score':result=run(args.max_workers)
    elif args.stage=='aggregate':result=aggregate()
    else:
        r=registration();pc=no_response_positive_control()
        result=dict(status='CPU_INTERFACE_CHECK_PASS',requests=len(r['requests']),queries=len(r['queries']),
            analytic_check=pc['status'],definition_files=len(definition_hashes()),model_calls=0,geometry_scans=0)
    print(json.dumps({k:v for k,v in result.items() if k not in ('receipts','definition_hashes','files')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

