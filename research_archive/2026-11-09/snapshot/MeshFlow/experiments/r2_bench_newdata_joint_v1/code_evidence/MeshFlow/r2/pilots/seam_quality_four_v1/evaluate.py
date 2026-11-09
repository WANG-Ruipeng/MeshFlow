"""Frozen original CPU quality plus explicitly versioned seam-to-bulk graph.

No training, model import, new query draw, geometry edit or semantic adjudication.
Run prepare before scoring; incomplete attempted cells require explicit recovery.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import collections
import concurrent.futures
import importlib
import json
import multiprocessing
import resource
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from . import eval_common as C

VERSION = 'seam_quality_four_original_quality_free_only_graph_v1'
ILLEGAL = {'proper_transverse_penetration', 'coplanar_positive_area_overlap'}
META = ('request_id', 'condition_id', 'parent_id', 'noise_index', 'N', 'K', 'split')
INTERNAL_FIELDS = ('retained_seam_faces', 'retained_bulk_faces', 'retained_seam_area',
    'retained_bulk_area', 'legal_CF_seed_seam_faces', 'legal_CF_seed_seam_area',
    'free_only_reached_seam_faces', 'free_only_reached_bulk_faces',
    'free_only_reached_bulk_area', 'free_only_reached_bulk_area_fraction',
    'legal_seam_bulk_edge_count', 'legal_seam_bulk_edge_length',
    'seed_connected_legal_seam_bulk_edge_count', 'seed_connected_legal_seam_bulk_edge_length',
    'bulk_reach_ge_0_30_descriptive', 'deleted_seam_faces')


def helpers():
    old = importlib.import_module('r2.pilots.user_condition_finetune_v1.evaluation')
    A = importlib.import_module('r2.tools.avmr_pmr_v1.run')
    G = importlib.import_module('r2.pilots.anchor_aware_denoiser_v1.evaluation')
    H = importlib.import_module('r2.tools.c_restore_coupling_v1.evaluation')
    Q = importlib.import_module('r2.tools.postprocess_avmr_pmr_v1.quality')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    return old, A, G, H, Q


def verify(files):
    for path, digest in files.items():
        if C.sha(path) != digest:
            raise ValueError('Frozen file changed: ' + str(path))


def save_npz(path, arrays):
    path = C.resolve(path)
    if path.exists():
        raise ValueError('Refuse overwrite NPZ ' + str(path))
    with path.open('wb') as handle:
        np.savez_compressed(handle, **arrays)


def manifest(root):
    root = C.resolve(root)
    value = C.read(root / 'eval_manifest.json')
    if tuple(value['arms']) != C.ARMS or len(value['conditions']) != 4 or len(value['requests']) != 16:
        raise ValueError('Expected exactly four TRAIN C, sixteen requests and three registered arms')
    if len({q['request_id'] for q in value['requests']}) != 16:
        raise ValueError('Duplicate request')
    for condition in value['conditions']:
        rows = [q for q in value['requests'] if q['condition_id'] == condition['condition_id']]
        if len(rows) != 4 or {q['noise_index'] for q in rows} != set(range(4)):
            raise ValueError('Unpaired noise panel')
        if condition['split'] != 'TRAIN':
            raise ValueError('This is a TRAIN-only local pilot')
    return value


def source_hashes():
    old, A, G, H, Q = helpers()
    paths = [Path(__file__), Path(C.__file__), Path(A.__file__), Path(G.__file__),
             Path(H.__file__), Path(Q.__file__), Path(old.__file__)]
    return {**old.scorer_sources(), **{str(p): C.sha(p) for p in paths}}


def prepare(root):
    root = C.resolve(root); m = manifest(root); old, A, _, _, _ = helpers()
    protected = {}; inputs = []
    for q in m['requests']:
        for field in ('inference_npz', 'metadata_npz', 'noise_npz'):
            protected[q[field]] = C.sha(q[field])
        inp = C.arrays(q['inference_npz']); meta = C.arrays(q['metadata_npz'])
        cc = inp['C_vertices'][inp['C_faces_model']]
        original = inp['C_vertices'][inp['C_faces_original']]
        if cc.shape != (q['K'], 3, 3) or cc.dtype != np.float32:
            raise ValueError('C shape/dtype differs')
        query = q['query_descriptor']; reference = old.load_queries(query)
        if query['identity']['C_sha256'] != old.array_sha(cc):
            raise ValueError('Frozen query belongs to another C')
        if len(reference['arrays']['C_points']) != 2048:
            raise ValueError('Query count differs')
        protected[query['path']] = query['sha256']
        source_ids = meta['C_source_vertex_ids'][inp['C_faces_original']]
        _, _, audit = A.topology(original, source_ids)
        inputs.append(dict(request_id=q['request_id'], C_sha256=old.array_sha(cc),
            C_original_sha256=old.array_sha(original), query_sha256=query['sha256'],
            input_sha256=C.sha(q['inference_npz']), noise_sha256=C.sha(q['noise_npz']),
            original_source_boundary_audit=audit))
    for condition in m['conditions']:
        protected[condition['labels_npz']] = C.sha(condition['labels_npz'])
    value = dict(version=VERSION, status='FROZEN', manifest_sha256=C.sha(root/'eval_manifest.json'),
        source_hashes=source_hashes(), protected=protected, inputs=inputs,
        cells=96, registered_requests=16, arms=list(C.ARMS), stages=['RAW', 'POST'],
        quality_definition=old.cpu_math()['control'].frozen_definition(), L=C.L,
        boundary_settings=[dict(setting=n, threshold_model=t, scale=s, tau=v) for n,t,s,v in A.SETTINGS],
        topology='Original actual_valid_graph on saved actual indexed output, legal and opposed separate; RAW index domain recorded by producer.',
        candidate_graph='Retained seam faces incident to legal CF are seeds. BFS admits only legal FF edges and never traverses C. Denominator is area of all retained bulk faces; empty/zero-area bulk is NA.',
        candidate_identity='Original generated RAW slot rows, never target/source GT assignment.',
        bulk_fraction_marker=0.30, marker_interpretation='Descriptive only; not completeness/adoption certification.',
        occupancy='Four C-only layouts, not sixteen independent trials. Compare to all-positive 62/66, positive recall 1, Null recall 0, layouts 2/4.',
        semantics='Actual AI_PROXY image review separate; no automated adoption labels.',
        model_calls=0, new_query_draws=0, GT_rank='NA_OPEN_ENDED_COMPLETION')
    path = root/'evaluation/score_registration.json'
    C.write(path, value)
    return value


def load_registration(root):
    root = C.resolve(root); p = root/'evaluation/score_registration.json'
    value = C.read(p)
    if value['source_hashes'] != source_hashes() or value['manifest_sha256'] != C.sha(root/'eval_manifest.json'):
        raise ValueError('Scorer/manifest changed after registration')
    verify(value['protected'])
    return value


def artifact(root, arm, request_id, stage):
    path = C.resolve(root)/'evaluation'/arm/request_id/stage.lower()/(stage.lower()+'.npz')
    if not path.exists():
        raise FileNotFoundError(path)
    receipt = path.parent/'receipt.json'
    if not receipt.exists():
        raise FileNotFoundError('Mesh receipt missing: '+str(receipt))
    r = C.read(receipt)
    if r.get('status') not in ('COMPLETE', 'VERIFIED_CACHE', 'PASS'):
        raise ValueError('Mesh has no successful completion receipt: '+str(receipt))
    if r.get('arm', r.get('label')) != arm or r.get('request_id') != request_id:
        raise ValueError('Producer receipt identity differs')
    verify(r['files'])
    if str(path) not in r['files'] or r['files'][str(path)] != C.sha(path):
        raise ValueError('Mesh is not bound by its producer receipt')
    return path, receipt


def seam_to_bulk(tri, K, edges, seam_mask, raw_ids, raw_seam_mask, applicable=True):
    """Derived graph only: no reclassification, no distances, no C traversal."""
    if not applicable:
        return dict(status='NA_NOT_APPLICABLE_FULL_ARM', **{k: None for k in INTERNAL_FIELDS}), []
    seam_mask = np.asarray(seam_mask, bool); raw_ids = np.asarray(raw_ids, np.int64)
    if seam_mask.shape != (len(tri),) or seam_mask[:K].any():
        raise ValueError('Invalid retained seam mask')
    if len(set(raw_ids.tolist())) != len(raw_ids) or not np.array_equal(seam_mask, np.asarray(raw_seam_mask,bool)[raw_ids]):
        raise ValueError('Lost/ambiguous original RAW slot identity')
    area = .5*np.linalg.norm(np.cross(tri[:,1].astype(float)-tri[:,0], tri[:,2].astype(float)-tri[:,0]), axis=1)
    seam = set(np.flatnonzero(seam_mask).tolist()); bulk = set(range(K,len(tri)))-seam
    seeds = set(); adj = collections.defaultdict(set); cross = []
    for edge in edges:
        if not edge['valid_edge']:
            continue
        i,j = map(int,edge['incident_face_ids'])
        if min(i,j)<K:
            free = max(i,j)
            if free in seam: seeds.add(free)
            continue
        adj[i].add(j); adj[j].add(i)
        if (i in seam) != (j in seam):
            cross.append(edge)
    seen = set(seeds); todo = list(sorted(seeds))
    while todo:
        i=todo.pop()
        for j in sorted(adj[i]-seen):
            if j<K: raise AssertionError('C traversal forbidden')
            seen.add(j); todo.append(j)
    rb=sorted(seen&bulk); rs=sorted(seen&seam); bulk_ids=sorted(bulk)
    denominator=float(area[bulk_ids].sum()); numerator=float(area[rb].sum())
    fraction=numerator/denominator if denominator>0 else None
    connected_cross=[e for e in cross if all(int(i) in seen for i in e['incident_face_ids'])]
    summary=dict(status='PASS' if denominator>0 else 'NA_EMPTY_OR_ZERO_AREA_BULK',
        retained_seam_faces=len(seam), retained_bulk_faces=len(bulk),
        retained_seam_area=float(area[sorted(seam)].sum()), retained_bulk_area=denominator,
        legal_CF_seed_seam_faces=len(seeds), legal_CF_seed_seam_area=float(area[sorted(seeds)].sum()),
        free_only_reached_seam_faces=len(rs), free_only_reached_bulk_faces=len(rb),
        free_only_reached_bulk_area=numerator, free_only_reached_bulk_area_fraction=fraction,
        legal_seam_bulk_edge_count=len(cross), legal_seam_bulk_edge_length=sum(e['length_model'] for e in cross),
        seed_connected_legal_seam_bulk_edge_count=len(connected_cross),
        seed_connected_legal_seam_bulk_edge_length=sum(e['length_model'] for e in connected_cross),
        bulk_reach_ge_0_30_descriptive=None if fraction is None else int(fraction>=.30),
        deleted_seam_faces=int(np.asarray(raw_seam_mask,bool).sum())-len(seam))
    rows=[dict(output_face_id=i, raw_slot_id=int(raw_ids[i]), role='SEAM' if i in seam else 'BULK',
        area_model2=float(area[i]), legal_CF_seed=i in seeds, reached_free_only=i in seen) for i in range(K,len(tri))]
    return summary,rows


def score_one(job):
    root,arm,q,stage,registration_sha=job; root=C.resolve(root)
    dest=root/'evaluation/scores'/arm/q['request_id']/stage; final=dest/'score.json'
    path,receipt=artifact(root,arm,q['request_id'],stage)
    identity=dict(version=VERSION, arm=arm, request_id=q['request_id'], stage=stage,
        mesh_path=str(path),mesh_sha256=C.sha(path),receipt_sha256=C.sha(receipt),
        registration_sha256=registration_sha)
    if final.exists():
        result=C.read(final)
        if result['identity']!=identity: raise ValueError('Completed score identity changed')
        verify(result['files'])
        return dict(path=str(final),sha256=C.sha(final),reused=True,seconds=0,
                    peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if dest.exists(): raise RuntimeError('Incomplete attempted cell; explicit recovery required '+str(dest))
    C.write(dest/'started.json',identity); tick=time.perf_counter()
    try:
        old,A,G,H,Q=helpers(); mod=old.cpu_math(); a=C.arrays(path)
        inp=C.arrays(q['inference_npz']); metadata=C.arrays(q['metadata_npz'])
        cc=inp['C_vertices'][inp['C_faces_model']]; original=inp['C_vertices'][inp['C_faces_original']]
        K=int(q['K']); tri=np.asarray(a['output']).reshape(-1,3,3)
        vertices=np.asarray(a['vertices']); faces=np.asarray(a['faces'],np.int64)
        if tri.dtype!=np.float32 or not np.isfinite(tri).all() or not old.byte_equal(vertices[faces],tri):
            raise ValueError('Nonfinite/nonFP32 or indexed/output mismatch')
        if not old.byte_equal(tri[:K],original): raise ValueError('Original C orientation/coordinates changed')
        if stage=='RAW' and len(tri)!=q['N']: raise ValueError('RAW total face budget changed')
        F=tri[K:]
        if len(F): metrics,cf,ff=mod['geometry'].scan_geometry(cc,F,C.L)
        else:
            cf,ff=Q._empty_scans(K);metrics={}
        pairs={'CF':cf,'FF':ff}; reference=old.load_queries(q['query_descriptor'])
        trusted=mod['control'].trusted_cf(cc,F,C.L,cf_scan=cf)
        im,iv=mod['control'].conflict_metrics(F,{'C':cc,'L0':C.L},reference,trusted)
        if abs(im['ConflictCov_C']-im['ConflictCov_C_cross']-im['ConflictCov_C_dup']+im['ConflictCov_C_cross_and_dup'])>1e-12:
            raise AssertionError('I union identity fails')
        metrics.update(im)
        for name,scan in (('CF',cf),('FF',ff)):
            illegal=[r for r in scan['contacts'] if r['classification'] in ILLEGAL]
            affected={int(r['pair'][1]) for r in illegal} if name=='CF' else {int(i) for r in illegal for i in r['pair']}
            metrics.update({name+'_illegal_pair_count':len(illegal),name+'_illegal_affected_free_fraction':len(affected)/len(F) if len(F) else None})
        shape={name:mod['shape'].shape_quality(mesh,C.L) for name,mesh in (('C',cc),('free',F),('whole',tri))}
        for name,s in shape.items():
            metrics.update({name+'_minimum_angle_lt5_fraction':s['min_angle_lt5_fraction'],name+'_q_gt10_fraction':s['q_actual_gt10_fraction'],
                name+'_near_degenerate_count':s['near_degenerate_area_count'],name+'_exact_zero_area_count':s['exact_zero_area_count'],name+'_total_surface_area':s['total_area']})
        metrics.update(mod['geometry'].topology(tri,K),actual_free_faces=len(F),deleted_free_faces=q['N']-len(tri))
        graph,edges,reached=G.actual_valid_graph(vertices,faces,K,pairs)
        graph.update(H.opposite_reach(tri,K,edges),C_F_has_opposite_winding_edge=int(graph['C_F_valid_opposite_winding_edge_count']>0),
            domain=str(a.get('index_domain','ACTUAL_SAVED_INDEX_SOURCE_C_RESTORED')))
        raw_ids=np.asarray(a['raw_slot_ids'],np.int64); raw_seam=np.asarray(a['raw_seam_slot_mask'],bool)
        if len(raw_ids)!=len(tri) or not np.array_equal(raw_ids[:K],np.arange(K)) or np.any(raw_ids<0) or np.any(raw_ids>=q['N']):
            raise ValueError('Invalid original RAW slot identity')
        seam=raw_seam[raw_ids]
        internal,face_rows=seam_to_bulk(tri,K,edges,seam,raw_ids,raw_seam,arm in C.ARMS)
        source_ids=metadata['C_source_vertex_ids'][inp['C_faces_original']]
        boundary_ids,points,ba=A.topology(original,source_ids);dist,nearest=A.nearest(points,F)
        boundary=[dict(setting=n,threshold_model=t,scale=s,tau=v,domain='ALL_C_SOURCE_TOPOLOGY_OPEN_BOUNDARY',**A.metric(dist,t)) for n,t,s,v in A.SETTINGS]
        vertex_rows=[dict(source_vertex_id=int(i),x=float(p[0]),y=float(p[1]),z=float(p[2]),
            distance_model=float(d) if np.isfinite(d) else None,
            distance_L_fraction=float(d/C.L) if np.isfinite(d) else None,
            nearest_free_face=int(n)//3 if n>=0 else None,nearest_free_corner=int(n)%3 if n>=0 else None)
            for i,p,d,n in zip(boundary_ids,points,dist,nearest)]
        distance={key:float(fn(dist)) if len(dist) and np.isfinite(dist).all() else None for key,fn in (
            ('boundary_distance_mean_model',np.mean),('boundary_distance_median_model',np.median),
            ('boundary_distance_p95_model',lambda x:np.quantile(x,.95)),('boundary_distance_max_model',np.max))}
        C.write(dest/'pairs.json',pairs);C.write(dest/'shape_per_face.json',shape)
        C.write(dest/'topology.json',graph);C.write(dest/'seam_to_bulk.json',internal)
        C.csvwrite(dest/'valid_edges.csv',edges);C.csvwrite(dest/'seam_bulk_faces.csv',face_rows)
        C.csvwrite(dest/'boundary_vertices.csv',vertex_rows);C.csvwrite(dest/'AVMR_PMR_all8.csv',boundary)
        save_npz(dest/'vectors.npz',iv)
        result=dict(**{k:q[k] for k in META},label=arm,stage=stage,status='PASS',identity=identity,
            metrics=metrics,topology=graph,seam_to_bulk=internal,boundary=boundary,distance=distance,
            C_preserved_exact=True,C_original_order_preserved=True,C_query_count=2048,
            C_query_weight_denominator=float(reference['arrays']['C_weights'].sum()),
            free_face_denominator=len(F),CF_tested_pair_denominator=K*len(F),
            FF_tested_pair_denominator=len(F)*(len(F)-1)//2,boundary_audit=ba,
            source_identity_domain='generated_RAW_slot_not_GT',model_calls=0,full_quality_scans=1,new_queries=0,
            status_by_metric=dict(I=im['ConflictCov_C_status'],S='NA_NO_REQUIRED_INTERFACE',E='NA_NO_EXPOSURE',
                GT='NA_NOT_UNIQUE_GENERATIVE_TARGET',semantic_use='AI_PROXY_REVIEW_SEPARATE'),
            seconds=time.perf_counter()-tick,files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
        if 'torch' in sys.modules: raise RuntimeError('CPU scorer imported model framework')
        C.write(final,result)
        return dict(path=str(final),sha256=C.sha(final),reused=False,seconds=result['seconds'],
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException as exc:
        C.write(dest/'failure.json',dict(status='STOPPED',error=repr(exc),traceback=traceback.format_exc()))
        raise


def run(root,workers=2):
    root=C.resolve(root);reg=load_registration(root);m=manifest(root)
    final=root/'evaluation/score_completion.json'
    if final.exists():
        previous=C.read(final)
        if previous['registration_sha256']!=C.sha(root/'evaluation/score_registration.json'): raise ValueError('Score registration changed')
        for rec in previous['receipts']: verify({rec['path']:rec['sha256']});verify(C.read(rec['path'])['files'])
        return previous
    # All registered RAW and POST must exist before this complete paired scoring stage.
    for q in m['requests']:
        for arm in C.ARMS:
            for stage in ('RAW','POST'):artifact(root,arm,q['request_id'],stage)
    snapshot_path=root/'evaluation'/('score_resources_'+str(time.time_ns())+'.json')
    C.resources(snapshot_path);tick=time.perf_counter();rh=C.sha(root/'evaluation/score_registration.json')
    jobs=[(str(root),arm,q,stage,rh) for q in m['requests'] for arm in C.ARMS for stage in ('RAW','POST')]
    first=score_one(jobs[0]);snap=C.resources()
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (snap.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    parallel=(os.cpu_count() or 1)>=4 and mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024)
    parallel=parallel and (snap['loadavg'] or [999])[0]<(os.cpu_count() or 1)-2
    count=2 if workers==2 and parallel else 1
    C.write(root/'evaluation'/('score_dispatch_'+str(time.time_ns())+'.json'),dict(workers=count,threads_per_worker=1,
        first_actual_cell=first,resources=snap,reason='Measured first authorized score; no probe or added cell'))
    receipts=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        # Bounded dispatch: failures stop new work, already running cells finish safely.
        todo=iter(jobs[1:]);active={}
        for _ in range(count):
            job=next(todo,None)
            if job is not None:active[pool.submit(score_one,job)]=job
        while active:
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            results=[future.result() for future in done]
            for future,result in zip(done,results):
                del active[future];receipts.append(result)
                C.write(root/'evaluation/score_progress.json',dict(status='RUNNING',completed=len(receipts),registered=96),replace=True)
                print(json.dumps(dict(scored=len(receipts),registered=96)),flush=True)
                job=next(todo,None)
                if job is not None:active[pool.submit(score_one,job)]=job
    receipts.sort(key=lambda r:r['path'])
    value=dict(status='COMPLETE',cells=96,receipts=receipts,registration_sha256=rh,workers=count,
        seconds=time.perf_counter()-tick,model_calls=0,new_queries=0,full_quality_scans=sum(not r['reused'] for r in receipts))
    C.write(final,value);return value



def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage', choices=('prepare','score'), required=True)
    p.add_argument('--root', default=str(C.OUT))
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    a=p.parse_args()
    result=prepare(a.root) if a.stage=='prepare' else run(a.root,a.workers)
    print(json.dumps(dict(status=result['status'],stage=a.stage)),flush=True)

if __name__=='__main__':main()
