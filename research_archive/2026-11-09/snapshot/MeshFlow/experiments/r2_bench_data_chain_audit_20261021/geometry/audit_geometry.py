"""Read-only Bench source/pair audit. NumPy CPU only; no model imports or repair.

Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1.
Exact equality is numerical equality in the stored dtype, never a tolerance weld.
Outputs live only alongside this script; source assets are hashed before/after.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT/'experiments/r2_bench_data_chain_audit_20261021/geometry'
MAN = ROOT/'experiments/r2_bench_clean_multipart_seam_v1/data/manifest.json'
OLD = ROOT/'experiments/r2_bench_control_transfer_v1/data/manifest.json'
INPUTS = {}
CHECKS = collections.Counter()

def resolve(p):
    return Path(str(p).replace('E:\\MeshFlow','/mnt/e/MeshFlow').replace('E:/MeshFlow','/mnt/e/MeshFlow'))

def sha(p):
    h=hashlib.sha256()
    with resolve(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def protect(p, expected=None):
    p=resolve(p); h=sha(p)
    if expected is not None: check(h==expected,'registered_hash_match')
    INPUTS[str(p)]=h
    return p

def read(p): return json.loads(protect(p).read_text(encoding='utf-8-sig'))

def arrays(p, expected=None):
    with np.load(protect(p,expected),allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def check(ok,name):
    if not ok: raise ValueError('CHECK FAILED: '+name)
    CHECKS[name]+=1

def json_write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def csv_write(name,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def id_keys(faces): return [tuple(sorted(map(int,f))) for f in faces]

def geom_keys(triangles):
    # Python float preserves exact float32 values. -0.0 == +0.0 by design.
    return [tuple(sorted(tuple(map(float,v)) for v in t)) for t in triangles]

def cycle(t):
    t=tuple(t)
    return min(t,t[1:]+t[:1],t[2:]+t[:2])

def winding(a,b):
    if len(set(a))<3 or len(set(b))<3: return 'DEGENERATE_AMBIGUOUS'
    if cycle(a)==cycle(b): return 'SAME'
    if cycle(a)==cycle(tuple(reversed(b))): return 'OPPOSITE'
    return 'NOT_SAME_UNORDERED_VERTICES'

def dup_metrics(keys,oriented):
    book=collections.defaultdict(list)
    for i,k in enumerate(keys): book[k].append(i)
    counts=collections.Counter()
    for ids in book.values():
        if len(ids)<2: continue
        counts['groups']+=1; counts['excess_faces']+=len(ids)-1; counts['involved_faces']+=len(ids)
        for a,b in itertools.combinations(ids,2):
            counts['pairs']+=1; counts[winding(tuple(oriented[a]),tuple(oriented[b]))]+=1
    return dict(groups=counts['groups'],excess_faces=counts['excess_faces'],involved_faces=counts['involved_faces'],pairs=counts['pairs'],same_winding_pairs=counts['SAME'],opposite_winding_pairs=counts['OPPOSITE'],degenerate_ambiguous_pairs=counts['DEGENERATE_AMBIGUOUS'])

def area2(t):
    t=np.asarray(t,np.float64)
    return np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)

def prefix(d,p): return {p+k:v for k,v in d.items()}

def parent_audit(row,source):
    f=source['source_faces']; mv=source['model_vertices']; sv=source['source_vertices']; canon=source['full_target']
    orig=mv[f]; ids=source['source_face_vertex_ids']; perms=source['source_corner_permutations']
    check(np.array_equal(np.take_along_axis(f,perms,axis=1),ids),'source_corner_permutation_maps_original_ids')
    check(np.array_equal(mv[ids],canon),'canonical_triangle_exactly_matches_source_vertices')
    check(np.array_equal(np.sort(f,axis=1),np.sort(ids,axis=1)),'source_face_order_preserved')
    keys=id_keys(f); gkeys=geom_keys(orig)
    oriented_geoms=[tuple(tuple(map(float,p)) for p in t) for t in orig]
    legacy_area=np.linalg.norm(np.cross(orig[:,1]-orig[:,0],orig[:,2]-orig[:,0]),axis=1)
    d=dup_metrics(keys,f); gd=dup_metrics(gkeys,oriented_geoms)
    check(d['excess_faces']==row['source_exact_duplicate_ID_faces'],'historical_duplicate_count_match')
    check(int(np.count_nonzero(legacy_area<=1e-12))==row['source_degenerate_faces'],'historical_degenerate_count_match')
    result=dict(parent_id=row['parent_id'],group=row['group'],official_split=row['official_split'],N=len(f),
                source_case_path=row['case_path'],source_case_sha256=row['case_sha256'],
                model_area_total=float(area2(orig).sum()/2),
                repeated_source_id_faces=sum(len(set(map(int,x)))<3 for x in f),
                exact_zero_area_faces_source64=int(np.count_nonzero(area2(sv[f])==0)),
                exact_zero_area_faces_model64=int(np.count_nonzero(area2(orig)==0)),
                exact_zero_area_faces_canonical64=int(np.count_nonzero(area2(canon)==0)),
                legacy_area2_le1e12_model32_faces=int(np.count_nonzero(legacy_area<=1e-12)),
                full_selfintersection='NOT_RUN',**prefix(d,'source_ID_'),**prefix(gd,'exact_geometry_'))
    return result

def main():
    start=time.time(); OUT.mkdir(parents=True,exist_ok=True)
    json_write('execution_started.json',dict(time_unix=start,pid=os.getpid(),numeric_threads=1,CPU_workers=1,model_calls=0,GPU_calls=0,source_mutations=0,scope='Existing arrays; exact duplicates/degeneracy and C/Y complement checks only'))
    manifest=read(MAN); old=read(OLD)
    rows=manifest['rows']
    active_ids={r['condition_id'] for r in manifest['train']+manifest['dev']}
    check((len(manifest['train']),len(manifest['dev']))==(195,39),'current_manifest_condition_counts')
    active=collections.Counter(r['parent_id'] for r in manifest['train']+manifest['dev'])
    old_rows=old['train']+old['dev']
    sources={}; parents=[]
    for r in old_rows:
        s=arrays(r['case_path'],r['case_sha256']); sources[r['parent_id']]=s
        p=parent_audit(r,s); p.update(current_active_conditions=active[r['parent_id']],current_active=bool(active[r['parent_id']]))
        parents.append(p)
    cond=[]; cross_pairs=[]
    for r in rows:
        c=arrays(r['case_path'],r['case_sha256']); pair=arrays(r['paired_path'],r['paired_sha256']); s=sources[r['parent_id']]
        known=c['known_ids']; free=c['free_ids']; full=c['full_target']; f=s['source_faces']; orig=s['model_vertices'][f]
        check(np.array_equal(full,s['full_target']),'current_target_matches_untouched_source')
        check(np.array_equal(np.sort(np.r_[known,free]),np.arange(len(full))),'C_Y_source_face_partition')
        check(np.array_equal(c['C'],full[known]) and np.array_equal(c['Y'],full[free]),'C_Y_exact_target_partition')
        for key in ('C','Y','known_ids','free_ids','source_face_ids','free_source_face_ids','C_source_faces_original','C_source_corner_permutations'):
            check(np.array_equal(c[key],pair[key]),'independent_pair_matches_case_'+key)
        check(np.array_equal(known,c['source_face_ids']) and np.array_equal(free,c['free_source_face_ids']),'face_ID_identity_mapping')
        check(np.array_equal(c['C_source_faces_original'],f[known]),'C_original_winding_matches_source')
        check(np.array_equal(c['C_vertices'][c['C_faces_model']],c['C']),'C_indexed_canonical_matches_target')
        check(np.array_equal(c['C_vertices'][c['C_faces_original']],orig[known]),'C_indexed_original_matches_source')
        yc=id_keys(f[free]); cc=id_keys(f[known]); yg=geom_keys(orig[free]); cg=geom_keys(orig[known])
        ygeom_oriented=[tuple(tuple(map(float,p)) for p in t) for t in orig[free]]
        yd=dup_metrics(yc,f[free]); ygd=dup_metrics(yg,ygeom_oriented)
        ydmap=collections.defaultdict(list); ygmap=collections.defaultdict(list)
        for idx,key in zip(free,yc): ydmap[key].append(int(idx))
        for idx,key in zip(free,yg): ygmap[key].append(int(idx))
        counts=collections.Counter(); touched_id=set(); touched_geom=set(); free_id=set(); free_geom=set()
        for ci,(si,ki,kg) in enumerate(zip(known,cc,cg)):
            for fi in ygmap.get(kg,[]):
                same_id=ki==tuple(sorted(map(int,f[fi])))
                wa=winding(tuple(map(int,f[si])),tuple(map(int,f[fi]))) if same_id else winding(tuple(tuple(map(float,p)) for p in orig[si]),tuple(tuple(map(float,p)) for p in orig[fi]))
                counts['geometry_pairs']+=1; touched_geom.add(ci); free_geom.add(fi)
                if same_id:
                    counts['ID_pairs']+=1; counts['ID_'+wa]+=1; touched_id.add(ci); free_id.add(fi)
                else: counts['geometry_only_pairs']+=1
                cross_pairs.append(dict(condition_id=r['condition_id'],parent_id=r['parent_id'],group=r['group'],part_type=r['part_type'],C_local_face=ci,C_source_face=int(si),Y_source_face=fi,same_source_vertex_ID_set=same_id,original_winding_relation=wa,overlap='EXACT_SAME_UNORDERED_TRIANGLE_COORDINATES',C_triangle_area_model=float(area2(orig[si:si+1])[0]/2)))
        ca=area2(orig[known])/2; ya=area2(orig[free])/2
        result=dict(condition_id=r['condition_id'],parent_id=r['parent_id'],group=r['group'],part_type=r['part_type'],active_current_pool=r['condition_id'] in active_ids,N=int(c['N']),K=int(c['K']),Y_faces=len(free),
                    C_source_components=r.get('source_edge_components','NA'),
                    C_exact_geometry_duplicate_excess=dup_metrics(cg,[tuple(tuple(map(float,p)) for p in t) for t in orig[known]])['excess_faces'],
                    Y_exact_zero_area_model64_faces=int(np.count_nonzero(ya==0)),
                    C_Y_ID_overlap_pairs=counts['ID_pairs'],C_Y_ID_overlap_C_faces=len(touched_id),C_Y_ID_overlap_Y_faces=len(free_id),
                    C_Y_ID_overlap_same_winding_pairs=counts['ID_SAME'],C_Y_ID_overlap_opposite_winding_pairs=counts['ID_OPPOSITE'],
                    C_Y_exact_geometry_overlap_pairs=counts['geometry_pairs'],C_Y_exact_geometry_overlap_C_faces=len(touched_geom),C_Y_exact_geometry_overlap_Y_faces=len(free_geom),
                    C_Y_geometry_only_distinct_ID_overlap_pairs=counts['geometry_only_pairs'],
                    C_area_model=float(ca.sum()),C_exactly_duplicated_in_Y_area_model=float(ca[list(touched_geom)].sum()) if touched_geom else 0.,
                    C_exactly_duplicated_in_Y_area_fraction=float(ca[list(touched_geom)].sum()/ca.sum()) if touched_geom and ca.sum()>0 else 0.,
                    source_reconstruction_as_success=False,full_Y_selfintersection='NOT_RUN',
                    **prefix(yd,'Y_source_ID_'),**prefix(ygd,'Y_exact_geometry_'))
        cond.append(result)
    groups={}
    for group in ('TRAIN','DEV'):
        ps=[p for p in parents if p['group']==group and p['current_active']]; cs=[c for c in cond if c['group']==group and c['active_current_pool']]
        groups[group]=dict(parents=len(ps),conditions=len(cs),N_sum=sum(p['N'] for p in ps),
            parents_with_ID_duplicates=sum(p['source_ID_excess_faces']>0 for p in ps),ID_duplicate_excess=sum(p['source_ID_excess_faces'] for p in ps),
            ID_duplicate_same_winding_pairs=sum(p['source_ID_same_winding_pairs'] for p in ps),ID_duplicate_opposite_winding_pairs=sum(p['source_ID_opposite_winding_pairs'] for p in ps),
            ID_duplicate_degenerate_ambiguous_pairs=sum(p['source_ID_degenerate_ambiguous_pairs'] for p in ps),
            parents_with_exact_coordinate_duplicates=sum(p['exact_geometry_excess_faces']>0 for p in ps),exact_coordinate_duplicate_excess=sum(p['exact_geometry_excess_faces'] for p in ps),
            parents_with_legacy_degenerates=sum(p['legacy_area2_le1e12_model32_faces']>0 for p in ps),legacy_degenerate_faces=sum(p['legacy_area2_le1e12_model32_faces'] for p in ps),
            exact_zero_area_faces_model64=sum(p['exact_zero_area_faces_model64'] for p in ps),
            conditions_with_Y_ID_duplicates=sum(c['Y_source_ID_excess_faces']>0 for c in cs),
            conditions_with_C_Y_ID_overlap=sum(c['C_Y_ID_overlap_pairs']>0 for c in cs),parents_with_C_Y_ID_overlap=len({c['parent_id'] for c in cs if c['C_Y_ID_overlap_pairs']>0}),
            conditions_with_C_Y_exact_geometry_overlap=sum(c['C_Y_exact_geometry_overlap_pairs']>0 for c in cs),
            conditions_with_C_Y_geometry_only_overlap=sum(c['C_Y_geometry_only_distinct_ID_overlap_pairs']>0 for c in cs),
            C_Y_ID_overlap_pairs_across_conditions=sum(c['C_Y_ID_overlap_pairs'] for c in cs),
            C_Y_ID_opposite_pairs_across_conditions=sum(c['C_Y_ID_overlap_opposite_winding_pairs'] for c in cs),
            C_Y_ID_same_pairs_across_conditions=sum(c['C_Y_ID_overlap_same_winding_pairs'] for c in cs),
            C_duplicated_area_fraction_condition_mean=float(np.mean([c['C_exactly_duplicated_in_Y_area_fraction'] for c in cs])))
    historical={}
    for group in ('TRAIN','DEV'):
        ps=[p for p in parents if p['group']==group]
        historical[group]=dict(parents=len(ps),parents_with_duplicate_ID_faces=sum(p['source_ID_excess_faces']>0 for p in ps),duplicate_ID_faces=sum(p['source_ID_excess_faces'] for p in ps),parents_with_degenerate_faces=sum(p['legacy_area2_le1e12_model32_faces']>0 for p in ps),degenerate_faces=sum(p['legacy_area2_le1e12_model32_faces'] for p in ps))
    stats=read(ROOT/'experiments/r2_bench_control_transfer_v1/data/dataset_card_statistics.json')
    for g,d in historical.items():
        for k,v in d.items(): check(stats['groups'][g][k]==v,'full72_matches_historical_statistics_'+k)
    csv_write('per_parent.csv',parents); csv_write('per_condition.csv',cond); csv_write('C_Y_exact_overlap_pairs.csv',cross_pairs)
    for path,expected in INPUTS.items(): check(sha(path)==expected,'input_hash_after_unchanged')
    json_write('input_hashes.json',INPUTS)
    summary=dict(schema='BENCH_STORED_DATA_EXACT_GEOMETRY_AUDIT_V1',status='COMPLETE',current_groups=groups,historical72_verification=historical,
        definitions=dict(source_ID_duplicates='Unordered equal original source vertex-ID triples; excess=multiplicity-1; pair counts are combinatorial and distinct from excess.',
        winding='Cyclic equality SAME, cyclic reversed equality OPPOSITE, repeated-vertex triangles AMBIGUOUS; geometric orientation does not establish author intent.',
        exact_geometry='Unordered numerical equality of three stored model-coordinate positions; no epsilon, quantization or welding. May include distinct source vertex IDs.',
        C_Y_overlap='Whole triangles exactly coincide; does NOT include partial coplanar overlaps, close parallel layers or proper intersections.',
        degeneracy='exact cross-product zero in float64 on stored floats; historical threshold separately reproduced as float32 twice-area<=1e-12 model units squared.',
        per_condition_totals='Repeat source geometry across alternative conditions. Do not treat condition-summed defects as independent parents.'),
        checks=dict(CHECKS),full_triangle_pair_selfintersection='NOT_RUN',source_repair=False,model_calls=0,GPU_calls=0,training_updates=0,
        script_sha256=sha(__file__),protected_inputs=len(INPUTS),wall_seconds=time.time()-start)
    json_write('summary.json',summary)
    json_write('zero_gpu_ledger.json',dict(status='COMPLETE',started_unix=start,finished_unix=time.time(),wall_seconds=time.time()-start,pid=os.getpid(),CPU_workers=1,numeric_threads=1,NumPy_version=np.__version__,GPU_calls=0,model_imports=0,model_forwards=0,training_updates=0,new_generation=0,new_random_noise=0,OT_calls=0,geometry_repairs=0,source_mutations=0,all_input_hashes_unchanged=True,scope='Static array arithmetic; 72 original parents plus 316 original multipart conditions; summary restricted to 195 active TRAIN and 39 DEV',full_selfintersection='NOT_RUN'))
    print(json.dumps(groups,indent=2))

if __name__=='__main__': main()
