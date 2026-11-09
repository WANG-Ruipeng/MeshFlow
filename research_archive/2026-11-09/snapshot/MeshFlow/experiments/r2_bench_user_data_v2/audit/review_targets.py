"""Independent read-only target lineage/area check; no new pair scorer."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,hashlib,json,time
from pathlib import Path
import numpy as np
BASE=Path('/mnt/e/MeshFlow/experiments/r2_bench_user_data_v2');HASHES={};CHECKS=collections.Counter()
def check(ok,name):
    if not ok:raise AssertionError(name)
    CHECKS[name]+=1
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def protect(p,expected=None):
    p=Path(p);h=sha(p);check(expected is None or h==expected,'hash_identity');HASHES[str(p)]=h;return p
def read(p):return json.loads(protect(p).read_text('utf-8-sig'))
def load(p):
    with np.load(protect(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def area(v,f):
    t=np.asarray(v,np.float64)[f];return np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
def gkey(t):return tuple(sorted(tuple(map(float,x)) for x in t))
def main():
    started=time.time();manifest=read(BASE/'parents_manifest.json');rows=manifest['parents'];check(len(rows)==32,'32_proposals')
    results=[];reason_sets=collections.Counter();reason_counts=collections.Counter()
    for row in rows:
        for p,h in row['artifact_hashes'].items():protect(p,h)
        protect(row['source_case_path'],row['source_case_sha256'])
        s=load(row['source_case_path']);target=load(row['target_case_path']);mesh=load(row['target_mesh_path']);orig=load(row['original_mesh_path']);lin=read(row['lineage_path']);geo=read(row['geometry_path'])
        v0=s['model_vertices'];f0=s['source_faces'];v=target['model_vertices'];f=target['source_faces'];n=len(f)
        check(np.array_equal(orig['vertices'],v0) and np.array_equal(orig['faces'],f0),'original_mesh_exact_source')
        check(np.array_equal(mesh['vertices'],v) and np.array_equal(mesh['faces'],f),'target_mesh_exact_case')
        for key in ('model_vertices','normalized_vertices','source_vertices'):
            check(target[key][:len(s[key])].dtype==s[key].dtype and target[key][:len(s[key])].tobytes()==s[key].tobytes(),'original_vertex_bytes_'+key)
        check(target['transform_json'].tobytes()==s['transform_json'].tobytes(),'normalization_exact')
        check(np.array_equal(target['pre_ot_full']*np.float32(2),target['full_target']),'canonical_half_scale')
        check(np.array_equal(v[target['source_face_vertex_ids']],target['full_target']),'canonical_exact_coordinates')
        oldarea=area(v0,f0);sourcearea=area(s['source_vertices'],f0);newarea=area(v,f)
        removed=set(lin['removed_zero_area_source_face_ids']);expectedzero=set(np.flatnonzero((oldarea==0)|(sourcearea==0)).tolist())
        check(removed==expectedzero,'removed_exact_zero_definition')
        check(not any(oldarea[x]>0 and sourcearea[x]>0 for x in removed),'no_two_domain_positive_area_removed_as_zero')
        nto=lin['new_to_old_face_ids'];otn=lin['old_to_new_face_ids'];check(len(nto)==n and len(otn)==len(f0),'lineage_shapes')
        covered=set();max_surface_error=0.;max_area_error=0.;max_relative_area_error=0.;seen_groups=set();area_expected=0.
        for ti,olds in enumerate(nto):
            check(bool(olds),'nonempty_ancestry');check(len(set(olds))==len(olds),'unique_ancestors');covered.update(olds)
            keys={gkey(v0[f0[x]]) for x in olds};check(len(keys)==1,'merged_ancestors_exact_same_surface')
            for old in olds:check(ti in otn[old],'bidirectional_lineage_forward')
            tri=v0[f0[olds[0]]].astype(np.float64);pts=v[f[ti]].astype(np.float64)
            e0,e1=tri[1]-tri[0],tri[2]-tri[0];normal=np.cross(e0,e1);norm=np.linalg.norm(normal)
            check(norm>0,'retained_source_positive_area')
            plane=np.abs((pts-tri[0])@normal)/norm;max_surface_error=max(max_surface_error,float(plane.max()))
            ab=np.linalg.lstsq(np.column_stack([e0,e1]),(pts-tri[0]).T,rcond=None)[0];bary=np.column_stack([1-ab.sum(0),ab.T])
            check(bary.min()>=-1e-5 and bary.max()<=1+1e-5,'descendant_inside_source_triangle')
            group=tuple(olds)
            if group not in seen_groups:
                seen_groups.add(group);desc=otn[olds[0]];expected=float(oldarea[olds[0]]);actual=float(newarea[desc].sum());err=abs(actual-expected)
                area_expected+=expected;max_area_error=max(max_area_error,err);max_relative_area_error=max(max_relative_area_error,err/expected)
        for old,children in enumerate(otn):
            check(all(old in nto[c] for c in children),'bidirectional_lineage_reverse')
            check(bool(children)==(old not in removed),'all_nonzero_source_faces_represented')
        check(covered==set(range(len(f0)))-removed,'no_missing_positive_support')
        check(max_surface_error<=1e-6*(np.sqrt(3)*2/.3762),'surface_plane_tolerance')
        check(max_relative_area_error<=1e-5,'descendant_area_preserved')
        check(n==row['N'] and row['zero_faces_removed']==len(removed),'manifest_counts')
        check(int(np.count_nonzero(newarea==0))==len(geo['exact_zero_area_faces']),'reported_zero_area_matches')
        check(geo['pair_count']==n*(n-1)//2 and sum(geo['classifications'].values())==geo['pair_count'],'cached_full_pair_accounting')
        check(len(geo['illegal_pairs'])==row['illegal_pairs'] and len(geo['unresolved_pairs'])==row['unresolved_pairs'],'cached_pair_counts')
        book=collections.defaultdict(list)
        for fi,face in enumerate(f):
            for j in range(3):
                a,b=map(int,(face[j],face[(j+1)%3]));book[tuple(sorted((a,b)))].append((fi,a,b))
        nonmanifold=sum(len(a)>2 for a in book.values());same=sum(len(a)==2 and a[0][1:]==a[1][1:] for a in book.values())
        check(nonmanifold==len(geo['topology']['nonmanifold_edges']),'independent_nonmanifold_count')
        check(same==len(geo['topology']['same_winding_double_edges']),'independent_same_direction_count')
        reasons=[]
        if not 128<=n<=256:reasons.append('N_OUTSIDE_ADAPTER')
        if geo['illegal_pairs']:reasons.append('FULL_TARGET_ILLEGAL_FINITE_PAIRS')
        if geo['unresolved_pairs']:reasons.append('FULL_TARGET_GEOMETRY_UNRESOLVED')
        if nonmanifold:reasons.append('FULL_TARGET_NONMANIFOLD_EDGES')
        if lin['orientation']['parity_conflicts']:reasons.append('UNSOLVABLE_WINDING')
        if same:reasons.append('TARGET_SAME_DIRECTION_DOUBLE_EDGE')
        if np.any(newarea==0):reasons.append('TARGET_ZERO_AREA')
        check(reasons==row['reasons'],'independent_gate_decision_matches')
        passed=not reasons;check(passed==(row['status']=='TARGET_GEOMETRY_PASS_PENDING_VISUAL'),'pass_status_matches')
        if passed:
            check(row['duplicate_excess_removed']==row['refined_new_vertices']==row['faces_flipped']==0,'six_pass_only_zero_removal')
            kept=[i for i in range(len(f0)) if i not in removed]
            check(np.array_equal(f,f0[kept]),'pass_faces_exact_original_subset')
            check(all(oldarea[x]==0 for x in removed),'pass_no_positive_model_area_removed')
            check(abs(float(newarea.sum()-oldarea.sum()))<1e-12,'pass_total_effective_area_unchanged')
        else:reason_sets[' + '.join(reasons)]+=1;reason_counts.update(reasons)
        results.append(dict(parent_id=row['parent_id'],status=row['status'],reasons=reasons,N_original=len(f0),N=n,
            zero_faces_removed=len(removed),model_positive_but_source_exact_zero_removed=[x for x in sorted(removed) if oldarea[x]>0],
            duplicate_excess_removed=row['duplicate_excess_removed'],refined_new_vertices=row['refined_new_vertices'],faces_flipped=row['faces_flipped'],
            original_effective_area_sum=float(oldarea.sum()),single_surface_expected_area=area_expected,target_area_sum=float(newarea.sum()),
            max_descendant_plane_error_model=max_surface_error,max_descendant_relative_area_error=max_relative_area_error,
            all_original_vertices_byte_equal=True,normalization_unchanged=True,all_positive_source_support_has_descendant=True,
            full_finite_pair_result='HASH_VERIFIED_REUSED_NOT_RERUN'))
    check(sum(not r['reasons'] for r in results)==6,'six_PASS');check(sum(bool(r['reasons']) for r in results)==26,'26_BLOCKED')
    for p,h in HASHES.items():check(sha(p)==h,'input_unchanged_after')
    result=dict(status='PASS',scope='Independent array, lineage, surface support, area, topology and gate accounting; frozen finite-pair results reused without rescore.',
        parents=results,geometric_pass_parents=6,blocked_parents=26,blocked_reason_combinations=dict(reason_sets),blocked_reason_counts_nonexclusive=dict(reason_counts),
        pass_total_zero_faces_removed=sum(r['zero_faces_removed'] for r in results if not r['reasons']),pass_positive_area_removed=0,
        notes=['Six geometric PASS targets are still pending semantic visual acceptance; this review is not visual review.',
               'All six PASS have no duplicate consolidation, new vertices or face flips; they only remove exact zero-area faces or are unchanged.',
               'No full pair detector rerun, new repair, data expansion or model call.'],checks=dict(CHECKS),input_hashes=HASHES,
        wall_seconds=time.time()-started,model_calls=0,GPU_compute=0,new_pair_scores=0,source_mutations=0)
    output=BASE/'audit/target_independent_review.json'
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','geometric_pass_parents','blocked_parents','blocked_reason_combinations','blocked_reason_counts_nonexclusive','pass_total_zero_faces_removed','wall_seconds')}))
if __name__=='__main__':main()
