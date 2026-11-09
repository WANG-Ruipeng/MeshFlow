"""Independent validation of saved source-interface records; no original scorer imports.
Only hashes, saved per-face areas, integer incidence/BFS and stored boundary distances.
Zero model, pair_scan, orientation solver, all-pairs distance or mesh writes.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import collections, csv, datetime, hashlib, json, math, time, traceback
import numpy as np

OUT=Path(__file__).resolve().parents[1]
CACHE={};CHECKS=collections.Counter();RECORDS=[];FAILURES=[];MAX_ERROR=collections.defaultdict(float)
LEGAL={'shared_edge_contact','shared_vertex_contact','coplanar_boundary_or_tolerance_contact',
       'noncoplanar_edge_or_tangent_contact','point_or_tangent_contact'}
ILLEGAL={'proper_transverse_penetration','coplanar_positive_area_overlap'}
def check(test, message):
    CHECKS[message]+=1
    if not test:raise AssertionError(message)
def near(a,b,message,rtol=2e-12,atol=2e-12):
    if a is None or b is None:check(a is b,message);return
    error=abs(float(a)-float(b));MAX_ERROR[message]=max(MAX_ERROR[message],error)
    check(math.isclose(float(a),float(b),rel_tol=rtol,abs_tol=atol),message)
def load(path,expected=None,npz=False):
    path=str(path)
    if path not in CACHE:
        p=Path(path);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if npz:
            with np.load(p,allow_pickle=False) as z:
                wanted={'full_target','model_vertices','source_faces','source_face_vertex_ids','source_corner_permutations',
                        'retained_to_original_face','C','C_original','C_original_source_vertex_ids','C_source_vertex_ids',
                        'C_source_corner_permutations','source_face_ids','free_source_ids'}
                data={k:z[k].copy() for k in z.files if k in wanted}
        else:data=json.loads(raw.decode('utf-8-sig'))
        CACHE[path]=(data,digest,len(raw))
    data,digest,size=CACHE[path]
    if expected:check(digest==expected,'source_file_sha256')
    return data
def bits(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def graph_walk(edges,K,oriented=False,original_flags=None):
    adjacency=collections.defaultdict(set)
    for e in edges:
        key=(e['vertex_u'],e['vertex_v'])
        opposed=original_flags[key] if original_flags is not None else e['winding_opposite']
        if e['valid_edge'] and (not oriented or opposed):
            i,j=e['incident_face_ids'];adjacency[i].add(j);adjacency[j].add(i)
    seen=set(range(K));todo=list(seen)
    while todo:
        node=todo.pop()
        for other in adjacency[node]-seen:seen.add(other);todo.append(other)
    return sorted(i for i in seen if i>=K)
def area_sum(rows,area_by_source,order):
    return math.fsum(area_by_source[int(order[i])] for i in rows)
def validate_one(job,reg):
    cid=job['condition_id'];fp=OUT/'conditions'/cid/'result.json'
    result=load(fp);check(result['status']=='PASS','result_status')
    check(result['metadata']==job and result['identity']['job']==job,'result_registered_identity')
    check(result['identity']['entry_sha256']==reg['entry_sha256'],'result_entry_hash')
    check(result['new_pair_tests']==0,'zero_new_pair_scan')
    refs=job['inputs']
    p=load(refs['parent']['path'],refs['parent']['sha256'],True)
    c=load(refs['condition']['path'],refs['condition']['sha256'],True)
    q=load(refs['quality']['path'],refs['quality']['sha256'])
    N,K=job['N'],job['K'];cids=c['source_face_ids'];yids=c['free_source_ids'];order=np.r_[cids,yids]
    tri=p['full_target'].reshape(N,3,3);V=p['model_vertices'];F=p['source_faces'][order]
    check(q['input_npz_sha256']==refs['parent']['sha256'],'quality_parent_hash')
    check(hashlib.sha256(tri.tobytes()).hexdigest()==q['full_target_sha256'],'quality_coordinate_hash')
    check(q['N']==N and q['L']==reg['L'] and q['independent_verification'] is True,'quality_scope')
    check(q['receipt']['new_unique_pair_tests']==N*(N-1)//2,'cached_complete_pair_budget')
    for file,sha in q['identity']['detector_source_hashes'].items():
        check(sha==reg['frozen_code']['scorers/'+Path(file).name]['sha256'],'cached_detector_version')
    check(len(cids)==K and np.array_equal(np.sort(order),np.arange(N)),'C_Y_partition')
    check(bits(V[p['source_face_vertex_ids']],tri),'canonical_indexed_coordinates')
    check(bits(tri[cids],c['C']) and bits(V[F][:K],c['C_original']),'C_prefix_coordinates')
    check(np.array_equal(F[:K],c['C_original_source_vertex_ids']),'C_prefix_original_vertex_IDs')
    check(np.array_equal(p['source_face_vertex_ids'][cids],c['C_source_vertex_ids']),'C_canonical_vertex_IDs')
    check(np.array_equal(p['source_corner_permutations'][cids],c['C_source_corner_permutations']),'C_saved_permutation')
    check(bits(np.take_along_axis(V[F],p['source_corner_permutations'][order][...,None],axis=1),tri[order]),'whole_source_winding_inverse')
    cset=set(map(int,cids));yset=set(map(int,yids))
    pairmap={tuple(map(int,x['pair'])):x['classification'] for x in q['pair_contacts']}
    check(len(pairmap)==len(q['pair_contacts']),'unique_cached_pairs')
    cf=[key for key in pairmap if (key[0] in cset)!=(key[1] in cset)]
    ff=[key for key in pairmap if key[0] in yset and key[1] in yset]
    check(result['cached_CF_contact_faces']==len(cf) and result['cached_FF_contact_faces']==len(ff),'cached_C_Y_face_remap_count')
    area_by_source={int(row['source_face_id']):float(row['area']) for row in q['basic_geometry']['per_face']}
    check(set(area_by_source)==set(range(N)),'cached_per_face_area_identity')
    denominator=math.fsum(area_by_source[int(i)] for i in yids)
    graph_records=[]
    for domain,g in result['graphs'].items():
        if domain=='SOURCE_INDEX':v,f=V,F
        else:
            check(domain=='EXACT_COORD_UINT32','graph_domain')
            flat=np.ascontiguousarray(V[F]).reshape(-1,3)
            _,first,inverse=np.unique(flat.view(np.uint32).reshape(-1,3),axis=0,return_index=True,return_inverse=True)
            v=flat[first];f=inverse.reshape(-1,3)
            check(bits(v[f],V[F]),'coordinate_index_preserves_coordinates')
        incidence=collections.defaultdict(list);dups=collections.defaultdict(list)
        for faceid,face in enumerate(f):
            dups[tuple(sorted(map(int,face)))].append(faceid)
            for u,w in zip(face,np.roll(face,-1)):
                incidence[tuple(sorted((int(u),int(w))))].append((faceid,int(u),int(w)))
        duplicate_faces={i for rows in dups.values() if len(rows)>1 for i in rows}
        eligible={key:val for key,val in incidence.items() if any(i>=K for i,_,_ in val)}
        saved=g['edges'];keys=[(e['vertex_u'],e['vertex_v']) for e in saved]
        check(len(keys)==len(set(keys)) and set(keys)==set(eligible),'saved_graph_edge_coverage')
        original_flags={}
        for e in saved:
            key=(e['vertex_u'],e['vertex_v']);inc=eligible[key];faceids=[i for i,_,_ in inc]
            check(e['incidence']==len(inc),'saved_edge_incidence')
            check(e['incident_face_ids']==faceids,'saved_edge_face_incidence')
            check(e['source_face_ids']==[int(order[i]) for i in faceids],'saved_edge_source_face_remap')
            iscf=any(i<K for i in faceids) and any(i>=K for i in faceids)
            check(e['is_C_F']==iscf,'edge_C_Y_roles')
            expected=[];cls='NA_NONMANIFOLD_OR_BOUNDARY';opposed=None
            if len(inc)!=2:expected=['NOT_TWO_INCIDENT_FACES']
            else:
                (i,a,b),(j,cc,dd)=inc;opposed=(a==dd and b==cc)
                if i==j or key[0]==key[1] or e['length_model']<=0 or area_by_source[int(order[i])]<=0 or area_by_source[int(order[j])]<=0:
                    expected.append('DEGENERATE')
                if i in duplicate_faces or j in duplicate_faces:expected.append('DUPLICATE_FACE')
                sourcepair=tuple(sorted((int(order[i]),int(order[j]))))
                cls=pairmap.get(sourcepair,'UNKNOWN_MISSING_SHARED_EDGE_PAIR')
                if cls not in LEGAL:expected.append('ILLEGAL' if cls in ILLEGAL else 'UNKNOWN_OR_NOT_CONTACT')
            original_flags[key]=opposed
            check(e['pair_classification']==cls,'saved_edge_cached_legality_remap')
            check(e['rejected_reasons']==expected and e['valid_edge']==(not expected),'saved_edge_validity_rules')
        shared=[e for e in saved if e['is_C_F']];valid=[e for e in shared if e['valid_edge']]
        bybefore=g['as_stored'];byafter=g['export_oriented']
        base=dict(C_F_shared_edge_count=len(shared),C_F_shared_edge_total_length=math.fsum(e['length_model'] for e in shared),
            C_F_valid_edge_count=len(valid),C_F_has_valid_edge=int(bool(valid)),
            C_F_valid_edge_total_length=math.fsum(e['length_model'] for e in valid),
            C_F_nonmanifold_edge_count=sum(e['incidence']>2 for e in shared),
            C_F_rejected_duplicate_edge_count=sum('DUPLICATE_FACE' in e['rejected_reasons'] for e in shared),
            C_F_rejected_illegal_edge_count=sum('ILLEGAL' in e['rejected_reasons'] for e in shared),
            C_F_rejected_unknown_edge_count=sum('UNKNOWN_OR_NOT_CONTACT' in e['rejected_reasons'] for e in shared),
            graph_valid_edges=sum(e['valid_edge'] for e in saved),exact_duplicate_faces=len(duplicate_faces),
            nonmanifold_edges=sum(len(a)>2 for a in incidence.values()),boundary_edges=sum(len(a)==1 for a in incidence.values()))
        reach=graph_walk(saved,K);saved_sources=[int(order[i]) for i in reach]
        check(saved_sources==g['reached_Y_source_faces'],'undirected_BFS_saved_indices')
        for prefix,summary,flags in [('AS_STORED',bybefore,original_flags),('EXPORT_ORIENTED',byafter,None)]:
            for key,num in base.items():near(summary[key],num,'graph_scalar_'+key)
            near(summary['free_area'],denominator,'source_Y_area_denominator')
            check(summary['reached_free_faces']==len(reach),'undirected_reached_count')
            area=area_sum(reach,area_by_source,order)
            near(summary['reached_free_area'],area,'undirected_reached_area')
            near(summary['reached_free_area_fraction'],area/denominator,'undirected_reached_fraction')
            directed=graph_walk(saved,K,True,flags);darea=area_sum(directed,area_by_source,order)
            check(summary['opposite_winding_reached_free_faces']==len(directed),'directed_BFS_count')
            near(summary['opposite_winding_reached_free_area'],darea,'directed_reached_area')
            near(summary['opposite_winding_reached_free_area_fraction'],darea/denominator,'directed_reached_fraction')
            opposite=sum(bool(flags[(e['vertex_u'],e['vertex_v'])] if flags is not None else e['winding_opposite']) for e in valid)
            check(summary['C_F_valid_opposite_winding_edge_count']==opposite,'directed_CF_edge_count')
        direction_only={'C_F_valid_opposite_winding_edge_count','opposite_winding_reached_free_faces',
                        'opposite_winding_reached_free_area','opposite_winding_reached_free_area_fraction'}
        check(all(bybefore[k]==byafter[k] for k in bybefore if k not in direction_only),'direction_undirected_invariants')
        diag=g['diagnostic'];ori=g['orientation_summary']
        check(diag['same_unordered_faces_after_orientation'] and diag['C_original_order_unchanged'],'saved_orientation_contract')
        for field in ['C_face_rows_bytewise_preserved','C_triangle_coordinate_order_bytewise_preserved','coordinates_unchanged','face_count_unchanged','face_vertex_sets_unchanged']:
            check(ori[field] is True,'orientation_receipt_'+field)
        check(diag['orientation_flips']==ori['flipped_free_faces'],'orientation_flips_receipt')
        countC=collections.Counter(tuple(sorted((int(u),int(w)))) for face in f[:K] for u,w in zip(face,np.roll(face,-1)))
        boundarykeys={key for key,n in countC.items() if n==1}
        linked=[e for e in valid if (e['vertex_u'],e['vertex_v']) in boundarykeys]
        check(diag['C_boundary_edge_count']==len(boundarykeys),'C_boundary_edge_index_count')
        check(diag['valid_interface_on_C_boundary_count']==len(linked),'valid_interface_boundary_count')
        total=math.fsum(e['length_model'] for e in linked)
        near(diag['valid_interface_boundary_length_model'],total,'valid_interface_boundary_length_sum')
        near(diag['valid_interface_boundary_length_fraction'],total/diag['C_boundary_length_model'] if diag['C_boundary_length_model'] else None,'boundary_coverage_fraction')
        if domain=='SOURCE_INDEX':
            near(diag['C_boundary_length_model'],result['boundary_audit']['boundary_length_model'],'source_C_boundary_denominator')
        graph_records.append(dict(domain=domain,saved_edges=len(saved),valid_CF_edges=len(valid),reached_Y_faces=len(reach),
                                  source_Y_area=denominator,orientation_flips=diag['orientation_flips']))
    # Direct source-index open-boundary membership, no coordinate welding.
    ce=collections.Counter(tuple(sorted((int(u),int(w)))) for face in F[:K] for u,w in zip(face,np.roll(face,-1)))
    boundary_ids=sorted({i for e,n in ce.items() if n==1 for i in e})
    bs=result['boundary'];check([x['source_vertex_id'] for x in bs]==boundary_ids,'saved_boundary_source_IDs')
    check(len(bs)==result['boundary_audit']['boundary_vertex_count']==result['source_index_boundary_vertex_denominator'],'boundary_denominator')
    yvertices=set(map(int,F[K:].reshape(-1)))
    check(result['source_index_boundary_vertices_in_Y']==sum(i in yvertices for i in boundary_ids),'boundary_Y_source_ID_matching')
    zeros=0
    for row in bs:
        check(row['nearest_Y_source_face'] in yset,'boundary_nearest_face_is_Y')
        corner=row['nearest_Y_canonical_corner'];check(0<=corner<3,'boundary_nearest_corner_range')
        d=row['distance_model'];check(d>=0 and math.isfinite(d),'saved_boundary_distance_finite')
        near(row['distance_L_fraction'],d/reg['L'],'boundary_distance_scale')
        if d==0:
            zeros+=1
            check(np.array_equal(V[row['source_vertex_id']],tri[row['nearest_Y_source_face'],corner]),'zero_distance_Y_coordinate_witness')
    check(len(result['AVMR'])==8,'all8_thresholds')
    for saved,setting in zip(result['AVMR'],reg['settings']):
        name,t,scale,tau=setting
        check((saved['setting'],saved['threshold_model'],saved['scale'],saved['tau'])==(name,t,scale,tau),'threshold_identity')
        matches=sum(x['distance_model']<t for x in bs)
        check(saved['boundary_vertex_count']==len(bs) and saved['matched_count']==matches,'threshold_exact_counts')
        if bs:
            near(saved['VMR'],matches/len(bs),'VMR_from_exact_counts')
            check(saved['perfect']==int(matches==len(bs)),'PMR_from_exact_counts')
        else:check(saved['VMR'] is None and saved['perfect'] is None and saved['status']=='NA_EMPTY_BOUNDARY','empty_boundary_NA')
    dev=job['parent_role']=='NEW_DEV32'
    if dev:
        strict=result['AVMR'][0]
        check(strict['setting']=='raw_tau_1e-6' and strict['perfect']==1,'DEV_strict_PMR_one')
        check(len(bs)>0 and zeros==len(bs),'DEV_all_boundary_zero_witness')
    return dict(condition_id=cid,parent_role=job['parent_role'],status='PASS',boundary_vertices=len(bs),zero_distances=zeros,
                source_Y_faces=N-K,graphs=graph_records)
def main():
    start=time.perf_counter();reg=load(OUT/'protocol/registration.json')
    actual_entry=hashlib.sha256((OUT/'code/audit.py').read_bytes()).hexdigest()
    check(actual_entry==reg['entry_sha256'],'adapter_source_hash')
    for _,row in reg['frozen_code'].items():
        check(hashlib.sha256(Path(row['local_snapshot']).read_bytes()).hexdigest()==row['sha256'],'frozen_function_hash')
    check(len(reg['jobs'])==258 and len({j['condition_id'] for j in reg['jobs']})==258,'registered258_unique')
    for j in reg['jobs']:
        try:RECORDS.append(validate_one(j,reg))
        except Exception as exc:FAILURES.append(dict(condition_id=j['condition_id'],error=repr(exc),traceback=traceback.format_exc()))
    status='PASS' if not FAILURES and len(RECORDS)==258 else 'FAIL'
    scope=dict(new_pair_scans=0,model_calls=0,original_metric_function_calls=0,orientation_solver_calls=0,
               all_pairs_distance_calls=0,source_mesh_writes=0,
               verification='Source hashes/bytes and integer incidence; BFS saved edges; source cached per-face area sums; arithmetic over saved distances; zero-distance coordinate witnesses only.',
               limitation='No independent replay of orientation solver or detector. Oriented edges are saved, so directed reach is independently checked; post-orientation faces/flips were not stored, so no new solver-optimality claim. Nonzero nearest distances are checked arithmetically, not by a new full nearest-neighbor search.')
    result=dict(status=status,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start,
                checked_conditions=len(RECORDS),registered_conditions=258,graph_domains_checked=2*len(RECORDS),
                DEV_conditions_checked=sum(x['parent_role']=='NEW_DEV32' for x in RECORDS),
                total_saved_edges_checked=sum(g['saved_edges'] for x in RECORDS for g in x['graphs']),
                checks=dict(CHECKS),max_abs_numeric_differences=dict(MAX_ERROR),failures=FAILURES,records=RECORDS,
                scope=scope,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                input_hashes=[dict(path=p,sha256=h,bytes=n) for p,(_,h,n) in CACHE.items()])
    dest=OUT/'protocol/independent_validation.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={k:result[k] for k in ['status','checked_conditions','graph_domains_checked','DEV_conditions_checked','total_saved_edges_checked','wall_seconds']}
    print(json.dumps(summary))
    if FAILURES:print(json.dumps(FAILURES[:3],ensure_ascii=False))
    if status!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
