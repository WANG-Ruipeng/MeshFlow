"""Independent saved-artifact verification; NumPy only, no model/GEO/scorer calls."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, time
from pathlib import Path
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_source_winding_export_v1'
AUDIT=OUT/'independent_audit'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def norm(a):return np.sqrt(np.sum(np.asarray(a,dtype=np.float64)**2,axis=-1))
def near(a,b,atol=2e-12,rtol=2e-11):
    assert np.shape(a)==np.shape(b) and np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=False),'Numerical mismatch'
def checked(p):
    p=Path(p).absolute()
    assert not any(q.is_symlink() for q in (p,*p.parents)),'Symlink output'
    p=p.resolve();assert p!=AUDIT.resolve() and p.is_relative_to(AUDIT.resolve()),'Outside owned audit'
    return p
def atomic(p,fn):
    p=checked(p);p.parent.mkdir(parents=True,exist_ok=True);temp=checked(p.with_name(p.name+'.tmp'))
    with temp.open('w',encoding='utf-8',newline='') as f:fn(f);f.flush();os.fsync(f.fileno())
    temp.replace(p)
def write(p,obj):atomic(p,lambda f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False))
def csvwrite(p,data):
    def emit(f):
        fields=list(dict.fromkeys(k for r in data for k in r)) or ['status']
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(data)
    atomic(p,emit)
def verify(mapping):
    for p,h in mapping.items():assert sha(p)==h,p
    return len(mapping)


def permute(a,p):
    a=np.asarray(a);p=np.asarray(p,dtype=np.int64)
    assert a.shape[:2]==p.shape and p.shape[1]==3
    idx=p.reshape(p.shape+(1,)*(a.ndim-2))
    return np.take_along_axis(a,idx,axis=1)

def check_source_restoration(condition,parent,model,restored,K):
    """Derive the inverse without calling restoration or orientation code."""
    cp=condition['C_source_corner_permutations'];ids=condition['C_source_vertex_ids']
    retained=condition['source_face_ids'].astype(np.int64)
    original=condition['original_source_face_ids'].astype(np.int64)
    assert cp.shape==(K,3) and np.array_equal(np.sort(cp,axis=1),np.tile(np.arange(3),(K,1)))
    # Explicit inverse map (rather than the production argsort implementation).
    inverse=np.empty((K,3),dtype=np.int64)
    for face in range(K):
        for old_slot,source_slot in enumerate(cp[face]):
            inverse[face,int(source_slot)]=old_slot
    assert np.array_equal(permute(cp,inverse),np.tile(np.arange(3),(K,1)))
    assert np.array_equal(parent['retained_to_original_face'][retained],original)
    assert np.array_equal(parent['source_corner_permutations'][retained],cp)
    assert np.array_equal(parent['source_face_vertex_ids'][retained],ids)
    original_ids=parent['original_source_faces'][original]
    assert np.array_equal(permute(original_ids,cp),ids)
    C=condition['C'];source_C=parent['model_vertices'][original_ids]
    assert same(parent['full_target'][retained].reshape(K,3,3),C)
    assert same(parent['model_vertices'][ids],C) and same(permute(source_C,cp),C)
    assert same(permute(C,inverse),source_C)
    assert same(model['output'][:K],C)
    n=len(model['faces']);allperm=np.tile(np.arange(3,dtype=np.int64),(n,1));allperm[:K]=inverse
    expected_faces=permute(model['faces'],allperm)
    assert same(restored['vertices'],model['vertices']) and same(restored['faces'],expected_faces)
    assert same(restored['output'],permute(model['output'],allperm))
    assert same(restored['C'],source_C) and same(restored['output'][:K],source_C)
    assert same(restored['output'][K:],model['output'][K:])
    assert same(restored['source_face_ids'],model['source_face_ids'])
    assert same(restored['source_corner_ids'],permute(model['source_corner_ids'],allperm))
    assert same(restored['vertices'][restored['faces']],restored['output'])
    assert np.array_equal(np.sort(restored['faces'],axis=1),np.sort(model['faces'],axis=1))
    assert same(restored['free'],model['output'][K:])
    changed=np.any(inverse!=np.arange(3),axis=1)
    odd=np.asarray([sum(int(p[i])>int(p[j]) for i in range(3) for j in range(i+1,3))%2 for p in inverse])
    return allperm,dict(C_faces=K,source_original_face_ids_verified=True,source_vertex_ids_verified=True,
        source_C_exact_coordinates_and_corner_order=True,source_C_changed_corner_order_faces=int(changed.sum()),
        source_C_odd_permutation_faces=int(odd.sum()),source_C_even_nonidentity_permutation_faces=int((changed&(odd==0)).sum()),
        free_bitwise_unchanged=True,vertex_array_bytes_unchanged=True,
        unordered_index_triangles_unchanged=True,source_face_ids_unchanged=True)

def check_corner_attributes(original,output,final_permutation):
    """Check output-corner gathers from the retained original-input audit domain."""
    mapping={
        'corner_displacement':'output_corner_displacement',
        'corner_displacement_vector':'output_corner_displacement_vector',
        'corner_target_input_ids':'output_corner_target_input_ids',
        'corner_vertex_ids':'output_corner_vertex_ids',
        'corner_status':'output_corner_status',
        'corner_C_candidate_count':'output_corner_C_candidate_count',
    }
    # The final source_corner_ids remain identifiers in the original input
    # triangle's canonical order. Scalar target IDs likewise stay original.
    sf=original['source_face_ids'];sc=permute(original['source_corner_ids'],final_permutation)
    assert same(output['source_face_ids'],sf) and same(output['source_corner_ids'],sc)
    checked_count=0
    for source,target in mapping.items():
        if source not in original:continue
        preserved='input_'+source
        assert preserved in output and target in output,(preserved,target)
        assert same(output[preserved],original[source])
        expected=original[source][sf[:,None],sc]
        assert same(output[target],expected),(source,target)
        checked_count+=1
    if 'vertex_source_input_corner_ids' in output:
        assert same(output['vertex_source_input_corner_ids'],original['vertex_source_input_corner_ids'])
    return checked_count

def direction_checks(faces,K):
    def count(part):
        inc=collections.defaultdict(list)
        for fi,f in enumerate(part):
            for j in range(3):
                u,v=map(int,(f[j],f[(j+1)%3]));inc[tuple(sorted((u,v)))].append((fi,u,v))
        eligible=[v for v in inc.values() if len(v)==2 and v[0][0]!=v[1][0]]
        return len(eligible),sum(e[0][1:]==e[1][1:] for e in eligible),sum(len(e)>2 for e in inc.values())
    c=count(faces[:K]);whole=count(faces)
    return dict(C_C_local_two_face_edges=c[0],C_C_local_same_direction_edges=c[1],
        C_C_local_nonmanifold_edges=c[2],whole_two_face_same_direction_edges=whole[1])

def ply_read(path):
    data=Path(path).read_bytes();marker=b'end_header\n';end=data.index(marker)+len(marker)
    header=data[:end].decode('ascii').splitlines();assert header[0]=='ply'
    n=int(next(x for x in header if x.startswith('element vertex ')).split()[-1])
    m=int(next(x for x in header if x.startswith('element face ')).split()[-1])
    assert [x for x in header if x.startswith('property ')]==['property float x','property float y','property float z','property list uchar int vertex_indices']
    if header[1]=='format binary_little_endian 1.0':
        assert len(data)-end==n*12+m*13
        v=np.frombuffer(data,dtype='<f4',count=n*3,offset=end).reshape(n,3).copy()
        face_dtype=np.dtype([('count','u1'),('ids','<i4',(3,))],align=False)
        records=np.frombuffer(data,dtype=face_dtype,count=m,offset=end+n*12)
        assert np.all(records['count']==3);f=records['ids'].astype(np.int64)
    else:
        assert header[1]=='format ascii 1.0';lines=data[end:].decode('ascii').splitlines()
        assert len(lines)==n+m
        v=np.asarray([[float(x) for x in line.split()] for line in lines[:n]],dtype=np.float32)
        f=np.asarray([[int(x) for x in line.split()[1:]] for line in lines[n:]],dtype=np.int64)
        assert all(line.split()[0]=='3' for line in lines[n:])
    return v,f


def audit():
    from r2.tools.c_restore_bounded_v1.independent_audit import check_orientation
    from r2.tools.c_restore_coupling_v1.independent_audit import graph_audit,eq,num
    t=time.monotonic();reg=read(OUT/'registration.json');done=read(OUT/'execution_completion.json')
    assert done['status']=='PASS' and done['outputs']==done['source_C_restorations']==done['orientation_solves']==36
    for k in ['model_loads','model_forwards','training_updates','new_GEO_C','new_full_quality_scores','new_images']:assert done[k]==0
    assert done['frozen_sources_unchanged'] and done['scientific_stage_stopped']
    assert reg['stages']==['MODEL_POST','RESTORED_C_ONLY','ORIENTED']
    records=reg['records'];assert len(records)==36 and len({r['condition_id'] for r in records})==6
    assert collections.Counter(r['method'] for r in records)=={'D0_BASE':12,'RBF':12,'BLEND':12}
    verify(reg['runner_hashes']);assert sha(reg['orientation_solver_path'])==reg['orientation_solver_sha256']
    small={p:h for p,h in reg['protected'].items() if Path(p).suffix.lower() not in {'.pt','.pth','.ckpt'}}
    weights={p:h for p,h in reg['protected'].items() if p not in small};verify(small)
    assert done['protected_files_unchanged']==len(reg['protected'])
    assert len(list((OUT/'events').glob('*/*_started.json')))==len(list((OUT/'events').glob('*/*_complete.json')))==36
    assert len(list((OUT/'outputs').glob('*/*/orientation_started.json')))==36
    controls=read(OUT/'restoration_checks/completion.json');assert controls['status']=='PASS'
    verify(controls.get('source_hashes',{}));verify(controls.get('files',{}))
    case_rows=[];graph_checks=0;attribute_checks=0;boundary_checks=0;metric_checks=0;pair_checks=0
    top_table={(r['method'],r['request_id'],r['stage']):r for r in rows(OUT/'topology_per_output.csv')}
    qtable={(r['method'],r['request_id'],r['stage']):r for r in rows(OUT/'quality_inherited_per_output.csv')}
    pairtable={(r['method'],r['request_id']):r for r in rows(OUT/'paired_per_parent_noise.csv')}
    vmrows=rows(OUT/'AVMR_PMR_per_output.csv');vtable={(r['method'],r['request_id'],r['stage'],r['setting']):r for r in vmrows}
    assert len(top_table)==len(qtable)==108 and len(vtable)==864 and len(pairtable)==36
    for r in records:
        rid=r['request_id'];method=r['method'];K=int(r['K']);folder=OUT/'outputs'/method/rid
        a=arrays(r['post_path']);condition=arrays(r['condition_source']);parent=arrays(r['parent_prepared_path'])
        assert sha(r['post_path'])==r['post_sha256'] and sha(r['parent_prepared_path'])==r['parent_prepared_sha256']
        q=read(folder/'receipt.json');d=read(r['diagnostic_path']);verify(q['files']);verify(d['quality']['files'])
        assert q['status']=='PASS' and q['identity']['post_sha256']==sha(r['post_path'])
        assert q['identity']['condition_sha256']==sha(r['condition_source']) and q['identity']['parent_sha256']==sha(r['parent_prepared_path'])
        assert q['identity']['registration_sha256']==sha(OUT/'registration.json')
        assert q['quality_reference']==d['quality'] and q['original_metrics']==d['metrics'] and q['original_boundary_vertices']==d['vertices']
        assert q['orientation_solves']==1 and q['model_forwards']==q['new_full_quality_scores']==q['new_GEO_C']==0
        restored=arrays(folder/'restored_C.npz');final=arrays(folder/'oriented.npz')
        first_perm,rest_result=check_source_restoration(condition,parent,a,restored,K)
        assert np.array_equal(restored['corner_permutation_to_MODEL_POST'],first_perm) and not restored['free_face_flips'].any()
        relperm=np.tile(np.arange(3,dtype=np.int64),(len(a['faces']),1));relperm[final['free_face_flips']]=[0,2,1]
        composed=permute(first_perm,relperm)
        assert np.array_equal(final['corner_permutation_to_MODEL_POST'],composed)
        adapter=dict(final,flips=final['free_face_flips'],corner_permutation=relperm)
        details=read(folder/'orientation_details.json');assert details['summary']==q['orientation']
        or_result=check_orientation(a['vertices'],restored['faces'],K,adapter,details['components'],details['edges'],details['summary'])
        assert same(final['C'],restored['C']) and same(final['faces'][:K],restored['faces'][:K])
        for name,state,p in [('restored_C',restored,first_perm),('oriented',final,composed)]:
            attribute_checks+=check_corner_attributes(a,state,p)
            assert same(state['input_faces'],a['input_faces']) and int(state['known_faces'])==K
            for key in ['vertex_source_input_corner_ids','source_face_kept','source_face_reason','duplicate_of_source_face']:
                assert same(state[key],a[key]),key
            assert same(state['faces'],state['output_corner_vertex_ids'])
            assert np.array_equal(state['C_original_source_vertex_ids'],parent['original_source_faces'][condition['original_source_face_ids']])
            assert same(state['C_original_source_face_ids'],condition['original_source_face_ids'])
            assert same(state['C_canonical_source_vertex_ids'],condition['C_source_vertex_ids'])
            assert same(state['C_source_corner_permutation'],condition['C_source_corner_permutations'])
            assert np.array_equal(state['C_inverse_corner_permutation'],first_perm[:K])
            v,f=ply_read(folder/(name+'.ply'));assert same(v,state['vertices']) and same(f,state['faces'])
        pairdata=read(d['quality']['pair_audit_path'])
        bv=rows(folder/'boundary_vertices.csv');vv={(x['stage'],int(x['source_vertex_id'])):x for x in bv}
        percasevm={(x['stage'],x['setting']):x for x in rows(folder/'AVMR_PMR.csv')}
        stage_data=[('MODEL_POST',a,np.tile(np.arange(3,dtype=np.int64),(len(a['faces']),1))),('RESTORED_C_ONLY',restored,first_perm),('ORIENTED',final,composed)]
        for stage,state,p in stage_data:
            top=q['topology'][stage]
            graph_checks+=graph_audit(state['output'],state['vertices'],state['faces'],K,rows(folder/('valid_edges_'+stage+'.csv')),top,pairdata)
            for key,value in direction_checks(state['faces'],K).items():eq(top[key],value)
            for key,value in top.items():
                if isinstance(value,(int,float)) or value is None:eq(top_table[method,rid,stage].get(key),value)
            if stage!='MODEL_POST':assert top['C_C_local_same_direction_edges']==0
            assert qtable[method,rid,stage]['value_status']=='INHERITED_GEOMETRY_EQUIVALENCE'
            for key,value in d['quality']['metrics'].items():
                if isinstance(value,(int,float)) or value is None:eq(qtable[method,rid,stage].get(key),value)
                else:assert qtable[method,rid,stage][key]==str(value)
            eq(qtable[method,rid,stage]['free_face_denominator'],len(a['faces'])-K)
            for old in d['vertices']:
                new=vv[stage,int(old['source_vertex_id'])]
                for field in ['source_vertex_id','x','y','z','distance_model','distance_L_fraction','nearest_free_face']:eq(new[field],old[field])
                j=old['nearest_free_face'];corner=old['nearest_free_corner']
                mapped=next(k for k in range(3) if int(p[K+j,k])==corner)
                assert int(new['nearest_free_corner'])==mapped and int(new['original_MODEL_POST_nearest_free_corner'])==corner
                assert new['witness_status']=='REMAP_OF_ORIGINAL_CHOSEN_WITNESS_NOT_NEW_SEARCH'
                assert same(state['output'][K+j,mapped],a['output'][K+j,corner])
                point=np.array([old['x'],old['y'],old['z']],float)
                eq(float(norm(point-state['output'][K+j,mapped])),old['distance_model'])
                boundary_checks+=10
            for metric in d['metrics']:
                for table in [percasevm[stage,metric['setting']],vtable[method,rid,stage,metric['setting']]]:
                    for key,value in metric.items():
                        if isinstance(value,(int,float)) or value is None:eq(table[key],value)
                        else:assert table[key]==str(value)
                    assert table['value_status']=='INHERITED_AFTER_GEOMETRY_EQUIVALENCE';metric_checks+=1
        t0,t1,t2=(q['topology'][stage] for stage in reg['stages'])
        expected=dict(C_C_same_before=t0['C_C_local_same_direction_edges'],C_C_same_after=t2['C_C_local_same_direction_edges'],
            valid_edges_unchanged=t2['C_F_valid_edge_count'],opposed_MODEL_POST=t0['C_F_valid_opposite_winding_edge_count'],
            opposed_RESTORED_C_ONLY=t1['C_F_valid_opposite_winding_edge_count'],opposed_ORIENTED=t2['C_F_valid_opposite_winding_edge_count'],
            unoriented_reach=t0['reached_free_area_fraction'],opposed_reach_MODEL_POST=t0['opposite_winding_reached_free_area_fraction'],
            opposed_reach_RESTORED_C_ONLY=t1['opposite_winding_reached_free_area_fraction'],opposed_reach_ORIENTED=t2['opposite_winding_reached_free_area_fraction'],
            reach_gain=t2['opposite_winding_reached_free_area_fraction']-t0['opposite_winding_reached_free_area_fraction'],
            parity_conflict_components=q['orientation']['parity_conflict_components'],fixed_source_C_conflict_components=q['orientation']['fixed_C_anchor_conflict_components'],
            flipped_free_faces=q['orientation']['flipped_free_faces'],nonmanifold_edges=q['orientation']['nonmanifold_edges'])
        for key,value in expected.items():eq(pairtable[method,rid][key],value);pair_checks+=1
        eq(q['restoration']['C_rows_reordered'],rest_result['source_C_changed_corner_order_faces'])
        eq(q['restoration']['C_winding_reversed_from_canonical'],rest_result['source_C_odd_permutation_faces'])
        case_rows.append(dict(method=method,request_id=rid,status='PASS',**rest_result,**or_result))
    summary_checks=0;alltop=list(top_table.values())
    for s in rows(OUT/'topology_summary.csv'):
        field={'REGION':'region_type','PARENT':'parent_id','NOISE':'candidate'}.get(s['group'])
        sub=[x for x in alltop if x['method']==s['method'] and x['stage']==s['stage'] and (field is None or x[field]==s['value'])]
        eq(s['n'],len(sub))
        for key,value in s.items():
            if key.endswith('_mean'):eq(value,float(np.mean([num(x[key[:-5]]) for x in sub])));summary_checks+=1
        for name,key in [('outputs_with_valid_edges','C_F_valid_edge_count'),('outputs_with_opposed_edges','C_F_valid_opposite_winding_edge_count')]:
            eq(s[name],sum(num(x[key])>0 for x in sub));summary_checks+=1
        for name,key in [('valid_edge_total','C_F_valid_edge_count'),('opposed_edge_total','C_F_valid_opposite_winding_edge_count')]:
            eq(s[name],sum(num(x[key]) for x in sub));summary_checks+=1
    csvwrite(AUDIT/'source_restoration_and_orientation_per_output.csv',case_rows)
    result=dict(status='PASS',outputs=36,stages=3,stage_topology_rows=108,old_method_cached_POST_counts={'D0_BASE':12,'RBF':12,'BLEND':12},
        inherited_quality_source_outputs=36,inherited_quality_stage_rows=108,formal_source_restorations=36,formal_orientation_solves=36,source_restoration_independent_inverse='PASS',
        fixed_restored_C_orientation_independent_union_find='PASS',source_face_corner_attribute_mapping='PASS',
        actual_PLY_roundtrips=72,corner_attribute_fields_checked=attribute_checks,actual_legal_graph_edges_checked=graph_checks,
        inherited_AVMR_rows=len(vtable),AVMR_inheritance_comparisons=metric_checks,nearest_corner_witness_checks=boundary_checks,
        paired_checks=pair_checks,summary_stat_checks=summary_checks,protected_nonweight_files_rehashed=len(small),
        protected_weights_inherited_execution_hashes=weights,weight_hash_verification_basis=dict(path=str(OUT/'execution_completion.json'),sha256=sha(OUT/'execution_completion.json'),protected_count=done['protected_files_unchanged']),
        coordinate_and_connectivity_equivalence_all=True,source_C_order_fixed_after_restoration=True,
        no_geometry_benefit_claim_from_winding=True,quality_status='INHERITED_ORIGINAL_UNORIENTED_EVIDENCE_NOT_RERUN',
        audit_source_sha256=sha(__file__),registration_sha256=sha(OUT/'registration.json'),
        source_dependencies={str(ROOT/'r2/tools/c_restore_bounded_v1/independent_audit.py'):sha(ROOT/'r2/tools/c_restore_bounded_v1/independent_audit.py'),str(ROOT/'r2/tools/c_restore_coupling_v1/independent_audit.py'):sha(ROOT/'r2/tools/c_restore_coupling_v1/independent_audit.py')},
        seconds=time.monotonic()-t,model_loads=0,model_forwards=0,training_updates=0,new_GEO=0,new_quality_scans=0,formal_orientation_solver_calls=0)
    write(AUDIT/'completion.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
    for target in [ROOT/'WORKSPACE.md',OUT/'registration.json',ROOT/'experiments/r2_c_restore_bounded_v1/REPORT.md']:
        try:checked(target)
        except AssertionError:pass
        else:raise AssertionError('Negative output guard failed')
    if a.check_only:print(json.dumps(dict(status='PASS',negative_output_guards=3,model_calls=0)));return
    print(json.dumps(audit(),ensure_ascii=False))
if __name__=='__main__':main()
