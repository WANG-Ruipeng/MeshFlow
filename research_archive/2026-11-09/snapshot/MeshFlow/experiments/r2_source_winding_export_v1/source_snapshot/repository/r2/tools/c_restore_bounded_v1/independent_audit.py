"""Independent saved-artifact verification; NumPy only, no model/GEO/scorer calls."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, time
from pathlib import Path
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_c_restore_bounded_v1'
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

def check_blend(c,proposal,ids,output,detail,diag,L):
    """Reconstruct source multiplicity, subconvex field, cap and FP32 endpoint."""
    k=len(c);centers=proposal[:k].reshape(-1,3).astype(float)
    targets=c.reshape(-1,3).astype(float);free=proposal[k:].reshape(-1,3)
    flat_ids=ids.reshape(-1);counts=collections.Counter(map(int,flat_ids))
    weights=np.array([1.0/counts[int(i)] for i in flat_ids]);b=targets-centers;bn=norm(b)
    q=norm(free.astype(float)[:,None,:]-centers[None,:,:])/(.1*L)
    kernel=np.zeros(q.shape,dtype=float);inside=q<1.;s0=1.-q[inside]
    kernel[inside]=(s0*s0)*(s0*s0)*(4.*q[inside]+1.)
    s=kernel*weights[None,:];support=s>0;ss=s.sum(1);alpha=s/np.maximum(1.,ss)[:,None]
    bmax=np.max(np.where(support,bn[None,:],0.),axis=1);bound=np.minimum(.02*L,bmax)
    field=np.einsum('ij,jk->ik',alpha,b,optimize=False)
    expected=dict(anchor_source_vertex_ids=flat_ids,anchor_proposal_model=centers,
        anchor_target_model=targets,anchor_weight=weights,anchor_displacement_target=b,
        anchor_displacement_target_norm=bn,free_kernel=kernel,free_weighted_support=s,
        free_alpha=alpha,free_support_sum=ss,free_alpha_sum=alpha.sum(1),
        free_support_count=support.sum(1),free_supported=support,
        free_participating_b_max_norm=bmax,free_actual_bound=bound)
    for name,value in expected.items():near(detail[name],value)
    near(detail['free_displacement_unclipped'],field)
    # Different independent contraction order can differ by FP64 roundoff.
    # Once verified, replay the saved field to require bitwise FP32 endpoints.
    field=detail['free_displacement_unclipped'];length=norm(field);clip=length>bound
    factor=np.ones(len(free));factor[clip]=bound[clip]/length[clip]
    applied=field*factor[:,None];endpoint=free.copy();moving=np.any(applied!=0,axis=1)
    endpoint[moving]=(free[moving].astype(float)+applied[moving]).astype(np.float32)
    rounding=norm(endpoint.astype(float)-free)>bound
    endpoint[rounding]=np.nextafter(endpoint[rounding],free[rounding])
    actual=endpoint.astype(float)-free;actual_norm=norm(actual)
    for name,value in dict(free_clip_factor=factor,free_clipped=clip,
        free_displacement_clipped=applied,free_safety_cap_active=length>.02*L,
        free_FP32_cap_rounding_adjusted=rounding,free_actual_displacement_FP32=actual,
        free_actual_displacement_norm=actual_norm,free_actual_bound_margin=bound-actual_norm).items():near(detail[name],value)
    assert same(output[:k],c) and same(output[k:].reshape(-1,3),endpoint)
    no_support=support.sum(1)==0
    assert same(output[k:].reshape(-1,3)[no_support],free[no_support])
    assert np.all(actual_norm<=bound) and np.all(alpha>=0) and np.all(alpha.sum(1)<=1+8*np.finfo(float).eps)
    assert np.all(length<=bmax+32*np.finfo(float).eps*np.maximum(1.,bmax))
    for vertex in counts:
        copies=np.flatnonzero(flat_ids==vertex)
        assert all(same(c.reshape(-1,3)[i],c.reshape(-1,3)[copies[0]]) for i in copies)
        near(weights[copies].sum(),1.)
    for key,value in dict(anchor_corners=len(centers),source_vertices=len(counts),free_corners=len(free),
        supported_free_corners=int((~no_support).sum()),no_support_free_corners=int(no_support.sum()),
        clipped_free_corners=int(clip.sum()),safety_cap_active_free_corners=int((length>.02*L).sum()),
        FP32_cap_rounding_adjusted_corners=int(rounding.sum()),model_calls=0,new_noise=0,matrix_solves=0).items():assert diag[key]==value,key
    near(diag['minimum_actual_bound_margin'],float(np.min(bound-actual_norm)))
    return dict(C_bitwise=True,source_ID_weights_verified=True,subconvex_formula_verified=True,
        output_reconstructed_bitwise=True,unclipped_bound_verified=True,actual_FP32_bound_verified=True,
        no_support_corners=int(no_support.sum()),supported_corners=int((~no_support).sum()),
        safety_cap_active_corners=int((length>.02*L).sum()),rounding_adjusted_corners=int(rounding.sum()),
        max_free_displacement=float(actual_norm.max()),minimum_actual_bound_margin=float(np.min(bound-actual_norm)))


class ParityUnionFind:
    """Independent union-by-size XOR relation checker, not the solver's BFS."""
    def __init__(self,n):
        self.parent=list(range(n));self.size=[1]*n;self.parity=[0]*n
    def find(self,x):
        if self.parent[x]!=x:
            p=self.parent[x];root,parity=self.find(p)
            self.parity[x]^=parity;self.parent[x]=root
        return self.parent[x],self.parity[x]
    def union(self,a,b,wanted):
        ra,pa=self.find(a);rb,pb=self.find(b)
        if ra==rb:return (pa^pb)==wanted
        if self.size[ra]<self.size[rb]:ra,rb=rb,ra
        self.parent[rb]=ra;self.parity[rb]=pa^pb^wanted;self.size[ra]+=self.size[rb]
        return True

def check_orientation(vertices,faces,K,oriented,components,edge_rows,summary):
    n=len(faces);tri=vertices[faces].astype(float)
    area=.5*norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]))
    degenerate=np.array([len(set(map(int,f)))<3 for f in faces])|(area==0)
    incidence=collections.defaultdict(list);dups=collections.Counter(tuple(sorted(map(int,f))) for f in faces)
    for i,face in enumerate(faces):
        for corner in range(3):
            a,b=map(int,(face[corner],face[(corner+1)%3]))
            incidence[tuple(sorted((a,b)))].append((i,corner,a,b))
    assert len(edge_rows)==len(incidence)
    edges=[];uf=ParityUnionFind(n)
    for eid,((a,b),entries) in enumerate(sorted(incidence.items())):
        ids=[x[0] for x in entries];reason=[]
        if a==b:reason.append('SELF_LOOP_EDGE')
        if len(entries)==1:reason.append('BOUNDARY_SINGLE_INCIDENCE')
        elif len(entries)>2:reason.append('NONMANIFOLD_MORE_THAN_TWO_INCIDENCES')
        elif len(entries)!=2:reason.append('NOT_TWO_INCIDENCES')
        if len(entries)==2 and len(set(ids))!=2:reason.append('TWO_INCIDENCES_SAME_FACE')
        if any(degenerate[i] for i in set(ids)):reason.append('DEGENERATE_INCIDENT_FACE')
        eligible=not reason
        opposite=(entries[0][2]==entries[1][3] and entries[0][3]==entries[1][2]) if len(entries)==2 and len(set(ids))==2 and a!=b else None
        required=int(not opposite) if eligible else None
        domain='C_F' if any(i<K for i in ids) and any(i>=K for i in ids) else 'C_C' if all(i<K for i in ids) else 'F_F'
        row=edge_rows[eid]
        for key,value in dict(edge_id=eid,vertex_u=a,vertex_v=b,incidence=len(entries),distinct_incident_faces=len(set(ids)),
            incident_face_ids=ids,incident_local_edges=[x[1] for x in entries],incident_directions=[[x[2],x[3]] for x in entries],
            domain=domain,original_opposite_direction=opposite,eligible_constraint=eligible,required_flip_XOR=required,exclusion_reasons=reason).items():assert row[key]==value,(eid,key)
        near(row['length'],float(norm(vertices[a].astype(float)-vertices[b])))
        edges.append((ids,eligible,required,opposite,domain,reason))
        if eligible:uf.union(ids[0],ids[1],required)
    groups=collections.defaultdict(list)
    for i in range(n):groups[uf.find(i)[0]].append(i)
    groups=sorted(groups.values(),key=lambda g:g[0]);assert len(groups)==len(components)
    membership={i:cid for cid,g in enumerate(groups) for i in g}
    flips=np.zeros(n,bool);parity_conflicts=0;anchor_conflicts=0;satisfiable_count=0
    for cid,members in enumerate(groups):
        c=components[cid];assert c['face_ids']==members and c['component_id']==cid and c['root_face']==members[0]
        constraints=[(eid,e) for eid,e in enumerate(edges) if e[1] and membership[e[0][0]]==cid]
        excluded=[eid for eid,e in enumerate(edges) if not e[1] and any(i in members for i in e[0])]
        root_parity=uf.find(members[0])[1];relative={i:uf.find(i)[1]^root_parity for i in members}
        conflict=any((relative[e[0][0]]^relative[e[0][1]])!=e[2] for _,e in constraints)
        anchors=[i for i in members if i<K];gauges=sorted({relative[i] for i in anchors})
        fixed_conflict=None if conflict else len(gauges)>1
        bad=[i for i in members if degenerate[i]];satisfiable=not conflict and not fixed_conflict and not bad
        parity_conflicts+=conflict;anchor_conflicts+=fixed_conflict is True;satisfiable_count+=satisfiable
        assert c['parity_conflict']==conflict and c['fixed_C_anchor_conflict']==fixed_conflict
        assert c['C_face_ids']==anchors and c['degenerate_face_ids']==bad and c['satisfiable']==satisfiable
        assert c['eligible_constraint_edges']==len(constraints) and c['excluded_incident_edge_ids']==excluded
        if not conflict:assert c['C_anchor_required_gauges']==gauges
        if satisfiable:
            gauge=gauges[0] if anchors else 0
            for i in members:flips[i]=relative[i]^gauge
            assert c['gauge']==gauge
        expected_status='SATISFIABLE_ANCHORED' if satisfiable and anchors else 'SATISFIABLE_UNANCHORED_GAUGE_ONLY' if satisfiable else 'UNCHANGED_PARITY_CONFLICT' if conflict else 'UNCHANGED_FIXED_C_CONFLICT' if fixed_conflict else 'UNCHANGED_DEGENERATE_FACE'
        assert c['repair_status']==expected_status
        assert c['flipped_free_face_ids']==[i for i in members if flips[i]]
        assert c['outward_orientation_certified'] is False
    changed=faces.copy();changed[flips]=faces[flips][:,[0,2,1]]
    perm=np.tile(np.array([0,1,2],np.int8),(n,1));perm[flips]=[0,2,1]
    assert same(oriented['vertices'],vertices) and same(oriented['faces'],changed)
    assert np.array_equal(oriented['flips'],flips) and np.array_equal(oriented['corner_permutation'],perm)
    assert same(changed[:K],faces[:K]) and not flips[:K].any()
    assert np.array_equal(np.sort(changed,axis=1),np.sort(faces,axis=1))
    assert same(oriented['output'],vertices[changed]) and same(oriented['C'],vertices[faces[:K]])
    assert same(oriented['free'],vertices[changed[K:]])
    opposite_after=0;cf_before=0;cf_after=0;unsatisfied=0;eligible_count=0
    for eid,(ids,eligible,required,opposite,domain,reason) in enumerate(edges):
        saved=edge_rows[eid];assert saved['incident_faces_flipped']==[bool(flips[i]) for i in ids]
        new_opposite=(bool(opposite)^bool(flips[ids[0]])^bool(flips[ids[1]])) if opposite is not None else None
        assert saved['repaired_opposite_direction']==new_opposite
        if eligible:
            satisfied=(int(flips[ids[0]])^int(flips[ids[1]]))==required
            assert saved['repaired_constraint_satisfied']==satisfied
            eligible_count+=1;opposite_after+=bool(new_opposite);unsatisfied+=not satisfied
            if domain=='C_F':cf_before+=bool(opposite);cf_after+=bool(new_opposite)
    for key,value in dict(vertices=len(vertices),faces=n,C_faces=K,free_faces=n-K,constraint_components=len(groups),
        satisfiable_components=satisfiable_count,parity_conflict_components=parity_conflicts,
        fixed_C_anchor_conflict_components=anchor_conflicts,degenerate_faces=int(degenerate.sum()),
        nonmanifold_edges=sum(len(e)>2 for e in incidence.values()),boundary_edges=sum(len(e)==1 for e in incidence.values()),
        duplicate_indexed_face_groups=sum(count>1 for count in dups.values()),eligible_two_incidence_edges=eligible_count,
        opposite_two_incidence_edges_after=opposite_after,opposite_C_F_edges_before=cf_before,opposite_C_F_edges_after=cf_after,
        unsatisfied_eligible_edges_after=unsatisfied,flipped_free_faces=int(flips.sum()),outward_orientation_certified=False,model_calls=0).items():assert summary[key]==value,key
    near(summary['flipped_free_surface_area'],float(area[flips].sum()))
    return dict(components=len(groups),edges=len(edges),eligible_constraints=eligible_count,
        parity_conflicts=parity_conflicts,fixed_C_conflicts=anchor_conflicts,flipped_free_faces=int(flips.sum()),
        source_geometry_unchanged=True,C_bytes_and_winding_unchanged=True,independent_XOR_union_find='PASS',
        deterministic_free_flip_solution='PASS',unsatisfied_eligible_edges_after=unsatisfied)

def audit():
    from r2.tools.c_restore_coupling_v1.independent_audit import graph_audit, reduction, eq, num
    started=time.monotonic();reg=read(OUT/'registration.json');done=read(OUT/'execution_completion.json')
    assert done['status']=='PASS' and done['torch_imported'] is False
    for key,value in dict(new_model_forwards=0,training_updates=0,new_generator_outputs=0,new_noise=0,
        new_blend_outputs=12,new_GEO_C=12,new_POST_quality=12,orientation_outputs=36).items():assert done[key]==value,key
    records=reg['records'];assert len(records)==12 and len({r['parent_id'] for r in records})==6
    verify(reg['runner_hashes'])
    # Root already read every protected byte before and after execution; avoid a
    # third large-weight pass. Independently rehash every nonweight asset here.
    small={p:h for p,h in reg['protected'].items() if Path(p).suffix.lower() not in {'.pt','.pth','.ckpt'}}
    weight_receipts={p:h for p,h in reg['protected'].items() if p not in small}
    verify(small)
    assert done['original_files_unchanged']==len(reg['protected']) and done['frozen_sources_unchanged']
    for kind,count in [('ORIENTATION',36),('GEO',12),('QUALITY',12)]:
        for suffix in ['started','complete']:assert len(list((OUT/'events').glob('*/*/'+kind+'_'+suffix+'.json')))==count
    ledger=[json.loads(s) for s in (OUT/'ledger.jsonl').read_text().splitlines() if s]
    for name in ['BLEND_STARTED','BLEND_COMPLETE']:
        ll=[e['request_id'] for e in ledger if e['event']==name]
        assert len(ll)==len(set(ll))==12 and set(ll)=={r['request_id'] for r in records}
    blends=[]
    for r in records:
        folder=OUT/'pre_geo/BLEND'/r['request_id'];saved=arrays(folder/'denoised.npz')
        source=arrays(r['condition_source']);raw=arrays(r['raw_path']);base=arrays(r['D0_denoised_path'])
        proof=read(folder/'receipt.json');verify(proof['files']);assert proof['output_sha256']==sha(folder/'denoised.npz')
        assert proof['identity']['registration_sha256']==sha(OUT/'registration.json')
        assert same(source['C'],raw['C']) and same(saved['C'],raw['C'])
        assert same(saved['official_unconstrained_proposal'],base['official_unconstrained_proposal'])
        details=arrays(folder/'details.npz');diag=read(folder/'diagnostics.json')
        blends.append(dict(request_id=r['request_id'],status='PASS',**check_blend(source['C'],saved['official_unconstrained_proposal'],source['C_source_vertex_ids'],saved['output'],details,diag,reg['fixed_L'])))
    controls=read(OUT/'blend_checks/completion.json');assert controls['status']=='PASS' and controls['checks']==8 and controls['model_calls']==0
    verify(controls['source_hashes'])
    controls=read(OUT/'orientation_checks/completion.json');assert controls['status']=='PASS' and len(controls['cases'])==7 and controls['model_calls']==0
    verify(controls['source_hashes']);verify(controls['files'])
    index=read(OUT/'post_index.json');assert len(index)==36 and collections.Counter(r['method'] for r in index)=={'D0_BASE':12,'RBF':12,'BLEND':12}
    audits=[];edge_checks=0;boundary_checks=0;decision_checks=0
    decisions=rows(OUT/'decision_per_output.csv');dm={(r['method'],r['request_id']):r for r in decisions};assert len(dm)==36
    metrics_rows=rows(OUT/'AVMR_PMR_per_output.csv');mm={(r['method'],r['request_id'],r['phase'],r['setting']):r for r in metrics_rows};assert len(mm)==576
    topology_table={(r['method'],r['request_id'],r['phase']):r for r in rows(OUT/'topology_per_output.csv')}
    for r in index:
        rid=r['request_id'];method=r['method'];k=int(r['K']);post=arrays(r['post_path']);pre=arrays(r['denoised_path'])['output'];raw=arrays(r['raw_path'])['raw'].reshape(-1,3,3)
        assert sha(r['post_path'])==r['post_sha256'] and same(post['output'],post['vertices'][post['faces']])
        assert same(post['output'][:k],raw[:k]) and same(pre[:k],raw[:k])
        d=read(r['diagnostic_path']);verify(d['quality']['files'])
        assert d['status']=='PASS' and d['identity']['post_sha256']==sha(r['post_path']) and d['pre_GEO_quality_status']=='NA_NOT_BUDGETED'
        assert d['quality']['identity']['processed_sha256']==sha(r['post_path'])
        assert d['quality']['free_face_denominator']==len(post['output'])-k
        folder=OUT/'oriented'/method/rid;proof=read(folder/'receipt.json');od=read(folder/'orientation_details.json');oriented=arrays(folder/'oriented.npz')
        assert proof['identity']['post_sha256']==sha(r['post_path']) and proof['identity']['registration_sha256']==sha(OUT/'registration.json')
        verify(proof['files']);assert proof['summary']==od['summary']
        checked_orientation=check_orientation(post['vertices'],post['faces'],k,oriented,od['components'],od['edges'],od['summary'])
        assert same(oriented['source_face_ids'],post['source_face_ids'])
        assert same(oriented['source_corner_ids'],np.take_along_axis(post['source_corner_ids'],oriented['corner_permutation'],axis=1))
        assert same(oriented['output'],np.take_along_axis(post['output'],oriented['corner_permutation'][:,:,None],axis=1))
        assert proof['equivalence']['quality_reference_POST_sha256']==sha(r['post_path'])
        assert proof['equivalence']['quality_scalar_status']=='REUSED_UNORIENTED_GEOMETRY_EQUIVALENCE; no claim of a new scorer execution'
        pairs=read(d['quality']['pair_audit_path'])
        edge_checks+=graph_audit(oriented['output'],oriented['vertices'],oriented['faces'],k,rows(folder/'valid_edges_after.csv'),proof['after'],pairs)
        for key,value in proof['after'].items():
            if isinstance(value,(int,float)) or value is None:eq(topology_table[method,rid,'POST_ORIENTED'].get(key),value)
        source=arrays(r['condition_source']);source_ids=source['C_source_vertex_ids'];cedges=collections.Counter()
        coords={}
        for face,ids in zip(source['C'],source_ids):
            for corner,vid in enumerate(ids):
                if int(vid) in coords:assert same(coords[int(vid)],face[corner])
                coords[int(vid)]=face[corner]
            for u,v in zip(ids,np.roll(ids,-1)):cedges[tuple(sorted((int(u),int(v))))]+=1
        bid=sorted({v for e,count in cedges.items() if count==1 for v in e})
        for phase,tri,vkey,mkey in [('POST',post['output'],'vertices','metrics'),('PRE_GEO',pre,'pre_GEO_vertices','pre_GEO_metrics')]:
            verts=d[vkey];assert [v['source_vertex_id'] for v in verts]==bid
            pts=np.asarray([coords[v] for v in bid],float);free=tri[k:].reshape(-1,3).astype(float)
            d2=((pts[:,None,:]-free[None,:,:])**2).sum(2);nearest=d2.argmin(1);dist=np.sqrt(d2[np.arange(len(bid)),nearest])
            for i,(v,value) in enumerate(zip(verts,dist)):
                near([v['x'],v['y'],v['z']],pts[i]);eq(v['distance_model'],value);eq(v['distance_L_fraction'],value/reg['fixed_L'])
                assert int(v['nearest_free_face'])*3+int(v['nearest_free_corner'])==int(nearest[i]);boundary_checks+=4
            for item in d[mkey]:
                count=int((dist<float(item['threshold_model'])).sum())
                expected=dict(boundary_vertex_count=len(bid),matched_count=count,VMR=count/len(bid),perfect=int(count==len(bid)))
                for key,value in expected.items():eq(item[key],value);eq(mm[method,rid,phase,item['setting']][key],value);boundary_checks+=2
            # Rebuild actual or exact-coordinate graph using saved original pair classifications only.
            if phase=='POST':vertices,faces=post['vertices'],post['faces'];graph_pairs=pairs
            else:
                flat=np.ascontiguousarray(tri,dtype=np.float32).reshape(-1,3)
                _,first,inverse=np.unique(flat.view(np.uint32).reshape(-1,3),axis=0,return_index=True,return_inverse=True)
                vertices,faces=flat[first],inverse.reshape(-1,3);graph_pairs=read(Path(r['diagnostic_path']).parent/'PRE_edge_local_pairs.json')
            ep=Path(r['diagnostic_path']).parent/(('valid_edges_' if method=='BLEND' else 'valid_edge_details_')+('POST' if phase=='POST' else 'PRE')+'.csv')
            top=d['topology' if phase=='POST' else 'pre_GEO_topology'];edge_checks+=graph_audit(tri,vertices,faces,k,rows(ep),top,graph_pairs)
            for key,value in top.items():
                if isinstance(value,(int,float)) or value is None:eq(topology_table[method,rid,phase].get(key),value)
        q=d['quality']['metrics'];strict=next(v for v in d['metrics'] if v['setting']=='raw_tau_1e-6');loose=next(v for v in d['metrics'] if v['setting']=='L_tau_0.02')
        expected=dict(CF=q['CF_proper_affected_free_fraction'],CF_illegal=q['CF_illegal_affected_free_fraction'],FF=q['FF_illegal_affected_free_fraction'],I=q['ConflictCov_C'],
            small_angle_lt5=q['free_minimum_angle_lt5_fraction'],strict_AVMR=strict['VMR'],strict_PMR=strict['perfect'],AVMR_02L=loose['VMR'],
            distance_mean_model=float(np.mean([v['distance_model'] for v in d['vertices']])),pre_distance_mean_model=float(np.mean([v['distance_model'] for v in d['pre_GEO_vertices']])),
            valid_edges=d['topology']['C_F_valid_edge_count'],opposed_before=d['topology']['C_F_valid_opposite_winding_edge_count'],opposed_after=proof['after']['C_F_valid_opposite_winding_edge_count'],
            reachable_area=d['topology']['reached_free_area_fraction'],opposed_reach_before=d['topology']['opposite_winding_reached_free_area_fraction'],opposed_reach_after=proof['after']['opposite_winding_reached_free_area_fraction'],
            free_faces=len(post['output'])-k,deleted_free_faces=int(r['N'])-len(post['output']))
        for key,value in expected.items():eq(dm[method,rid][key],value);decision_checks+=1
        audits.append(dict(method=method,request_id=rid,status='PASS',**checked_orientation))
    pairs_checked=0
    for p in rows(OUT/'paired_per_parent_noise.csv'):
        a=dm[p['method'],p['request_id']];b=dm[p['reference'],p['request_id']]
        for key,value in p.items():
            if key.endswith('_delta'):eq(value,num(a[key[:-6]])-num(b[key[:-6]]));pairs_checked+=1
    summary_checks=0;meta_keys={'method','request_id','label','parent_id','condition_id','region_type','tier','candidate','N','K'}
    for s in rows(OUT/'decision_summary.csv'):
        field={'REGION':'region_type','PARENT':'parent_id','NOISE':'candidate'}.get(s['group'])
        subset=[r for r in decisions if r['method']==s['method'] and (field is None or r[field]==s['value'])]
        eq(s['n'],len(subset))
        for key in decisions[0]:
            if key not in meta_keys:eq(s[key],float(np.mean([float(r[key]) for r in subset])));summary_checks+=1
        for key,f in [('outputs_with_valid_edges','valid_edges'),('outputs_with_opposed_edges_before','opposed_before'),('outputs_with_opposed_edges_after','opposed_after')]:
            eq(s[key],sum(float(r[f])>0 for r in subset));summary_checks+=1
    summary_checks+=reduction(metrics_rows,rows(OUT/'AVMR_PMR_summary.csv'))
    csvwrite(AUDIT/'blend_per_output.csv',blends);csvwrite(AUDIT/'orientation_per_output.csv',audits)
    result=dict(status='PASS',BLEND_outputs=12,orientation_outputs=36,eligible_graph='Full indexed two-incidence nondegenerate C-C/C-F/F-F, regardless of intersection classification',
        independent_parity_algorithm='Union-by-size XOR disjoint sets; execution solver uses BFS',actual_legal_graph_edges_checked=edge_checks,
        boundary_and_threshold_checks=boundary_checks,AVMR_rows=len(mm),decision_value_checks=decision_checks,paired_delta_checks=pairs_checked,summary_stat_checks=summary_checks,
        protected_nonweight_files_rehashed=len(small),protected_weights_from_execution_receipt=weight_receipts,
        weight_verification_basis=dict(path=str(OUT/'execution_completion.json'),sha256=sha(OUT/'execution_completion.json'),full_execution_protected_count=done['original_files_unchanged']),
        blend_controls=8,orientation_controls=7,all_C_coordinates_and_winding_preserved=True,source_corner_and_face_identity_verified=True,
        scalar_quality_reuse='Unordered per-face geometry equivalence, frozen original scorer evidence; no assertion of rerun FP64 bitwise identity',
        full_PRE_quality='NA_NOT_BUDGETED',audit_source_sha256=sha(__file__),registration_sha256=sha(OUT/'registration.json'),
        pure_graph_helper_path=str(ROOT/'r2/tools/c_restore_coupling_v1/independent_audit.py'),
        pure_graph_helper_sha256=sha(ROOT/'r2/tools/c_restore_coupling_v1/independent_audit.py'),
        seconds=time.monotonic()-started,model_forwards=0,training_updates=0,new_GEO_runs=0,new_quality_scans=0)
    write(AUDIT/'completion.json',result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    for p in [ROOT/'WORKSPACE.md',OUT/'registration.json',ROOT/'experiments/r2_c_restore_coupling_v1/REPORT.md']:
        try:checked(p)
        except AssertionError:pass
        else:raise AssertionError('Negative output guard failed')
    if args.check_only:
        print(json.dumps(dict(status='PASS',negative_output_guards=3,model_calls=0)));return
    result=audit();print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
