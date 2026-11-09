"""Independent saved split-trial audit; no model or contact-detector calls.

Children have distinct snapshot IDs and explicit original-parent lineage.
Old legal edges are conserved by actual endpoint indices and parent incidences,
not by the necessarily changed child face ID. No old result is rewritten.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import csv
import json
import time
from pathlib import Path
import numpy as np
from r2.pilots.seam_hinge_quality_v1 import hinge as OLD
from r2.pilots.seam_hinge_quality_v1.hinge_audit import geometry
from r2.tools.hinge_opposite_v1.audit import topo

C=OLD.C
OUT=C.ROOT/'experiments/r2_hinge_subsegment_v1'
BASE=C.ROOT/'experiments/r2_hinge_opposite_v1/cells/HINGE_OPP015'
ILLEGAL=frozenset(('proper_transverse_penetration','coplanar_positive_area_overlap'))


def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def indexed_edges(f):
    inc=collections.defaultdict(list)
    for i,face in enumerate(f):
        for a,b in ((0,1),(1,2),(2,0)):
            inc[tuple(sorted((int(face[a]),int(face[b]))))].append((i,a,b))
    return inc


def parent_edge_key(edge,occ,parents):
    return (tuple(edge),tuple(sorted(int(parents[i]) for i,a,b in occ)))


def lineage_topology(m,pairs=None):
    n=len(m['faces']);parents=m['parent_post_face_id']
    pseudo=dict(m,source_face_ids=m['child_face_id'],source_corner_ids=np.tile(np.arange(3,dtype=np.int64),(n,1)))
    top=topo(pseudo,pairs)
    legal_by_index={(r['vertex_u'],r['vertex_v']):r for r in top['legal'].values()}
    result=dict(two={},legal={},nonmanifold=set(),internal_edges=[])
    for edge,occ in indexed_edges(m['faces']).items():
        key=parent_edge_key(edge,occ,parents)
        if len(occ)>2:result['nonmanifold'].add(key)
        if len(occ)!=2:continue
        (i,a,b),(j,c,d)=occ;f=m['faces']
        opposite=bool(i!=j and edge[0]!=edge[1] and f[i,a]==f[j,d] and f[i,b]==f[j,c])
        info=dict(vertex_ids=edge,parent_post_faces=key[1],child_faces=[i,j],opposite=opposite,
                  internal_subdivision=bool(parents[i]==parents[j]),legal=edge in legal_by_index)
        result['two'][key]=info
        if info['legal']:result['legal'][key]=info
        if info['internal_subdivision']:result['internal_edges'].append(info)
    return result


def basic_lineage(m,base):
    f,v=m['faces'],m['vertices'];K=int(base['known_faces'])
    parents=np.asarray(m['parent_post_face_id']);raw=np.asarray(m['parent_source_raw_slot'])
    assert len(f)==len(parents)==len(raw)==len(m['child_face_id'])
    assert len(set(map(int,m['child_face_id'])))==len(f)
    assert set(map(int,parents))==set(range(len(base['faces'])))
    assert np.all(parents>=0) and np.all(parents<len(base['faces']))
    assert np.array_equal(raw,base['source_face_ids'][parents])
    assert np.array_equal(m['child_seam_mask'],base['raw_seam_slot_mask'][raw])
    assert np.array_equal(parents[:K],np.arange(K))
    assert same(f[:K],base['faces'][:K]) and same(v[f[:K]],base['output'][:K])
    assert same(m['output'],v[f])
    assert same(m['base_parent_source_corner_ids'],base['source_corner_ids'])
    bary=np.asarray(m['corner_parent_barycentric'],np.float64)
    assert bary.shape==(len(f),3,3) and np.isfinite(bary).all()
    assert np.allclose(bary.sum(axis=-1),1.,rtol=0,atol=2e-12)
    assert np.all(bary>=-2e-12) and np.all(bary<=1+2e-12)
    anchor=np.full(f.shape,-1,np.int64)
    for i,parent in enumerate(parents):
        for j,weights in enumerate(bary[i]):
            for a in range(3):
                if np.array_equal(weights,np.eye(3)[a]):anchor[i,j]=base['output_anchor_ids'][parent,a]
    assert same(anchor,m['output_anchor_ids'])
    # Existing parent corners use their current representative positions. New
    # side points may differ from the FP64 linear combination by FP32 rounding.
    parent_tri=v[base['faces'][parents]].astype(np.float64)
    reconstructed=np.einsum('fij,fjk->fik',bary,parent_tri)
    error=np.linalg.norm(reconstructed-m['output'].astype(np.float64),axis=-1)
    tolerance=8*np.finfo(np.float32).eps*max(1.,float(abs(v).max()))
    assert error.max()<=tolerance
    fixed=set(map(int,base['faces'][:K].ravel()))
    assert same(v[sorted(fixed)],base['vertices'][sorted(fixed)])
    # Every original hard seam corner remains at the identical C vertex ID.
    for parent in np.flatnonzero(base['raw_seam_slot_mask'][base['source_face_ids']]):
        children=np.flatnonzero(parents==parent)
        for corner in np.flatnonzero(base['output_anchor_ids'][parent]>=0):
            anchor=int(base['faces'][parent,corner])
            assert any(anchor in f[i] for i in children)
            assert same(v[anchor],base['vertices'][anchor])
    counts=np.bincount(parents,minlength=len(base['faces']))
    assert np.all(counts>=1)
    return dict(parent_faces=len(counts),child_faces=len(f),split_parent_faces=int(np.sum(counts>1)),
                added_faces=len(f)-len(base['faces']),maximum_barycentric_reconstruction_error=float(error.max()),
                barycentric_reconstruction_tolerance=tolerance)


def parent_pairs(pairs,m):
    K=int(m['known_faces']);parents=m['parent_post_face_id'];result={k:set() for k in ('CF','FF')};siblings=[];unknown={k:set() for k in result}
    for kind in result:
        for row in pairs[kind]['contacts']:
            a,b=map(int,row['pair']);i=a if kind=='CF' else K+a;j=K+b
            pa,pb=int(parents[i]),int(parents[j]);key=(pa,pb) if kind=='CF' else tuple(sorted((pa,pb)))
            if row['classification'] in ILLEGAL:
                result[kind].add(key)
                if pa==pb:siblings.append((i,j,row['classification']))
            if any(z in row['classification'].lower() for z in ('unknown','unresolved','degenerate')) or row.get('detector',{}).get('resolved') is False:
                unknown[kind].add(key)
    return result,siblings,unknown


def snapshot(base):
    n=len(base['faces'])
    return dict(vertices=base['vertices'].copy(),faces=base['faces'].copy(),output=base['output'].copy(),
        known_faces=base['known_faces'].copy(),child_face_id=np.arange(n,dtype=np.int64),
        parent_post_face_id=np.arange(n,dtype=np.int64),parent_source_raw_slot=base['source_face_ids'].copy(),
        child_seam_mask=base['raw_seam_slot_mask'][base['source_face_ids']].copy(),
        corner_parent_barycentric=np.broadcast_to(np.eye(3),(n,3,3)).copy())


def rebuild(current,candidate,base,pre,cap):
    """Independently reconstruct the unique registered side split in memory."""
    f=current['faces'];v=current['vertices'];pp=current['parent_post_face_id']
    h,far,u=[int(candidate[k]) for k in ('hinge_vertex','seam_far_vertex','bulk_endpoint_vertex')]
    parent=int(candidate['seam_post_face_id']);bulk_parent=int(candidate['bulk_post_face_id'])
    rows=[i for i in range(len(f)) if pp[i]==parent and h in f[i] and far in f[i]]
    assert len(rows)==1
    i=rows[0];inc=indexed_edges(f);se=tuple(sorted((h,far)));be=tuple(sorted((h,u)))
    assert len(inc[se])==len(inc[be])==1
    si,a,b=inc[se][0];j,c,d=inc[be][0];assert si==i and pp[j]==bulk_parent
    assert (f[i,a]==h)!=(f[j,c]==h)
    other=next(z for z in range(3) if z not in (a,b))
    axis=v[far].astype(float)-v[h];ray=v[u].astype(float)-v[h]
    t=float(np.dot(axis,ray)/np.dot(axis,axis));projection=v[h].astype(float)+t*axis
    target=projection.astype(np.float32)
    angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(axis,ray))/(np.linalg.norm(axis)*np.linalg.norm(ray)),-1,1))))
    assert .15<=t<=.85 and angle<=20+1e-12
    assert np.linalg.norm(ray-t*axis)<=cap+1e-12
    original_members=np.flatnonzero(base['corner_vertex_ids'].ravel()==u)
    maximum=float(np.linalg.norm(pre.reshape(-1,3)[original_members].astype(float)-target,axis=1).max())
    assert maximum<=cap+1e-12
    fixed=set(map(int,base['faces'][:int(base['known_faces'])].ravel()))
    assert u not in fixed or same(target,v[u])
    r={k:z.copy() for k,z in current.items()};r['vertices'][u]=target
    new_faces=np.asarray([[f[i,a],u,f[i,other]],[u,f[i,b],f[i,other]]],np.int64)
    hrow=int(np.flatnonzero(f[i]==h)[0]);frow=int(np.flatnonzero(f[i]==far)[0])
    oldbar=current['corner_parent_barycentric'][i]
    pointbar=(1-t)*oldbar[hrow]+t*oldbar[frow]
    newbar=np.asarray([[oldbar[a],pointbar,oldbar[other]],[pointbar,oldbar[b],oldbar[other]]])
    child_ids=np.asarray([current['child_face_id'].max()+1,current['child_face_id'].max()+2],np.int64)
    additions=dict(faces=new_faces,child_face_id=child_ids,parent_post_face_id=np.asarray([parent,parent]),
        parent_source_raw_slot=np.repeat(current['parent_source_raw_slot'][i],2),child_seam_mask=np.ones(2,bool),
        corner_parent_barycentric=newbar)
    for key,extra in additions.items():r[key]=np.concatenate((np.delete(current[key],i,axis=0),extra),axis=0)
    r['output']=r['vertices'][r['faces']]
    changed_old={int(current['child_face_id'][z]) for z in range(len(f)) if u in f[z]}
    affected=[z for z,child in enumerate(r['child_face_id']) if int(child) in changed_old or child in child_ids]
    return r,dict(parent=parent,moved=u,selected_old_row=i,affected=affected,new_ids=child_ids,
        intended_edge=be,maximum_PRE_GEO=maximum,t=t,angle=angle,projection=projection,target=target,
        original_members=original_members)


def duplicate_parents(m):
    groups=collections.defaultdict(list)
    for i,tri in enumerate(m['output']):
        key=tuple(sorted(tuple(map(int,z)) for z in np.ascontiguousarray(tri).view(np.uint32)))
        groups[key].append(i)
    return {tuple(sorted((int(m['parent_post_face_id'][i]),int(m['parent_post_face_id'][j]))))
        for group in groups.values() for a,i in enumerate(group) for j in group[a+1:]}


def check_trial(current,attempt,info,base,bp,cp,tp,record):
    K=int(base['known_faces']);parents=attempt['parent_post_face_id'];children=set(map(int,info['new_ids']))
    shapes=geometry(attempt['output']);oldshape=geometry(current['output']);original=geometry(base['output'])
    old_index={int(cid):i for i,cid in enumerate(current['child_face_id'])}
    failures=[]
    recorded_shapes={int(r['child_face_id']):r for r in record['affected_face_shapes']}
    for i in info['affected']:
        cid=int(attempt['child_face_id'][i]);parent=int(parents[i]);before=old_index.get(cid);new=cid in children
        ref=info['selected_old_row'] if before is None else before
        local=[]
        if len(set(map(int,attempt['faces'][i])))<3 or shapes['area'][i]<=1e-12*C.L*C.L:local.append('COLLAPSED_OR_DEGENERATE')
        if new:
            if shapes['angle'][i]<5-1e-10:local.append('CHILD_ANGLE_LT5')
            if shapes['q'][i]>10+1e-10:local.append('CHILD_Q_GT10')
        else:
            if shapes['angle'][i]<5 and (shapes['angle'][i]<oldshape['angle'][ref]-1e-10 or shapes['angle'][i]<original['angle'][parent]-1e-10):local.append('ANGLE_NEW_OR_WORSE')
            if shapes['q'][i]>10 and (shapes['q'][i]>oldshape['q'][ref]-1e-10 or shapes['q'][i]>original['q'][parent]+1e-10):local.append('Q_NEW_OR_WORSE')
        original_cos=float(shapes['normal'][i]@original['normal'][parent])
        previous_cos=float(shapes['normal'][i]@oldshape['normal'][before]) if before is not None else original_cos
        if min(original_cos,previous_cos)<.5-1e-12:local.append('NORMAL_ROTATION_GT60')
        saved=recorded_shapes[cid]
        assert local==saved['failures'],(cid,local,saved['failures'])
        for field,key in [('area','area'),('minimum_angle','angle'),('q','q')]:
            assert np.isclose(saved[field],shapes[key][i],rtol=1e-11,atol=1e-10)
        failures.extend(local)
    ct=lineage_topology(current,cp);tt=lineage_topology(attempt,tp)
    if tt['nonmanifold']-ct['nonmanifold']:failures.append('NEW_NONMANIFOLD')
    newtwo=set(tt['two'])-set(ct['two'])
    if any(not tt['two'][key]['opposite'] for key in newtwo):failures.append('NEW_SAME_WINDING')
    if duplicate_parents(attempt)-duplicate_parents(current):failures.append('NEW_DUPLICATE')
    bad,siblings,unknown=parent_pairs(tp,attempt)
    oldbad,_,baseunknown=parent_pairs(bp,snapshot(base));curbad,_,curunknown=parent_pairs(cp,current)
    for kind in ('CF','FF'):
        newbad=bad[kind]-(oldbad[kind]&curbad[kind])
        assert sorted(newbad)==[tuple(x) for x in record['new_illegal_parent_pairs'][kind]]
        if newbad:failures.append('NEW_'+kind+'_ILLEGAL')
    newunknown={kind:unknown[kind]-curunknown[kind] for kind in ('CF','FF')}
    if any(newunknown.values()):failures.append('NEW_UNRESOLVED_CONTACT')
    lost=set(ct['legal'])-set(tt['legal'])
    saved_lost={(tuple(x['vertex_ids']),tuple(x['parent_incidents'])) for x in record['lost_legal_edges']}
    assert saved_lost==lost
    if lost:failures.append('LOST_LEGAL_EDGE')
    target=[r for key,r in tt['two'].items() if tuple(r['vertex_ids'])==info['intended_edge']]
    if len(target)!=1 or not target[0]['legal'] or not target[0]['opposite']:failures.append('SPLIT_TARGET_NOT_LEGAL_OPPOSITE')
    failures=list(dict.fromkeys(failures))
    assert failures==record['all_guard_failures'],(failures,record['all_guard_failures'])
    assert record['decision']==(failures[0] if failures else 'ACCEPTED')
    return dict(decision=record['decision'],verified_all_guard_failures=failures,illegal_sibling_pairs=len(siblings),
        new_two_face_edges=len(newtwo),new_internal_subdivision_edges=sum(tt['two'][k]['internal_subdivision'] for k in newtwo),
        lost_legal_edges=len(lost),new_illegal_parent_pairs=sum(len(bad[k]-(oldbad[k]&curbad[k])) for k in bad),
        new_unknown_parent_pairs=sum(map(len,newunknown.values())),
        maximum_PRE_GEO_displacement=info['maximum_PRE_GEO'])


def main():
    start=time.perf_counter();dest=OUT/'independent_split';done=OUT/'independent_split_audit.json'
    if done.exists():
        r=C.read(done);OLD.B.verify(r['files']);print(json.dumps(dict(status=r['status'],already_complete=True)));return
    receipt=C.read(OUT/'trials/completion.json');assert receipt['status']=='COMPLETE'
    OLD.B.verify(receipt['files']);OLD.B.verify(receipt['code'])
    registration=C.read(OUT/'split_protocol.json');OLD.B.verify(registration['protected'])
    verified_attempts=[];cells=[]
    for q in OLD.requests():
        base=C.arrays(BASE/q['request_id']/'post.npz');bp=C.read(BASE/q['request_id']/'pairs.json')
        pre=C.arrays(OLD.paths('PLAIN_S2000',q)['pre'])['output'];basestate=snapshot(base)
        base_top=lineage_topology(basestate,bp)
        for delta in (.015,.025):
            label='SPLIT'+str(round(delta*1000)).zfill(3);folder=OUT/'trials'/label/q['request_id']
            records=C.read(folder/'candidate_records.json');cell=C.read(folder/'complete.json')
            current=snapshot(base);cp=bp;accepted=0
            for record in records:
                candidate=record['candidate'];expected,info=rebuild(current,candidate,base,pre,delta*C.L)
                attempt=C.arrays(folder/('attempt_%03d.npz'%record['rank']))
                for key in expected:assert same(expected[key],attempt[key]),(q['request_id'],key)
                identity=basic_lineage(attempt,base)
                tp=C.read(folder/('attempt_%03d_pairs.json'%record['rank']))
                result=check_trial(current,attempt,info,base,bp,cp,tp,record)
                verified_attempts.append(dict(request_id=q['request_id'],split=q['split'],variant=label,rank=record['rank'],**identity,**result))
                if result['decision']=='ACCEPTED':current=expected;cp=tp;accepted+=1
            final=C.arrays(folder/'post.npz');fp=C.read(folder/'pairs.json')
            for key in current:assert same(current[key],final[key]),(q['request_id'],label,key)
            identity=basic_lineage(final,base);assert identity['added_faces']==accepted==cell['accepted']
            nt=lineage_topology(final,fp);lost=set(base_top['legal'])-set(nt['legal']);assert not lost
            assert not nt['nonmanifold']-base_top['nonmanifold']
            newtwo=set(nt['two'])-set(base_top['two']);assert all(nt['two'][k]['opposite'] for k in newtwo)
            oldbad,_,oldunknown=parent_pairs(bp,basestate);newbad,siblings,newunknown=parent_pairs(fp,final)
            assert not siblings
            for kind in ('CF','FF'):
                assert not newbad[kind]-oldbad[kind]
                assert not newunknown[kind]-oldunknown[kind]
            cv=base['corner_vertex_ids'].ravel();changed=np.flatnonzero(np.any(final['vertices']!=base['vertices'],axis=1))
            maximum=0.
            for vertex in changed:
                members=np.flatnonzero(cv==vertex)
                distance=np.linalg.norm(pre.reshape(-1,3)[members].astype(float)-final['vertices'][vertex],axis=1)
                maximum=max(maximum,float(distance.max()));assert distance.max()<=delta*C.L+1e-12
            cells.append(dict(request_id=q['request_id'],split=q['split'],variant=label,registered_attempts=len(records),
                accepted=accepted,**identity,original_legal_edges_lost=0,new_illegal_parent_pairs=0,new_same_direction_two_face_edges=0,
                C_and_original_seam_anchors_preserved=True,maximum_cumulative_PRE_GEO_displacement=maximum,
                no_op_geometry_exact=same(final['vertices'],base['vertices']) and same(final['faces'],base['faces']) if not accepted else False))
        print(json.dumps(dict(audited_request=q['request_id'],cells=len(cells))),flush=True)
    assert len(cells)==56 and len(verified_attempts)==8
    OLD.B.verify(receipt['files']);OLD.B.verify(receipt['code']);OLD.B.verify(registration['protected'])
    C.csvwrite(dest/'per_cell.csv',cells);C.write(dest/'verified_trials.json',verified_attempts)
    result=dict(status='PASS_INDEPENDENT_SPLIT_AUDIT',cells=56,candidate_attempts=8,
        accepted=sum(r['accepted'] for r in cells),rejected=sum(r['decision']!='ACCEPTED' for r in verified_attempts),
        decisions=dict(collections.Counter(r['decision'] for r in verified_attempts)),
        original_parent_faces_lost=0,original_legal_edges_lost=0,new_illegal_parent_pairs=0,
        illegal_sibling_pairs_at_accepted_endpoints=0,new_same_direction_two_face_edges_at_accepted_endpoints=0,
        maximum_barycentric_reconstruction_error=max(r['maximum_barycentric_reconstruction_error'] for r in cells),
        new_detector_calls=0,model_calls=0,mesh_edits=0,seconds=time.perf_counter()-start,
        limitations=['Saved finite endpoint collision classifications are reused, not rerun.',
            'Parent-level conflict accounting is explicit; child sibling conflicts remain separate and forbidden.',
            'Floating-point projection/area residuals are measured, not declared mathematically zero.',
            'Preserved topology and guards do not certify semantic usefulness or all-metric improvement.'],
        audit_source_sha256=C.sha(__file__),files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
    C.write(done,result);print(json.dumps({k:v for k,v in result.items() if k not in ('limitations','files')}),flush=True)


if __name__=='__main__':
    main()

