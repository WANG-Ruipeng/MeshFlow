"""Independent HINGE V2 saved-state audit; no engine or detector calls.

Replays accepted target assignments and append-only triangles from their logs.
Checks endpoint contacts using already saved pair classifications. Rejected
collision trials are not rescanned and are not represented as independently
verified intermediate detector results.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import json
from pathlib import Path
import sys
import time
import numpy as np
from r2.tools.hinge_opposite_v1.audit import topo, same, incidence, ILLEGAL
from r2.pilots.seam_hinge_quality_v1.hinge_audit import geometry
from .diagnose import arrays, read, sha, write_csv, L

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_hinge_v2_cpu_v1'
BASE=ROOT/'experiments/r2_hinge_opposite_v1/cells/HINGE_OPP015'
ARMS=('MIDPOINT_REPLAY','ALPHA','HOLE','ALPHA_HOLE')


def view(mesh):
    """Temporary audit IDs are stable output-face rows, never invented RAW IDs."""
    n=len(mesh['faces'])
    sc=mesh.get('source_corner_ids',mesh.get('base_source_corner_ids'))
    assert sc is not None and len(sc)<=n
    if len(sc)<n:sc=np.concatenate([sc,np.tile(np.arange(3),(n-len(sc),1))])
    return dict(mesh,source_face_ids=np.arange(n,dtype=np.int64),source_corner_ids=sc)


def bad_sets(pairs,mesh,unknown=False):
    K=int(mesh['known_faces']);F=len(mesh['faces'])-K;result={}
    for kind in ('CF','FF'):
        scan=pairs[kind]
        expected=K*F if kind=='CF' else F*(F-1)//2
        assert scan['expected_full_pairs']==scan['tested_pairs']==expected
        assert scan['affected_subset'] is None and scan['independent_verification'] is True
        selected=set();seen=set()
        for r in scan['contacts']:
            a,b=map(int,r['pair']);assert (a,b) not in seen;seen.add((a,b))
            assert 0<=b<F and 0<=a<(K if kind=='CF' else F)
            label=r['classification'].lower();d=r.get('detector',{})
            flag=(any(t in label for t in ('unknown','unresolved','degenerate')) or d.get('resolved') is False or d.get('verified') is False) if unknown else r['classification'] in ILLEGAL
            if flag:selected.add((a if kind=='CF' else K+a,K+b))
        result[kind]=selected
    return result


def duplicate_pairs(tri):
    groups=collections.defaultdict(list)
    for i,t in enumerate(tri):
        groups[tuple(sorted(tuple(map(float,x)) for x in t))].append(i)
    return {(x[i],x[j]) for x in groups.values() for i in range(len(x)) for j in range(i+1,len(x))}


def endpoint(base,bp,mesh,pairs):
    assert same(mesh['vertices'][mesh['faces']],mesh['output'])
    K=int(base['known_faces']);assert int(mesh['known_faces'])==K
    assert same(mesh['output'][:K],base['output'][:K])
    assert same(mesh['C'],mesh['output'][:K]) and same(mesh['free'],mesh['output'][K:])
    a,b=topo(view(base),bp),topo(view(mesh),pairs)
    lost=set(a['legal'])-set(b['legal']);added=set(b['legal'])-set(a['legal'])
    newtwo=set(b['two'])-set(a['two'])
    assert not lost,('lost legal edge',lost)
    assert not b['nm']-a['nm'],'new nonmanifold'
    assert all(b['two'][k]['winding_opposite'] for k in newtwo),'new same-winding two-face edge'
    for k in set(a['two'])&set(b['two']):
        assert a['two'][k]['winding_opposite']==b['two'][k]['winding_opposite']
    assert not duplicate_pairs(mesh['output'])-duplicate_pairs(base['output'])
    oldbad,newbad=bad_sets(bp,base),bad_sets(pairs,mesh)
    oldu,newu=bad_sets(bp,base,True),bad_sets(pairs,mesh,True)
    for kind in ('CF','FF'):
        assert not newbad[kind]-oldbad[kind],(kind,'new illegal')
        assert not newu[kind]-oldu[kind],(kind,'new unresolved')
    return dict(old_legal_edges=len(a['legal']),lost_old_legal_edges=len(lost),new_legal_edges=len(added),
                new_two_face_edges=len(newtwo),new_same_winding_edges=0,new_nonmanifold_edges=0,
                new_CF_illegal=0,new_FF_illegal=0,new_unknown=0,
                removed_CF_illegal=len(oldbad['CF']-newbad['CF']),removed_FF_illegal=len(oldbad['FF']-newbad['FF']))


def alpha_replay(base,pre,result,records,multi):
    v=base['vertices'].copy();f=base['faces'];roots=np.arange(len(v),dtype=np.int64)
    K=int(base['known_faces']);fixed=set(map(int,f[:K].reshape(-1)))
    cv=base['corner_vertex_ids'].reshape(-1);initial=geometry(base['output']);maximum=0.
    hard=base['output_anchor_ids']>=0;accepted=0;attempts=0;additional=0
    original_topo=topo(view(base));old_dups=duplicate_pairs(base['output'])
    for row in records:
        aa,bb=row['endpoint_original_vertices'];a,b=int(roots[aa]),int(roots[bb])
        choices=row['location_attempts'];attempts+=len(choices)
        if not choices:
            assert row['decision']!='ACCEPTED';continue
        assert a!=b
        aset=set(np.flatnonzero(roots==a).tolist());bset=set(np.flatnonzero(roots==b).tolist())
        af,bf=bool(aset&fixed),bool(bset&fixed);assert not (af and bf)
        group=aset|bset;fixed_target=v[a] if af else v[b] if bf else None
        first=fixed_target.copy() if fixed_target is not None else ((v[a].astype(float)+v[b].astype(float))/2).astype(np.float32)
        expected=[('ORIGINAL',None,first)];seen={first.tobytes()}
        if multi and fixed_target is None:
            for alpha in (0.,.25,.5,.75,1.):
                target=((1-alpha)*v[a].astype(float)+alpha*v[b].astype(float)).astype(np.float32)
                if target.tobytes() in seen:continue
                seen.add(target.tobytes());expected.append(('ALPHA',alpha,target))
        assert row['distinct_proposal_count']==len(expected)
        assert len(choices)<=len(expected)
        corners=np.flatnonzero(np.isin(cv,list(group)))
        expected_root=a if af else b if bf else min(a,b)
        for order,trial in enumerate(choices):
            kind,alpha,target=expected[order]
            assert trial['location_order']==order and trial['proposal_kind']==kind and trial['alpha']==alpha
            assert len(trial['merged_clusters'])==1;op=trial['merged_clusters'][0]
            assert same(target,np.asarray(op['target'],np.float32))
            assert set(op['original_cluster_ids'])==group and op['new_root']==expected_root
            assert op['fixed_C']==bool(af or bf) and op['original_corner_count']==len(corners)
            disp=np.linalg.norm(pre.reshape(-1,3)[corners].astype(float)-target,axis=1)
            assert abs(disp.max()-op['maximum_PRE_GEO_displacement'])<1e-12
            assert abs(disp.max()-trial['maximum_PRE_GEO_displacement'])<1e-12
            if trial['decision']!='ACCEPTED':continue
            assert order==len(choices)-1 and row['decision']=='ACCEPTED'
            assert row['accepted_proposal_kind']==kind and row['accepted_alpha']==alpha
            assert np.linalg.norm(v[a].astype(float)-v[b])<=.015*L+1e-12
            assert disp.max()<=.015*L+1e-12
            previous_tri=v[roots[f]].copy();previous=geometry(previous_tri)
            before=topo(view(dict(base,vertices=v,faces=roots[f],output=previous_tri)))
            v[list(group)]=target;roots[list(group)]=expected_root
            ff=roots[f];tri=v[ff];assert same(tri[:K],base['output'][:K])
            assert same(tri[hard],base['output'][hard])
            aff=np.flatnonzero(np.any(np.isin(f,list(group)),axis=1));aff=aff[aff>=K]
            assert sorted(base['source_face_ids'][aff].tolist())==sorted(trial['affected_source_raw_slots'])
            assert all(len(set(map(int,ff[i])))==3 for i in aff)
            shape=geometry(tri)
            assert np.all(shape['area'][aff]>1e-12*L*L)
            for ref in (initial,previous):
                assert not np.any((shape['angle'][aff]<5)&(shape['angle'][aff]<ref['angle'][aff]-1e-10))
                assert not np.any((shape['q'][aff]>10)&(shape['q'][aff]>ref['q'][aff]+1e-10))
                assert np.all((shape['normal'][aff]*ref['normal'][aff]).sum(1)>=.5-1e-12)
            after=topo(view(dict(base,vertices=v,faces=ff,output=tri)))
            assert all(after['two'][k]['winding_opposite'] for k in set(after['two'])-set(before['two']))
            assert not after['nm']-original_topo['nm']
            assert not duplicate_pairs(tri)-old_dups
            maximum=max(maximum,float(disp.max()));accepted+=1;additional+=int(kind=='ALPHA')
        if row['decision']!='ACCEPTED':assert len(choices)==len(expected)
    assert same(v,result['vertices']) and same(roots[f],result['faces']) and same(v[roots[f]],result['output'])
    assert same(roots,result['HINGE_OPP015_vertex_to_ALPHA_vertex'])
    assert same(roots[base['original_POST_vertex_to_hinge_vertex']],result['original_POST_vertex_to_hinge_vertex'])
    mapped=base['corner_vertex_ids'].copy();valid=mapped>=0;mapped[valid]=roots[mapped[valid]]
    assert same(mapped,result['corner_vertex_ids'])
    for name in ('source_face_ids','source_corner_ids','raw_slot_ids','source_face_kept','source_face_reason',
                 'duplicate_of_source_face','raw_anchor_ids','output_anchor_ids','raw_seam_slot_mask','known_faces'):
        assert same(base[name],result[name]),name
    changed=[i for i in range(len(v)) if roots[i]!=i or not same(v[i],base['vertices'][i])]
    corners=np.flatnonzero(np.isin(cv,changed))
    final_disp=np.linalg.norm(pre.reshape(-1,3)[corners].astype(float)-v[roots[cv[corners]]],axis=1)
    assert not len(corners) or final_disp.max()<=.015*L+1e-12
    return dict(accepted=accepted,accepted_alpha=additional,verified_location_attempts=attempts,
        max_trial_PRE_GEO_displacement=maximum,final_max_PRE_GEO_displacement=float(final_disp.max()) if len(corners) else 0.)


def holes_replay(base,result,records):
    n=len(base['faces']);K=int(base['known_faces']);v=base['vertices'];f=base['faces'].copy()
    assert same(result['vertices'],v) and same(result['faces'][:n],f)
    assert same(result['output'][:n],base['output'])
    assert int(result['base_output_face_count'])==n and int(result['base_free_face_count'])==n-K
    roles=['C' if i<K else 'SEAM' if base['raw_seam_slot_mask'][base['source_face_ids'][i]] else 'BULK' for i in range(n)]
    accepted=0
    for row in records:
        if row['decision']!='ACCEPTED':continue
        cycle=tuple(row['cycle_vertices']);tri=tuple(row['oriented_bridge_vertices'])
        assert len(set(cycle))==3 and set(cycle)==set(tri)
        inc=incidence(f);adj=[]
        for j in range(3):
            a,b=tri[j],tri[(j+1)%3];occ=inc[tuple(sorted((a,b)))];assert len(occ)==1
            i,x,y=occ[0];assert f[i,x]==b and f[i,y]==a
            assert roles[i]!='C';adj.append(i)
        assert 'SEAM' in [roles[i] for i in adj]
        assert len(set(adj))==3
        assert tuple(sorted(cycle)) not in {tuple(sorted(map(int,t))) for t in f}
        shape=geometry(v[np.asarray([tri])])
        assert shape['area'][0]>1e-12*L*L and shape['angle'][0]>=5 and shape['q'][0]<=10
        assert row['new_bridge_output_face_id']==len(f)
        assert not row['all_guard_failures']
        f=np.concatenate([f,np.asarray([tri],f.dtype)]);roles.append('BRIDGE');accepted+=1
    assert same(f,result['faces']) and len(f)==n+accepted
    assert np.array_equal(result['output_face_id'],np.arange(len(f)))
    assert result['face_role'].astype(str).tolist()==roles
    assert np.array_equal(result['bridge_mask'],np.arange(len(f))>=n)
    assert np.array_equal(result['parent_post_face_id'],np.r_[np.arange(n),np.full(accepted,-1)])
    assert np.array_equal(result['parent_source_raw_slot'],np.r_[base['source_face_ids'],np.full(accepted,-1)])
    for name in ('source_face_ids','source_corner_ids','raw_slot_ids','output_anchor_ids'):
        stored='base_'+name if accepted else name
        assert same(result[stored],base[name])
        if accepted:assert name not in result
    assert same(result['corner_vertex_ids'],base['corner_vertex_ids'])
    return dict(accepted=accepted,new_BRIDGE_faces=accepted,deleted_original_faces=0,moved_vertices=0)


def run():
    tick=time.perf_counter();reg=read(OUT/'protocol.json')
    assert reg['cells']==112 and tuple(reg['arms'])==ARMS
    assert read(OUT/'geometry_complete.json')['status']=='COMPLETE'
    results=[];inputs={str(OUT/'protocol.json'):sha(OUT/'protocol.json')}
    for q in reg['requests']:
        rid=q['request_id'];base=arrays(BASE/rid/'post.npz');bp=read(BASE/rid/'pairs.json')
        pre=arrays(ROOT/'experiments/r2_seam_quality_wave1_wave2_v1/evaluation/PLAIN_S2000'/rid/'post/pre_geo.npz')['output']
        for arm in ARMS:
            p=OUT/'cells'/arm/rid;done=read(p/'complete.json');assert done['status']=='COMPLETE'
            for name,h in done['files'].items():assert sha(name)==h
            inputs[str(p/'complete.json')]=sha(p/'complete.json')
            mesh=arrays(p/'post.npz');pairs=read(p/'pairs.json');records=read(p/'records.json');summary=read(p/'summary.json')
            if arm in ('MIDPOINT_REPLAY','ALPHA'):
                replay=alpha_replay(base,pre,mesh,records,arm=='ALPHA');parent=base;parent_pairs=bp
                assert replay['accepted_alpha']==summary['accepted_alpha']
                assert replay['verified_location_attempts']==summary['location_attempts']
                assert abs(replay['final_max_PRE_GEO_displacement']-summary['max_PRE_GEO_displacement'])<1e-12
            else:
                ap=OUT/'cells/ALPHA'/rid
                parent=arrays(ap/'post.npz') if arm=='ALPHA_HOLE' else base
                parent_pairs=read(ap/'pairs.json') if arm=='ALPHA_HOLE' else bp
                replay=holes_replay(parent,mesh,records)
            assert replay['accepted']==summary['accepted']
            row=dict(request_id=rid,split=q['split'],arm=arm,**replay,**endpoint(base,bp,mesh,pairs))
            if arm=='ALPHA_HOLE':
                direct=endpoint(parent,parent_pairs,mesh,pairs)
                row.update(hole_only_new_legal_edges=direct['new_legal_edges'])
            results.append(row)
    assert len(results)==112 and 'torch' not in sys.modules
    dest=OUT/'independent_audit';dest.mkdir(parents=True,exist_ok=True)
    write_csv(dest/'per_cell.csv',results)
    totals=[]
    for arm in ARMS:
        rr=[r for r in results if r['arm']==arm]
        totals.append(dict(arm=arm,cells=len(rr),accepted=sum(r['accepted'] for r in rr),
            new_legal_edges=sum(r['new_legal_edges'] for r in rr),new_two_face_edges=sum(r['new_two_face_edges'] for r in rr),
            new_CF_illegal=sum(r['new_CF_illegal'] for r in rr),new_FF_illegal=sum(r['new_FF_illegal'] for r in rr),
            removed_CF_illegal=sum(r['removed_CF_illegal'] for r in rr),removed_FF_illegal=sum(r['removed_FF_illegal'] for r in rr)))
    result=dict(status='PASS_INDEPENDENT_GEOMETRY_AUDIT',cells=112,totals=totals,
        old_faces_deleted=0,C_and_seam_anchor_changes=0,new_nonmanifold=0,new_same_winding_two_face_edges=0,
        old_legal_edge_losses=0,new_illegal_CF_FF=0,new_unknown_pairs=0,
        seconds=time.perf_counter()-tick,new_model_calls=0,new_pair_scans=0,
        scope='All accepted target formulas, cumulative original cluster motion, shapes, topology, bridge lineage and final cached contact guards independently checked; rejected collision scans not rerun.',
        code_sha256=sha(__file__),source_hashes=inputs,
        files={str(dest/'per_cell.csv'):sha(dest/'per_cell.csv')})
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','files')},indent=2));return result

def main():return run()
if __name__=='__main__':main()
