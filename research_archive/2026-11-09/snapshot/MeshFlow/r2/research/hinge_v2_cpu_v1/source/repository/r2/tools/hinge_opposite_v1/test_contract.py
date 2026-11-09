"""CPU engineering checks, not extra model samples or a threshold search."""
from __future__ import annotations
import argparse,ast,json,time
from pathlib import Path
import numpy as np
from . import engine as E


def _ids(n):
    return np.arange(n,dtype=np.int64)*11+7,np.tile(np.arange(3,dtype=np.int64),(n,1))


def _equal_arrays(left,right):
    assert set(left)==set(right)
    for key in left:
        a,b=left[key],right[key]
        assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),key


def _json(value):
    return json.dumps(E.C.clean(value),sort_keys=True,allow_nan=False)


def constructions():
    names=[]
    # The nominated hinge 0--1/2 becomes opposite, but whole-cluster merging
    # simultaneously joins 1--3 and 2--3 in the SAME direction.
    f=np.asarray([[0,1,4],[2,0,5],[1,3,6],[2,3,7]],np.int64)
    sf,sc=_ids(len(f));trial=f.copy();trial[trial==2]=1
    additions=E.new_two_face_edges(f,trial,sf,sc,0)
    assert len(additions)==2 and sum(r['winding_opposite'] for r in additions)==1
    bad=E.new_same_winding_edges(f,trial,sf,sc,0)
    assert len(bad)==1 and bad[0]['post_face_ids']==[2,3] and bad[0]['kind']=='FF'
    names.append('opposite_main_hinge_does_not_hide_incidental_same_winding_FF')
    # A C/free collateral edge is equally protected. No contact or valid-graph
    # information enters this test, so even an illegal edge cannot be filtered.
    order=[2,0,1,3];ff=f[order];tt=trial[order];sff=sf[order];scc=sc[order]
    bad=E.new_same_winding_edges(ff,tt,sff,scc,1)
    assert len(bad)==1 and bad[0]['kind']=='CF'
    names.append('incidental_CF_checked_without_geometry_validity_filter')
    # Same-direction contacts already present are allowed to remain. A stable
    # source halfedge identity survives arbitrary representative renumbering.
    old=np.asarray([[0,1,2],[0,1,3],[4,5,6],[7,4,8]],np.int64)
    sfo,sco=_ids(4);new=old.copy();new[new==7]=5
    assert len(E.new_two_face_edges(old,new,sfo,sco,0))==1
    assert not E.new_same_winding_edges(old,new,sfo,sco,0)
    assert sum(not r['winding_opposite'] for r in E.two_face_edges(new,sfo,sco,0).values())==1
    names.append('old_same_winding_defect_not_rejected')
    remap=np.asarray([71,2,54,8,19,3,45,13,9])
    assert not E.new_same_winding_edges(new,remap[new],sfo,sco,0)
    assert set(E.two_face_edges(new,sfo,sco,0))==set(E.two_face_edges(remap[new],sfo,sco,0))
    names.append('vertex_representative_renumbering_not_new_source_edge')
    # Two existing shared-vertex shells can acquire a new same-winding contact
    # using reused numeric IDs; the source halfedge pair still exposes it.
    assert E.new_same_winding_edges(f,trial,sf,sc,0)[0]['source_edge_key']==bad[0]['source_edge_key']
    names.append('source_identity_stable_under_output_face_reordering')
    # Explicit per-face cyclic permutations must carry corner identities.
    roll=np.asarray([1,2,0]);fr=f[:,roll];tr=trial[:,roll];scr=sc[:,roll]
    assert {r['source_edge_key'] for r in E.new_same_winding_edges(fr,tr,sf,scr,0)}=={r['source_edge_key'] for r in E.new_same_winding_edges(f,trial,sf,sc,0)}
    names.append('original_corner_identity_carried_through_cyclic_order')
    # Existing incidence3 contacts are outside the two-face definition; their
    # creation is covered by the unchanged independent nonmanifold guard.
    nm=np.asarray([[0,1,2],[1,0,3],[0,1,4]],np.int64);sfn,scn=_ids(3)
    assert not E.two_face_edges(nm,sfn,scn,0)
    names.append('incidence_two_scope_keeps_separate_nonmanifold_guard')
    return names


def historical():
    from r2.pilots.seam_hinge_quality_v1 import hinge as H
    from r2.pilots.seam_hinge_quality_v1.hinge_check import mesh
    # AST equality establishes unchanged queue construction, matching and all
    # original helper contracts, independent of formatting/comments.
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text(encoding='utf-8')) .body if isinstance(n,ast.FunctionDef)}
    oldf,newf=functions(H.__file__),functions(E.__file__)
    for name in ('inventory','edge_keys','nm_keys','scan'):assert oldf[name]==newf[name],name
    x=mesh();tri=x['output'];pairs=H.scan(tri,1)
    old=H.decode(x,tri.copy(),tri.copy(),pairs,.015)
    off=E.decode(x,tri.copy(),tri.copy(),pairs,.015,require_opposite_winding=False)
    on=E.decode(x,tri.copy(),tri.copy(),pairs,.015,require_opposite_winding=True)
    _equal_arrays(old[0],off[0]);_equal_arrays(old[0],on[0]);assert on[2]['accepted']==1
    for pos in (1,3,4,5):assert _json(old[pos])==_json(off[pos])
    checks=[dict(test='frozen_helper_AST_equal',functions=['inventory','edge_keys','nm_keys','scan']),
            dict(test='clean_opposite_hinge_preserved',accepted=on[2]['accepted'])]
    protected={str(Path(H.__file__)):H.C.sha(H.__file__)}
    q={r['request_id']:r for r in H.requests()}
    # Replay only previously accepted operations. This does not construct a new
    # queue, run the detector or optimize a new mesh.
    cases=[('2d08a64e_n3',.015),('608b46b0_n0',.015),('85b16941_n3',.015),('8c9b2023_n2',.025)]
    for request,delta in cases:
        p=H.paths('PLAIN_S2000',q[request]);base=H.C.arrays(p['post']);pre=H.C.arrays(p['pre'])['output'];raw=H.C.arrays(p['raw'])['output']
        folder=H.OUT/'cells'/H.label('PLAIN_S2000',delta)/request
        recpath=folder/'candidate_records.json';records=H.C.read(recpath)
        for path in [*p.values(),recpath,folder/'post.npz']:protected[str(path)]=H.C.sha(path)
        v=base['vertices'].copy();roots=np.arange(len(v),dtype=np.int64);f0=base['faces'];sf,sc=base['source_face_ids'],base['source_corner_ids'];K=int(base['known_faces'])
        _,_,members,fixed=H.inventory(base,pre,raw,delta);bad_operations=[]
        for row in records:
            if row['decision']!='ACCEPTED':continue
            state,why=H.B.state_merge(v,roots,members,pre.reshape(-1,3).astype(float),fixed,[row['endpoint_original_vertices']]);assert why is None
            assert _json(state['operations'])==_json(row['merged_clusters'])
            bad=E.new_same_winding_edges(roots[f0],state['roots'][f0],sf,sc,K)
            if bad:bad_operations.append(dict(rank=row['rank'],source_raw_slots=row['source_raw_slots'],new_same_winding_edges=bad))
            v,roots=state['vertices'],state['roots']
        saved=H.C.arrays(folder/'post.npz')
        assert np.array_equal(v,saved['vertices']) and np.array_equal(roots[f0],saved['faces'])
        assert bad_operations,request
        checks.append(dict(test='historical_accepted_operation_caught',request_id=request,delta=delta,violating_operations=bad_operations))
    # One authorized known case fully replays False, comparing every output
    # array byte, candidate record and legacy summary field with saved results.
    request='608b46b0_n0';delta=.015;p=H.paths('PLAIN_S2000',q[request]);base=H.C.arrays(p['post']);pre=H.C.arrays(p['pre'])['output'];raw=H.C.arrays(p['raw'])['output'];pairs=H.C.read(p['pairs'])
    folder=H.OUT/'cells'/H.label('PLAIN_S2000',delta)/request
    off=E.decode(base,pre,raw,pairs,delta,require_opposite_winding=False)
    _equal_arrays(off[0],H.C.arrays(folder/'post.npz'))
    assert _json(off[1])==_json(H.C.read(folder/'candidate_records.json'))
    previous=H.C.read(folder/'hinge.json')
    assert {k:v for k,v in off[2].items() if k not in ('version','require_opposite_winding')}=={k:v for k,v in previous.items() if k!='version'}
    checks.append(dict(test='False_reproduces_saved_complete_case_bitwise',request_id=request,delta=delta,accepted=off[2]['accepted'],summary_exceptions=['version','require_opposite_winding']))
    H.B.verify(protected)
    return checks,protected


def main():
    p=argparse.ArgumentParser();p.add_argument('--historical',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args()
    start=time.perf_counter();tests=constructions();more,protected=historical() if args.historical else ([],{})
    result=dict(status='PASS',constructive_tests=tests,historical_checks=more,protected_hashes=protected,
        source_hashes={str(Path(E.__file__)):E.C.sha(E.__file__),str(Path(__file__)):E.C.sha(__file__)},
        seconds=time.perf_counter()-start,model_calls=0,GPU_calls=0,
        scope='Engineering fixtures plus explicitly selected saved operations; not formal 56-cell experiment.')
    if args.output:
        if args.output.exists():raise FileExistsError('Do not overwrite engineering receipt '+str(args.output))
        E.C.write(args.output,result)
    print(json.dumps(dict(status='PASS',constructive_tests=len(tests),historical_checks=len(more),seconds=result['seconds'],output=str(args.output)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
