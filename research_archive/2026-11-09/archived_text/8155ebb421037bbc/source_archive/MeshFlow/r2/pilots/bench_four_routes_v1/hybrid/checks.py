"""Meaningful CPU fixtures for fixed selector semantics; no mesh scoring/model."""
from . import common as C,selector as S
import numpy as np

def run():
    from meshflow_control import geo_c
    out=C.OUT/'audits/cpu_checks.json'
    if out.exists():return C.read(out)
    C.write(C.OUT/'resources_before_checks.json',dict(utc=C.utc(),resources=C.resources(),workers=1,threads=1,GPU_workers=0))
    # A chain with only point contacts makes ring depth explicit.
    raw=np.asarray([[[20,0,0],[21,0,0],[20,1,0]],
        [[0,0,0],[1,0,0],[0,1,0]],[[1,0,0],[2,0,0],[2,1,0]],
        [[2,1,0],[3,1,0],[3,2,0]],[[7,0,0],[8,0,0],[7,1,0]]],np.float32)
    geo=geo_c.process(raw,1);pre=raw.copy();pre[1,2]=[0,.1,0];pre[2:,:,2]=.05
    oldraw=raw.tobytes();oldpre=pre.tobytes();r=S.select(raw,pre,1,geo,L=1.)
    checks=[]
    def ok(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(dict(name=name,status='PASS'))
    ok('seed_and_exactly_one_vertex_ring',np.array_equal(np.flatnonzero(r['selected_RAW']),[1,2]))
    ok('no_recursive_second_ring',not r['selected_RAW'][3] and not r['selected_RAW'][4])
    ok('whole_selected_faces_RAW',r['output'][r['selected_RAW']].tobytes()==raw[r['selected_RAW']].tobytes())
    ok('unselected_faces_D0',r['output'][~r['selected_RAW']].tobytes()==pre[~r['selected_RAW']].tobytes())
    ok('C_exact_and_input_not_mutated',r['output'][:1].tobytes()==raw[:1].tobytes() and raw.tobytes()==oldraw and pre.tobytes()==oldpre)
    ok('outer_mixed_boundary_not_hidden',r['summary']['split_mixed_boundary_clusters']>=1)
    same=S.select(raw,raw.copy(),1,geo,L=1.)
    ok('identity_no_seed',not same['selected_RAW'].any() and same['output'].tobytes()==raw.tobytes())
    invalid=raw.copy();invalid[4]=invalid[4,0];ig=geo_c.process(invalid,1)
    inv=S.select(invalid,invalid.copy(),1,ig,L=1.)
    ok('degenerate_RAW_not_triggered',not inv['seed'][4] and inv['per_face'][4]['area_ratio'] is None)
    # Long highly skinny D0 face can cross area threshold without min-edge threshold.
    ar=raw.copy();ar[1]=[[0,0,0],[1,0,0],[.5,.1,0]]
    a=S.select(raw,ar,1,geo,L=1.)
    ok('area_trigger_OR_independent',a['per_face'][1]['reason']=='SEED_AREA')
    # A short edge with a taller triangle keeps area while crossing edge threshold.
    er=raw.copy();er[1]=[[0,0,0],[.1,0,0],[0,10,0]]
    e=S.select(raw,er,1,geo,L=1.)
    ok('edge_trigger_OR_independent',e['per_face'][1]['reason']=='SEED_LENGTH')
    # GEO target identities include coordinates of subsequently deleted faces.
    deleted=np.concatenate((raw,raw[1:2]),axis=0);dg=geo_c.process(deleted,1)
    dp=deleted.copy();dp[1,2]=[0,.1,0];dr=S.select(deleted,dp,1,dg,L=1.)
    ok('deleted_duplicate_corners_still_in_neighborhood',dr['selected_RAW'][-1] and not dg['source_face_kept'][-1])
    ids=S.cluster_ids(raw,geo);actual=raw.reshape(-1,3)[geo['corner_target_input_ids'].reshape(-1)]
    ok('cluster_identity_matches_exact_original_GEO',all(np.array_equal(actual[ids.reshape(-1)==c],np.broadcast_to(actual[np.flatnonzero(ids.reshape(-1)==c)[0]],actual[ids.reshape(-1)==c].shape)) for c in np.unique(ids)))
    # A newly attached corner must not feed expansion back into the second ring.
    ok('deterministic_repeated_selection',S.select(raw,pre,1,geo,L=1.)['output'].tobytes()==r['output'].tobytes())
    try:S.select(raw,pre+np.float32(.1),1,geo,L=1.)
    except ValueError:checks.append(dict(name='reject_unrestored_C',status='PASS'))
    else:raise AssertionError('Unrestored C accepted')
    result=dict(status='PASS',checks=checks,fixture_GEO_calls=3,model_calls=0,new_noise=0,real_requests_processed=0,
        code={str(p):C.sha(p) for p in (C.CODE/'selector.py',C.CODE/'checks.py')})
    C.write(out,result);return result

if __name__=='__main__':print(run())
