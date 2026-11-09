"""Small geometric/selection tests; no model/OT/scoring calls."""
import numpy as np
from .selector import measure, nearest, choose, boundary_samples


def run_tests():
    checks=[]
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float32);f=np.array([[0,1,2]])
    C=v[f];raw=np.concatenate((C,C.copy()))
    r=measure(raw,v,f,2,1,.2,.4,'b')
    assert r['boundary_cover']==1 and abs(r['rank']-.2)<1e-12;checks.append('open_C_three_equal_terms')
    d=nearest(np.array([[0.,0.,2.],[2.,0.,0.]]),C)
    assert np.allclose(d,[2.,1.]);checks.append('finite_triangle_distance_not_squared_or_infinite_plane')
    far=raw.copy();far[1]+=np.array([0,0,10])
    rfar=measure(far,v,f,2,1,0.,0.,'a')
    assert rfar['boundary_cover']==0 and np.isclose(rfar['rank'],1/3);checks.append('far_F_not_zero_penalty')
    tetra=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    closed=measure(np.concatenate((tetra[faces],C)),tetra,faces,5,4,.2,.4,'z')
    assert closed['boundary_cover'] is None and closed['rank_term_count']==2 and np.isclose(closed['rank'],.3)
    checks.append('closed_C_cover_NA_not_perfect')
    invalid=raw.copy();invalid[1]=0
    assert not measure(invalid,v,f,2,1,0,0,'x')['eligible'];checks.append('all_degenerate_ineligible')
    invalid=raw.copy();invalid[1,0,0]=np.nan
    assert not measure(invalid,v,f,2,1,0,0,'x')['eligible'];checks.append('nonfinite_ineligible')
    invalid=raw.copy();invalid[0,0,0]=.1
    assert not measure(invalid,v,f,2,1,0,0,'x')['eligible'];checks.append('C_change_ineligible')
    assert choose([dict(r,noise_sha256='c'),dict(r,noise_sha256='a')])['noise_sha256']=='a'
    checks.append('tie_noise_hash_not_index')
    assert choose([dict(eligible=False,rank=None)]) is None;checks.append('all_invalid_no_fallback_or_redraw')
    points,weights,edges=boundary_samples(v,f)
    assert len(points)==24 and np.isclose(weights.sum(),2+2**.5);checks.append('length_weighted_source_open_boundary')
    scaled=measure(raw*7,v*7,f,2,1,.2,.4,'b')
    assert scaled['boundary_cover']==r['boundary_cover'] and scaled['rank']==r['rank'];checks.append('C_scale_covariance')
    return dict(status='PASS',count=len(checks),checks=checks,model_calls=0,OT_calls=0)
