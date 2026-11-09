from pathlib import Path
from copy import deepcopy
import json,inspect,math
import numpy as np
import pytest
import torch
from experiments.a_radial.models import ShapeAE,ShapeFM,fit_pca_train_only
from experiments.a_radial.io_utils import array_hash,model_hash,rng_state,restore_rng,seed_all
from experiments.a_radial.mesh_ops import bisect_to_faces,topology,ray_hits
from experiments.a_radial.registry import validate_internal
from experiments.a_radial.official_train import ema_update

def test_locked_parameter_counts():
    assert sum(p.numel() for p in ShapeAE().parameters())==5383234
    assert sum(p.numel() for p in ShapeFM().parameters())==50240

def test_decoder_only_z():
    assert list(inspect.signature(ShapeAE.decode_nodes).parameters)==['self','z']
    m=ShapeAE();z=torch.randn(2,64);m.eval()
    assert torch.equal(m.decode_nodes(z),m.decode_nodes(z));assert m.decode_nodes(z).shape==(2,2050)

def test_small_pca_rank_and_padding():
    seed_all(71);m=ShapeAE();y=torch.randn(16,2050)*.1
    info=fit_pca_train_only(m,y);assert info['numeric_rank']==15
    assert torch.count_nonzero(m.basis[:,15:])==0
    with torch.no_grad():r,z=m(y)
    assert torch.max(abs(r-y))<1e-6
    assert z.shape==(16,64)

def test_identical_training_pca_rejected():
    with pytest.raises(ValueError):fit_pca_train_only(ShapeAE(),torch.ones(16,2050))

def test_normalization_once_and_constant_ode():
    m=ShapeFM()
    for p in m.parameters():p.data.zero_()
    m.net[-1].bias.data.fill_(.25);m.mu.fill_(2.);m.sd.fill_(3.)
    e=torch.zeros(3,64);zs=m.sample_standardized(e,32);assert torch.allclose(zs,torch.full_like(e,.25))
    assert torch.allclose(m.unstandardize(zs),torch.full_like(e,2.75))
    assert not torch.equal(m.unstandardize(m.unstandardize(zs)),m.unstandardize(zs))

def test_model_hash_includes_scalar_buffers():
    m=ShapeAE();h=model_hash(m);m.scale.mul_(2);assert h!=model_hash(m)

def test_rng_restore():
    import random
    seed_all(100);s=rng_state();a=(np.random.randn(4),torch.randn(4),random.random());restore_rng(s)
    b=(np.random.randn(4),torch.randn(4),random.random());assert np.array_equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]

def test_exact_ema_coefficient():
    m=torch.nn.Linear(1,1,bias=False);e=deepcopy(m);e.weight.data.fill_(1);m.weight.data.zero_()
    for _ in range(512):ema_update(e,m,.9999)
    assert abs(float(e.weight.detach())-.9999**512)<2e-5
    assert .9999**50000<.01

def cube():
    import trimesh
    m=trimesh.creation.box();return np.asarray(m.vertices),np.asarray(m.faces)

def test_cube_preserving_subdivision():
    v,f=cube();sv,sf,parent,ops,check=bisect_to_faces(v,f,512)
    assert len(sf)==512 and len(ops)==250
    assert np.array_equal(v,sv[:len(v)]) and topology(sv,sf)['closed_oriented_g0']
    assert check['positive_child_orientation'];assert check['plane_distance_bbox_max']<1e-12

def test_subdivision_rejects_wrong_target_and_open_input():
    v,f=cube()
    for target in (13,10):
        with pytest.raises(ValueError):bisect_to_faces(v,f,target)
    with pytest.raises(ValueError):bisect_to_faces(v,f[:-1],512)

def test_shared_edge_hit_not_multiple_layers():
    v,f=cube();d=np.array([[1.,0,0],[1,1,0],[1,1,1]]);d/=np.linalg.norm(d,axis=1)[:,None]
    ct,r=ray_hits(v,f,d);assert np.all(ct==1)
    v2=np.vstack([v,v*.5]);f2=np.vstack([f,f[:,::-1]+8]);ct,_=ray_hits(v2,f2,d);assert (ct>1).all()

def ledger():
    r={'sample_id':'00000','status':'ok','source_bundle_sha256':'b','source_nodes_sha256':'n','ae_sha256':'ae','fm_sha256':'fm','requested_faces':512}
    return {k:{'requested_samples':1,'records':[dict(r)]} for k in ['fixed','a','qem']}

def test_strict_shared_latents():assert validate_internal(ledger())['attempts']==1

def test_mismatched_checkpoint_caught():
    d=ledger();d['qem']['records'][0]['fm_sha256']='different'
    with pytest.raises(ValueError):validate_internal(d)

def test_dropped_failure_caught():
    d=ledger();d['a']['records']=[]
    with pytest.raises(ValueError):validate_internal(d)

def test_duplicate_attempt_caught():
    d=ledger();d['fixed']['records']*=2;d['fixed']['requested_samples']=2
    with pytest.raises(ValueError):validate_internal(d)


def test_token_geometry_permutation_and_duplication_gate():
    from experiments.a_radial.official_contract import compare_token_geometry
    t=np.array([[[0.,0,0],[1,0,0],[0,1,0]],[[0,0,0],[0,1,0],[0,0,1]]])
    shuffled=t[::-1,::-1].astype(np.float32)
    assert compare_token_geometry(t,shuffled)==0
    bad=shuffled.copy();bad[0]=bad[1]
    with pytest.raises(AssertionError):compare_token_geometry(t,bad)
