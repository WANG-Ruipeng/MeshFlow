"""Synthetic teacher/cache contracts; no real teacher/model forward."""
import json
from pathlib import Path
import numpy as np
import pytest
from meshflow_control.teachers.siglip_render import array_hash,camera_from_training,render,RENDER_CONFIG
from meshflow_control.teachers.siglip_targets import normalize_multiview,feature_statistics,TargetStore
from meshflow_control.teachers.siglip_assets import digest,MODEL_ID,REVISION

def tetra():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    return v[np.array([[0,1,2],[0,2,3],[0,3,1],[1,3,2]])]

def test_fixed_camera_has_margin_for_all_four_views():
    meshes=[tetra(),tetra()*np.float32(2)]
    camera=camera_from_training(meshes)
    assert camera["margin_factor"]==1.1
    assert camera["half_extent"]==camera["unpadded_projected_half_extent"]*1.1
    for mesh in meshes:
        before=mesh.tobytes()
        for v in range(4):
            image,audit=render(mesh,camera,v)
            assert image.size==(256,256) and image.mode=="RGB"
            assert audit["foreground_pixels"]>0
            assert min(audit["foreground_bbox"][:2])>0 and max(audit["foreground_bbox"][2:])<255
        assert mesh.tobytes()==before
    assert RENDER_CONFIG["azimuth_degrees"]==[45.,135.,225.,315.]

def test_two_sided_winding_and_deterministic_render():
    mesh=tetra();camera=camera_from_training([mesh])
    for v in range(4):
        a,_=render(mesh,camera,v);b,_=render(mesh[:,::-1].copy(),camera,v);c,_=render(mesh,camera,v)
        assert np.array_equal(a,b) and np.array_equal(a,c)
    with pytest.raises(ValueError):render(mesh*100,camera,0)

def test_full_geometry_hash_not_uid_or_shape_only():
    a=tetra();b=a.copy();b[-1,0,0]+=np.float32(.01)
    assert array_hash(a)!=array_hash(b)
    assert array_hash(a)==array_hash(a.copy())
    with pytest.raises(ValueError):camera_from_training([])

def test_normalize_views_before_equal_mean():
    p=np.zeros((4,768),np.float32);p[0,0]=100;p[1,1]=1;p[2,1]=2;p[3,1]=3
    views,target=normalize_multiview(p)
    expected=np.zeros(768,np.float32);expected[:2]=[1,3];expected/=np.linalg.norm(expected)
    assert np.allclose(target,expected,atol=1e-7)
    assert np.allclose(np.linalg.norm(views,axis=1),1)
    assert target.dtype==np.float32 and target.shape==(768,)
    assert not np.allclose(target,p.mean(0)/np.linalg.norm(p.mean(0)))

@pytest.mark.parametrize("kind",["zero","nan","wrong_dim","float64"])
def test_invalid_official_pooler_rejected(kind):
    p=np.ones((4,768),np.float32)
    if kind=="zero":p[0]=0
    if kind=="nan":p[0,0]=np.nan
    if kind=="wrong_dim":p=p[:3]
    if kind=="float64":p=p.astype(np.float64)
    with pytest.raises(ValueError):normalize_multiview(p)

def fixture_store(tmp_path):
    full=tetra();gh=array_hash(full);target=np.zeros(768,np.float32);target[0]=1
    path=tmp_path/"feature.npz";np.savez(path,target=target)
    manifest=dict(status="COMPLETE",inputs=dict(tasks={"task":dict(geometry_sha256=gh,uid="parent",N=4,K=1)},
        alpha1_samples={"3":dict(task_id="task",geometry_sha256=gh)}),rows=[dict(geometry_sha256=gh,path=str(path),sha256=digest(path),target_array_sha256=array_hash(target))])
    (tmp_path/"teacher_manifest.json").write_text(json.dumps(manifest))
    ids=np.array([2,0,3,1]);corners=np.array([[2,0,1],[1,2,0],[0,1,2],[2,0,1]])
    xyz=full[ids[:,None],corners];known=ids==0
    sample=dict(alpha=1.,task_id="task",object_id="parent",sample_index=3,y=np.asarray(4),known_mask=known,valid_mask=np.ones(4,bool),
        source_face_ids=ids,corner_permutations=corners,context=np.where(known[:,None,None],xyz,0).reshape(4,9),
        x1=np.where(~known[:,None,None],xyz,0).reshape(4,9))
    return TargetStore(tmp_path),sample,target

def test_teacher_reconstructed_from_actual_C_and_x1_after_permutations(tmp_path):
    store,sample,target=fixture_store(tmp_path)
    assert np.array_equal(store.for_sample(sample),target)
    sample["x1"][0,0]+=.001
    with pytest.raises(ValueError):store.for_sample(sample)

def test_augmented_unregistered_or_wrong_parent_sample_rejected(tmp_path):
    store,sample,_=fixture_store(tmp_path)
    for key,value in [("alpha",.95),("object_id","other"),("sample_index",4)]:
        changed=dict(sample);changed[key]=value
        with pytest.raises(ValueError):store.for_sample(changed)

def test_teacher_cache_change_rejected(tmp_path):
    _,sample,_=fixture_store(tmp_path)
    path=tmp_path/"feature.npz";np.savez(path,target=np.zeros(768,np.float32))
    with pytest.raises(ValueError):TargetStore(tmp_path)

def test_constant_baseline_diagnostic_no_success_claim():
    values=np.zeros((2,768),np.float32);values[0,0]=1;values[1,1]=1
    stats=feature_statistics(values,[1,3])
    assert not stats["all_targets_identical"]
    assert stats["pairwise_cosine"]["mean"]==0
    assert np.isclose(stats["constant_unique_teacher_mean"]["mean_alignment_loss"],1-1/np.sqrt(2))
    assert feature_statistics(np.repeat(values[:1],2,axis=0),[1,1])["all_targets_identical"]
    assert MODEL_ID=="google/siglip-base-patch16-256" and len(REVISION)==40
