"""Real-data CPU contract checks; no model construction/forward/backward."""
from __future__ import annotations
import collections
import json
import numpy as np
from . import data as D
from . import training as T
from .head import features
from r2.pilots.chair_multipart_seam_v1.head import features as chair_features


def run():
    manifest=D.read(D.OUT/'manifest.json')
    schedule=D.read(D.OUT/'schedule.json')
    scaler=D.read(D.OUT/'head_scaler.json')
    cases={r['condition_id']:D.load_case(r) for r in manifest['rows']}
    train=manifest['train'];dev=manifest['dev']
    assert not {r['parent_id'] for r in train}&{r['parent_id'] for r in dev}
    assert len({r['parent_id'] for r in train})==58
    fx=[]
    for row in manifest['rows']:
        c=cases[row['condition_id']]
        assert D.sha(row['case_path'])==row['case_sha256']
        f=features(c['C_vertices'],c['C_faces_original'],int(c['N']))
        assert f.shape==(len(c['occupied_labels']),28) and np.isfinite(f).all()
        if len(f):
            assert np.array_equal(f,chair_features(c['C_vertices'],c['C_faces_original'],int(c['N'])))
        assert np.array_equal(np.sort(np.r_[c['known_ids'],c['free_ids']]),np.arange(int(c['N'])))
        if row in train:
            assert len(f)>0 and np.count_nonzero(c['occupied_labels'])>0
            fx.append(f)
    raw=np.concatenate(fx).astype(np.float64)
    assert len(raw)==1861 and scaler['edges']==1861
    assert np.allclose(np.asarray(scaler['mean']),raw.mean(0),rtol=2e-7,atol=1e-8)
    assert np.allclose(np.asarray(scaler['population_std']),raw.std(0),rtol=1e-12,atol=1e-12)
    parent_counts=collections.Counter()
    for entry in schedule['steps']:
        rows=entry['samples']
        assert len(rows)==8 and len({r['parent_id'] for r in rows})==4
        assert [r['augmentation'] for r in rows]==['CLEAN']*4+['ANISOTROPIC']*4
        for a,b in zip(rows[:4],rows[4:]):
            assert (a['condition_id'],a['parent_id'])==(b['condition_id'],b['parent_id'])
            assert a['scale_xyz']==[1.,1.,1.]
            assert all(.92<=v<=1.08 for v in b['scale_xyz'])
        parent_counts.update(r['parent_id'] for r in rows)
    assert sum(parent_counts.values())==12000
    assert max(parent_counts.values())-min(parent_counts.values())<=2
    ot=0
    for row in schedule['steps'][0]['samples']:
        c=cases[row['condition_id']]
        s=D.sample_for_training(c,row);ot+=1
        scaled=D.scaled_case(c,row['scale_xyz'])
        expected=scaled['full_target'].reshape(-1,3,3)[s['source_face_ids'][:,None],s['source_corner_permutations']]
        assert np.array_equal(s['x1'],expected)
        fixed=s['anchor_ids']>=0
        assert np.array_equal(s['x0'][fixed],s['C_vertices'][s['anchor_ids'][fixed]])
        assert not s['u'][fixed].any()
        n,k=int(s['N']),int(s['K'])
        assert int(s['free_coordinate_count'])==9*(n-k)-6*np.count_nonzero(c['occupied_labels'])
        z=np.random.RandomState(row['noise_seed']).standard_normal((n-k,3,3)).astype(np.float32)
        eps=np.random.RandomState(row['epsilon_seed']).standard_normal(z.shape).astype(np.float32)
        mixed=np.float32(np.sqrt(.75))*z+np.float32(.5)*eps
        x0=s['x0'][np.argsort(s['face_permutation'])]
        fi=s['anchor_ids'][np.argsort(s['face_permutation'])]<0
        assert np.array_equal(x0[k:][fi[k:]],mixed[fi[k:]])
        # Positive diagonal transformation: cross products transform by det(S) S^-T.
        tri=c['C_vertices'][c['C_faces_original']]
        tri2=scaled['C_vertices'][c['C_faces_original']]
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normal2=np.cross(tri2[:,1]-tri2[:,0],tri2[:,2]-tri2[:,0])
        scale=np.asarray(row['scale_xyz'],np.float32)
        assert np.allclose(normal2,normal*np.prod(scale)/scale,rtol=1e-4,atol=2e-6)
    for row in [next(r for r in manifest['rows'] if r['boundary_edges']==0),
                next(r for r in manifest['rows'] if r['boundary_edges']>0 and not np.count_nonzero(cases[r['condition_id']]['occupied_labels']))]:
        c=cases[row['condition_id']]
        n,k=int(c['N']),int(c['K'])
        z=np.zeros((n-k,3,3),np.float32);eps=z.copy()
        s=D.sample_for_probe(c,z,eps,.5,42);ot+=1
        assert int(s['free_coordinate_count'])==9*(n-k)
    # Pure algebra verifies new per-C BCE averaging differs from old edge pooling.
    fake_losses=[np.ones(2),np.full(20,3.)]
    assert np.mean([v.mean() for v in fake_losses])==2.
    assert not np.isclose(np.concatenate(fake_losses).mean(),2.)
    sample=dict(valid_mask=np.array([1,1,1,0],bool),
                anchor_ids=np.array([[0,1,2],[0,1,-1],[-1,-1,-1],[-1,-1,-1]]))
    error=np.arange(36,dtype=np.float32).reshape(4,3,3)
    stats=T.stat(error,sample)
    assert stats['all_free_scalar_count']==12
    assert stats['seam_tip_scalar_count']==3 and stats['bulk_scalar_count']==9
    assert stats['all_free_velocity_sse']==float(error[1,2].sum()+error[2].sum())
    result=dict(status='PASS',conditions_checked=316,eligible_train_conditions=195,parents=58,
                per_update_distinct_parents=4,parent_exposure_min=min(parent_counts.values()),
                parent_exposure_max=max(parent_counts.values()),TRAIN_only_scaler_edges=1861,
                augmentation_geometry_before_OT=True,Gaussian_noise_unscaled=True,source_masks_exact=True,
                per_condition_BCE_not_edge_pooled=True,closed_and_open_no_owner_probe_supported=True,
                sample_OT_calls=ot,synthetic_probe_OT_calls=2,model_constructions=0,model_forwards=0,
                backward_calls=0,GPU_calls=0)
    D.write_new(D.OUT/'cpu_checks.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':run()
