"""TRAIN47 owner-only extension; inference remains the original C-only layout.

Same FM/OT/hybrid as meshflow_control.seam.slot_data.supervised_sample.
Difference: source third vertices already in C remain ordinary free tips;
multi-edge source faces are represented once by the preselected owner edge.
No additional GT anchor or inference-only source topology is introduced.
"""
import numpy as np
from meshflow_control.seam.slot_data import inference_layout
from meshflow_control.training.coupling import nested_ot_map, to_noise_slots


def supervised_sample(case, raw_noise, epsilon, t, permutation_seed):
    full = np.asarray(case['full_target'], np.float32).reshape(-1,3,3)
    pre = np.asarray(case['pre_ot_full'], np.float32)
    source_vertices = np.asarray(case['source_face_vertex_ids'], np.int64)
    known_ids = np.asarray(case['known_ids'], np.int64)
    free_ids = np.asarray(case['free_ids'], np.int64)
    n, k = len(full), len(known_ids)
    assert np.array_equal(pre*np.float32(2),full)
    z, eps = np.asarray(raw_noise), np.asarray(epsilon)
    assert z.shape == eps.shape == (n-k,3,3) and np.isfinite(z).all() and np.isfinite(eps).all()
    assert 0 <= float(t) <= 1
    labels = np.asarray(case['occupied_labels'],bool)
    layout = inference_layout(case['C_vertices'],case['C_faces_original'],case['C_faces_model'],n,labels)
    anchor = layout['anchor_ids']
    target = np.zeros((n,3,3),np.float32)
    ids = np.full(n,-1,np.int64)
    corner = np.full((n,3),-1,np.int64)
    target[:k] = full[known_ids]
    ids[:k] = known_ids
    corner[:k] = np.arange(3)
    owners = {}
    for ei in np.flatnonzero(labels):
        fi = int(case['edge_owner_face'][ei])
        if fi in owners.values():
            raise ValueError('Source owner assigned twice; owner-only packaging failed')
        source_triple = np.asarray(case['edge_source_triple'][ei],np.int64)
        # Deliberately no third-in-C hard reference: this remains a free tip.
        cp = np.asarray([np.flatnonzero(source_vertices[fi] == vi).item() for vi in source_triple],np.int64)
        slot = k+int(ei)
        owners[slot] = fi
        ids[slot],corner[slot],target[slot] = fi,cp,full[fi,cp]
    bulk_slots = layout['bulk_slots']
    bulk_faces = np.asarray([fi for fi in free_ids if fi not in set(owners.values())],np.int64)
    if len(bulk_faces) != len(bulk_slots):
        raise ValueError('Owner plus returned alias/Null bulk violates original N')
    rng = np.random.RandomState(int(permutation_seed))
    bulk_faces = bulk_faces[rng.permutation(len(bulk_faces))]
    cp0 = np.asarray([[0,1,2],[1,2,0],[2,0,1]],np.int64)[rng.randint(0,3,len(bulk_faces))]
    face_map, corner_map = nested_ot_map(pre[bulk_faces[:,None],cp0],z[bulk_slots-k])
    ids[bulk_slots] = bulk_faces[np.argsort(face_map)]
    corner[bulk_slots] = to_noise_slots(cp0,face_map,corner_map)
    target[bulk_slots] = to_noise_slots(full[bulk_faces[:,None],cp0],face_map,corner_map)
    mixed = np.float32(np.sqrt(.75))*z.astype(np.float32)+np.float32(.5)*eps.astype(np.float32)
    x0 = np.zeros_like(target)
    x0[k:] = mixed
    fixed = anchor >= 0
    x0[fixed] = layout['C_vertices'][anchor[fixed]]
    assert np.array_equal(target[fixed],x0[fixed])
    velocity = target-x0
    xt = (np.float32(1)-np.float32(t))*x0+np.float32(t)*target
    xt[fixed] = x0[fixed]
    order = np.random.RandomState(int(permutation_seed)^0xD00D).permutation(n)
    vals = dict(x0=x0[order],x1=target[order],xt=xt[order],u=velocity[order],
                anchor_ids=anchor[order],known_mask=layout['known_mask'][order],
                valid_mask=layout['valid_mask'][order],free_corner_mask=layout['free_corner_mask'][order],
                source_face_ids=ids[order],source_corner_permutations=corner[order],
                face_permutation=order,ot_face_map=face_map,ot_corner_map=corner_map,
                t=np.asarray(t,np.float32),C_vertices=layout['C_vertices'],
                free_coordinate_count=np.asarray(layout['free_coordinate_count']),
                occupied_labels=labels,occupied_for_FM=labels,
                edge_features=layout['features'],N=np.asarray(n),K=np.asarray(k),
                arm=np.asarray('SEAM_SLOT_C200'),production_OT_calls=np.asarray(1))
    assert np.array_equal(np.sort(ids),np.arange(n))
    assert np.array_equal(full[vals['source_face_ids'][:,None],vals['source_corner_permutations']],vals['x1'])
    assert not vals['u'][vals['anchor_ids']>=0].any()
    assert int(vals['free_coordinate_count']) == 9*(n-k)-6*int(labels.sum())
    return vals
