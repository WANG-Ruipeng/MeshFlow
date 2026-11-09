"""Original owner-only SEAM, independent natural-N128..799 compatibility.
No new parameters, loss, source-shared noise, head training, or state projection.
Only explicit N bounds differ from the old SEAM adapter. Closed C is Native.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'r2/src'))
import numpy as np
import torch
from r2.pilots.bench_natural_n_recipe_v1.runtime import (
    NativeModel, build_model, native_execution, edge_features, N_MIN, N_MAX)
from meshflow_control.models.native import assert_native_runtime, _face_select, _corners
from meshflow_control.models.geometry import inject_context
from meshflow_control.seam.model_adapter import _corner_select, _validate_corner_masks, corner_masked_fm_loss
from meshflow_control.training.coupling import nested_ot_map, to_noise_slots
VERSION='bench_four_routes_original_seam128_799_v1'

def inference_layout(C_vertices, C_faces_original, C_faces_model, total_N, occupied):
    """Create a fixed-N layout from C and predicted booleans only.

    C_faces_model is an internal corner permutation of C_faces_original, not a
    different surface. Original C order is saved for final export.
    """
    v = np.asarray(C_vertices, np.float32)
    original = np.asarray(C_faces_original, np.int64)
    canonical = np.asarray(C_faces_model, np.int64)
    if original.shape != canonical.shape or not np.array_equal(np.sort(original,1),np.sort(canonical,1)):
        raise ValueError('Model C changes original faces')
    edges, features = edge_features(v, original, total_N)
    o = np.asarray(occupied)
    if o.dtype != np.bool_ or o.shape != (len(edges),):
        raise ValueError('An explicit predicted bool for each C edge is required')
    n, k = int(total_N), len(original)
    if len(edges)>n-k:raise ValueError('Original boundary slot budget exceeded')
    anchor = np.full((n,3), -1, np.int64)
    anchor[:k] = canonical
    for ei, (_, u, w) in enumerate(edges):
        if o[ei]:
            anchor[k+ei,:2] = [w,u]
    known = np.arange(n) < k
    free_corner = anchor < 0
    return dict(anchor_ids=anchor, known_mask=known, valid_mask=np.ones(n,bool),
                free_corner_mask=free_corner, occupied=o.copy(), edge_records=edges,
                features=features, free_coordinate_count=3*int(free_corner.sum()),
                seam_slots=k+np.flatnonzero(o), bulk_slots=np.flatnonzero(~known & free_corner.all(1)),
                C_vertices=v, C_faces_original=original, C_faces_model=canonical,
                N=n, K=k)

def supervised_sample(case, raw_noise, epsilon, t, permutation_seed, arm='SEAM'):
    if arm!='SEAM':raise ValueError('Only original SEAM is registered')
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

def sample_for_training(case, row, arm="SEAM"):
    copied = {key: value.copy() for key, value in case.items()}
    alpha = np.float32(row["alpha"])
    for key in ("full_target", "pre_ot_full", "C_vertices"):
        copied[key][...,0] *= alpha
    n, k = len(copied["full_target"]), len(copied["known_ids"])
    z = np.random.RandomState(row["noise_seed"]).standard_normal((n-k,3,3)).astype(np.float32)
    eps = np.random.RandomState(row["epsilon_seed"]).standard_normal(z.shape).astype(np.float32)
    normal = float(np.random.RandomState(row["time_seed"]).standard_normal())
    t = float(1/(1+np.exp(-normal)))
    return supervised_sample(copied,z,eps,t,row["permutation_seed"],arm)

def verify_sample(sample, stage):
    """Check role invariants on an already required training sample; no extra OT."""
    n, k = int(sample["N"]), int(sample["K"])
    if not 128 <= n <= 799 or not 2 <= k < n:
        raise ValueError("Frozen N128..799/K>=2 completion contract exceeded")
    anchor = np.asarray(sample["anchor_ids"])
    fixed = anchor >= 0
    known = np.asarray(sample["known_mask"], bool)
    labels = np.asarray(sample["occupied_labels"], bool)
    active = np.asarray(sample["occupied_for_FM"], bool)
    wanted = labels if stage == "SEAM" else np.zeros_like(labels)
    if not np.array_equal(active, wanted):
        raise ValueError("NATIVE has no owner constraint; SEAM uses source owners only")
    if known.sum() != k or not np.array_equal(known, fixed.all(1)) or not np.asarray(sample["valid_mask"]).all():
        raise ValueError("Known/valid role masks differ")
    seam_count = int((fixed.sum(1) == 2).sum())
    expected = 9 * (n - k) - 6 * int(wanted.sum())
    if (seam_count != int(wanted.sum()) or int(sample["free_coordinate_count"]) != expected
            or int((~fixed).sum()) * 3 != expected or expected <= 0):
        raise ValueError("Whole-batch free scalar denominator differs")
    if stage == "NATIVE" and np.any(fixed[~known]):
        raise ValueError("Native free corners must have no owner anchors")
    return dict(source_owner_count=int(labels.sum()), active_seam_slots=seam_count,
                C_boundary_edges=len(labels), closed_C=not len(labels),
                free_coordinate_count=expected,
                policy="NATIVE_ALL_FREE_OT" if stage == "NATIVE" else "SOURCE_OWNER_SEAM_OT")

def seam_forward(model, x, t, y, valid_mask, C_face_mask, corner_known):
    """Reuse original Native parameters with mixed known/free seam corners.

    ``corner_known`` is bool[B,N,3], independent of the C-face context mask.
    The unmodified no-seam branch is a literal call to the original model, so
    it preserves that model's arithmetic and validation rather than emulating
    a supposedly equivalent forward. No production forward is run by tests.
    """
    _validate_corner_masks(x, valid_mask, C_face_mask, corner_known)
    ordinary = C_face_mask[..., None].expand_as(corner_known)
    if torch.equal(corner_known, ordinary):
        return model(x, t, y, valid_mask=valid_mask, known_mask=C_face_mask)

    assert_native_runtime(model, coordinates=(x, t))
    if getattr(model, "readout_mode", "none") != "none" or len(getattr(model, "readouts", {})):
        raise ValueError("This prototype accepts Native/T1 + role + Geo without READOUT")
    batch, faces, _ = x.shape
    if (t.shape != (batch,) or y.shape != (batch,) or y.dtype != torch.int64
            or t.device != x.device or y.device != x.device):
        raise ValueError("Colocated free t[B] and integer total N[B] required")
    if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()):
        raise ValueError("Nonfinite coordinates/time")
    if bool(((t < 0) | (t > 1)).any()):
        raise ValueError("Free time outside [0,1]")
    if not torch.equal(valid_mask.sum(1), y) or bool(((y < 128) | (y > 799)).any()):
        raise ValueError("y must count real faces only, N128..799")
    if bool((C_face_mask.sum(1) == 0).any()):
        raise ValueError("Every sample requires original C faces")
    if bool(((valid_mask[..., None] & ~corner_known).sum((1, 2)) == 0).any()):
        raise ValueError("Every sample requires a free coordinate")

    backbone = model.backbone
    _, hidden = backbone.x_embedder(x)
    hidden = hidden.view(batch, faces * 3, -1)
    role = model.role_embedding[corner_known.long()].to(hidden.dtype)
    hidden = hidden + role.reshape_as(hidden)
    if model.context_encoder is not None:
        delta = model.context_encoder(
            x, C_face_mask, valid_mask, observer=model.context_observer)
        hidden = inject_context(hidden, delta, C_face_mask, observer=model.context_observer)

    c_free = backbone.t_embedder(t)
    c_known = backbone.t_embedder(torch.ones_like(t))
    count = backbone.y_embedder(y // backbone.face_bin, False)
    c_free, c_known = c_free + count, c_known + count
    for block in backbone.layers:
        free_mods = block.adaLN_modulation(c_free).chunk(6, dim=1)
        known_mods = block.adaLN_modulation(c_known).chunk(6, dim=1)
        shift_a, scale_a, gate_a = (
            _face_select(f, k, C_face_mask)
            for f, k in zip(free_mods[:3], known_mods[:3]))
        shift_m, scale_m, gate_m = (
            _corner_select(f, k, corner_known)
            for f, k in zip(free_mods[3:], known_mods[3:]))
        face_hidden = hidden.reshape(batch, faces, 3, -1).mean(dim=2)
        face_hidden = gate_a * block.attn(
            block.norm1(face_hidden) * (1 + scale_a) + shift_a, mask=valid_mask)
        hidden = hidden + _corners(face_hidden)
        hidden = hidden + gate_m * block.mlp(
            block.norm2(hidden) * (1 + scale_m) + shift_m, mask=valid_mask)

    final = backbone.final_layer
    free_mods = final.adaLN_modulation(c_free).chunk(2, dim=1)
    known_mods = final.adaLN_modulation(c_known).chunk(2, dim=1)
    shift, scale = (
        _corner_select(f, k, corner_known) for f, k in zip(free_mods, known_mods))
    hidden = final.norm_final(hidden) * (1 + scale) + shift
    return final.linear(hidden).view(batch, faces, 9)

