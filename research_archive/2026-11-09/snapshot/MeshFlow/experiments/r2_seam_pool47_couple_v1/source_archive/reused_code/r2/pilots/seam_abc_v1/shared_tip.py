"""C-only initial-state tip references and source-supervised training transport.

Source vertex IDs occur only in ``prepare_training``/``audit_sources``.  The
inference feature/decoder APIs accept C, N, a predicted SEAM layout and its
initial Gaussian state.  A reference guarantees one moving vertex identity,
not a shared edge, a legal face pair, or whole-body reachability.
"""
from __future__ import annotations

import numpy as np

VERSION = 'seam_abc_shared_tip_v1'
L = float(np.sqrt(3.) * 2 / .3762)
QUERY_DIM = 14
PAIR_DIM = 44


def layout_features(layout, x0):
    """Return pointer features before any alias is imposed; no source-Y API.

    layout must have anchor_ids, known_mask, C_vertices, edge_features (or
    features), and slot_edge_ids, mapping each active seam slot to a C edge.
    No slot/parent identity is a neural feature.  IDs are decoder metadata.
    """
    x0 = np.asarray(x0)
    anchors = np.asarray(layout['anchor_ids'])
    known = np.asarray(layout['known_mask'])
    cv = np.asarray(layout['C_vertices'])
    edge_features = np.asarray(layout.get('edge_features', layout.get('features')))
    edge_ids = np.asarray(layout['slot_edge_ids'])
    n = len(x0); k = int(known.sum())
    if (x0.shape != (n, 3, 3) or x0.dtype != np.float32 or
            anchors.shape != (n, 3) or known.shape != (n,) or
            edge_ids.shape != (n,) or not np.isfinite(x0).all()):
        raise ValueError('Invalid C-only pointer state/layout')
    fixed = anchors >= 0
    if not fixed[known].all() or np.any(fixed.sum(1)[~known] == 1):
        raise ValueError('Expected original C, two-anchor seam or ordinary bulk')
    seam = ~known & (fixed.sum(1) == 2)
    bulk = ~known & ~fixed.any(1)
    if np.any(~known & ~seam & ~bulk):
        raise ValueError('Unsupported variable corner layout')
    tip_slots = np.flatnonzero(seam)
    tip_slots = tip_slots[np.argsort(edge_ids[tip_slots], kind='stable')]
    if len(np.unique(edge_ids[tip_slots])) != len(tip_slots) or np.any(edge_ids[tip_slots] < 0):
        raise ValueError('Each seam requires a unique C edge identity')
    tip_corner = np.asarray([np.flatnonzero(~fixed[s]).item() for s in tip_slots], np.int64)
    tips = (3 * tip_slots + tip_corner).astype(np.int64)
    queries = np.flatnonzero(np.repeat(bulk, 3)).astype(np.int64)
    flat = x0.reshape(-1, 3)
    center = (cv.min(0) + cv.max(0)) * np.float32(.5)
    span = np.ptp(cv, axis=0)
    q = np.concatenate((flat[queries] / L,
                        x0.mean(1)[queries // 3] / L,
                        np.broadcast_to(center / L, (len(queries), 3)),
                        np.broadcast_to(span / L, (len(queries), 3)),
                        np.broadcast_to(np.asarray([n / 256., k / 256.], np.float32), (len(queries), 2))), axis=1).astype(np.float32)
    e = edge_features[edge_ids[tip_slots]].astype(np.float32)
    p = np.concatenate((np.broadcast_to(q[:, None], (len(q), len(tips), QUERY_DIM)),
                        np.broadcast_to(e[None], (len(q), len(tips), 24)),
                        np.broadcast_to((flat[tips] / L)[None], (len(q), len(tips), 3)),
                        (flat[queries, None] - flat[None, tips]) / L), axis=2).astype(np.float32)
    if q.shape != (len(queries), QUERY_DIM) or p.shape != (len(queries), len(tips), PAIR_DIM):
        raise AssertionError('Pointer feature schema changed')
    return dict(pointer_null_features=q, pointer_pair_features=p,
                pointer_query_corner_ids=queries, pointer_tip_corner_ids=tips,
                pointer_tip_edge_ids=edge_ids[tip_slots].astype(np.int64))


def validate_alias(alias_corner_ids, anchor_ids, known_mask):
    alias = np.asarray(alias_corner_ids)
    anchors, known = np.asarray(anchor_ids), np.asarray(known_mask)
    n = len(known)
    if alias.shape != (n, 3) or alias.dtype != np.int64 or np.any(alias < -1):
        raise ValueError('alias_corner_ids must be int64[N,3], -1 or owner flat corner')
    fixed = anchors >= 0
    seam = ~known & (fixed.sum(1) == 2)
    bulk = ~known & ~fixed.any(1)
    aliased = alias >= 0
    if np.any(aliased & ~bulk[:, None]):
        raise ValueError('Only ordinary bulk corners can alias a seam tip')
    owners = alias[aliased]
    if len(owners):
        if np.any(owners >= n * 3) or np.any(fixed.reshape(-1)[owners]) or np.any(~seam[owners // 3]):
            raise ValueError('References must address an independent active seam tip')
        if np.any(alias.reshape(-1)[owners] >= 0):
            raise ValueError('Alias graph must be one-hop and acyclic')
    for row in alias:
        v = row[row >= 0]
        if len(v) != len(set(map(int, v))):
            raise ValueError('Two corners of one face cannot reference one tip')
    independent = ~fixed & ~aliased
    return dict(alias_mask=aliased, train_free_corner_mask=independent,
                train_free_coordinate_count=3 * int(independent.sum()),
                aliased_corner_count=int(aliased.sum()))


def clamp_alias(x, alias_corner_ids):
    out = np.array(x, copy=True)
    refs = np.asarray(alias_corner_ids).reshape(-1)
    valid = refs >= 0
    out.reshape(-1, 3)[valid] = np.asarray(x).reshape(-1, 3)[refs[valid]]
    return out


def clamp_alias_tensor(x, alias_corner_ids):
    """Differentiable copy for [1,N,9] or [N,3,3], no extra forward."""
    import torch
    refs = torch.as_tensor(alias_corner_ids, dtype=torch.long, device=x.device).reshape(-1)
    flat = x.reshape(-1, 3)
    if len(refs) != len(flat):
        raise ValueError('Tensor reference shape mismatch')
    mask = refs >= 0
    out = flat.clone()
    out[mask] = flat[refs[mask]]
    return out.reshape_as(x)


def shared_tip_forward(model, x, t, y, valid_mask, C_face_mask, corner_known, alias_corner_ids):
    """One existing SEAM forward; alias corners retain time=t/free role."""
    from r2.prototypes.seam_slot_c200_v1.model_adapter import seam_forward
    shared_x = clamp_alias_tensor(x, alias_corner_ids)
    velocity = seam_forward(model, shared_x, t, y, valid_mask, C_face_mask, corner_known)
    return clamp_alias_tensor(velocity, alias_corner_ids)


def independent_fm_loss(velocity, target, train_free_corner_mask, denominator):
    import torch
    mask = torch.as_tensor(train_free_corner_mask, dtype=torch.bool, device=velocity.device)
    count = int(mask.sum()) * 3
    if denominator < count or count <= 0:
        raise ValueError('Independent scalar denominator invalid')
    difference = (velocity.float() - target.float()).reshape(*mask.shape, 3)
    return difference[mask].square().sum() / int(denominator)


def decode_predictions(logits, features, anchor_ids, known_mask):
    """Argmax including Null=class0, with deterministic same-face collision veto.

    For conflicting corners, retain maximum logit margin over Null (tie: flat
    corner ID); reject the others to Null.  No geometry/GT-based threshold.
    """
    scores = np.asarray(logits, np.float32)
    query = features['pointer_query_corner_ids']; tips = features['pointer_tip_corner_ids']
    if scores.shape != (len(query), 1 + len(tips)) or not np.isfinite(scores).all():
        raise ValueError('Finite pointer scores [bulk corners, Null+active tips] required')
    decision = np.argmax(scores, axis=1)
    alias = np.full(np.asarray(anchor_ids).shape, -1, np.int64)
    proposals = []
    for qi in np.flatnonzero(decision):
        c = int(query[qi]); tip = int(tips[decision[qi] - 1])
        proposals.append((-(float(scores[qi, decision[qi]]) - float(scores[qi, 0])), c, tip, int(qi)))
    seen = set(); rejected = []
    for negmargin, c, tip, qi in sorted(proposals):
        key = (c // 3, tip)
        if key in seen:
            rejected.append(dict(query_corner_id=c, tip_corner_id=tip, reason='SAME_FACE_TIP_COLLAPSE'))
        else:
            seen.add(key); alias.reshape(-1)[c] = tip
    facts = validate_alias(alias, anchor_ids, known_mask)
    return alias, dict(**facts, pointer_argmax_nonnull=len(proposals),
                      pointer_conflict_null=len(rejected), rejected=rejected)


def prepare_training(sample, source_face_vertex_ids):
    """Teacher-forced source correspondence after the already-frozen OT.

    Preserve old loader metadata/free_corner_mask/count.  Add a separate FM
    independent mask/count; original free mask remains usable for diagnostics.
    Every initial feature is captured BEFORE imposing a teacher reference.
    """
    s = dict(sample)
    source = np.asarray(source_face_vertex_ids, np.int64)
    ids, cp = s['source_face_ids'], s['source_corner_permutations']
    transported = source[ids[:, None], cp]
    k = int(s['K']); anchors = s['anchor_ids']; known = s['known_mask']
    seam = ~known & ((anchors >= 0).sum(1) == 2)
    edge_ids = np.full(len(known), -1, np.int64)
    edge_ids[seam] = np.asarray(s['face_permutation'])[seam] - k
    layout = dict(anchor_ids=anchors, known_mask=known, C_vertices=s['C_vertices'],
                  edge_features=s['edge_features'], slot_edge_ids=edge_ids)
    features = layout_features(layout, s['x0'])
    alias = np.full(anchors.shape, -1, np.int64)
    tips = features['pointer_tip_corner_ids']; q = features['pointer_query_corner_ids']
    tip_source = transported.reshape(-1)[tips]
    labels = np.zeros(len(q), np.int64)
    # tips are ordered by C edge ID.  Multiple true seam tips may represent
    # the same source vertex; the smallest edge is the one teacher root.
    for qi, flat in enumerate(q):
        matched = np.flatnonzero(tip_source == transported.reshape(-1)[flat])
        if len(matched):
            labels[qi] = int(matched[0]) + 1
            alias.reshape(-1)[flat] = tips[matched[0]]
    facts = validate_alias(alias, anchors, known)
    if not np.array_equal(clamp_alias(s['x1'], alias), s['x1']):
        raise ValueError('Source-ID shared corners do not have bitwise equal targets')
    s['x0'] = clamp_alias(s['x0'], alias)
    s['u'] = s['x1'] - s['x0']
    t = np.float32(s['t'])
    s['xt'] = (np.float32(1) - t) * s['x0'] + t * s['x1']
    fixed = anchors >= 0
    s['xt'][fixed] = s['x0'][fixed]
    if not np.array_equal(clamp_alias(s['xt'], alias), s['xt']) or not np.array_equal(clamp_alias(s['u'], alias), s['u']):
        raise ValueError('Training path/velocity broke live reference')
    s.update(features, **facts, alias_corner_ids=alias, pointer_labels=labels,
             slot_edge_ids=edge_ids, pointer_label_contract='source_ID_after_OT_minimum_C_edge_owner')
    return s


def inference_aliases(head, layout, x0):
    """layout + original Gaussian x0 + learned head -> fixed references.

    Caller already obtained layout via the existing C-only occupancy head.
    This routine performs one tiny pointer-head call, zero generator calls.
    Caller controls autocast and the registered call ledger.
    """
    import torch
    features = layout_features(layout, x0)
    device = next(head.parameters()).device
    pair = torch.as_tensor(features['pointer_pair_features'], device=device)
    null = torch.as_tensor(features['pointer_null_features'], device=device)
    with torch.no_grad():
        logits = head(pair, null)
    alias, facts = decode_predictions(logits.float().cpu().numpy(), features,
                                      layout['anchor_ids'], layout['known_mask'])
    return alias, features, logits.float().cpu().numpy(), facts
