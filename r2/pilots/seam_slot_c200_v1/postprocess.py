"""Single official D0, exact C/seam restoration, unchanged GEO_C, source export.

All 32 RAW receipts must exist before the first D0 call.  Restored seam
coordinates are not assumed to imply shared IDs: every retained anchor is
checked against its C vertex after cleanup and source-winding transport.
Deleted seam faces remain deleted with the original GEO_C reason recorded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback
import json
import numpy as np

from . import generate as G

VERSION = 'r2_seam_slot_anchor_protected_D0_GEO_source_export_v1'


def all_raw(root, manifest):
    records = []
    for arm in G.ARMS:
        complete = root / 'evaluation' / arm / 'raw_complete.json'
        result = G.checked_receipt(complete)
        if result['outputs'] != 16: raise ValueError('Each arm must have all sixteen RAW outputs')
        for q in manifest['requests']:
            p = root / 'evaluation' / arm / q['request_id'] / 'raw/receipt.json'
            r = G.checked_receipt(p)
            if r['arm'] != arm or r['request_id'] != q['request_id'] or r['generator_forwards'] != 50:
                raise ValueError('RAW identity/budget mismatch')
            records.append(p)
    if len(records) != 32: raise ValueError('All-RAW-first barrier failed')
    return records


def validate_anchor_indices(post, raw):
    """Verify retained source corners reference the same vertex as original C."""
    k = int(raw['known_faces']); n = int(raw['input_faces'])
    original = raw['C_faces_original']
    source_faces, source_corners = post['source_face_ids'], post['source_corner_ids']
    if not np.array_equal(source_faces[:k], np.arange(k)):
        raise ValueError('C face rows changed')
    mapping = {}
    for compact_id, post_id in zip(original.reshape(-1), post['faces'][:k].reshape(-1)):
        compact_id, post_id = int(compact_id), int(post_id)
        if compact_id in mapping and mapping[compact_id] != post_id:
            raise ValueError('C vertex identity split during GEO')
        mapping[compact_id] = post_id
    if len(set(mapping.values())) != len(mapping): raise ValueError('Distinct C vertex identities merged')
    if set(mapping) != set(range(len(raw['C_vertices']))): raise ValueError('Missing original C vertex')
    for ci, pi in mapping.items():
        if not G.same(post['vertices'][pi], raw['C_vertices'][ci]): raise ValueError('C vertex bytes changed')
    transported = raw['raw_anchor_ids'][source_faces[:, None], source_corners]
    fixed = transported >= 0
    expected = np.full_like(transported, -1)
    for ci, pi in mapping.items(): expected[transported == ci] = pi
    if not np.array_equal(expected[fixed], post['faces'][fixed]):
        raise ValueError('Retained seam anchor coordinates did not preserve the required C index')
    kept = post['source_face_kept']; seam = raw['raw_seam_slot_mask']
    if kept.shape != (n,) or seam.shape != (n,): raise ValueError('Wrong RAW slot identity domain')
    return dict(output_anchor_ids=transported,
                raw_anchor_ids=raw['raw_anchor_ids'], raw_seam_slot_mask=seam,
                raw_slot_edge_id=raw['raw_slot_edge_id'], raw_slot_ids=source_faces.copy(),
                retained_seam_mask=seam[source_faces],
                compact_C_vertex_to_post=np.asarray([mapping[i] for i in range(len(mapping))], np.int64)), dict(
        status='PASS', C_vertex_ids_bijective=True, retained_anchor_corners=int(fixed[k:].sum()),
        input_active_seam_faces=int(seam.sum()), retained_seam_faces=int((seam & kept).sum()),
        deleted_seam_faces=int((seam & ~kept).sum()), deleted_seam_records=[
            dict(raw_slot_id=int(i), edge_id=int(raw['raw_slot_edge_id'][i]),
                 reason=str(post['source_face_reason'][i]),
                 duplicate_of_raw_slot=int(post['duplicate_of_source_face'][i]))
            for i in np.flatnonzero(seam & ~kept)],
        identity_domain='generated_RAW_slot_not_GT', final_anchor_vertex_identity_verified=True,
        geometry_quality='Not certified by anchor identity checks')


def geometry_from_proposal(raw, proposal, source):
    """CPU-only geometry finish; exact existing GEO_C and source solver."""
    from meshflow_control import geo_c, geo_io
    from meshflow_control.source_winding import export_source_winding
    canonical = raw['raw']; k = int(raw['known_faces'])
    if proposal.dtype != np.float32 or proposal.shape != canonical.shape or not np.isfinite(proposal).all():
        raise ValueError('Malformed official proposal')
    anchor = raw['raw_anchor_ids']; fixed = anchor >= 0
    pre = proposal.copy()
    pre[fixed] = raw['C_vertices'][anchor[fixed]]
    if not G.same(pre[:k], canonical[:k]): raise ValueError('Canonical C not preserved')
    geometry = geo_c.process(pre, k)
    canonical_post = geo_io._validate_result(pre, k, geometry)
    oriented = export_source_winding(canonical_post, source)
    post = oriented['oriented_arrays']
    extra, anchors = validate_anchor_indices(post, raw)
    final = dict(post, **extra)
    final.update(identity_domain=np.asarray('generated_RAW_slot_not_GT'),
                 occupied=raw['occupied'], occupancy_logits=raw['occupancy_logits'],
                 edge_records=raw['edge_records'], C_vertices=raw['C_vertices'],
                 C_faces_original=raw['C_faces_original'], C_faces_model=raw['C_faces_model'])
    if not G.same(final['vertices'][final['faces']], final['output']): raise ValueError('Export geometry mismatch')
    if not G.same(final['C'], source['C_original']): raise ValueError('Original source C not restored')
    return pre, canonical_post, final, geometry, oriented, anchors


def run(root, arm):
    if arm not in G.ARMS: raise ValueError('Unknown arm')
    root, reg, manifest = G.load_registered(root, require_execution_freeze=True)
    barrier = all_raw(root, manifest)
    armroot = root / 'evaluation' / arm
    complete = armroot / 'post_complete.json'
    if complete.exists(): return G.checked_receipt(complete)
    pending = []
    for q in manifest['requests']:
        dest = armroot / q['request_id'] / 'post'
        if (dest / 'receipt.json').exists(): G.checked_receipt(dest / 'receipt.json')
        elif dest.exists():
            if not (dest / 'proposal_receipt.json').exists() or (dest / 'geometry_attempt.json').exists():
                raise RuntimeError('Incomplete D0/geometry attempt; no implicit replay: ' + str(dest))
            G.checked_receipt(dest / 'proposal_receipt.json'); pending.append(q)
        else: pending.append(q)
    if not pending: raise RuntimeError('No pending outputs, missing aggregate; explicit finalize required')
    stage = armroot / ('postprocess_attempt_' + str(time.time_ns())); stage.mkdir()
    G.write(stage / 'all_RAW_barrier.json', dict(status='PASS', RAW=32, files={str(p):G.sha(p) for p in barrier}))
    checkpoint = reg['d0_checkpoint']
    if checkpoint['sha256'] != '6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36':
        raise ValueError('Only fixed official D0 allowed')
    need_model = any(not (armroot/q['request_id']/'post/proposal_receipt.json').exists() for q in pending)
    model = None; calls = count = 0; started = time.perf_counter()
    if need_model:
        snap = G.resources(); G.write(stage / 'resources_before.json', snap)
        G.dispatch_guard(snap, reg.get('resources', {}).get('D0_estimate_mib', 2600))
        from meshflow_control.denoise_model import load_denoiser, predict_proposal
        model, model_identity = load_denoiser(checkpoint['path'])
        G.write(stage / 'model_identity.json', model_identity)
        G.write(stage / 'resources_resident.json', G.resources())
    try:
        from meshflow_control.geo_io import write_indexed_ply
        for q in pending:
            source_folder = armroot/q['request_id']/'raw'; raw_receipt = G.checked_receipt(source_folder/'receipt.json')
            raw = G.load_npz(source_folder/'raw.npz'); source = G.load_npz(source_folder/'source_condition.npz')
            dest = armroot/q['request_id']/'post'
            if not dest.exists():
                dest.mkdir()
                G.write(dest/'attempt.json', dict(arm=arm, request=q, raw_receipt_sha256=G.sha(source_folder/'receipt.json'),
                    D0_checkpoint_sha256=checkpoint['sha256'], D0_budget=1, GEO_calls=1, unix=time.time(),
                    contract='D0 -> restore original C and predicted seam anchors -> original GEO_C -> source winding'))
            if (dest/'proposal_receipt.json').exists():
                receipt = G.checked_receipt(dest/'proposal_receipt.json')
                if receipt['raw_sha256'] != G.sha(source_folder/'raw.npz'): raise ValueError('Proposal RAW mismatch')
                proposal = G.load_npz(dest/'proposal.npz')['official_unconstrained_proposal']
            else:
                import torch
                G.write(dest/'resources_before_D0.json', G.resources())
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick = time.perf_counter()
                G.event(dest/'calls.jsonl', 'D0_FORWARD_STARTED', budget=1)
                proposal = predict_proposal(model, raw['raw'])
                torch.cuda.synchronize(); calls += 1
                G.event(dest/'calls.jsonl', 'D0_FORWARD_RETURNED', seconds=time.perf_counter()-tick)
                G.save_npz(dest/'proposal.npz', dict(official_unconstrained_proposal=proposal))
                G.write(dest/'proposal_receipt.json', dict(status='COMPLETE', raw_sha256=G.sha(source_folder/'raw.npz'),
                    official_checkpoint_sha256=checkpoint['sha256'], D0_forwards=1, seconds=time.perf_counter()-tick,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_scope='one D0 forward, resident model included; not total process VRAM',
                    files={str(dest/'proposal.npz'):G.sha(dest/'proposal.npz')}))
            G.write(dest/'geometry_attempt.json', dict(unix=time.time(), geometry_calls=1, source_orientation_calls=1))
            pre, canonical, final, geometry, oriented, anchors = geometry_from_proposal(raw, proposal, source)
            G.save_npz(dest/'pre_geo.npz', dict(output=pre, C=raw['raw'][:int(raw['known_faces'])],
                       known_faces=raw['known_faces'], raw_anchor_ids=raw['raw_anchor_ids'], raw_seam_slot_mask=raw['raw_seam_slot_mask']))
            G.save_npz(dest/'post_canonical.npz', canonical); G.save_npz(dest/'post.npz', final)
            write_indexed_ply(dest/'post.ply', final['vertices'], final['faces'])
            G.write(dest/'geometry.json', dict(stats=geometry['stats'], face_records=geometry['face_records']))
            G.write(dest/'anchor_identity.json', anchors)
            G.write(dest/'source_restoration.json', oriented['restoration_summary'])
            orient = oriented['orientation_result']
            G.write(dest/'orientation.json', dict(summary=orient['summary'], components=orient['component_rows'], edges=orient['edge_rows']))
            files = {str(p):G.sha(p) for p in dest.iterdir() if p.is_file()}
            G.write(dest/'receipt.json', dict(status='COMPLETE', schema=VERSION, arm=arm, request_id=q['request_id'],
                condition_id=q['condition_id'], raw_receipt_sha256=G.sha(source_folder/'receipt.json'),
                raw_sha256=G.sha(source_folder/'raw.npz'), official_D0_sha256=checkpoint['sha256'],
                raw_path=str(source_folder/'raw.npz'), post_path=str(dest/'post.npz'), post_sha256=G.sha(dest/'post.npz'),
                pre_geo_path=str(dest/'pre_geo.npz'), pre_geo_sha256=G.sha(dest/'pre_geo.npz'),
                D0_forwards=1, GEO_calls=1, orientation_solves=1, anchors=anchors,
                geometry_contract='Original GEO_C reused; no threshold/merge/cleanup modification',
                source_labels_used=False, files=files))
            count += 1
            print(json.dumps(dict(stage='POST', arm=arm, request_id=q['request_id'], completed=count,
                                 retained_seam=anchors['retained_seam_faces'], deleted_seam=anchors['deleted_seam_faces'])), flush=True)
        receipts = [armroot/q['request_id']/'post/receipt.json' for q in manifest['requests']]
        for p in receipts: G.checked_receipt(p)
        G.write(stage/'resources_finished.json', G.resources())
        rows = [dict(q, arm=arm, label=arm, post_path=str(armroot/q['request_id']/'post/post.npz'),
                     post_sha256=G.sha(armroot/q['request_id']/'post/post.npz'),
                     raw_path=str(armroot/q['request_id']/'raw/raw.npz'),
                     raw_sha256=G.sha(armroot/q['request_id']/'raw/raw.npz')) for q in manifest['requests']]
        result = dict(status='COMPLETE', arm=arm, outputs=16, new_D0_forwards=calls, new_GEO_calls=count, rows=rows,
                      wall_seconds=time.perf_counter()-started, files={str(p):G.sha(p) for p in receipts})
        G.write(complete, result)
        return result
    except BaseException as error:
        G.write(stage/'failure.json', dict(status='STOPPED', error=repr(error), traceback=traceback.format_exc(),
                D0_returns=calls, completed_requests=count, no_automatic_model_replay=True))
        raise
    finally:
        if model is not None:
            import torch
            del model; torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=G.ARMS, required=True)
    p.add_argument('--root', default='experiments/r2_seam_slot_c200_v1')
    a = p.parse_args(); print(json.dumps(G.clean(run(a.root, a.arm)), indent=2))


if __name__ == '__main__': main()
