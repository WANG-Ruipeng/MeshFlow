"""Independent, CPU-only final evidence audit for the fixed A/B/C experiment.

Reads saved arrays, receipts and logs. Does not import torch, execute scoring,
render images, or deserialize weights. --partial reports readiness without
reading mutable training logs. A complete audit is explicit and fail-closed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import collections
import csv
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
from . import eval_common as C

TRAIN_ARMS = ('A_COUPLE', 'B_POOL47', 'C_SHARED_TIP')
LABELS = (*C.ARMS, 'SEAM200')
BASE = C.ROOT / 'experiments/r2_seam_slot_c200_v1'
LONG_ROOT = C.ROOT / 'experiments/r2_seam_quality_four_v1'
ILLEGAL = {'proper_transverse_penetration', 'coplanar_positive_area_overlap'}
META = ('request_id', 'condition_id', 'parent_id', 'noise_index', 'N', 'K', 'split')


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def jsonlines(path):
    with C.resolve(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def csvrows(path):
    with C.resolve(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def scalar_equal(text, value):
    if value is None:
        return text == ''
    if isinstance(value, bool):
        return text == str(value)
    if isinstance(value, (int, float)):
        try:
            return math.isclose(float(text), float(value), rel_tol=1e-11, abs_tol=1e-12)
        except ValueError:
            return False
    return text == str(value)


def csv_numeric(text):
    return float(text == 'True') if text in ('True', 'False') else float(text)


class Audit:
    def __init__(self):
        self.hashed = {}
        self.checks = []

    def hash(self, path, expected=None):
        path = C.resolve(path).resolve()
        key = str(path)
        if key not in self.hashed:
            self.hashed[key] = C.sha(path)
        if expected is not None:
            need(self.hashed[key] == expected, 'Hash mismatch: ' + key)
        return self.hashed[key]

    def mapping(self, values):
        for path, digest in values.items():
            self.hash(path, digest)

    def receipt(self, path):
        value = C.read(path)
        self.hash(path)
        self.mapping(value.get('files', {}))
        return value

    def record(self, name, **fields):
        self.checks.append(dict(check=name, status='PASS', **fields))


def readiness(root):
    missing = []
    incomplete = []
    required = [root / 'evaluation/score_completion.json',
                root / 'evaluation/summary_completion.json',
                root / 'evaluation/gallery/render_completion.json',
                root / 'evaluation/visual_review_receipt.json']
    for arm in C.ARMS:
        required.extend(root / 'evaluation' / arm / (s + '_complete.json') for s in ('raw', 'post'))
    for path in required:
        if not path.exists():
            missing.append(str(path))
    for arm in TRAIN_ARMS:
        path = root / 'training' / arm / 'status.json'
        if not path.exists() or C.read(path).get('status') != 'COMPLETE':
            incomplete.append(arm)
    review_files = list((root / 'evaluation/reviews').glob('*.csv'))
    for prefix in ('3f','73','85','8c','2d','610','608'):
        path = root / 'evaluation/reviews' / (prefix+'.csv')
        if not path.exists():
            missing.append(str(path))
    return dict(status='READY' if not missing and not incomplete else 'NOT_READY',
                missing=missing, unfinished_training=incomplete, review_files=[str(p) for p in review_files],
                new_model_calls=0, new_scores=0)


def frozen(a, root):
    reg = C.read(root / 'registration.json')
    manifest = C.read(root / 'eval_manifest.json')
    need(tuple(reg['arms']) == C.ARMS and tuple(manifest['arms']) == C.ARMS, 'Frozen arm list')
    need(reg['updates_per_arm'] == 800 and reg['new_rollouts'] == reg['new_D0_forwards'] == 96, 'Frozen budget')
    need(reg['expected_occupancy_forwards'] == 24 and reg['expected_pointer_forwards'] == 28, 'Frozen head budget')
    need(len(manifest['requests']) == 28 and len(manifest['conditions']) == 7, 'Frozen panel size')
    need(collections.Counter(q['split'] for q in manifest['requests']) == {'TRAIN': 16, 'DEV': 12}, 'Frozen panel split')
    need(len({q['request_id'] for q in manifest['requests']}) == 28, 'Repeated requests')
    for cond in manifest['conditions']:
        qs = [q for q in manifest['requests'] if q['condition_id'] == cond['condition_id']]
        need(len(qs) == 4 and sorted(q['noise_index'] for q in qs) == [0, 1, 2, 3], 'Four fixed noises per C')
    a.hash(root / 'registration.json')
    a.hash(root / 'eval_manifest.json', reg['eval_manifest_sha256'])
    a.hash(root / 'PROTOCOL.md', reg['protocol_sha256'])
    a.hash(reg['source_checkpoint'], reg['source_checkpoint_sha256'])
    a.hash(reg['d0_checkpoint']['path'], reg['d0_checkpoint']['sha256'])
    freeze = a.receipt(root / 'execution_freeze.json')
    for key in ('code_hashes', 'input_hashes'):
        a.mapping(freeze[key])
    for arm in TRAIN_ARMS:
        folder = root / 'training_inputs' / arm
        f = a.receipt(folder / 'execution_freeze.json')
        need(f['status'] == 'FROZEN_BEFORE_GPU' and f['source_gate_status'] == 'PASS', 'Training freeze ' + arm)
        need(f['generator_forward_before_freeze'] == f['D0_forward_before_freeze'] == 0, 'Pre-freeze GPU call')
        a.hash(folder / 'registration.json', f['registration_sha256'])
        for key in ('code_hashes', 'input_hashes'):
            a.mapping(f[key])
    cpu = a.receipt(root / 'training_cpu_checks.json')
    need(cpu['status'] == 'PASS' and cpu['GPU_calls'] == cpu['neural_model_calls'] == cpu['optimizer_updates'] == 0, 'A CPU gate')
    a.hash(C.ROOT / 'r2/pilots/seam_abc_v1/coupling.py', cpu['coupling_sha256'])
    proof = a.receipt(root / 'shared_tip_cpu/completion.json')
    need(proof['status'] == 'COMPLETE' and proof['GPU_calls'] == proof['OT_calls'] == proof['generator_forwards'] == proof['source_rescoring'] == 0, 'C CPU topology proof')
    a.mapping(proof['source_hashes'])
    a.mapping(proof['output_hashes'])
    cp = C.read(root / 'shared_tip_cpu/cpu_checks.json')
    need(cp['status'] == 'PASS' and cp['head_forward_calls'] == 4 and cp['head_backward_calls'] == 1 and cp['algebra_backward_calls'] == 1, 'CPU tiny-head call accounting')
    need(cp['optimizer_updates'] == cp['generator_forwards'] == cp['D0_forwards'] == 0, 'CPU test crossed budget boundary')
    b = a.receipt(root / 'B_DATA_MANIFEST.json')
    need(b['status'] == 'COMPLETE' and b['samples'] == b['OT_logical_calls'] == 6400, 'B OT budget')
    for key in ('all_C_seam_anchors_exact', 'all_source_faces_exact_once', 'all_source_winding_roundtrips', 'all_third_in_C_owner_tips_free', 'old_assets_unchanged'):
        need(b[key] is True, 'B data gate ' + key)
    a.record('frozen_contract', panel_requests=28, new_endpoint_rollouts=96, new_D0=96,
             CPU_pointer_forward=4, CPU_pointer_backward=1, CPU_algebra_backward=1,
             CPU_generator_and_D0=0, new_B_OT=6400, C_proof_conditions=47)
    return reg, manifest


def training(a, root):
    rows = []
    by_arm = {}
    common = dict(train_forward=6400, train_backward=6400, optimizer_update=800,
                  head_forward=6400, head_backward=6400, head_optimizer_update=800)
    for arm in TRAIN_ARMS:
        folder = root / 'training' / arm
        inp = root / 'training_inputs' / arm
        reg, data, gate = (C.read(inp / n) for n in ('registration.json', 'data_manifest.json', 'data_gate.json'))
        need(gate['status'] == 'PASS' and len(data['steps']) == 800, 'Data gate/steps')
        need([x['source_step'] for x in data['steps']] == list(range(201, 1001)), 'Stream cursor')
        status = a.receipt(folder / 'status.json')
        need((status['status'], status['local_step'], status['global_step']) == ('COMPLETE', 800, 13000), 'Training incomplete ' + arm)
        expected = dict(common)
        if arm == 'C_SHARED_TIP':
            expected.update(pointer_forward=6400, pointer_backward=6400, pointer_optimizer_update=800)
        for name in ('checkpoint', 'inference_export', 'occupancy_export', *(['pointer_export'] if arm == 'C_SHARED_TIP' else [])):
            rec = status[name]
            a.hash(rec['path'], rec['sha256'])
        ck = status['checkpoint']
        need(ck['generator_state_sha256'] == status['inference_export']['generator_state_sha256'], 'Generator export mismatch')
        need(ck['head_state_sha256'] == status['occupancy_export']['head_state_sha256'], 'Occupancy export mismatch')
        if arm == 'C_SHARED_TIP':
            need(ck['pointer_state_sha256'] == status['pointer_export']['pointer_state_sha256'], 'Pointer export mismatch')
        need(not status['calls']['failed'] and not status['calls']['pending_call_ids'], 'Unfinished model call')
        for key in ('attempted', 'successful'):
            actual = {k: v for k, v in status['calls'][key].items() if v}
            need(actual == expected, 'Status call budget ' + arm + ':' + key)
        calls = jsonlines(folder / 'calls.jsonl')
        a.hash(folder / 'calls.jsonl')
        begins, ends = {}, {}
        for r in calls:
            need(r['event'] in ('CALL_BEGIN', 'CALL_DONE'), 'Failed or unexpected call event')
            table = begins if r['event'] == 'CALL_BEGIN' else ends
            need(r['call_id'] not in table, 'Repeated call ID')
            table[r['call_id']] = r
        need(set(begins) == set(ends), 'Unpaired model calls')
        need(dict(collections.Counter(x['kind'] for x in begins.values())) == expected, 'Actual call budget ' + arm)
        for cid, r in begins.items():
            done = ends[cid]
            need((r['kind'], r['context']) == (done['kind'], done['context']) and done['time_unix'] >= r['time_unix'], 'Call identity/chronology')
        need([r['context']['global_step'] for r in begins.values() if r['kind'] == 'optimizer_update'] == list(range(12201, 13001)), 'Skipped/replayed generator update')
        files = sorted(folder.glob('attempts/*/initialization.json'))
        need(len(files) == 1, 'Multiple initialization attempts require explicit recovery audit')
        ini = a.receipt(files[0])
        for field, key in (('generator_state_sha256', 'source_generator_sha256'), ('optimizer_state_sha256', 'source_optimizer_sha256'), ('rng_sha256', 'source_rng_sha256'), ('head_state_sha256', 'source_head_sha256')):
            need(ini[field] == reg[key], 'Initial restore differs ' + field)
        need(ini['global_step'] == 12200 and ini['exact_generator_optimizer_RNG_restoration'] and not ini['head_fresh'] and not ini['EMA'], 'Wrong source continuation')
        if arm == 'C_SHARED_TIP':
            need(ini['pointer_fresh'] and ini['pointer_parameters'] == 3970 and ini['pointer_optimizer_step'] == 0 and ini['pointer_seed'] == 2026101703, 'Fresh pointer contract')
        exposure = jsonlines(folder / 'exposure.jsonl')
        logs = jsonlines(folder / 'training_log.jsonl')
        heads = jsonlines(folder / 'occupancy_training.jsonl')
        diag = jsonlines(folder / 'geometry_diagnostics.jsonl')
        for name in ('exposure', 'training_log', 'occupancy_training', 'geometry_diagnostics'):
            a.hash(folder / (name + '.jsonl'))
        need(len(exposure) == len(heads) == len(diag) == 6400 and len(logs) == 800, 'Training evidence rows')
        for i, r in enumerate(exposure):
            step, micro = divmod(i, 8)
            rec = data['steps'][step]['samples'][arm][micro]
            need((r['local_step'], r['global_step'], r['microbatch']) == (step + 1, 12201 + step, micro), 'Exposure cursor')
            need(C.resolve(r['path']) == C.resolve(rec['path']) and r['uid'] == rec['uid'] and r['alpha'] == rec['alpha'], 'Exposure sample identity')
            a.hash(rec['path'], rec['sha256'])
        pointers = jsonlines(folder / 'pointer_training.jsonl') if arm == 'C_SHARED_TIP' else []
        if pointers:
            need(len(pointers) == 6400, 'Pointer training row count')
            a.hash(folder / 'pointer_training.jsonl')
        for i, log in enumerate(logs):
            exp = exposure[i * 8:(i + 1) * 8]
            hlog = heads[i * 8:(i + 1) * 8]
            dlog = diag[i * 8:(i + 1) * 8]
            need((log['local_step'], log['global_step']) == (i + 1, 12201 + i), 'Training log cursor')
            original = sum(r['free_coordinate_count'] for r in exp)
            independent = sum(r.get('independent_free_coordinate_count', r['free_coordinate_count']) for r in exp)
            need(log['free_coordinate_denominator'] == independent, 'FM denominator')
            need(log['occupancy_edge_denominator'] == sum(r['edge_count'] for r in hlog), 'Head denominator')
            need(math.isclose(log['occupancy_BCE'], sum(r['weighted_BCE_contribution'] for r in hlog), rel_tol=1e-10, abs_tol=1e-12), 'Head loss aggregation')
            need(math.isclose(log['FM'], sum(r['FM_contribution'] for r in dlog), rel_tol=1e-10, abs_tol=1e-12), 'FM aggregation')
            if arm == 'C_SHARED_TIP':
                p = pointers[i * 8:(i + 1) * 8]
                alias_count = sum(r['alias_corner_count'] for r in exp)
                need(log['original_free_coordinate_denominator'] == original and original - independent == 3 * alias_count == 3 * log['alias_corners'], 'Shared reference independent-variable count')
                need(log['pointer_query_denominator'] == sum(r['query_count'] for r in p), 'Pointer CE denominator')
                need(math.isclose(log['pointer_CE'], sum(r['CE_contribution'] for r in p), rel_tol=1e-10, abs_tol=1e-12), 'Pointer CE aggregation')
                for dr in dlog:
                    need(dr['all_free_diagnostic_includes_aliases'] and dr['batch_free_coordinate_denominator'] == independent, 'C diagnostic/FM distinction')
                    need(math.isclose(dr['independent_velocity_sse'] / independent, dr['FM_contribution'], rel_tol=3e-6, abs_tol=1e-7), 'Independent SSE/loss')
            elif arm == 'B_POOL47':
                need(log.get('coupling', 0) == 0, 'A loss leaked into B')
        by_arm[arm] = dict(status=status, exposure=exposure, heads=heads, data=data)
        rows.append(dict(arm=arm,updates=800,forward=6400,backward=6400,head_updates=800,
                         pointer_forward=6400 if pointers else 0,pointer_updates=800 if pointers else 0,
                         source_global_step=12200,final_global_step=13000,checkpoint_sha256=ck['sha256']))
    def shared_identity(r):
        return tuple(r[k] for k in ('local_step','global_step','microbatch','sample_index','uid','alpha','path','t','N','K','free_coordinate_count'))
    need([shared_identity(r) for r in by_arm['A_COUPLE']['exposure']] == [shared_identity(r) for r in by_arm['C_SHARED_TIP']['exposure']], 'A/C same input stream')
    def no_clock(r):
        return {k:v for k,v in r.items() if k not in ('pid','time_unix')}
    need([no_clock(r) for r in by_arm['A_COUPLE']['heads']] == [no_clock(r) for r in by_arm['C_SHARED_TIP']['heads']], 'A/C identical occupancy training')
    old = C.read(LONG_ROOT / 'training/LONG/status.json')
    for arm in ('A_COUPLE','C_SHARED_TIP'):
        for key in ('head_state_sha256','head_optimizer_state_sha256'):
            need(by_arm[arm]['status']['checkpoint'][key] == old['checkpoint'][key], 'Same-stream head/AdamW differs from LONG')
    need(len({r['uid'] for r in by_arm['B_POOL47']['exposure']}) == 47, 'B actual parent coverage')
    need(len({r['uid'] for r in by_arm['A_COUPLE']['exposure']}) == 4, 'A parent scope')
    a.record('training',generator_updates=2400,generator_forward=19200,generator_backward=19200,
             occupancy_updates=2400,pointer_updates=800,pointer_forward=6400,pointer_backward=6400,
             extra_preflight=0,A_C_same_stream_and_head=True,B_actual_parents=47,
             tensor_state_recomputation='NA: byte hashes and recorded exact restore; no torch loading')
    return rows


def outputs(a, root, reg, manifest):
    rows = []
    raw_last, d0_first = [], []
    calls = collections.Counter()
    pointer_summary = collections.Counter()
    for arm in C.ARMS:
        folder = root / 'evaluation' / arm
        for stage in ('raw', 'post'):
            comp = a.receipt(folder / (stage + '_complete.json'))
            need(comp['status'] == 'COMPLETE' and comp['outputs'] == 28, 'Incomplete endpoint arm/stage')
        layout_paths = list((folder / 'layouts').glob('*/receipt.json'))
        need(len(layout_paths) == 7, 'Seven C-only occupancy layouts')
        for path in layout_paths:
            r = a.receipt(path)
            need(r['source_labels_read'] is False and r['occupancy_threshold_logit'] == 0, 'Occupancy inference contract')
            cached = 'cache_origin_receipt' in r
            need(r['occupancy_forward_calls'] == (0 if cached else 1), 'Cached occupancy counted as new')
            need(not cached or arm == 'LONG', 'Unexpected cached layout arm')
            calls['occupancy'] += r['occupancy_forward_calls']
            a.hash(r['head']['path'], r['head']['sha256'])
        for q in manifest['requests']:
            rid, k, n = q['request_id'], q['K'], q['N']
            rd, pd = folder / rid / 'raw', folder / rid / 'post'
            rr, pr = a.receipt(rd / 'receipt.json'), a.receipt(pd / 'receipt.json')
            cached = arm == 'LONG' and q['split'] == 'TRAIN'
            need(rr['arm'] == pr['arm'] == arm and rr['request_id'] == pr['request_id'] == rid, 'Output registered identity')
            need(rr['generator_forwards'] == 50 and rr['D0_forwards'] == 0 and rr['source_labels_read'] is False, 'RAW contract')
            need(pr['D0_forwards'] == pr['GEO_calls'] == pr['orientation_solves'] == 1 and pr['source_labels_used'] is False, 'POST contract')
            need(rr['new_rollouts'] == (0 if cached else 1), 'RAW cache accounting')
            if cached:
                need(rr['newly_generated'] is False and pr['newly_generated'] is False and rr['new_D0_forwards'] == pr['new_D0_forwards'] == 0, 'Cached historical call flags')
                for dest, receipt, meshname in ((rd, rr, 'raw.npz'), (pd, pr, 'post.npz')):
                    original = C.resolve(receipt['cache_origin'])
                    a.hash(original / 'receipt.json', receipt['cache_origin_receipt_sha256'])
                    need(a.hash(dest / meshname) == a.hash(original / meshname), 'Cache geometry changed')
                calls['cached_LONG'] += 1
            prop = a.receipt(pd / 'proposal_receipt.json')
            need(prop['D0_forwards'] == 1 and prop['official_checkpoint_sha256'] == reg['d0_checkpoint']['sha256'], 'D0 fixed model identity')
            rc, pc = jsonlines(rd / 'calls.jsonl'), jsonlines(pd / 'calls.jsonl')
            pointer_events = [r for r in rc if r['event'].startswith('POINTER_')]
            forward_events = [r for r in rc if r['event'] in ('FORWARD_STARTED','FORWARD_RETURNED')]
            need(len(forward_events) == 100 and len(rc) == 100 + len(pointer_events), 'Euler actual count')
            for i in range(50):
                start, end = forward_events[2*i:2*i+2]
                need(start['event'] == 'FORWARD_STARTED' and end['event'] == 'FORWARD_RETURNED' and start['euler_step'] == end['euler_step'] == i and end['unix'] >= start['unix'], 'Euler actual call identity')
            need([r['event'] for r in pc] == ['D0_FORWARD_STARTED','D0_FORWARD_RETURNED'] and pc[1]['unix'] >= pc[0]['unix'], 'Single D0 actual ledger')
            if arm == 'C_SHARED_TIP':
                need([r['event'] for r in pointer_events] == ['POINTER_FORWARD_STARTED','POINTER_FORWARD_RETURNED'], 'Pointer actual calls')
                pointer = a.receipt(rd / 'pointer_audit.json')
                need(pointer['source_labels_read'] is False and pointer['pointer_forward_calls'] == rr['pointer_forwards'] == 1, 'Pointer source visibility')
                a.hash(pointer['identity']['path'], pointer['identity']['sha256'])
                calls['pointer'] += 1
            else:
                need(not pointer_events and rr.get('pointer_forwards', 0) == 0, 'Pointer leaked into another arm')
            if not cached:
                calls['rollouts'] += 1
                calls['generator'] += 50
                calls['D0'] += 1
                raw_last.append(forward_events[-1]['unix'])
                d0_first.append(pc[0]['unix'])
            raw = C.arrays(rd / 'raw.npz')
            post = C.arrays(pd / 'post.npz')
            pre = C.arrays(pd / 'pre_geo.npz')['output']
            source = C.arrays(rd / 'source_condition.npz')
            trajectory = C.arrays(rd / 'trajectory.npz')['state']
            anchor = raw['raw_anchor_ids']
            fixed = anchor >= 0
            vals = raw['C_vertices'][anchor[fixed]]
            need(trajectory.shape == (51,n,3,3) and exact(trajectory[-1],raw['raw']), 'Trajectory terminal identity')
            need(all(exact(t[fixed], vals) for t in trajectory) and exact(pre[fixed], vals), 'C/seam coordinate moved')
            need(exact(raw['output'][:k],source['C_original']) and exact(post['output'][:k],source['C_original']), 'Original C order changed')
            z = C.arrays(q['noise_npz'])['z']
            initial = np.zeros_like(raw['raw'])
            initial[k:] = z
            initial[fixed] = vals
            alias = raw.get('raw_alias_corner_ids', np.full(anchor.shape,-1,np.int64))
            selected = np.flatnonzero(alias.reshape(-1) >= 0)
            targets = alias.reshape(-1)[selected]
            need(arm == 'C_SHARED_TIP' or len(selected) == 0, 'Alias in ordinary representation')
            if arm == 'C_SHARED_TIP':
                pointer_arrays = C.arrays(rd / 'pointer.npz')
                need(np.array_equal(pointer_arrays['alias_corner_ids'],alias), 'Inference pointer decode mismatch')
                logits = pointer_arrays['logits']
                need(logits.ndim == 2 and np.isfinite(logits).all(), 'Pointer logits')
                raw_nonnull = int((np.argmax(logits,axis=1) != 0).sum())
                need(raw_nonnull == pointer['facts']['pointer_argmax_nonnull'], 'Pointer argmax tally')
                need(len(selected) == pointer['facts']['aliased_corner_count'], 'Pointer accepted alias tally')
                pointer_summary['queries'] += len(logits)
                pointer_summary['argmax_nonnull'] += raw_nonnull
                pointer_summary['decoded_nonnull'] += len(selected)
                pointer_summary['all_Null_requests'] += int(not raw_nonnull and not len(selected))
                pointer_summary['actual_reference_outputs'] += int(bool(len(selected)))
            if len(selected):
                need(np.all(selected//3 >= k) and np.all(targets//3 >= k), 'C corner alias illegal')
                need(np.all(~raw['raw_seam_slot_mask'][selected//3]) and np.all(raw['raw_seam_slot_mask'][targets//3]), 'Bulk-to-tip role violation')
                need(np.all(anchor.reshape(-1)[selected] < 0) and np.all(anchor.reshape(-1)[targets] < 0) and np.all(alias.reshape(-1)[targets] < 0), 'Alias anchor/cycle violation')
                initial.reshape(-1,3)[selected] = initial.reshape(-1,3)[targets]
                need(all(exact(t.reshape(-1,3)[selected],t.reshape(-1,3)[targets]) for t in trajectory), 'Alias separated during rollout')
                need(exact(pre.reshape(-1,3)[selected],pre.reshape(-1,3)[targets]), 'D0 alias separated')
                canonical = C.arrays(pd / 'post_canonical.npz')
                mapping = canonical['corner_vertex_ids'].reshape(-1)
                need(np.array_equal(mapping[selected],mapping[targets]), 'GEO actual alias index mismatch')
                retained = canonical['source_face_kept'][selected//3]
                need(np.all(mapping[selected[retained]] >= 0), 'Retained alias has no final vertex ID')
            need(exact(initial,trajectory[0]), 'Saved Gaussian/alias initial state changed')
            if not cached:
                audit = rr['audit']
                original_scalar = 3 * int((~fixed).sum())
                need(audit['original_free_scalar_count'] == original_scalar and audit['free_scalar_count'] == original_scalar - 3*len(selected), 'Inference independent scalar count')
                need(audit['referenced_bulk_corners'] == len(selected), 'Inference alias count')
            mapping = {}
            for ci,vi in zip(raw['C_faces_original'].flat,post['faces'][:k].flat):
                ci,vi = int(ci),int(vi)
                need(ci not in mapping or mapping[ci] == vi, 'C source vertex split')
                mapping[ci] = vi
            need(len(mapping) == len(set(mapping.values())) == len(raw['C_vertices']), 'C vertex identity collapse')
            src,corner = post['source_face_ids'],post['source_corner_ids']
            transport = anchor[src[:,None],corner]
            for fi,co in np.argwhere(transport >= 0):
                need(int(post['faces'][fi,co]) == mapping[int(transport[fi,co])], 'Retained seam not sharing C index')
            need(exact(post['vertices'][post['faces']],post['output']), 'POST indexed-coordinate mismatch')
            need(np.array_equal(raw['occupied'],raw['occupancy_logits'] >= 0), 'Occupancy threshold differs')
            rows.append(dict(arm=arm,request_id=rid,split=q['split'],cached=cached,new_RAW=int(not cached),
                             new_D0=int(not cached),new_generator_calls=0 if cached else 50,
                             pointer_calls=int(arm=='C_SHARED_TIP'),aliases=len(selected),
                             C_preserved=True,all51_anchors_exact=True,
                             alias_index_status='PASS' if len(selected) else 'NA_NO_PREDICTED_REFERENCE',
                             raw_sha256=a.hash(rd/'raw.npz'),post_sha256=a.hash(pd/'post.npz')))
        barriers = list(folder.glob('postprocess_attempt_*/all_RAW_barrier.json'))
        need(barriers, 'Missing complete RAW barrier')
        for p in barriers:
            b = a.receipt(p)
            need(b['status'] == 'PASS' and b['RAW'] == len(b['files']) == 112, 'Global all-RAW barrier')
    need(dict(calls) == dict(occupancy=24,cached_LONG=16,rollouts=96,generator=4800,D0=96,pointer=28), 'Actual new/cached inference budget')
    need(max(raw_last) < min(d0_first), 'D0 started before all new RAW finished')
    a.record('endpoint_inference',**calls,all_RAW_before_D0=True,
             last_new_RAW_unix=max(raw_last),first_new_D0_unix=min(d0_first),
             C_source_order_and_index_preserved=True,shared_tip_equality_not_edge_legality=True)
    a.record('observed_pointer_endpoint_decisions',**pointer_summary,
             interpretation='No-reference outputs do not empirically test inference-time live sharing; CPU and teacher-layout training checks are separate.')
    return rows


def numerical(a, root, manifest):
    completion = a.receipt(root/'evaluation/score_completion.json')
    need(completion['status'] == 'COMPLETE' and completion['cells'] == len(completion['receipts']) == 224, '224 current score cells')
    score_reg = C.read(root/'evaluation/score_registration.json')
    a.hash(root/'evaluation/score_registration.json',completion['registration_sha256'])
    a.hash(root/'eval_manifest.json',score_reg['manifest_sha256'])
    a.mapping(score_reg['source_hashes'])
    a.mapping(score_reg['protected'])
    for rec in completion['receipts']:
        a.hash(rec['path'],rec['sha256'])
    need(sum(len(list((root/'evaluation/scores'/arm).glob('*/*/score.json'))) for arm in C.ARMS) == 224, 'Extra/missing score files')
    actual = csvrows(root/'evaluation/per_output.csv')
    a.hash(root/'evaluation/per_output.csv')
    need(len(actual) == 256, '224 current +32 historical score rows')
    lookup = {(r['label'],r['request_id'],r['stage']):r for r in actual}
    need(len(lookup) == 256, 'Repeated score row')
    expected = {}
    totals = collections.defaultdict(collections.Counter)
    for label in LABELS:
        rroot = BASE if label == 'SEAM200' else root
        arm = 'SEAM_SLOT_C200' if label == 'SEAM200' else label
        for q in manifest['requests']:
            if label == 'SEAM200' and q['split'] != 'TRAIN':
                continue
            for stage in ('RAW','POST'):
                key = label,q['request_id'],stage
                folder = rroot/'evaluation/scores'/arm/q['request_id']/stage
                s = a.receipt(folder/'score.json')
                a.hash(s['identity']['mesh_path'],s['identity']['mesh_sha256'])
                need(s['full_quality_scans'] == 1 and s['model_calls'] == s['new_queries'] == 0, 'Saved scoring contract')
                row = {k:s[k] for k in META}
                row.update(label=label,stage=stage,split=q['split'])
                for field in ('metrics','topology','distance','seam_to_bulk'):
                    row.update({k:v for k,v in s[field].items() if not isinstance(v,(dict,list)) and k != 'status'})
                row.update(seam_to_bulk_status=s['seam_to_bulk']['status'],C_preserved_exact=s['C_preserved_exact'],I_status=s['status_by_metric']['I'],
                           free_face_denominator=s['free_face_denominator'],CF_tested_pair_denominator=s['CF_tested_pair_denominator'],
                           FF_tested_pair_denominator=s['FF_tested_pair_denominator'],C_query_weight_denominator=s['C_query_weight_denominator'])
                for boundary in s['boundary']:
                    row['A_VMR__'+boundary['setting']] = boundary['VMR']
                    row['PMR__'+boundary['setting']] = boundary['perfect']
                mesh = C.arrays(s['identity']['mesh_path'])
                ids = mesh['raw_slot_ids']
                seam = mesh['raw_seam_slot_mask'][ids]
                count = collections.Counter()
                for kind,scan in C.read(folder/'pairs.json').items():
                    for contact in scan['contacts']:
                        if contact['classification'] not in ILLEGAL:
                            continue
                        i,j = map(int,contact['pair'])
                        if kind == 'FF':
                            i += q['K']; j += q['K']
                            role = 'seam-seam' if seam[i] and seam[j] else 'seam-bulk' if seam[i] or seam[j] else 'bulk-bulk'
                        else:
                            need(kind == 'CF','Unknown contact domain')
                            j += q['K']
                            role = 'C-seam' if seam[j] else 'C-bulk'
                        count[role] += 1
                for role in ('seam-seam','seam-bulk','bulk-bulk','C-seam','C-bulk'):
                    row['illegal_pairs_'+role] = count[role]
                need(sum(count[r] for r in ('seam-seam','seam-bulk','bulk-bulk')) == s['metrics']['FF_illegal_pair_count'], 'FF role partition')
                need(count['C-seam']+count['C-bulk'] == s['metrics']['CF_illegal_pair_count'], 'CF role partition')
                for field,value in row.items():
                    need(scalar_equal(lookup[key].get(field,''),value),'Per-output mismatch '+str(key)+' '+field)
                expected[key] = row
                totals[label,q['split'],stage].update(count)
    need([totals['SEAM200','TRAIN','POST'][r] for r in ('bulk-bulk','seam-bulk','seam-seam')] == [338,4,1], 'Historical role reconstruction')
    summaries = csvrows(root/'evaluation/summary.csv')
    a.hash(root/'evaluation/summary.csv')
    summary_keys = set()
    for r in summaries:
        key = r['label'],r['split'],r['stage'],r['metric']
        need(key not in summary_keys,'Repeated summary cell')
        summary_keys.add(key)
        selected = [v for k,v in expected.items() if k[0] == r['label'] and k[2] == r['stage'] and (r['split'] == 'ALL' or v['split'] == r['split'])]
        values = [v[r['metric']] for v in selected if isinstance(v.get(r['metric']),(int,float,bool))]
        registered = {'TRAIN':16,'DEV':12,'ALL':28}[r['split']]
        need(int(r['n_registered']) == registered == len(selected) and int(r['n_valid']) == len(values), 'Summary denominators')
        need(r['status'] == ('PASS' if len(values) == registered else 'NA_OR_PARTIAL'), 'Summary status')
        for field,func in (('mean',np.mean),('median',np.median),('min',np.min),('max',np.max),('sum',np.sum)):
            need(scalar_equal(r[field],float(func(values)) if values else None),'Summary statistic '+str(key)+' '+field)
    paired = csvrows(root/'evaluation/paired_per_noise.csv')
    a.hash(root/'evaluation/paired_per_noise.csv')
    pair_keys = set()
    for r in paired:
        key = tuple(r[k] for k in ('baseline','candidate','request_id','stage','metric'))
        need(key not in pair_keys,'Duplicate noise comparison')
        pair_keys.add(key)
        av = expected[r['baseline'],r['request_id'],r['stage']].get(r['metric'])
        bv = expected[r['candidate'],r['request_id'],r['stage']].get(r['metric'])
        valid = isinstance(av,(int,float,bool)) and isinstance(bv,(int,float,bool))
        need(scalar_equal(r['baseline_value'],av) and scalar_equal(r['candidate_value'],bv),'Paired values')
        need(scalar_equal(r['delta'],float(bv)-float(av) if valid else None),'Paired delta')
        need(r['status'] == ('PASS' if valid else 'NA'),'Paired status')
    parents = csvrows(root/'evaluation/paired_per_parent.csv')
    a.hash(root/'evaluation/paired_per_parent.csv')
    groups = collections.defaultdict(list)
    fields = ('baseline','candidate','parent_id','split','stage','metric')
    for r in paired:
        groups[tuple(r[k] for k in fields)].append(r)
    seen = set()
    for r in parents:
        key = tuple(r[k] for k in fields)
        need(key not in seen,'Repeated parent comparison')
        seen.add(key)
        group = groups[key]
        valid = [v for v in group if v['status'] == 'PASS']
        need(len(group) == 4 and int(r['n_registered']) == 4 and int(r['n_valid']) == len(valid),'Parent denominator')
        for field in ('baseline_value','candidate_value','delta'):
            need(scalar_equal(r[field],float(np.mean([csv_numeric(v[field]) for v in valid])) if valid else None),'Parent mean')
    need(seen == set(groups),'Missing parent comparison')
    summary = a.receipt(root/'evaluation/summary_completion.json')
    a.mapping(summary['source_scores']); a.mapping(summary['summary_files'])
    need(summary['current_panel_scores'] == 224 and summary['historical_SEAM200_scores'] == 32 and summary['new_scans'] == 0, 'Summary scope')
    a.record('saved_score_reconstruction',current_scores=224,historical_scores=32,per_output_rows=256,
             recorded_new_scans=completion['full_quality_scans'],summary_rows=len(summaries),
             paired_noise_rows=len(paired),paired_parent_rows=len(parents),
             contact_indexing='CF free +K; FF both +K; kept output mapped to raw slot',
             collision_detector_reruns=0)
    return [dict(label=k[0],split=k[1],stage=k[2],**v) for k,v in sorted(totals.items())]


def visual(a, root, manifest):
    gallery = root/'evaluation/gallery'
    completion = a.receipt(gallery/'render_completion.json')
    need(completion['pages'] == 56 and completion['new_model_calls'] == completion['geometry_edits'] == 0, 'Gallery budget')
    a.mapping(completion['artifacts'])
    definition = C.read(gallery/'definition.json')
    a.hash(gallery/'definition.json',completion['definition_sha256'])
    a.mapping(definition['geometry_hashes'])
    need(len(definition['geometry_hashes']) == 224, 'Gallery input completeness')
    images = a.receipt(gallery/'manifest.json')['images']
    need(len(images) == 56,'56 image pages')
    lookup = {(r['request_id'],r['stage']):r for r in images}
    need(len(lookup) == 56,'Duplicate gallery page')
    for r in images:
        a.hash(r['path'],r['image_sha256'])
    expected = {(q['request_id'],stage,label) for q in manifest['requests'] for stage in ('RAW','POST') for label in C.ARMS}
    reviewed = {}
    for path in sorted((root/'evaluation/reviews').glob('*.csv')):
        rows = csvrows(path)
        if not rows or not {'request_id','stage','model','review_status'}.issubset(rows[0]):
            continue
        a.hash(path)
        for r in rows:
            key = r['request_id'],r['stage'],r['model']
            need(key in expected,'Unexpected visual review')
            need(key not in reviewed,'Duplicate visual review; aggregate files must be stored outside reviews/')
            need(r['review_status'] == 'ACTUALLY_VIEWED' and 'AI_PROXY' in r['reviewer'],'Not actual AI_PROXY review')
            need(r['review_kind'] == 'AI_PROXY' and r['blinding'].startswith('NONBLIND'),'Review identity/blinding')
            image = lookup[key[:2]]
            need(r['image_sha256'] == image['image_sha256'] and C.resolve(r['image_path']) == C.resolve(image['path']),'Reviewed image identity')
            for field in ('whole_body_complete','C_adoption','fragments_or_folded_layers','visible_join','visible_observations','uncertainty'):
                need(bool(r[field].strip()),'Blank actual review '+field)
            reviewed[key] = r
    need(set(reviewed) == expected,'Missing actual reviews')
    aggregate = a.receipt(root/'evaluation/visual_review_receipt.json')
    need(aggregate['status'] == 'PASS' and aggregate['row_count'] == aggregate['actual_view_record_count'] == 224 and aggregate['page_count'] == 56, 'Actual review aggregation')
    need(aggregate['labels_reassigned'] == aggregate['new_visual_judgments'] == aggregate['new_geometry_scores'] == aggregate['model_calls'] == 0, 'Aggregation changed evaluation')
    a.mapping(aggregate['protected_hashes'])
    for name,digest in aggregate['output_sha256'].items():
        a.hash(root/'evaluation'/name,digest)
    merged = csvrows(root/'evaluation/visual_review.csv')
    need(len(merged) == 224 and {(r['request_id'],r['stage'],r['model']):r for r in merged} == reviewed, 'Merged visual cells changed')
    a.record('actual_review_records',pages=56,model_stage_reviews=224,reviewer='AI_PROXY, nonblind',
             audit_does_not_claim_human_review_or_infer_semantics=True)
    return list(reviewed.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true',help='Read readiness only, no mutable logs/weight hashing')
    args = parser.parse_args()
    root = C.OUT
    ready = readiness(root)
    if args.partial or ready['status'] != 'READY':
        print(json.dumps(ready,ensure_ascii=False))
        return
    out = root/'audit'
    target = out/'final_execution.json'
    if target.exists():
        old = C.read(target); a = Audit()
        a.mapping(old['verified_files']); a.mapping(old['audit_artifacts'])
        print(json.dumps(dict(status='PASS_REVERIFIED',files=len(a.hashed),new_model_calls=0,new_scores=0)))
        return
    a = Audit(); started = time.perf_counter()
    reg, manifest = frozen(a,root)
    training_rows = training(a,root)
    output_rows = outputs(a,root,reg,manifest)
    role_rows = numerical(a,root,manifest)
    reviews = visual(a,root,manifest)
    need('torch' not in sys.modules,'Auditor imported model framework')
    a.hash(__file__)
    artifacts = {}
    for name,rows in (('training_budget',training_rows),('output_identity',output_rows),('illegal_roles',role_rows),('actual_review_registry',reviews)):
        path = out/(name+'.csv'); C.csvwrite(path,rows); artifacts[str(path)] = C.sha(path)
    value = dict(status='PASS',independent_auditor='seam_code_audit',new_model_calls=0,new_scores=0,new_geometry=0,
                 seconds=time.perf_counter()-started,checks=a.checks,verified_files=a.hashed,audit_artifacts=artifacts,
                 limits=['Weight tensors not deserialized: byte hashes and exact-restoration receipts verified.',
                         'Saved contact pairs and metric aggregates reconstructed; collision detector not rerun.',
                         'Actual AI_PROXY review records verified; auditor does not independently certify semantic ratings.',
                         'Shared vertex identity does not certify shared edges, manifoldness or geometric quality.'])
    C.write(target,value)
    print(json.dumps(dict(status='PASS',seconds=value['seconds'],files=len(a.hashed),checks=a.checks)))


if __name__ == '__main__':
    main()
