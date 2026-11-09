"""Fixed endpoint Euler50 for three SEAM quality continuations.

Only registered C bundles, requested N, saved Gaussian arrays and final
checkpoints enter generation.  Occupancy labels/source Y are never loaded.
One predicted layout per C is saved and shared by its four noise requests.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np

ARMS = ('LONG', 'A_COUPLE', 'B_POOL47', 'C_SHARED_TIP')
VERSION = 'r2_seam_abc_endpoint_inference_v1'
REPOSITORY = Path(__file__).resolve().parents[3]


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as handle:
        json.dump(clean(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def save_npz(path, arrays):
    if any(np.asarray(v).dtype.hasobject for v in arrays.values()): raise TypeError('Object arrays forbidden')
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
    loaded = load_npz(path)
    if loaded.keys() != arrays.keys() or any(not same(loaded[k], np.asarray(arrays[k])) for k in loaded):
        raise ValueError('NPZ byte roundtrip failed')


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def event(path, name, **fields):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(clean(dict(event=name, unix=time.time(), **fields)), allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def root_path(root):
    p = Path(root).resolve()
    expected = (REPOSITORY / 'experiments/r2_seam_abc_v1').resolve()
    if p != expected or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('This entrypoint owns only experiments/r2_seam_abc_v1')
    return p


def verify_files(files):
    for p, digest in files.items():
        if sha(p) != digest: raise ValueError('Frozen file changed: ' + str(p))


def load_registered(root, *, require_execution_freeze=False):
    root = root_path(root)
    reg = read(root / 'registration.json')
    freeze = root / 'execution_freeze.json'
    if require_execution_freeze and not freeze.is_file():
        raise ValueError('Freeze execution code and C-only input hashes before dispatch')
    if freeze.exists():
        f = read(freeze)
        for key in ('code_hashes', 'input_hashes'):
            if not isinstance(f.get(key), dict) or not f[key]: raise ValueError('Incomplete execution freeze: ' + key)
            verify_files(f[key])
    candidates = (root / 'eval_manifest.json', root / 'inputs/eval_manifest.json')
    manifests = [p for p in candidates if p.is_file()]
    if len(manifests) != 1: raise ValueError('Exactly one fixed eval_manifest required')
    path = manifests[0]; manifest = read(path)
    if reg.get('eval_manifest_sha256') and sha(path) != reg['eval_manifest_sha256']:
        raise ValueError('Evaluation manifest changed')
    if reg.get('status', 'FROZEN') != 'FROZEN': raise ValueError('Registration must be frozen')
    requests = manifest['requests']
    if len(requests) != 28 or len({q['request_id'] for q in requests}) != 28:
        raise ValueError('Exactly twenty-eight fixed request identities required')
    conditions = {q['condition_id'] for q in requests}
    if len(conditions) != 7 or any(sum(q['condition_id'] == c for q in requests) != 4 for c in conditions):
        raise ValueError('Seven conditions with four noises each required')
    for q in requests:
        if not 128 <= int(q['N']) <= 256 or not 2 <= int(q['K']) < int(q['N']):
            raise ValueError('Invalid registered N/K')
        if Path(q['request_id']).name != q['request_id'] or q['request_id'] in ('.', '..'):
            raise ValueError('Unsafe request identity')
        for key in ('inference_npz', 'noise_npz', 'metadata_npz'):
            if not Path(q[key]).is_file(): raise FileNotFoundError(q[key])
            expected = q.get(key + '_sha256') or q.get(key.removesuffix('_npz') + '_sha256')
            if expected and sha(q[key]) != expected: raise ValueError('Changed request asset: ' + key)
    for key in ('source_hashes', 'input_hashes'):
        if isinstance(manifest.get(key), dict): verify_files(manifest[key])
    return root, reg, manifest


def resources():
    def call(args):
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
    import psutil
    return dict(unix=time.time(), gpu=call(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']),
                processes=call(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv']),
                cpu_cores=os.cpu_count(), cpu_percent=psutil.cpu_percent(interval=.1),
                ram_available_bytes=psutil.virtual_memory().available,
                load_average=list(os.getloadavg()) if hasattr(os, 'getloadavg') else None)


def dispatch_guard(snapshot, estimate_mib):
    if snapshot['gpu']['returncode'] != 0: raise RuntimeError('nvidia-smi unavailable; no dispatch')
    row = snapshot['gpu']['stdout'].splitlines()[0].split(',')
    total, free = float(row[1]), float(row[3])
    headroom = max(2048., .1 * total)
    if free - headroom < 1.2 * float(estimate_mib):
        raise RuntimeError('RESOURCE_WAIT: insufficient measured GPU memory headroom')


def read_bundle(q):
    from r2.prototypes.seam_slot_c200_v1.inference import ALLOWED_KEYS
    bundle = load_npz(q['inference_npz'])
    if set(bundle) != ALLOWED_KEYS: raise ValueError('Only C/N inference bundle accepted')
    if int(bundle['total_N']) != int(q['N']) or len(bundle['C_faces_model']) != int(q['K']):
        raise ValueError('C bundle N/K mismatch')
    if bundle['C_vertices'].dtype != np.float32: raise ValueError('C vertices must be FP32')
    v = bundle['C_vertices']
    if len({p.tobytes() for p in v}) != len(v):
        raise ValueError('This four-C prototype requires injective C ID to coordinate bytes')
    if set(np.asarray(bundle['C_faces_original']).reshape(-1).tolist()) != set(range(len(v))):
        raise ValueError('C vertex table contains missing or unused source vertices')
    return bundle


def source_from_bundle(q, bundle):
    from meshflow_control.source_winding import capture_source, validate_source
    meta = load_npz(q['metadata_npz'])
    original, model = bundle['C_faces_original'], bundle['C_faces_model']
    if original.dtype != np.int64 or model.dtype != np.int64: raise ValueError('C indices must be int64')
    p = np.asarray([[int(np.flatnonzero(original[i] == v).item()) for v in row] for i, row in enumerate(model)], np.int64)
    supplied = meta.get('source_C_corner_permutations', meta.get('C_source_corner_permutations'))
    if supplied is None or not np.array_equal(p, supplied): raise ValueError('Source C corner permutation differs')
    table = meta['C_source_vertex_ids']
    if table.dtype != np.int64 or table.shape != (len(bundle['C_vertices']),) or len(np.unique(table)) != len(table):
        raise ValueError('Expected compact-C vertex to original source vertex identity table')
    canonical_ids = table[model]
    original_ids = canonical_ids[np.arange(len(original))[:, None], np.argsort(p, axis=1)]
    face_ids = meta.get('source_C_face_ids', meta.get('C_original_source_face_ids'))
    if face_ids is None: raise ValueError('Original C face identities missing')
    source = capture_source(bundle['C_vertices'][original], source_vertex_ids=original_ids,
                            corner_permutation=p, source_face_ids=face_ids)
    validate_source(source, bundle['C_vertices'][model])
    return source


def read_noise(q):
    z = load_npz(q['noise_npz'])
    if set(z) != {'z', 'seed'} or z['seed'].shape != () or int(z['seed']) != int(q['seed']):
        raise ValueError('Frozen noise/seed identity differs')
    a = z['z']
    if a.dtype != np.float32 or a.shape != (int(q['N'])-int(q['K']), 3, 3) or not np.isfinite(a).all():
        raise ValueError('Frozen Gaussian shape/precision differs')
    return a


def checked_receipt(path):
    r = read(path)
    if r.get('status') != 'COMPLETE': raise ValueError('Incomplete output receipt')
    verify_files(r['files'])
    return r


def _load_generator(root, arm):
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.models.native import build_model
    modelroot = REPOSITORY / 'experiments/r2_seam_quality_four_v1' if arm == 'LONG' else root
    path = modelroot / 'inference' / (arm + '_generator.pt')
    receipt = read(path.with_suffix('.json'))
    if sha(path) != receipt['sha256']: raise ValueError('Generator export changed')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (payload['schema'] != ('r2_seam_quality_four_generator_v1' if arm == 'LONG' else 'r2_seam_abc_generator_v1') or payload['arm'] != arm
            or int(payload['local_step']) != 800 or int(payload['seam_total_step']) != 1000 or int(payload['global_step']) != 13000
            or payload.get('EMA') is not False):
        raise ValueError('Only preregistered final live endpoint allowed')
    if any(k in payload for k in ('optimizer', 'rng', 'head_optimizer')):
        raise ValueError('Use inference generator export')
    if state_hash(payload['generator_state']) != payload['generator_state_sha256']:
        raise ValueError('Generator tensor hash mismatch')
    model = build_model(payload['model_config'], readout_mode='none', trainable=False)
    model.load_state_dict(payload['generator_state'], strict=True)
    model.to('cuda').set_trainable(False).eval()
    return model, payload, dict(path=str(path), sha256=receipt['sha256'], generator_state_sha256=payload['generator_state_sha256'])


def _layouts(root, arm, manifest):
    from r2.prototypes.seam_slot_c200_v1.slot_data import inference_layout, boundary_from_C
    conditions = {}
    for q in manifest['requests']:
        if q['condition_id'] in conditions:
            if q['inference_npz'] != conditions[q['condition_id']]['inference_npz']:
                raise ValueError('Four noises must share one C bundle')
        else: conditions[q['condition_id']] = q
    folder = root / 'evaluation' / arm / 'layouts'; folder.mkdir(parents=True, exist_ok=True)
    head = head_identity = None
    if arm in ARMS:
        import torch
        from meshflow_control.checkpoints import state_hash
        from r2.prototypes.seam_slot_c200_v1.occupancy import OccupancyHead
        modelroot = REPOSITORY / 'experiments/r2_seam_quality_four_v1' if arm == 'LONG' else root
        hp = modelroot / 'inference' / (arm + '_occupancy.pt'); hr = read(hp.with_suffix('.json'))
        if sha(hp) != hr['sha256']: raise ValueError('Occupancy export changed')
        payload = torch.load(hp, map_location='cpu', weights_only=True, mmap=True)
        if payload['schema'] != ('r2_seam_quality_four_occupancy_v1' if arm == 'LONG' else 'r2_seam_abc_occupancy_v1') or payload['local_step'] != 800 or payload['seam_total_step'] != 1000 or payload['global_step'] != 13000 or payload['arm'] != arm or payload['threshold_logit'] != 0.:
            raise ValueError('Fixed final occupancy/threshold required')
        if state_hash(payload['head_state']) != payload['head_state_sha256']: raise ValueError('Occupancy tensor hash mismatch')
        head = OccupancyHead(**payload['head_config']).eval().requires_grad_(False)
        head.load_state_dict(payload['head_state'], strict=True)
        head.to('cuda')
        head_identity = dict(path=str(hp), sha256=hr['sha256'], head_state_sha256=payload['head_state_sha256'])
    result = {}
    for ci, (condition, q) in enumerate(sorted(conditions.items())):
        bundle = read_bundle(q); source = source_from_bundle(q, bundle)
        dest = folder / f'C{ci:02d}'
        if (dest / 'receipt.json').exists():
            receipt = checked_receipt(dest / 'receipt.json')
            if receipt['condition_id'] != condition or receipt['head'] != head_identity or receipt['C_bundle_sha256'] != sha(q['inference_npz']):
                raise ValueError('Saved layout identity changed')
            a = load_npz(dest / 'layout.npz')
            layout = inference_layout(**bundle, occupied=a['occupied'])
            for name in ('anchor_ids', 'edge_records', 'seam_slots', 'bulk_slots'):
                if not same(a[name], layout[name]): raise ValueError('Cached layout changed')
            logits = a['occupancy_logits']
        else:
            if dest.exists(): raise RuntimeError('Incomplete occupancy attempt; no implicit replay')
            dest.mkdir()
            write(dest / 'attempt.json', dict(condition_id=condition, head=head_identity, unix=time.time()))
            if head is not None:
                from r2.prototypes.seam_slot_c200_v1.inference import predict_layout
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    layout, logits = predict_layout(head, bundle)
            else:
                edges = boundary_from_C(bundle['C_vertices'], bundle['C_faces_original'])
                layout = inference_layout(**bundle, occupied=np.zeros(len(edges), bool))
                logits = np.zeros(len(edges), np.float32)
            a = {k: np.asarray(v) for k, v in layout.items()}
            a['occupancy_logits'] = logits
            save_npz(dest / 'layout.npz', a); save_npz(dest / 'source_condition.npz', source)
            write(dest / 'receipt.json', dict(status='COMPLETE', condition_id=condition, C_bundle_sha256=sha(q['inference_npz']),
                  head=head_identity, occupancy_forward_calls=int(head is not None), occupancy_device='cuda' if head is not None else 'NA',
                  occupancy_precision='BF16 forward; FP32 parameters' if head is not None else 'NA_NO_HEAD',
                  occupancy_threshold_logit=0., predicted_layout_shared_by_noises=4,
                  source_labels_read=False, model_input_keys=sorted(bundle),
                  files={str(dest / n):sha(dest / n) for n in ('layout.npz', 'source_condition.npz')}))
        result[condition] = (layout, logits, source, dest)
    return result


def raw_arrays(raw, layout, logits, source):
    from r2.prototypes.seam_slot_c200_v1.corner_contract import decode_shared_indices
    n, k = layout['N'], layout['K']
    dec = decode_shared_indices(raw, layout['known_mask'], layout['valid_mask'], layout['anchor_ids'], layout['C_vertices'])
    alias = layout.get('alias_corner_ids', np.full((n, 3), -1, np.int64))
    if np.any(alias >= 0):
        from .shared_tip import validate_alias
        validate_alias(alias, layout['anchor_ids'], layout['known_mask'])
        ff = dec['faces'].reshape(-1)
        selected = alias.reshape(-1) >= 0
        ff[selected] = ff[alias.reshape(-1)[selected]]
        if not np.array_equal(dec['vertices'][dec['faces']], raw):
            raise ValueError('RAW shared-tip index decoder changed coordinates')
    corner_map = np.tile(np.arange(3, dtype=np.int64), (n, 1))
    corner_map[:k] = np.argsort(source['C_source_corner_permutations'], axis=1)
    faces = dec['faces'][np.arange(n)[:, None], corner_map]
    output = dec['vertices'][faces]
    seam = np.zeros(n, bool); seam[layout['seam_slots']] = True
    slot_edge = np.full(n, -1, np.int64)
    slot_edge[k:k+len(layout['occupied'])] = np.arange(len(layout['occupied']))
    return dict(raw=raw, output=output, C=output[:k].copy(), free=output[k:].copy(),
                raw_alias_corner_ids=alias,
                vertices=dec['vertices'], faces=faces, known_faces=np.asarray(k), input_faces=np.asarray(n),
                source_face_ids=np.arange(n, dtype=np.int64), raw_slot_ids=np.arange(n, dtype=np.int64),
                source_corner_ids=corner_map, identity_domain=np.asarray('generated_RAW_slot_not_GT'),
                raw_anchor_ids=layout['anchor_ids'], output_anchor_ids=layout['anchor_ids'][np.arange(n)[:, None], corner_map],
                raw_seam_slot_mask=seam, retained_seam_mask=seam.copy(), raw_slot_edge_id=slot_edge,
                occupied=layout['occupied'], occupancy_logits=logits, edge_records=layout['edge_records'],
                C_vertices=layout['C_vertices'], C_faces_original=layout['C_faces_original'], C_faces_model=layout['C_faces_model'],
                C_original_source_vertex_ids=source['C_original_source_vertex_ids'],
                C_source_corner_permutations=source['C_source_corner_permutations'],
                C_inverse_corner_permutation=np.argsort(source['C_source_corner_permutations'], axis=1))


def rollout(model, layout, z, ledger):
    import torch
    from meshflow_control.models.native import native_execution
    from r2.prototypes.seam_slot_c200_v1.model_adapter import seam_forward
    n, k = layout['N'], layout['K']; device = next(model.parameters()).device
    if z.dtype != np.float32 or z.shape != (n-k, 3, 3) or not np.isfinite(z).all(): raise ValueError('Invalid frozen Gaussian')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()): raise ValueError('Generator must be frozen')
    initial = np.zeros((n, 3, 3), np.float32); initial[k:] = z
    fixed = layout['anchor_ids'] >= 0
    initial[fixed] = layout['C_vertices'][layout['anchor_ids'][fixed]]
    alias = layout.get('alias_corner_ids')
    if alias is not None:
        from .shared_tip import clamp_alias, clamp_alias_tensor
        initial = clamp_alias(initial, alias)
    state = torch.from_numpy(initial.reshape(1, n, 9)).to(device)
    fixed_t = torch.from_numpy(fixed[None]).to(device)
    free_t = ~fixed_t
    condition = torch.from_numpy(initial[None]).to(device)
    known = torch.from_numpy(layout['known_mask'][None]).to(device)
    valid = torch.ones_like(known); y = torch.tensor([n], dtype=torch.int64, device=device)
    path = [initial.copy()]
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick = time.perf_counter()
    with torch.no_grad(), native_execution(model, coordinates=state):
        for step in range(50):
            t = torch.full((1,), step / 50, dtype=torch.float32, device=device)
            event(ledger, 'FORWARD_STARTED', kind='generator', euler_step=step)
            velocity = seam_forward(model, state, t, y, valid, known, fixed_t)
            torch.cuda.synchronize()
            event(ledger, 'FORWARD_RETURNED', kind='generator', euler_step=step)
            if velocity.dtype != torch.bfloat16 or velocity.shape != state.shape or not bool(torch.isfinite(velocity).all()):
                raise RuntimeError('Invalid BF16 generator velocity')
            current = state.reshape(1, n, 3, 3)
            next_state = current.clone()
            next_state[free_t] = current[free_t] + velocity.float().reshape_as(current)[free_t] * (1 / 50)
            next_state[fixed_t] = condition[fixed_t]
            if alias is not None:
                next_state = clamp_alias_tensor(next_state, alias)
            state = next_state.reshape(1, n, 9)
            if not bool(torch.isfinite(state).all()): raise RuntimeError('Nonfinite integration')
            path.append(next_state[0].cpu().numpy().copy())
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
        raise RuntimeError('Sampling unexpectedly consumed RNG')
    trajectory = np.stack(path)
    if alias is not None and any(not same(clamp_alias(p, alias), p) for p in trajectory):
        raise ValueError('Shared tip coordinate contract failed in rollout')
    if any(not same(p[fixed], initial[fixed]) for p in trajectory): raise ValueError('C/seam anchors changed across 51 states')
    return trajectory[-1].copy(), trajectory, dict(nfe=50, seconds=time.perf_counter()-tick,
        anchored_all51_bitwise=True, original_free_scalar_count=layout['free_coordinate_count'],
        free_scalar_count=layout['free_coordinate_count']-(3*int((alias>=0).sum()) if alias is not None else 0),
        referenced_bulk_corners=int((alias>=0).sum()) if alias is not None else 0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_scope='per rollout allocator peaks, resident generator included; not process VRAM',
        precision='BF16 forward; FP32 parameters and Euler integration', noise_policy='saved standard Gaussian; no HYBRID sampling noise')


def load_pointer(root):
    import torch
    from meshflow_control.checkpoints import state_hash
    from .pointer import TipPointerHead
    p = root/'inference/C_SHARED_TIP_pointer.pt'
    receipt = read(p.with_suffix('.json'))
    if sha(p) != receipt['sha256']:
        raise ValueError('Pointer export checksum changed')
    value = torch.load(p, map_location='cpu', weights_only=True)
    if value['local_step'] != 800 or value['arm'] != 'C_SHARED_TIP':
        raise ValueError('Only registered final pointer allowed')
    head = TipPointerHead().eval().requires_grad_(False)
    head.load_state_dict(value['pointer_state'], strict=True)
    if state_hash(head.state_dict()) != value['pointer_state_sha256']:
        raise ValueError('Pointer state mismatch')
    return head.to('cuda'), dict(path=str(p), sha256=receipt['sha256'], state_sha256=value['pointer_state_sha256'])


def run(root, arm):
    if arm not in ARMS: raise ValueError('Unknown arm')
    root, reg, manifest = load_registered(root, require_execution_freeze=True)
    armroot = root / 'evaluation' / arm; armroot.mkdir(parents=True, exist_ok=True)
    complete = armroot / 'raw_complete.json'
    if complete.exists(): return checked_receipt(complete)
    pending = []
    for q in manifest['requests']:
        dest = armroot / q['request_id'] / 'raw'
        if (dest / 'receipt.json').exists(): checked_receipt(dest / 'receipt.json')
        elif dest.exists(): raise RuntimeError('Unresolved rollout; no automatic retry: ' + str(dest))
        else:
            read_noise(q)
            source_from_bundle(q, read_bundle(q))
            pending.append(q)
    if not pending: raise RuntimeError('Missing aggregate with no pending work; explicit finalize required')
    snap = resources(); stage = armroot / ('generation_attempt_' + str(time.time_ns()))
    stage.mkdir(); write(stage / 'resources_before.json', snap)
    dispatch_guard(snap, reg.get('resources', {}).get('generator_estimate_mib', 2600))
    from meshflow_control import runtime
    runtime.configure_stable_runtime()
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.geo_io import write_indexed_ply
    model, payload, identity = _load_generator(root, arm)
    pointer, pointer_identity = load_pointer(root) if arm == 'C_SHARED_TIP' else (None, None)
    write(stage / 'model_identity.json', identity); write(stage / 'resources_resident.json', resources())
    count = 0; started = time.perf_counter()
    try:
        layouts = _layouts(root, arm, manifest)
        for q in pending:
            layout, logits, source, lp = layouts[q['condition_id']]
            dest = armroot / q['request_id'] / 'raw'; dest.mkdir(parents=True, exist_ok=False)
            write(dest / 'attempt.json', dict(request=q, arm=arm, checkpoint=identity, unix=time.time(), budget_forwards=50))
            snap = resources(); write(dest / 'resources_before.json', snap)
            z = read_noise(q)
            if pointer is not None:
                from .shared_tip import inference_aliases
                layout = dict(layout)
                initial = np.zeros((layout['N'], 3, 3), np.float32)
                initial[layout['K']:] = z
                fixed = layout['anchor_ids'] >= 0
                initial[fixed] = layout['C_vertices'][layout['anchor_ids'][fixed]]
                slot_edge_ids = np.full(layout['N'], -1, np.int64)
                slot_edge_ids[layout['seam_slots']] = np.asarray(layout['seam_slots']) - layout['K']
                layout['slot_edge_ids'] = slot_edge_ids
                event(dest/'calls.jsonl', 'POINTER_FORWARD_STARTED', budget=1)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    alias, features, pointer_logits, pointer_facts = inference_aliases(pointer, layout, initial)
                event(dest/'calls.jsonl', 'POINTER_FORWARD_RETURNED')
                layout['alias_corner_ids'] = alias
                save_npz(dest/'pointer.npz', dict(alias_corner_ids=alias, logits=pointer_logits, **features))
                write(dest/'pointer_audit.json', dict(identity=pointer_identity, source_labels_read=False,
                      pointer_forward_calls=1, facts=pointer_facts))
            raw, trajectory, audit = rollout(model, layout, z, dest / 'calls.jsonl')
            audit['noise_policy'] = 'saved standard Gaussian; C arm applies predicted shared-tip references; no local mean/scale change'
            arrays = raw_arrays(raw, layout, logits, source)
            save_npz(dest / 'raw.npz', arrays); save_npz(dest / 'trajectory.npz', dict(state=trajectory))
            save_npz(dest / 'source_condition.npz', source)
            write_indexed_ply(dest / 'raw.ply', arrays['vertices'], arrays['faces'])
            write(dest / 'audit.json', audit)
            files = {str(p): sha(p) for p in dest.iterdir() if p.is_file()}
            write(dest / 'receipt.json', dict(status='COMPLETE', schema=VERSION, arm=arm, request_id=q['request_id'],
                condition_id=q['condition_id'], checkpoint=identity, input_bundle_sha256=sha(q['inference_npz']),
                noise_sha256=sha(q['noise_npz']), layout_receipt=str(lp/'receipt.json'),
                layout_receipt_sha256=sha(lp/'receipt.json'), new_rollouts=1, generator_forwards=50,
                raw_path=str(dest/'raw.npz'), raw_sha256=sha(dest/'raw.npz'),
                D0_forwards=0, pointer_forwards=int(pointer is not None), source_labels_read=False, audit=audit, files=files))
            count += 1
            print(json.dumps(dict(stage='RAW', arm=arm, request_id=q['request_id'], completed=count, seconds=audit['seconds'])), flush=True)
        if state_hash(model.state_dict()) != payload['generator_state_sha256']: raise RuntimeError('Frozen generator changed')
        receipts = [armroot/q['request_id']/'raw/receipt.json' for q in manifest['requests']]
        for p in receipts: checked_receipt(p)
        write(stage / 'resources_finished.json', resources())
        rows = [dict(q, arm=arm, label=arm, raw_path=str(armroot/q['request_id']/'raw/raw.npz'),
                     raw_sha256=sha(armroot/q['request_id']/'raw/raw.npz')) for q in manifest['requests']]
        result = dict(status='COMPLETE', arm=arm, outputs=len(manifest['requests']), new_rollouts=count, new_generator_forwards=50*count, rows=rows,
                      wall_seconds=time.perf_counter()-started, files={str(p):sha(p) for p in receipts})
        write(complete, result)
        return result
    except BaseException as error:
        write(stage / 'failure.json', dict(status='STOPPED', error=repr(error), traceback=traceback.format_exc(),
              completed_requests=count, no_automatic_retry=True))
        raise
    finally:
        del model, pointer; torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--root', default='experiments/r2_seam_abc_v1')
    a = p.parse_args(); print(json.dumps(clean(run(a.root, a.arm)), indent=2))


if __name__ == '__main__': main()
