"""Bounded Bench D0 adaptation: 500 updates, then frozen one-pass evaluation.

This module owns CUDA only. It never imports geometry/scoring modules or performs
generator sampling. A partial attempt is never silently replayed. Intermediate
full checkpoints retain the exact optimizer and RNG states for explicit recovery.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_backend_shape_preservation_v1'
VERSION = 'bench_backend_shape_preservation_runtime_v1'
SCALE = 0.95 * 0.3762
UPDATES = 500
BATCH = 4
SEED = 2026102301
OFFICIAL_SHA = '67765c728c503b61c21624c05997ea026c9de486d1529052fafa134be3336b14'


def resolve(value):
    value = str(value).replace('\\', '/')
    if os.name != 'nt' and re.match(r'^[A-Za-z]:/', value):
        value = '/mnt/' + value[0].lower() + value[2:]
    if os.name == 'nt' and re.match(r'^/mnt/[a-z]/', value):
        value = value[5].upper() + ':/' + value[7:]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with resolve(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, payload, replace=False):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if read(path) != clean(payload):
            raise RuntimeError('Refusing changed output: ' + str(path))
        return
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(clean(payload), handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def event(ledger_path, kind, **payload):
    ledger_path = resolve(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(clean(dict(utc=utc(), event=kind, **payload)), allow_nan=False) + '\n')
        handle.flush()


def arrays(path):
    with np.load(resolve(path), allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def save_npz(path, payload):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('Refusing overwrite: ' + str(path))
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resources():
    def call(command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except Exception as error:
            return dict(error=repr(error))
    return dict(utc=utc(), gpu=call(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits']),
        processes=call(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv']),
        cpu_logical=os.cpu_count(), loadavg=list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
        memory=Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else None,
        cpu_work=call(['ps', '-eo', 'pid,pcpu,pmem,args', '--sort=-pcpu']))


def configure():
    current = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if current not in (None, ':4096:8'):
        raise RuntimeError('Unexpected CUBLAS_WORKSPACE_CONFIG')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # Resolve the maintained, source-audited wrapper ahead of installed or sibling
    # packages. Importing a geometry wrapper here could mutate CUDA visibility.
    for path in (ROOT, ROOT / 'r2/src'):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    import torch
    from meshflow_control import denoise_model as D
    if Path(D.__file__).resolve() != (ROOT / 'r2/src/meshflow_control/denoise_model.py').resolve():
        raise RuntimeError('Imported denoiser wrapper from an unexpected package')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    D.configure_runtime()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 is required')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return torch, D


def gpu_capacity(required_worker_gib=None):
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20, check=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError('This pilot requires exactly one explicitly identified GPU')
    total, free = [float(value.strip()) for value in lines[0].split(',')]
    headroom = max(2048.0, 0.1 * total)
    value = dict(total_mib=total, free_mib=free, required_headroom_mib=headroom,
        measured_whole_GPU_not_allocator=True, headroom_holds=free >= headroom)
    if required_worker_gib is not None:
        value.update(estimated_worker_mib=required_worker_gib * 1024,
            dispatch_requirement_mib=required_worker_gib * 1024 + headroom)
        if free < value['dispatch_requirement_mib']:
            raise RuntimeError('Insufficient observed GPU headroom before dispatch: ' + json.dumps(value))
    return value


def row_id(row, fallback):
    return row.get('row_id', row.get('id', row.get('request_id', str(fallback))))


def contract(manifest_path):
    manifest_path = resolve(manifest_path)
    manifest = read(manifest_path)
    rows = manifest['train_rows']
    stream = manifest['stream']
    if len(stream) != UPDATES or any(len(row) != BATCH for row in stream):
        raise ValueError('Exactly 500 updates x 4 registered instances required')
    if any(not isinstance(i, int) or i < 0 or i >= len(rows) for line in stream for i in line):
        raise ValueError('Bad training stream index')
    for row in rows + manifest['validation_rows'] + manifest.get('raw_rows', []):
        path = resolve(row['path'])
        expected = row.get('sha256')
        if expected is None or sha(path) != expected:
            raise ValueError('Input hash mismatch/missing hash: ' + str(path))
    return manifest, dict(manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path),
        runtime_sha256=sha(__file__), schema=VERSION,
        updates=UPDATES, batch=BATCH, microbatch=1, optimizer='fresh AdamW', lr=1e-5,
        betas=[0.9, 0.95], weight_decay=0.0, gradient_clip=1.0,
        objective='per-instance free-coordinate masked MSE in denoiser units; mean over 4 instances',
        scale=SCALE, face_count_dropout=0.0, training_seed=SEED,
        model='official Bench EMA full decoder; no new parameters/role/features',
        precision='FP32 parameters/residual/loss; BF16 forward; math SDPA',
        gradient_checkpointing=True, official_sha256=OFFICIAL_SHA)


def sample(row):
    data = arrays(row['path'])
    x, target, mask = data['x_model'], data['target_model'], data['loss_mask']
    known = int(data['known_faces'])
    if (x.dtype != np.float32 or target.dtype != np.float32 or mask.dtype != np.bool_ or
        x.shape != target.shape or x.ndim != 3 or x.shape[1:] != (3, 3) or
        mask.shape != x.shape[:2] or not 1 <= len(x) <= 800 or not 0 < known < len(x) or
        mask[:known].any() or not mask[known:].any() or not np.isfinite(x).all() or
        not np.isfinite(target).all() or not np.array_equal(x[:known], target[:known])):
        raise ValueError('Invalid fixed training sample: ' + str(row['path']))
    return data


def configure_train_model(model):
    model.train().requires_grad_(True)
    model.decoder.y_embedder.dropout_prob = 0.0
    for layer in model.decoder.layers:
        layer.gradient_checkpointing = True
    if len(model.decoder.layers) != 24:
        raise ValueError('Unexpected decoder depth')


def training_prediction(model, x_model, torch, D):
    """One differentiable decoder call; no encoder/posterior/no_grad helper."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    n = len(x_model)
    x = torch.from_numpy(np.ascontiguousarray(x_model).reshape(1, n, 9)).to('cuda') * SCALE
    condition = torch.tensor([n], device='cuda', dtype=torch.long)
    with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
        z = model.pe_nerf(x.reshape(1, n * 3, 3))
        residual = model.decode(z, condition, mask=None)
    # Exactly the official residual addition, before inverse scaling. The loss
    # deliberately stays in denoiser units, unlike the saved model-unit proposal.
    return (x.float() + residual.float()).reshape(n, 3, 3)


def rng_state(torch):
    npstate = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(name=npstate[0], keys=npstate[1].tolist(),
        pos=npstate[2], has_gauss=npstate[3], cached_gaussian=npstate[4]),
        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all())


def save_torch(path, value, torch):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('Refusing checkpoint overwrite: ' + str(path))
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def verified_receipt(path):
    result = read(path)
    if result['status'] != 'COMPLETE':
        raise ValueError('Receipt not complete')
    for filename, digest in result.get('files', {}).items():
        if sha(filename) != digest:
            raise ValueError('Completed output changed: ' + filename)
    return result


def boundary_preflight(root, manifest_path):
    """Accept only the recorded post-save logger error, never a failed update."""
    root = resolve(root)
    dest = root / 'training'
    if (dest / 'resume_boundary_attempt.json').exists():
        raise RuntimeError('The single authorized boundary recovery has already been attempted')
    failure = read(dest / 'failure.json')
    expected_error = 'TypeError("event() got multiple values for argument \'path\'")'
    if failure['error'] != expected_error or failure['counts'] != dict(
        forwards_returned=400, backwards_returned=400, updates_returned=100):
        raise ValueError('Failure is not the authorized post-step100 logger exception')
    records = [json.loads(line) for line in (dest / 'ledger.jsonl').read_text().splitlines() if line.strip()]
    counts = {name: sum(r['event'] == name for r in records) for name in
        ('FORWARD_STARTED', 'FORWARD_RETURNED', 'BACKWARD_STARTED', 'BACKWARD_RETURNED',
         'OPTIMIZER_STEP_STARTED', 'OPTIMIZER_STEP_RETURNED')}
    if counts != dict(FORWARD_STARTED=400, FORWARD_RETURNED=400, BACKWARD_STARTED=400,
        BACKWARD_RETURNED=400, OPTIMIZER_STEP_STARTED=100, OPTIMIZER_STEP_RETURNED=100):
        raise ValueError('Ledger contains missing, extra, or unfinished scientific calls')
    if any(r.get('update', 0) > 100 for r in records) or records[-1]['event'] != 'OPTIMIZER_STEP_RETURNED':
        raise ValueError('Scientific work happened after the boundary checkpoint')
    for name in ('FORWARD_STARTED', 'FORWARD_RETURNED', 'BACKWARD_STARTED', 'BACKWARD_RETURNED'):
        if [(r['update'], r['micro']) for r in records if r['event'] == name] != [
            (step, micro) for step in range(1, 101) for micro in range(BATCH)]:
            raise ValueError('Per-micro ledger does not match the exact first100 stream')
    losses = [json.loads(line) for line in (dest / 'losses.jsonl').read_text().splitlines() if line.strip()]
    if [row['update'] for row in losses] != list(range(1, 101)):
        raise ValueError('Loss log does not end exactly at completed update100')
    checkpoints = sorted((dest / 'checkpoints').glob('*.pt'))
    expected_path = dest / 'checkpoints/step0100.pt'
    if checkpoints != [expected_path] or list((dest / 'checkpoints').glob('*.partial')):
        raise ValueError('Unexpected checkpoint after the permitted boundary')
    saved = read(dest / 'checkpoints/step0100.json')
    if resolve(saved['path']) != expected_path or sha(expected_path) != saved['sha256']:
        raise ValueError('Boundary checkpoint hash/path mismatch')
    original = read(dest / 'attempt.json')['contract']
    archive = root / 'code_before_boundary_fix.py'
    if sha(archive) != original['runtime_sha256'] or sha(manifest_path) != original['manifest_sha256']:
        raise ValueError('Original code archive or frozen training manifest changed')
    return dict(status='PASS', checkpoint=saved, original_contract=original,
        old_code_sha256=sha(archive), new_code_sha256=sha(__file__),
        preserved_failure_sha256=sha(dest / 'failure.json'), original_ledger_sha256=sha(dest / 'ledger.jsonl'),
        original_losses_sha256=sha(dest / 'losses.jsonl'), counts=counts,
        next_update=101, terminal_update=500, replayed_updates=0, replayed_forwards=0,
        replayed_backwards=0, original_segment_seconds=failure['seconds'])


def restore_rng(state, torch):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value['name'], np.asarray(value['keys'], dtype=np.uint32),
        value['pos'], value['has_gauss'], value['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])


def train(root, manifest_path, resume_boundary=False):
    root = resolve(root)
    dest = root / 'training'
    complete = dest / 'complete.json'
    if complete.exists():
        return verified_receipt(complete)
    if dest.exists() and not resume_boundary:
        raise RuntimeError('Incomplete training attempt exists. No automatic replay. Inspect ledger/checkpoint before explicit recovery.')
    manifest, identity = contract(manifest_path)
    execution_code_sha = identity['runtime_sha256']
    resume = boundary_preflight(root, manifest_path) if resume_boundary else None
    if resume:
        previous = resume['original_contract']
        if {k:v for k,v in identity.items() if k != 'runtime_sha256'} != {k:v for k,v in previous.items() if k != 'runtime_sha256'}:
            raise ValueError('Boundary recovery cannot change scientific settings')
        identity = previous
    capacity = gpu_capacity(required_worker_gib=8.0)
    if resume:
        write(dest / 'resume_boundary_attempt.json', dict(utc=utc(), **resume, no_automatic_replay=True))
    else:
        dest.mkdir(parents=True)
        write(dest / 'attempt.json', dict(utc=utc(), contract=identity, no_automatic_replay=True))
    suffix = '_resume_boundary' if resume else ''
    write(dest / ('resources_before'+suffix+'.json'), resources())
    write(dest / ('dispatch_guard'+suffix+'.json'), capacity)
    ledger = dest / 'ledger.jsonl'
    calls = backward = updates = 0
    model = optimizer = None
    started = time.perf_counter()
    try:
        torch, D = configure()
        saved_checkpoint = None
        if resume:
            saved_checkpoint = torch.load(resolve(resume['checkpoint']['path']), map_location='cpu', mmap=True, weights_only=True)
            if saved_checkpoint['step'] != 100 or saved_checkpoint['counts'] != dict(forwards=400, backwards=400, updates=100) or saved_checkpoint['contract'] != identity:
                raise ValueError('Loaded boundary state/counts/contract differ')
            with torch.random.fork_rng(devices=[]):
                model = D.AutoencoderKL(**dict(D.INIT_KWARGS, fixed_std=0.05))
            model.load_state_dict(saved_checkpoint['model'], strict=True)
            for module in model.modules():
                if isinstance(module, D.SelfAttention):
                    module.mixed_precision = 'fp32'
            model = model.to('cuda')
            original = saved_checkpoint['official_identity']
            model._meshflow_denoiser_identity = original
        else:
            from r2.pilots.bench_control_transfer_v1.postprocess import load_bench_denoiser
            model, original = load_bench_denoiser()
        configure_train_model(model)
        write(dest / 'model_identity.json', original)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95),
            weight_decay=0.0, foreach=False, fused=False)
        if resume:
            optimizer.load_state_dict(saved_checkpoint['optimizer'])
        write(dest / ('resources_model_loaded'+suffix+'.json'), resources())
        cache = {i: sample(row) for i, row in enumerate(manifest['train_rows'])}
        if resume:
            restore_rng(saved_checkpoint['rng'], torch)
            initial_rng = saved_checkpoint['initial_rng']
            calls, backward, updates = 400, 400, 100
            current_rng = rng_state(torch)
            loaded_check = dict(status='PASS', restored_step=100,
                next_update=101, optimizer_state_entries=len(optimizer.state),
                all_optimizer_steps_100=all(int(s['step']) == 100 for s in optimizer.state.values()),
                python_rng_exact=current_rng['python'] == saved_checkpoint['rng']['python'],
                numpy_rng_exact=current_rng['numpy'] == saved_checkpoint['rng']['numpy'],
                cpu_rng_exact=torch.equal(torch.get_rng_state(), saved_checkpoint['rng']['torch_cpu']),
                cuda_rng_exact=len(current_rng['torch_cuda']) == len(saved_checkpoint['rng']['torch_cuda']) and all(
                    torch.equal(a,b) for a,b in zip(current_rng['torch_cuda'],saved_checkpoint['rng']['torch_cuda'])),
                checkpoint_sha256=resume['checkpoint']['sha256'], replayed_updates=0)
            if not all(loaded_check[key] for key in ('all_optimizer_steps_100', 'python_rng_exact',
                'numpy_rng_exact', 'cpu_rng_exact', 'cuda_rng_exact')):
                raise ValueError('Boundary optimizer or RNG restoration did not match exactly')
            if any(group['lr'] != 1e-5 or group['betas'] != (0.9, 0.95) or group['weight_decay'] != 0.0 for group in optimizer.param_groups):
                raise ValueError('Restored AdamW scientific hyperparameters differ')
            write(dest / 'resume_boundary_loaded.json', loaded_check)
            event(ledger, 'BOUNDARY_RESUME_LOADED', next_update=101, restored_updates=100,
                checkpoint_path=resume['checkpoint']['path'], replayed_updates=0)
            del saved_checkpoint
        else:
            initial_rng = rng_state(torch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        train_start = time.perf_counter()
        checkpoints = [resume['checkpoint']] if resume else []
        for step, indices in enumerate(manifest['stream'][updates:], updates + 1):
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            records = []
            for micro, index in enumerate(indices):
                row, data = manifest['train_rows'][index], cache[index]
                event(ledger, 'FORWARD_STARTED', update=step, micro=micro, row_index=index)
                prediction = training_prediction(model, data['x_model'], torch, D)
                target = torch.from_numpy(data['target_model']).to('cuda') * SCALE
                mask = torch.from_numpy(data['loss_mask']).to('cuda')
                loss = (prediction[mask] - target[mask]).square().mean()
                torch.cuda.synchronize()
                calls += 1
                event(ledger, 'FORWARD_RETURNED', update=step, micro=micro, cumulative=calls)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite masked free-coordinate MSE')
                value = float(loss.detach().cpu())
                event(ledger, 'BACKWARD_STARTED', update=step, micro=micro)
                (loss / BATCH).backward()
                torch.cuda.synchronize()
                backward += 1
                event(ledger, 'BACKWARD_RETURNED', update=step, micro=micro, cumulative=backward)
                records.append(dict(row_index=index, id=row_id(row, index), parent=row.get('parent', row.get('parent_id')),
                    kind=row['kind'], sigma=row['sigma'], N=len(data['x_model']), K=int(data['known_faces']),
                    supervised_corners=int(data['loss_mask'].sum()), mse_denoiser=value, mse_model=value / SCALE ** 2))
                del prediction, target, mask, loss
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            event(ledger, 'OPTIMIZER_STEP_STARTED', update=step)
            optimizer.step()
            torch.cuda.synchronize()
            updates += 1
            event(ledger, 'OPTIMIZER_STEP_RETURNED', update=step, cumulative=updates)
            D._assert_runtime()
            record = dict(update=step, cumulative_forwards=calls, cumulative_backwards=backward,
                mse_denoiser=float(np.mean([r['mse_denoiser'] for r in records])),
                mse_model=float(np.mean([r['mse_model'] for r in records])),
                gradient_norm_before_clip=float(gradient_norm.detach().cpu()),
                seconds=time.perf_counter() - tick, samples=records,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_scope='Current training phase including resident model/gradients/AdamW; allocator only, excludes context and other processes')
            event(dest / 'losses.jsonl', 'UPDATE', **record)
            if step <= 5 or (resume and step <= 105) or step % 50 == 0:
                write(dest / f'resources_update_{step:04d}.json', resources())
                write(dest / f'whole_gpu_capacity_update_{step:04d}.json', gpu_capacity())
                print(json.dumps(dict(stage='TRAIN', **{k: record[k] for k in ['update', 'mse_denoiser', 'seconds', 'peak_allocated_bytes', 'peak_reserved_bytes']})), flush=True)
            if step % 100 == 0:
                state = dict(schema=VERSION, step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    rng=rng_state(torch), initial_rng=initial_rng, contract=identity, official_identity=original,
                    counts=dict(forwards=calls, backwards=backward, updates=updates),
                    model_training=True, face_count_dropout=0.0)
                saved = save_torch(dest / 'checkpoints' / f'step{step:04d}.pt', state, torch)
                checkpoints.append(saved)
                write(dest / 'checkpoints' / f'step{step:04d}.json', saved)
                event(ledger, 'CHECKPOINT_SAVED', update=step, **saved)
                del state
        if (calls, backward, updates) != (UPDATES * BATCH, UPDATES * BATCH, UPDATES):
            raise ValueError('Training counts do not match fixed budget')
        if not torch.equal(initial_rng['torch_cpu'], torch.get_rng_state()) or any(
            not torch.equal(a, b) for a, b in zip(initial_rng['torch_cuda'], torch.cuda.get_rng_state_all())):
            raise RuntimeError('Unexpected training RNG consumption with fixed inputs and dropout=0')
        model.eval().requires_grad_(False)
        final = save_torch(dest / 'candidate_denoiser.pt', dict(schema=VERSION, model=model.state_dict(),
            step=UPDATES, contract=identity, official_identity=original), torch)
        write(dest / 'resources_finished.json', resources())
        if sha(manifest_path) != identity['manifest_sha256'] or sha(__file__) != execution_code_sha:
            raise ValueError('Frozen manifest/runtime changed during training')
        result = dict(status='COMPLETE', schema=VERSION, counts=dict(forwards=calls, backwards=backward, updates=updates),
            training_seconds=time.perf_counter() - train_start, total_seconds=time.perf_counter() - started + (resume['original_segment_seconds'] if resume else 0),
            checkpoints=checkpoints, inference_weights=final, contract=identity,
            execution_code_sha256=execution_code_sha, boundary_recovery=resume,
            training_seconds_scope='Current successful continuation segment' if resume else 'Single training segment',
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            no_extra_probe_forwards=True, no_extra_updates=True, no_automatic_replay=True,
            files={str(resolve(p)): sha(p) for p in [manifest_path, __file__, dest / 'losses.jsonl', dest / 'ledger.jsonl', final['path']]})
        result['files'].update({p['path']: p['sha256'] for p in checkpoints})
        write(complete, result)
        return result
    except BaseException as error:
        write(dest / ('resume_boundary_failure.json' if resume else 'failure.json'), dict(status='BLOCKED', error=repr(error), traceback=traceback.format_exc(),
            counts=dict(forwards_returned=calls, backwards_returned=backward, updates_returned=updates),
            no_automatic_retry=True, seconds=time.perf_counter() - started))
        raise
    finally:
        if model is not None:
            del model, optimizer
            torch.cuda.empty_cache()


def load_candidate(root, torch, D):
    done = verified_receipt(root / 'training/complete.json')
    path = resolve(done['inference_weights']['path'])
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    kwargs = dict(D.INIT_KWARGS, fixed_std=0.05)
    with torch.random.fork_rng(devices=[]):
        model = D.AutoencoderKL(**kwargs)
    model.load_state_dict(checkpoint['model'], strict=True)
    if sum(p.numel() for p in model.parameters()) != D.EXPECTED_PARAMETERS:
        raise ValueError('Candidate architecture changed')
    model.eval().requires_grad_(False)
    model.decoder.y_embedder.dropout_prob = 0.0
    for module in model.modules():
        if isinstance(module, D.SelfAttention):
            module.mixed_precision = 'fp32'
    identity = dict(checkpoint['official_identity'], candidate_checkpoint_path=str(path),
        candidate_checkpoint_sha256=done['inference_weights']['sha256'], candidate_step=UPDATES,
        adaptation_contract=checkpoint['contract'])
    model._meshflow_denoiser_identity = identity
    return model.to('cuda'), identity


def evaluate(root, manifest_path, arm):
    if arm not in ('OFFICIAL_SOURCE', 'CANDIDATE_SOURCE', 'CANDIDATE_RAW'):
        raise ValueError('Unregistered evaluation arm')
    root = resolve(root)
    manifest, frozen = contract(manifest_path)
    rows = manifest['validation_rows'] if arm.endswith('SOURCE') else manifest['raw_rows']
    if len(rows) != 30:
        raise ValueError('Registered evaluation requires exactly 30 samples')
    dest = root / 'evaluation' / arm
    complete = dest / 'complete.json'
    if complete.exists():
        return verified_receipt(complete)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'failure.json').exists():
        raise RuntimeError('Failed evaluation is not automatically retried')
    pending = []
    for i, row in enumerate(rows):
        rid = row_id(row, i)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', rid):
            raise ValueError('Unsafe/invalid row ID')
        folder = dest / rid
        if (folder / 'receipt.json').exists():
            verified_receipt(folder / 'receipt.json')
        elif folder.exists():
            raise RuntimeError('Incomplete forward cannot be silently replayed: ' + str(folder))
        else:
            pending.append((row, folder))
    if not pending:
        raise RuntimeError('All rows completed but aggregate missing; audit before finalization')
    model = None
    calls = 0
    start = time.perf_counter()
    write(dest / 'resources_before.json', resources())
    try:
        write(dest / 'dispatch_guard.json', gpu_capacity(required_worker_gib=3.0))
        torch, D = configure()
        if arm == 'OFFICIAL_SOURCE':
            from r2.pilots.bench_control_transfer_v1.postprocess import load_bench_denoiser
            model, model_identity = load_bench_denoiser()
        else:
            model, model_identity = load_candidate(root, torch, D)
        write(dest / 'model_identity.json', model_identity)
        write(dest / 'resources_model_loaded.json', resources())
        for row, folder in pending:
            folder.mkdir()
            data = arrays(row['path'])
            x = data['x_model'] if arm.endswith('SOURCE') else data['raw']
            write(folder / 'attempt.json', dict(utc=utc(), row=row, arm=arm, maximum_forward_calls=1))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            event(dest / 'ledger.jsonl', 'FORWARD_STARTED', arm=arm, row=row)
            proposal = D.predict_proposal(model, x)
            torch.cuda.synchronize()
            calls += 1
            event(dest / 'ledger.jsonl', 'FORWARD_RETURNED', arm=arm, row_id=folder.name, cumulative=calls)
            save_npz(folder / 'proposal.npz', dict(official_unconstrained_proposal=proposal,
                unconstrained_proposal=proposal, known_faces=data['known_faces']))
            write(folder / 'receipt.json', dict(status='COMPLETE', arm=arm, id=folder.name,
                D0_forwards=1, generator_forwards=0, new_noise_draws=0, seconds=time.perf_counter()-tick,
                input_sha256=row['sha256'], output_sha256=sha(folder / 'proposal.npz'),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                peak_scope='One evaluation forward including resident model; allocator only',
                files={str(folder / 'proposal.npz'): sha(folder / 'proposal.npz')}))
            if calls == 1 or calls % 10 == 0:
                write(dest / f'resources_output_{calls:04d}.json', resources())
            print(json.dumps(dict(stage=arm, completed=calls, row_id=folder.name)), flush=True)
        if sha(manifest_path) != frozen['manifest_sha256']:
            raise ValueError('Frozen input manifest changed during inference')
        paths = [dest / row_id(row, i) / 'receipt.json' for i, row in enumerate(rows)]
        write(dest / 'resources_finished.json', resources())
        result = dict(status='COMPLETE', arm=arm, outputs=len(paths), new_forwards=calls,
            generator_forwards=0, new_noise_draws=0, seconds=time.perf_counter()-start,
            manifest_sha256=frozen['manifest_sha256'], runtime_sha256=sha(__file__),
            files={str(path): sha(path) for path in paths})
        write(complete, result)
        return result
    except BaseException as error:
        write(dest / 'failure.json', dict(status='BLOCKED', error=repr(error), traceback=traceback.format_exc(),
            returned_forward_calls=calls, no_automatic_retry=True))
        raise
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate', 'verify'])
    parser.add_argument('--root', default=str(OUT))
    parser.add_argument('--manifest')
    parser.add_argument('--resume-boundary', action='store_true', help='Explicit one-time zero-replay recovery from the verified post-step100 logger failure only')
    parser.add_argument('--arm', choices=['OFFICIAL_SOURCE', 'CANDIDATE_SOURCE', 'CANDIDATE_RAW'])
    args = parser.parse_args()
    root = resolve(args.root)
    manifest_path = resolve(args.manifest) if args.manifest else root / 'data/manifest.json'
    if args.command == 'train':
        result = train(root, manifest_path, resume_boundary=args.resume_boundary)
    elif args.command == 'evaluate':
        if args.arm is None:
            parser.error('evaluate requires --arm')
        result = evaluate(root, manifest_path, args.arm)
    else:
        path = root / 'training/complete.json' if args.arm is None else root / 'evaluation' / args.arm / 'complete.json'
        result = verified_receipt(path)
    print(json.dumps(clean(dict(status=result['status'], command=args.command, arm=args.arm,
        counts=result.get('counts'), outputs=result.get('outputs'), new_forwards=result.get('new_forwards'),
        total_seconds=result.get('total_seconds', result.get('seconds')),
        inference_weights=result.get('inference_weights'), root=root)), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
