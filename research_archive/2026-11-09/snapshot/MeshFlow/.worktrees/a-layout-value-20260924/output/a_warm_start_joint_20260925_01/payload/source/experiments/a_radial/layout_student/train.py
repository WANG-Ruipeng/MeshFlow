"""Task-aligned training of the unchanged 29,907-parameter DomainVelocity.

Only the training-only label archive is opened here. The conditional input is
exactly the original 256 proxy points; teacher endpoints are supervision only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from ..legacy.networks import DomainVelocity, point_features


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(json.dumps([key, str(t.dtype), list(t.shape)], separators=(',', ':')).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def spherical_bridge(u0: torch.Tensor, u1: torch.Tensor,
                     t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized chord bridge and its exact derivative, not SLERP.

    Inputs: B x N x 3 endpoints and one time per parent (B or B x 1).
    Velocity is (I - u u^T) (u1-u0) / ||(1-t)u0+t u1||.
    """
    if u0.shape != u1.shape or u0.ndim != 3 or u0.shape[-1] != 3:
        raise ValueError('Bridge endpoints must share B x N x 3 shape')
    if t.numel() != u0.shape[0]:
        raise ValueError('One bridge time is required per parent')
    tt = t.reshape(-1, 1, 1)
    d = u1 - u0
    x = u0 + tt * d
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    if torch.any(norm < 1e-8):
        raise ValueError('Bridge passes too close to zero (antipodal endpoints)')
    u = x / norm
    velocity = (d - (d * u).sum(-1, keepdim=True) * u) / norm
    return u, velocity


def discrete_bridge(u0: torch.Tensor, u1: torch.Tensor, k: torch.Tensor,
                    steps: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deployable tangent targets for one normalize(u + velocity / 32) step.

    With u and q the consecutive normalized chord points, u + v / steps
    equals q / dot(u,q). This removes the continuous-Euler discretization gap
    before applying the original deployment guard. It does not imply that the
    chord trajectory itself preserves all quality constraints.
    """
    if k.numel() != u0.shape[0] or torch.is_floating_point(k):
        raise ValueError('One integer bridge step is required per parent')
    if steps < 1 or torch.any(k < 0) or torch.any(k >= steps):
        raise ValueError('Bridge step outside the fixed deployment interval')
    t = k.to(dtype=u0.dtype).reshape(-1) / steps
    u, _ = spherical_bridge(u0, u1, t)
    q, _ = spherical_bridge(u0, u1, t + 1.0 / steps)
    dot = (u * q).sum(-1, keepdim=True)
    if torch.any(dot <= 0):
        raise ValueError('Consecutive bridge endpoints are not in the same hemisphere')
    velocity = steps * (q / dot - u)
    # Preserve an exactly stationary teacher instead of normalization roundoff.
    stationary = (u0 == u1).all(dim=-1, keepdim=True)
    velocity = torch.where(stationary, torch.zeros_like(velocity), velocity)
    return u, velocity, q, t


class BalancedSampler:
    """Each parent appears once per shuffled epoch, with checkpointable state."""
    def __init__(self, count: int, seed: int):
        if count < 1:
            raise ValueError('Empty training set')
        self.count = int(count)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(count).tolist()
        self.cursor = 0
        self.completed_epochs = 0
        self.exposures = [0] * count

    def take(self, batch_size: int) -> list[int]:
        out = []
        for _ in range(batch_size):
            if self.cursor == self.count:
                self.completed_epochs += 1
                self.order = self.rng.permutation(self.count).tolist()
                self.cursor = 0
            idx = int(self.order[self.cursor])
            self.cursor += 1
            self.exposures[idx] += 1
            out.append(idx)
        return out

    def state_dict(self) -> dict[str, Any]:
        return {'count': self.count, 'rng': self.rng.bit_generator.state,
                'order': self.order.copy(), 'cursor': self.cursor,
                'completed_epochs': self.completed_epochs,
                'exposures': self.exposures.copy()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state['count'] != self.count:
            raise ValueError('Sampler parent count changed')
        if sorted(state['order']) != list(range(self.count)):
            raise ValueError('Invalid saved sampler permutation')
        self.rng.bit_generator.state = state['rng']
        self.order = list(state['order'])
        self.cursor = int(state['cursor'])
        self.completed_epochs = int(state['completed_epochs'])
        self.exposures = list(state['exposures'])


def rng_state() -> dict[str, Any]:
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if not torch.cuda.is_available():
            raise ValueError('Checkpoint contains CUDA RNG but CUDA is unavailable')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def configure_determinism(seed: int) -> None:
    # Set before the first CUDA BLAS operation. Do not silently permit TF32.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
        raise ValueError('Unsupported CUBLAS_WORKSPACE_CONFIG for deterministic training')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def _load_training_data(run: Path, protocol: dict[str, Any]):
    label_path = Path(protocol['paths']['training_labels'])
    if not label_path.is_absolute():
        label_path = run / label_path
    with np.load(label_path, allow_pickle=False) as a:
        uids = [str(x) for x in a['train_uids'].tolist()]
        u0 = np.asarray(a['directions0'], np.float64)
        u1 = np.asarray(a['directions1'], np.float64)
        proxies = np.asarray(a['proxy_points'], np.float64)
        cached_features = np.asarray(a['point_features']) if 'point_features' in a else None
    if uids != protocol['train_uids'] or len(uids) != len(set(uids)):
        raise ValueError('Training UID order/uniqueness disagrees with protocol')
    if set(uids) & set(protocol['internal_check_uids']):
        raise ValueError('Internal-check teacher leakage into training archive')
    n = len(uids)
    if u0.shape != (n, 258, 3) or u1.shape != u0.shape:
        raise ValueError('Teacher layout must have original F512 / V258 shape')
    if proxies.shape != (n, 256, 3):
        raise ValueError('Original proxy256 input contract violated')
    if not all(np.isfinite(x).all() for x in (u0, u1, proxies)):
        raise ValueError('Non-finite training data')
    if not (np.allclose(np.linalg.norm(u0, axis=-1), 1, atol=1e-7, rtol=0)
            and np.allclose(np.linalg.norm(u1, axis=-1), 1, atol=1e-7, rtol=0)):
        raise ValueError('Teacher directions must be unit vectors')
    if np.any(np.linalg.norm(proxies, axis=-1) <= 0):
        raise ValueError('Proxy contains zero-radius points')
    features = point_features(proxies)
    if cached_features is not None and not np.array_equal(features, cached_features):
        raise ValueError('Cached features differ from original point_features(proxy)')
    label_hash = file_sha256(label_path)
    expected = protocol.get('hashes', {}).get('training_labels_sha256')
    if expected is not None and expected != label_hash:
        raise ValueError('Training archive changed after protocol freeze')
    return uids, u0, u1, features, label_hash


def train_run(run: Path, seed: int, total_updates: int = 3000,
              resume: Path | None = None, *, device: str = 'cuda',
              checkpoint_every: int = 1000, stop_after: int | None = None,
              output_dir: Path | None = None) -> dict[str, Any]:
    """Train to the predeclared final update; intermediate files are resume only.

    `stop_after` emulates interruption while retaining the same final budget and
    scheduler. Resuming always writes a new `output_dir`, never overwrites files.
    """
    run = Path(run).resolve()
    protocol_path = run / 'RUN_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8-sig'))
    training = protocol.get('training', {})
    if 'updates' in training and training['updates'] != total_updates:
        raise ValueError('Requested total updates differ from frozen protocol')
    if 'seeds' in training and seed not in training['seeds']:
        raise ValueError('Seed not in frozen training protocol')
    for key, value in [('batch_size', 4), ('learning_rate', 3e-4), ('weight_decay', 1e-4)]:
        if key in training and training[key] != value:
            raise ValueError(f'Unexpected frozen training setting: {key}')
    if total_updates < 1 or checkpoint_every < 1:
        raise ValueError('Positive update/checkpoint budgets required')
    end_update = total_updates if stop_after is None else int(stop_after)
    if not 0 < end_update <= total_updates:
        raise ValueError('stop_after must be in [1, total_updates]')
    dev = torch.device(device)
    if dev.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU correctness tests and CUDA training are supported')
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    configure_determinism(seed)
    uids, u0, u1, features, label_hash = _load_training_data(run, protocol)
    initial = Path(protocol['paths']['original_a_checkpoint'])
    if not initial.is_absolute():
        initial = run / initial
    initial_hash = file_sha256(initial)
    expected_initial = protocol.get('hashes', {}).get('original_a_sha256')
    if expected_initial is not None and initial_hash != expected_initial:
        raise ValueError('Original A checkpoint changed')
    model = DomainVelocity()
    initial_state = torch.load(initial, map_location='cpu', weights_only=True)
    model.load_state_dict(initial_state, strict=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count != 29907:
        raise ValueError('Original DomainVelocity architecture changed')
    model = model.to(device=dev, dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4,
                                  foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    sampler = BalancedSampler(len(uids), seed)
    data = [torch.as_tensor(u0, dtype=torch.float64, device=dev),
            torch.as_tensor(u1, dtype=torch.float64, device=dev),
            torch.as_tensor(features, dtype=torch.float32, device=dev)]
    protocol_hash = file_sha256(protocol_path)
    config = {'seed': seed, 'total_updates': total_updates, 'batch_size': 4,
              'optimizer': 'AdamW', 'learning_rate': 3e-4, 'weight_decay': 1e-4,
              'scheduler': 'constant', 'dtype': 'float32', 'device': str(dev),
              'tf32': False, 'deterministic_algorithms': True,
              'bridge': 'discrete_normalized_linear_sphere_bridge',
              'bridge_steps': 32, 'bridge_computation_dtype': 'float64',
              'training_time': 'uniform_integer_0_to_31_div32',
              'objective': 'mean_squared_tangent_velocity',
              'checkpoint_selection': 'predeclared_final_update_only',
              'parameter_count': parameter_count, 'proxy_points': 256,
              'query_directions': 258, 'network_extra_inputs': [],
              'train_uids': uids, 'original_a_checkpoint_sha256': initial_hash,
              'training_labels_sha256': label_hash, 'protocol_sha256': protocol_hash,
              'source_sha256': {'train.py': file_sha256(Path(__file__)),
                  'legacy/networks.py': file_sha256(Path(__file__).parent.parent / 'legacy/networks.py')}}
    start_update = 0
    trace = []
    accumulated_seconds = 0.0
    if resume is not None:
        state = torch.load(Path(resume), map_location='cpu', weights_only=False)
        if state['config'] != config:
            raise ValueError('Resume config/data/protocol/source identity mismatch')
        if model_sha256(state['model']) != state['model_sha256']:
            raise ValueError('Saved model tensor hash mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        sampler.load_state_dict(state['sampler'])
        start_update = int(state['update'])
        trace = list(state['trace'])
        accumulated_seconds = float(state['elapsed_training_seconds'])
        restore_rng(state['rng'])
    if not start_update < end_update:
        raise ValueError('Resume must execute at least one new update')
    out = Path(output_dir) if output_dir is not None else run / 'training' / f'seed_{seed}'
    if not out.is_absolute():
        out = run / out
    out = out.resolve()
    if run not in out.parents:
        raise ValueError('Training output must remain below its new run directory')
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / 'TRAIN_CONFIG.json', config)
    _write_json(out / 'START.json', {'seed': seed, 'start_update': start_update,
        'end_update_this_invocation': end_update, 'final_budget': total_updates,
        'resume': str(Path(resume).resolve()) if resume is not None else None,
        'resume_sha256': file_sha256(Path(resume)) if resume is not None else None})
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    segment_checkpoint_seconds = 0.0

    def save_checkpoint(update: int, final: bool) -> Path:
        nonlocal segment_checkpoint_seconds
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)
        before = time.perf_counter()
        state = {'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'rng': rng_state(), 'sampler': sampler.state_dict(), 'update': update,
                 'config': config, 'trace': trace,
                 'elapsed_training_seconds': accumulated_seconds + before - started,
                 'is_predeclared_final': final,
                 'initial_checkpoint_sha256': initial_hash,
                 'training_labels_sha256': label_hash, 'protocol_sha256': protocol_hash}
        state['model_sha256'] = model_sha256(state['model'])
        path = out / ('final.pt' if final else f'step_{update:05d}.pt')
        with path.open('xb') as f:
            torch.save(state, f)
        segment_checkpoint_seconds += time.perf_counter() - before
        return path

    last_checkpoint = None
    try:
        for update in range(start_update + 1, end_update + 1):
            indices = sampler.take(4)
            index = torch.tensor(indices, dtype=torch.long, device=dev)
            k = torch.randint(0, 32, (4,), device=dev)
            start_batch, end_batch, proxy_features = [x[index] for x in data]
            positions, target_velocity, _, t = discrete_bridge(start_batch, end_batch, k)
            positions = positions.float()
            target_velocity = target_velocity.float()
            t = t.float()
            prediction = model(positions, t, proxy_features)
            loss = torch.mean((prediction - target_velocity).square())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at update {update}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'Non-finite/missing gradient at update {update}')
            optimizer.step()
            scheduler.step()
            trace.append({'update': update, 'loss': float(loss.detach().cpu()),
                          'uids': [uids[i] for i in indices],
                          'times': t.detach().cpu().tolist(), 'lr': optimizer.param_groups[0]['lr']})
            if update % checkpoint_every == 0 or update == end_update:
                last_checkpoint = save_checkpoint(update, update == total_updates)
                print(json.dumps({'seed': seed, 'update': update,
                    'loss': trace[-1]['loss'], 'checkpoint': str(last_checkpoint)}, ensure_ascii=False), flush=True)
    except Exception as exc:
        _write_json(out / 'FAILED.json', {'error_type': type(exc).__name__, 'error': str(exc),
            'completed_updates': trace[-1]['update'] if trace else start_update})
        raise
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)
    elapsed = accumulated_seconds + time.perf_counter() - started
    report = {'status': 'COMPLETE' if end_update == total_updates else 'INTERRUPTED_FOR_RESUME_TEST',
              'seed': seed, 'update': end_update, 'final_budget': total_updates,
              'checkpoint': str(last_checkpoint), 'checkpoint_sha256': file_sha256(last_checkpoint),
              'model_sha256': model_sha256(model.state_dict()),
              'parameter_count': parameter_count, 'training_labels_sha256': label_hash,
              'protocol_sha256': protocol_hash, 'initial_checkpoint_sha256': initial_hash,
              'elapsed_training_seconds': elapsed,
              'checkpoint_write_seconds_this_invocation': segment_checkpoint_seconds,
              'peak_gpu_allocated_bytes': int(torch.cuda.max_memory_allocated(dev)) if dev.type == 'cuda' else None,
              'peak_gpu_reserved_bytes': int(torch.cuda.max_memory_reserved(dev)) if dev.type == 'cuda' else None,
              'parent_exposures': dict(zip(uids, sampler.exposures)),
              'first_loss': trace[0]['loss'], 'last_loss': trace[-1]['loss'],
              'last_100_loss_mean': float(np.mean([x['loss'] for x in trace[-100:]])),
              'selection_uses_internal_check': False,
              'checkpoint_selection': 'predeclared_final_update_only',
              'internal_check_teacher_files_opened': 0,
              'training_only_archive_uids_verified': True}
    _write_json(out / 'TRAIN_TRACE.json', trace)
    _write_json(out / 'TRAIN_REPORT.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--updates', default=3000, type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint-every', default=1000, type=int)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    report = train_run(args.run, args.seed, args.updates, args.resume, device=args.device,
                       checkpoint_every=args.checkpoint_every, stop_after=args.stop_after,
                       output_dir=args.output_dir)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
