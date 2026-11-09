"""Independent CPU-only audit of the completed fixed training and recovery state."""
from __future__ import annotations

from collections import Counter
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import time

import numpy as np
from .common import OUT, read, resolve, sha, write


def jsonl(path):
    return [json.loads(line) for line in resolve(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def run():
    destination = OUT / 'audits/training_exact_budget.json'
    if destination.exists():
        result = read(destination)
        for path, digest in result['evidence_hashes'].items():
            if sha(path) != digest:
                raise ValueError('Audited training evidence changed: ' + path)
        return result
    start = time.perf_counter()
    root = OUT / 'training'
    done = read(root / 'complete.json')
    assert done['status'] == 'COMPLETE' and done['counts'] == dict(forwards=2000, backwards=2000, updates=500)
    manifest = read(OUT / 'data/manifest.json')
    ledger = jsonl(root / 'ledger.jsonl')
    losses = jsonl(root / 'losses.jsonl')
    wanted = [(step, micro) for step in range(1, 501) for micro in range(4)]
    counts = Counter(row['event'] for row in ledger)
    for event in ('FORWARD_STARTED', 'FORWARD_RETURNED', 'BACKWARD_STARTED', 'BACKWARD_RETURNED'):
        selected = [row for row in ledger if row['event'] == event]
        assert [(row['update'], row['micro']) for row in selected] == wanted
        if event.endswith('RETURNED'):
            assert [row['cumulative'] for row in selected] == list(range(1, 2001))
    for event in ('OPTIMIZER_STEP_STARTED', 'OPTIMIZER_STEP_RETURNED'):
        selected = [row for row in ledger if row['event'] == event]
        assert [row['update'] for row in selected] == list(range(1, 501))
        if event.endswith('RETURNED'):
            assert [row['cumulative'] for row in selected] == list(range(1, 501))
    assert [row['update'] for row in losses] == list(range(1, 501))
    exposure = Counter()
    kinds = Counter()
    observed_indices = []
    for step, row in enumerate(losses):
        assert len(row['samples']) == 4
        assert row['cumulative_forwards'] == row['cumulative_backwards'] == 4 * (step + 1)
        for micro, observed in enumerate(row['samples']):
            index = manifest['stream'][step][micro]
            target = manifest['train_rows'][index]
            assert observed['row_index'] == index and observed['id'] == target['row_id']
            assert observed['parent'] == target['parent_id'] and observed['kind'] == target['kind']
            assert observed['sigma'] == target['sigma']
            assert np.isfinite(observed['mse_denoiser']) and observed['mse_denoiser'] >= 0
            exposure[target['parent_id']] += 1
            kinds[target['kind']] += 1
            observed_indices.append(index)
    assert len(observed_indices) == len(set(observed_indices)) == 2000
    assert dict(exposure) == manifest['parent_exposure'] and set(exposure.values()) == {125}
    resume_positions = [i for i, row in enumerate(ledger) if row['event'] == 'BOUNDARY_RESUME_LOADED']
    assert len(resume_positions) == 1
    i = resume_positions[0]
    assert ledger[i - 1]['event'] == 'OPTIMIZER_STEP_RETURNED' and ledger[i - 1]['update'] == 100
    assert ledger[i + 1]['event'] == 'FORWARD_STARTED' and ledger[i + 1]['update'] == 101
    assert ledger[i]['replayed_updates'] == 0
    assert [row['update'] for row in ledger if row['event'] == 'CHECKPOINT_SAVED'] == [200, 300, 400, 500]
    restored = read(root / 'resume_boundary_loaded.json')
    assert all(restored[key] for key in ('all_optimizer_steps_100', 'python_rng_exact', 'numpy_rng_exact', 'cpu_rng_exact', 'cuda_rng_exact'))
    # CPU tensor deserialization is not a model/feature forward or architecture
    # instantiation. Validate the retained checkpoint can recover exact final state.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    full_path = root / 'checkpoints/step0500.pt'
    inference_path = resolve(done['inference_weights']['path'])
    full = torch.load(full_path, map_location='cpu', mmap=True, weights_only=True)
    inference = torch.load(inference_path, map_location='cpu', mmap=True, weights_only=True)
    assert full['step'] == inference['step'] == 500
    assert full['counts'] == dict(forwards=2000, backwards=2000, updates=500)
    assert full['contract'] == inference['contract'] == read(root / 'attempt.json')['contract']
    assert len(full['model']) == len(inference['model']) == 296
    assert all(value.dtype == torch.float32 and torch.equal(value, inference['model'][name]) for name, value in full['model'].items())
    assert len(full['optimizer']['state']) == 296
    assert all(int(state['step']) == 500 for state in full['optimizer']['state'].values())
    assert set(full['rng']) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
    assert full['rng']['python'] == full['initial_rng']['python']
    assert full['rng']['numpy'] == full['initial_rng']['numpy']
    assert torch.equal(full['rng']['torch_cpu'], full['initial_rng']['torch_cpu'])
    assert len(full['rng']['torch_cuda']) == len(full['initial_rng']['torch_cuda'])
    assert all(torch.equal(a,b) for a,b in zip(full['rng']['torch_cuda'], full['initial_rng']['torch_cuda']))
    assert not torch.cuda.is_initialized()
    full_receipt = read(root / 'checkpoints/step0500.json')
    assert sha(full_path) == full_receipt['sha256']
    assert sha(inference_path) == done['inference_weights']['sha256']
    result = dict(status='PASS', logical_full_model_forwards=2000, backwards=2000, optimizer_updates=500,
        checkpoint_activation_recomputation='Part of normal gradient-checkpointed backward; not additional full-model examples',
        all_started_calls_returned=True, updates_unique_1_to_500=True, actual_stream_exact=True,
        training_parents=len(exposure), actual_parent_exposure=dict(exposure), instance_kinds=dict(kinds),
        boundary_recovery=dict(from_step=100, next_step=101, replayed_updates=0, replayed_forwards=0, replayed_backwards=0,
            model_optimizer_rng_exact=True, original_failure_retained=True),
        final_model_tensors=296, final_optimizer_states=296, all_optimizer_steps_500=True,
        inference_weights_exactly_equal_full_checkpoint=True, original_runtime_rng_preserved=True,
        CPU_checkpoint_load=True, model_instantiated=False, CUDA_initialized=False, extra_model_forwards=0,
        peak_training_allocated_bytes=max(row['peak_allocated_bytes'] for row in losses),
        peak_training_reserved_bytes=max(row['peak_reserved_bytes'] for row in losses),
        allocator_peak_scope='Across both actual training segments; not total process or total GPU memory',
        summed_recorded_update_seconds=sum(row['seconds'] for row in losses), seconds=time.perf_counter()-start,
        evidence_hashes={str(path): sha(path) for path in [OUT / 'data/manifest.json', root / 'complete.json',
            root / 'ledger.jsonl', root / 'losses.jsonl', root / 'attempt.json', root / 'failure.json',
            root / 'resume_boundary_attempt.json', root / 'resume_boundary_loaded.json', OUT / 'engineering_amendment.json',
            root / 'checkpoints/step0500.json']})
    # Large retained artifacts were just fully hashed; store those digests once.
    result['evidence_hashes'][str(full_path)] = full_receipt['sha256']
    result['evidence_hashes'][str(inference_path)] = done['inference_weights']['sha256']
    write(destination, result)
    return result


if __name__ == '__main__':
    result = run()
    print('TRAINING_EXACT_BUDGET', result['status'], result['logical_full_model_forwards'], result['backwards'], result['optimizer_updates'])
