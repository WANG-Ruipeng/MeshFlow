"""Independent CPU-only audit of saved generation artifacts; never load a model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import time
import numpy as np
from r2.pilots.single_step_denoiser_distillation_v1 import common as C


def main():
    started = time.monotonic()
    manifest_path = C.OUT / 'manifest/data.json'
    completion_path = C.OUT / 'generation/completion.json'
    source_before = {str(p): C.sha(p) for p in (manifest_path, completion_path, C.CODE/'generate.py')}
    m, done = C.read(manifest_path), C.read(completion_path)
    assert done['status'] == 'COMPLETE'
    assert (done['rollouts'], done['forwards'], done['generator_training']) == (249, 12450, 0)
    plans = m['generation_plan']
    receipts = {r['sample_id']: r for r in done['rows']}
    assert len(plans) == len(receipts) == 249
    assert {r['sample_id'] for r in plans} == set(receipts)
    rows, input_hashes = [], {'train': set(), 'internal_val': set()}
    gaussian_hashes, source_files = set(), {}
    counts = {'train': 0, 'internal_val': 0}
    for plan in plans:
        sid, split = plan['sample_id'], plan['split']
        assert split in counts
        receipt = receipts[sid]
        assert receipt == C.read(C.OUT/'generation/receipts'/f'{sid}.json')
        n, k = plan['N'], plan['K']
        assert (receipt['N'], receipt['K'], receipt['forwards']) == (n, k, 50)
        assert receipt['generator_file_sha256'] == C.GENERATOR_EXPECTED
        assert receipt['generator_state_sha256'] == C.GENERATOR_STATE_EXPECTED
        assert receipt['C_byte_exact'] and receipt['all_51_C_byte_exact'] and receipt['no_rng_consumed']
        assert receipt['audit']['seed'] == plan['Gaussian_seed'] == plan['noise_seed']
        assert receipt['audit']['nfe'] == 50
        assert receipt['audit']['known_state_checks'] == 51
        assert receipt['generation_input_path'] == plan['generation_input_path']
        assert receipt['raw_path'] == plan['generated_raw_path']
        assert receipt['input_path'] == plan['input_path']
        ghash = C.sha(plan['generation_input_path'])
        rhash = C.sha(receipt['raw_path'])
        ihash = C.sha(receipt['input_path'])
        assert ghash == plan['generation_input_sha256'] == receipt['generation_input_sha256']
        assert rhash == receipt['raw_sha256'] and ihash == receipt['input_sha256']
        source = plan['source_npz']
        if source not in source_files:
            source_files[source] = C.sha(source)
        assert source_files[source] == plan['source_npz_sha256']
        with np.load(source, allow_pickle=False) as orig, np.load(plan['generation_input_path'], allow_pickle=False) as gen, np.load(receipt['raw_path'], allow_pickle=False) as raw, np.load(receipt['input_path'], allow_pickle=False) as inp:
            assert set(gen.files) == set(plan['generation_allowed_keys']) == {'C', 'N', 'K', 'known_mask', 'z'}
            original_c = orig['C']
            assert original_c.dtype == np.float32 and original_c.shape == (k, 3, 3)
            assert C.ah(original_c) == plan['source_C_sha256']
            for arrays in (gen, raw, inp):
                assert int(arrays['N']) == n and int(arrays['K']) == k
                assert arrays['C'].dtype == np.float32
                assert arrays['C'].shape == original_c.shape and arrays['C'].tobytes() == original_c.tobytes()
            known = np.arange(n) < k
            assert gen['known_mask'].dtype == np.bool_ and inp['known_mask'].dtype == np.bool_
            assert np.array_equal(gen['known_mask'], known) and np.array_equal(inp['known_mask'], known)
            z = gen['z']
            assert z.shape == (1, n, 9) and z.dtype == np.float32
            assert np.isfinite(z).all() and np.count_nonzero(z[:, :k]) == 0
            zhash = C.ah(z)
            assert zhash == plan['Gaussian_sha256'] == receipt['actual_Gaussian_sha256']
            assert raw['z'].dtype == np.float32 and raw['z'].shape == z.shape and raw['z'].tobytes() == z.tobytes()
            gaussian_hashes.add(zhash)
            mesh, trajectory = raw['raw'], raw['trajectory']
            assert mesh.dtype == trajectory.dtype == np.float32
            assert mesh.shape == (n, 3, 3) and trajectory.shape == (51, n, 3, 3)
            assert np.isfinite(mesh).all() and np.isfinite(trajectory).all()
            assert mesh[:k].tobytes() == original_c.tobytes()
            assert all(state[:k].tobytes() == original_c.tobytes() for state in trajectory)
            assert trajectory[-1].tobytes() == mesh.tobytes()
            assert trajectory[0, k:].tobytes() == z.reshape(n, 3, 3)[k:].tobytes()
            model, denoise = inp['input_model'], inp['input_d']
            assert model.dtype == denoise.dtype == np.float32
            assert model.shape == denoise.shape == (n, 3, 3)
            assert model.tobytes() == mesh.tobytes()
            expected = np.multiply(mesh, np.float32(C.SCALE), dtype=np.float32)
            assert np.isfinite(denoise).all() and expected.tobytes() == denoise.tobytes()
            dhash = C.ah(denoise)
            input_hashes[split].add(dhash)
            counts[split] += 1
            rows.append(dict(sample_id=sid, split=split, parent_id=plan['parent_id'], condition_id=plan['condition_id'], N=n, K=k,
                Gaussian_seed=plan['Gaussian_seed'], Gaussian_sha256=zhash,
                generation_input_sha256=ghash, raw_npz_sha256=rhash, input_npz_sha256=ihash,
                raw_array_sha256=C.ah(mesh), input_d_array_sha256=dhash, source_C_array_sha256=C.ah(original_c),
                trajectory_array_sha256=C.ah(trajectory), trajectory_C_states_verified=51,
                raw_C_byte_exact=True, input_model_RAW_byte_exact=True, FP32_scale_byte_exact=True, status='PASS'))
    assert counts == {'train': 224, 'internal_val': 25}
    assert len(input_hashes['train']) == 224 and len(input_hashes['internal_val']) == 25
    overlap = input_hashes['train'] & input_hashes['internal_val']
    assert not overlap and len(gaussian_hashes) == 249
    assert {r['parent_id'] for r in plans if r['split'] == 'train'} == {r['parent_id'] for r in plans if r['split'] == 'internal_val'}
    assert len({r['parent_id'] for r in plans}) == 25
    assert 'torch' not in sys.modules
    source_after = {p: C.sha(p) for p in source_before}
    assert source_before == source_after
    csv_path = C.OUT/'audit/generated_input_hashes.csv'
    C.csvwrite(csv_path, rows)
    report = dict(status='PASS', audit_kind='Independent CPU inspection of actual saved arrays',
        sample_counts=counts, unique_input_d_hash_counts={k:len(v) for k,v in input_hashes.items()},
        train_validation_input_d_hash_overlap=sorted(overlap), independent_Gaussian_hashes=249,
        train_parent_count=25, validation_parent_count=25, validation_scope='Held-out noise on training parents; not unseen-shape validation',
        raw_C_byte_exact=249, trajectory_C_states_byte_exact=249*51, initial_Gaussian_free_byte_exact=249,
        terminal_trajectory_RAW_byte_exact=249, FP32_scale_saved_input_d_byte_exact=249,
        saved_input_d_note='Authoritative saved GPU FP32 inputs were read and matched against independently computed CPU FP32 multiplication byte-for-byte. No fresh GPU verification was executed.',
        protected_source_manifest_before=source_before, protected_source_manifest_after=source_after,
        source_npz_files_verified=len(source_files), source_npz_hashes=source_files,
        audit_source=str(Path(__file__).resolve()), audit_source_sha256=C.sha(__file__),
        per_sample_csv=str(csv_path), per_sample_csv_sha256=C.sha(csv_path),
        generated_or_modified_scientific_inputs=0, new_Gaussian_draws=0, model_loads=0, model_forwards=0, GPU_calls=0,
        torch_imported=False, wall_seconds=time.monotonic()-started)
    C.write(C.OUT/'audit/generated_input_independent_audit.json', report)
    print({k:report[k] for k in ('status','sample_counts','unique_input_d_hash_counts','train_validation_input_d_hash_overlap','trajectory_C_states_byte_exact','FP32_scale_saved_input_d_byte_exact','wall_seconds')})


if __name__ == '__main__':
    main()
