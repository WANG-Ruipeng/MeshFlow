"""Independent CPU validation of registered count/noise/identity and budget."""
from __future__ import annotations
import ast
from collections import Counter
import json
from pathlib import Path
import numpy as np
from legacy import ROOT, OUT, OLD, sha, save, table

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def arr_sha(a):
    import hashlib
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()

def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

def main():
    p = OUT.parent / 'registration.json'
    r, old = read(p), read(OLD / 'registration.json')
    protected = {str(p): sha(p)}
    assert r['status'] == 'FROZEN'
    assert r['version'] == 'R2_DEDUP_N_SENSITIVITY_V1'
    assert r['original_requests'] == old['requests'] and r['queries'] == old['queries']
    assert r['conditions'] == old['conditions']
    assert r['d0_checkpoint'] == old['d0_checkpoint']
    expected_budget = dict(training_updates=0, backward=0, new_rollouts=48,
        new_sampling_forward=2400, new_D0_forward=48, new_quality_scans=96,
        cached_original_N_requests=48, cached_original_N_scores=96, total_score_rows=192)
    assert r['budget'] == expected_budget
    assert r['count_policy']['numerator'] == 2717 and r['count_policy']['denominator'] == 3047
    assert r['count_policy']['one_lower_count_only'] is True
    assert r['inference_contract']['source_GT_rank'] is False
    assert r['inference_contract']['all_48_RAW_before_D0'] is True
    assert r['inference_contract']['Euler_steps'] == 50
    assert r['inference_contract']['same_C_queries'] is True
    originals = {q['request_id']: q for q in old['requests']}
    assert len(originals) == len(r['requests']) == 24
    assert {q['request_id'] for q in r['requests']} == set(originals)
    counts = Counter(q['parent_id'] for q in r['requests'])
    assert len(counts) == 6 and set(counts.values()) == {4}
    rows = []
    for q in r['requests']:
        a = originals[q['request_id']]
        n, k, low = int(a['N']), int(a['K']), int(q['N'])
        # Independent exact integer half-up computation, without a 256 clamp.
        rounded_free = ((n-k)*2717*2 + 3047) // (2*3047)
        expected = max(128, k + rounded_free)
        assert low == expected and 128 <= low <= 256 and k < low < n
        for key in ('parent_id','condition_id','base_condition_id','split','panel','candidate','K',
                    'C_sha256','condition_source','condition_source_sha256','noise_seed','selection_order'):
            assert q[key] == a[key], (q['request_id'], key)
        assert q['original_input_path'] == a['input_path']
        assert q['original_input_sha256'] == a['input_sha256']
        assert q['original_N'] == n
        for path, digest in ((a['input_path'],a['input_sha256']), (q['input_path'],q['input_sha256']),
                             (a['condition_source'],a['condition_source_sha256'])):
            assert sha(path) == digest, path
            protected[path] = digest
        high, lowa = arrays(a['input_path']), arrays(q['input_path'])
        assert set(lowa) == {'C','z','N','K'}
        assert int(lowa['N']) == low and int(lowa['K']) == k
        assert int(high['N']) == n and int(high['K']) == k
        assert same(high['C'], lowa['C']) and arr_sha(lowa['C']) == q['C_sha256']
        assert high['z'].shape == (1,n,9) and lowa['z'].shape == (1,low,9)
        assert same(lowa['z'], high['z'][:,:low])
        assert same(lowa['z'][:,k:], high['z'][:,k:low])
        assert np.all(lowa['z'][:,:k] == 0)
        assert arr_sha(lowa['z']) == q['Gaussian_sha256']
        assert arr_sha(lowa['z'][:,k:]) == q['free_Gaussian_sha256']
        rows.append(dict(request_id=q['request_id'],parent_id=q['parent_id'],split=q['split'],candidate=q['candidate'],
            K=k,N_original=n,N_low=low,free_original=n-k,free_low=low-k,
            N_bin_original=n//20,N_bin_low=low//20,C_byte_exact=True,free_noise_prefix_byte_exact=True,
            source_identity_unchanged=True,input_sha256=q['input_sha256']))
    for parent in counts:
        assert sorted(q['candidate'] for q in r['requests'] if q['parent_id']==parent) == [0,1,2,3]
    assert Counter(q['split'] for q in r['requests']) == {'TRAIN':12,'DEV':12}
    assert len(r['queries']) == 6
    for q in r['queries']:
        assert sha(q['path']) == q['sha256']
        protected[q['path']] = q['sha256']
        assert q['query_count'] == 2048
        aa = arrays(q['path'])
        for key, digest in q['array_sha256'].items():
            assert arr_sha(aa[key]) == digest, (q['path'], key)
    for stem, prior_label in (('ORIGINAL','ORIGINAL_C200'),('DEDUP','DEDUP_C200')):
        prior = read(OLD / 'evaluation/bindings' / (prior_label+'.json'))
        for suffix in ('LOW_N','ORIG_N'):
            b = r['models'][stem+'_'+suffix]
            for key in ('path','sha256','generator_state_sha256','local_step','global_step'):
                assert b[key] == prior[key]
            assert b['local_step'] == 200 and b['global_step'] == 12200
            assert b['source_arm'] == prior_label
        assert len(r['cache'][stem+'_ORIG_N']) == 24
    gatepath = Path(r['gate_path'])
    assert sha(gatepath) == r['gate_sha256']
    protected[str(gatepath)] = sha(gatepath)
    gate = read(gatepath)
    assert gate['version'] == 'pilot_B_gate_v1'
    assert 'no B training' in gate['if_fail']
    assert 'after CPU owner/data/occupancy feasibility freeze' in gate['if_pass']
    code = ROOT / 'r2/pilots/dedup_n_sensitivity_v1'
    codehash = {}
    for name in ('common.py','prepare.py','evaluate.py'):
        f = code / name
        text = f.read_text(encoding='utf8')
        ast.parse(text, filename=str(f))
        codehash[str(f)] = sha(f)
    prepare = (code/'prepare.py').read_text(encoding='utf8')
    assert "a['z'][:,:low].copy()" in prepare
    assert 'np.random' not in prepare and 'torch.randn' not in prepare
    evaluator = (code/'evaluate.py').read_text(encoding='utf8')
    assert "rows={a:[inference(a).read_raw(q,a) for q in requests(a)] for a in TRAIN_LABELS}" in evaluator
    assert 'return rows[label]' in evaluator
    assert "z=inp['z']" in evaluator and 'for q in pending:' in evaluator
    assert "if label not in TRAIN_LABELS:raise ValueError('Baseline cannot generate')" in evaluator
    assert "if label not in TRAIN_LABELS:raise ValueError('Baseline cannot call D0')" in evaluator
    assert "No silent model retries" in r['failures']
    for path, digest in protected.items():
        assert sha(path) == digest
    table('registered_noise_prefix_audit.csv', rows)
    result = dict(status='PASS',scope='Frozen preflight only; execution budget is checked again after output completion',
        registration_sha256=sha(p),protected_small_inputs=protected,checked_code_hashes=codehash,
        audit_source_sha256=sha(__file__),requests=24,parents=6,noises_per_parent=4,source_queries_per_C=2048,
        C_and_source_identity_unchanged=True,byte_exact_old_free_noise_prefix_all=True,new_noise_draws=0,
        fixed_model_bindings_match_prior=True,large_weight_hash_note='Compared registered endpoint identities with prior binding; did not redundantly hash 7.29GB D0 or reload any models.',
        registered_scientific_budget=expected_budget,registered_gate=gate,
        count_interpretation='One preregistered lower count derived from training-pool free-count ratio, not source reconstruction or per-evaluation-parent dedup count. N embedding bins also change for all six parents; not a pure removal of redundant output faces.',
        gate_timing='B_GATE and registration existed and were hash-bound at this preflight; runtime ledger must establish prior-to-first-model-call chronology.',
        review_limits='No model calls, geometric scoring, new views, or semantic validation performed by this preflight.')
    save('preflight.json', result)
    print(json.dumps({'status':'PASS','requests':24,'parents':6,'N_pairs':sorted({(x['parent_id'][:6],x['N_original'],x['N_low']) for x in rows}), 'output':str(OUT/'preflight.json')}))

if __name__ == '__main__':
    main()
