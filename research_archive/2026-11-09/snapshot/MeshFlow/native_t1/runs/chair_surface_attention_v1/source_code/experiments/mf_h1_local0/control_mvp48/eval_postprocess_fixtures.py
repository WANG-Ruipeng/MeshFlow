"""Small synthetic CPU checks for unattended transaction and raw-audit boundaries."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from . import eval_postprocess as post
from .eval_common import EVALUATION, REGISTRATION, write_json, read_json
from .eval_scoring import METRIC_SPECS


def run():
    N, K = 5, 2
    C = np.arange(K * 9, dtype=np.float32).reshape(K, 3, 3)
    z = np.ones((1, N, 9), dtype=np.float32)
    known = np.arange(N) < K
    bank = dict(C=C, z=z, known_mask=known, valid_mask=np.ones(N, np.bool_))
    initial = z[0].copy()
    initial[:K] = C.reshape(K, 9)
    states = np.repeat(initial[None], 51, axis=0)
    raw = dict(raw=states[-1].copy(), states=states, C=C.copy(), known_mask=known,
               valid_mask=bank['valid_mask'], accepted_times=np.array([0.] + [(k+1)/50 for k in range(50)]))
    row = dict(N=N, K=K, nfe=50, status='PASS')
    assert post._raw_audit(row, bank, raw)['known_all51_bitwise']
    changed = {k: v.copy() for k, v in raw.items()}
    changed['states'][1, 0, 0] = -0.0
    assert post._raw_audit(row, bank, changed)['status'] == 'INVALID_OUTPUT'
    changed['states'][0, 0, 0] = -0.0
    try:
        post._raw_audit(row, bank, changed)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed initial C bits were accepted')
    partial = {k: v.copy() for k, v in raw.items()}
    partial.update(states=partial['states'][:4], accepted_times=partial['accepted_times'][:4])
    assert post._raw_audit(dict(row, nfe=3, status='FAIL'), bank, partial)['saved_state_count'] == 4
    with patch.object(post, '_input', return_value=bank):
        failed_without_raw = post._score_one(dict(row, status='FAIL', reason='Exact worker failure'), {}, {}, {})
    assert failed_without_raw['failure_reason'] == 'Exact worker failure'
    expected = [dict(request_id=f'p{p}_c{c}_n{n}_{m}', parent_id=f'parent{p}', condition_id=f'c{c}',
                     noise_id=n, model=m) for p in range(8) for c in range(2) for n in range(2) for m in post.MODELS]
    teachers = [dict(parent_id=f'validation{p}', condition_id='c0', noise_id=0, teacher_id=f't{p}',
                     path=f'teacher{p}.npz', sha256='synthetic') for p in range(8)]
    fixture = Path(tempfile.mkdtemp(prefix='eval_postprocess_fixture_', dir=EVALUATION))
    registration = fixture / 'registration'
    registration.mkdir()
    state_path = fixture / 'scoring_run.json'
    gpu_path = fixture / 'gpu_run.json'
    protocol = dict(status='FROZEN', scorer_identity={}, metric_specs=METRIC_SPECS,
                    data_identity=dict(data_ready_sha256='synthetic'))
    for name, value in (('eval_protocol.json', protocol), ('eval_requests.json', expected),
                        ('eval_control.json', dict(status='NOT_RUN', reason='Synthetic fixture'))):
        write_json(registration / name, value)
    write_json(fixture / 'TRAIN_READY.json', dict(status='SYNTHETIC'))
    ledger = dict(teacher_inputs=teachers, OT_attempts=0, OT_returns=0)
    first = dict(expected[0], status='PASS', nfe=50)
    write_json(gpu_path, dict(status='RUNNING', rows=[first], teacher_rows=[]))
    score_calls = []
    def score_ok(row, expected_row, preparation, counts):
        score_calls.append(row['request_id'])
        counts['synthetic_score_calls'] = counts.get('synthetic_score_calls', 0) + 1
        return dict(row, metrics={name: 0. for name in METRIC_SPECS})
    def score_error(row, expected_row, preparation, counts):
        score_calls.append(row['request_id'])
        counts['synthetic_score_calls'] += 1
        raise RuntimeError('Synthetic scoring failure; no retry permitted')
    with patch.multiple(post, RUN=fixture, EVALUATION=fixture, REGISTRATION=registration,
                        STATE=state_path, GPU=gpu_path, SCORES=fixture / 'scores'):
        with patch.object(post, '_contract', return_value=(protocol, ledger, expected)), patch.object(post, '_score_one', side_effect=score_ok):
            first_state = post.score_available()
            first_sha = first_state['scores'][first['request_id']]['sha256']
            second_state = post.score_available()
            assert len(score_calls) == 1 and second_state['scores'][first['request_id']]['sha256'] == first_sha
        second = dict(expected[1], status='PASS', nfe=50)
        write_json(gpu_path, dict(status='STOPPED', rows=[first, second], teacher_rows=[]))
        with patch.object(post, '_contract', return_value=(protocol, ledger, expected)), patch.object(post, '_score_one', side_effect=score_error):
            failed = post.score_available()
            assert failed['status'] == 'STOPPED' and failed['scoring_counts']['synthetic_score_calls'] == 2
            assert post.score_available()['status'] == 'STOPPED' and len(score_calls) == 2
        with patch.object(post, '_contract', return_value=(protocol, ledger, expected)), patch.object(
                post, '_fixed_figures', return_value=dict(status='SYNTHETIC_NOT_RENDERED')):
            final = post.finalize()
        result = read_json(fixture / 'results.json')
        assert len(result['per_output']) == 96 and len(result['per_parent']) == 8
        assert len(result['teacher']['rows']) == 24 and result['teacher']['attempted'] == 0
        assert result['formal_attempted'] == 2 and result['formal_passed'] == 1
        assert result['formal_failed'] == 1 and result['formal_pending'] == 94
        assert result['formal_nfe']['recorded_actual_sum'] == 100
        assert final['status'] == 'INCOMPLETE_OR_FAILED_MATRIX'
        assert (fixture / 'teacher.csv').read_bytes() == (fixture / 'teacher_validation.csv').read_bytes()
        assert all(result['conclusions'][f'q{i}'] for i in range(1, 5))
    result = dict(status='PASS', fixture_scope='Synthetic raw checks and mocked CPU-score transactions only',
                  raw_bitwise_C_check=True, initial_state_tamper_rejected=True,
                  partial_rollout_retained=True, duplicate_score_prevented=True,
                  interrupted_score_no_retry=True, final_tables=[96, 8, 24],
                  failed_and_pending_denominators_preserved=True, teacher_alias_synchronized=True, worker_failure_reason_preserved=True, synthetic_score_calls=len(score_calls),
                  model_forwards=0, formal_generation=0, formal_geometry_score_calls=0,
                  OT_calls=0, fixture_directory=str(fixture))
    write_json(REGISTRATION / 'eval_postprocess_fixtures.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False))
