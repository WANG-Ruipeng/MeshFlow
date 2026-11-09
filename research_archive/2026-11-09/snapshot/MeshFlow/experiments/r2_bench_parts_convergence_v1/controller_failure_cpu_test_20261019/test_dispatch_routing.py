"""CPU production Driver routing test; recorded jobs are never dispatched."""
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from r2.pilots.bench_parts_convergence_v1 import run_all as R
from r2.pilots.bench_parts_convergence_v1 import controller as C


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path(__file__).resolve().parent/('dispatch_routing_'+str(time.time_ns()))
    root = out/'fixture';root.mkdir(parents=True)
    files = [Path(R.__file__), Path(C.__file__), Path(R.__file__).with_name('train.py'),
             Path(R.__file__).with_name('resume_train.py')]
    before = {str(p): digest(p) for p in files}
    registration = {'status': 'FROZEN', 'common_updates': 12000,
        'minimum_stage_updates': 3000, 'maximum_stage_updates': 9000,
        'evaluation': {'process_common_steps': [2000, 6000, 10000],
                       'process_paired_steps': [13000, 17000, 19000]}}
    (root/'registration.json').write_text(json.dumps(registration), encoding='utf-8')
    results = []
    with patch.object(R.subprocess, 'Popen', side_effect=AssertionError('No real dispatch allowed')) as popen:
        # Real initializer creates and hashes the production identity including
        # the adapter; it performs no resource query or worker launch.
        driver = R.Driver(root)
        identity = C.read(root/'dispatch/identity.json')
        wrapper = Path(R.__file__).with_name('resume_train.py')
        assert identity['code_hashes'][str(wrapper)] == digest(wrapper)
        cases = [
            ('saved_offgrid_common', ('COMMON',), 6000, 5250, 'READY', 'resume_train', '--resume'),
            ('ordinary_milestone_common', ('COMMON',), 6000, 4000, 'READY', 'train', '--resume'),
            ('new_common', ('COMMON',), 2000, None, 'ABSENT', 'train', None),
            ('new_paired_forks', ('PLAIN', 'SEAM'), 13000, None, 'ABSENT', 'train', '--fork'),
            ('saved_offgrid_seam', ('SEAM',), 15000, 14250, 'INTERRUPTED_SAFE', 'resume_train', '--resume'),
        ]
        for name, arms, target, step, state_name, module, flag in cases:
            counts = {a: 0 for a in arms};recorded = []
            def wait(arm):
                counts[arm] += 1
                if counts[arm] > 1:
                    return {'state': 'READY', 'step': target, 'safe_to_start': True}
                return {'state': state_name, 'step': step, 'safe_to_start': True,
                        'checkpoint': str(root/'training'/arm/'rolling.pt')}
            def wave(jobs): recorded.extend(jobs)
            with patch.object(driver, 'wait_training', side_effect=wait), patch.object(driver, 'wave', side_effect=wave):
                driver.train_target(arms, target)
            assert len(recorded) == len(arms) and all(n == 2 for n in counts.values())
            for arm, job in zip(arms, recorded):
                cmd = job['command'];index = cmd.index('-m')
                assert cmd[index+1] == R.PREFIX+module
                assert cmd[cmd.index('--target')+1] == str(target)
                assert cmd[cmd.index('--arm')+1] == arm
                assert job['start_step'] == (12000 if state_name == 'ABSENT' and arm != 'COMMON'
                                             else 0 if state_name == 'ABSENT' else step)
                if flag == '--resume':
                    assert cmd[cmd.index(flag)+1] == str(root/'training'/arm/'rolling.pt') and '--fork' not in cmd
                elif flag == '--fork':
                    assert cmd[cmd.index(flag)+1] == str(root/'training/COMMON/step12000.pt') and '--resume' not in cmd
                else:
                    assert '--resume' not in cmd and '--fork' not in cmd
            results.append({'case': name, 'status': 'PASS', 'recorded_commands_only': recorded})
        recorded = []
        with patch.object(driver, 'wait_training', return_value={'state': 'READY', 'step': 6000}), \
             patch.object(driver, 'wave', side_effect=lambda jobs: recorded.extend(jobs)):
            driver.train_target(('COMMON',), 6000)
        assert not recorded
        results.append({'case': 'completed_milestone_has_no_new_job', 'status': 'PASS'})
        assert popen.call_count == 0
    after = {str(p): digest(p) for p in files}
    assert before == after, 'Production source changed during routing test'
    assert 'torch' not in sys.modules
    result = {'status': 'PASS', 'checks': results,
              'identity_contains_resume_train_hash': True,
              'production_hashes_before': before, 'production_hashes_after': after,
              'production_source_unchanged_during_test': True, 'scientific_jobs_launched': 0,
              'scientific_jobs_retried': 0, 'model_calls': 0, 'GPU_calls': 0,
              'torch_imported': False,
              'scope': 'Actual production Driver.__init__/train_target/cmd with wait-state fixtures and a capture-only wave; original trainer/adapter not executed'}
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'cases': len(results), 'identity_adapter_hash_verified': True,
                      'result': str(out/'results.json'), 'production_hashes': after}, indent=2))


if __name__ == '__main__': main()
