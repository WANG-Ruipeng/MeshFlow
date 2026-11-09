"""No-model tests for launcher state and measured-concurrency accounting."""
import ast,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from r2.pilots.bench_seat_coverage_v1 import run as R, controller as C
from r2.pilots.bench_seat_coverage_v1.common import OUT,sha,write_new

class Tests(unittest.TestCase):
    def test_normal_live_completed_states(self):
        with tempfile.TemporaryDirectory(dir=OUT/'engineering') as d:
            p=Path(d)
            self.assertEqual(R.job_status(p)['status'],'NOT_STARTED')
            write_new(p/'pid.json',dict(status='LAUNCHED',pid=123))
            self.assertEqual(R.job_status(p,alive=lambda *args:True)['status'],'LAUNCHED')
            write_new(p/'status.json',dict(status='COMPLETE',pid=123))
            self.assertEqual(R.job_status(p,alive=lambda *args:False)['status'],'COMPLETE')

    def test_dead_or_invisible_supervisor_stops(self):
        for alive in (False,None):
            with tempfile.TemporaryDirectory(dir=OUT/'engineering') as d:
                p=Path(d);write_new(p/'status.json',dict(status='RUNNING',pid=123,child_pid=124))
                r=R.job_status(p,alive=lambda *args:alive)
                self.assertEqual(r['status'],'STALE_PROCESS_REQUIRES_INSPECTION')
                self.assertFalse(r['automatic_retry'])

    def test_missing_pid_launch_never_relaunched(self):
        with tempfile.TemporaryDirectory(dir=OUT/'engineering') as d:
            out=Path(d);p=out/'workflow/train';write_new(p/'launch.json',dict(command=['synthetic']))
            with patch.object(R,'OUT',out),patch.object(R.subprocess,'Popen',side_effect=AssertionError('Must not launch')):
                self.assertEqual(R.launch('train')['status'],'LAUNCH_INCOMPLETE_REQUIRES_INSPECTION')

    def test_normal_child_exit_handoff_not_false_failure(self):
        with tempfile.TemporaryDirectory(dir=OUT/'engineering') as d:
            p=Path(d);write_new(p/'status.json',dict(status='RUNNING',pid=123,child_pid=124))
            r=R.job_status(p,alive=lambda pid,created:pid==123)
            self.assertEqual(r['status'],'RUNNING');self.assertFalse(r['child_process_alive'])

    def test_process_identity_no_signals(self):
        self.assertTrue(R.pid_alive(os.getpid()))
        self.assertFalse(R.pid_alive(os.getpid(),created=0))
        self.assertFalse(R.pid_alive(-1000))

    def test_real_sampler_observations_force_serial_when_WDDM_NA(self):
        with tempfile.TemporaryDirectory(dir=OUT/'engineering') as d:
            out=Path(d);base=out/'training/NATIVE';base.mkdir(parents=True)
            write_new(base/'first_formal_backward.json',dict(status='PASS',synthetic_CPU_test=True))
            write_new(base/'resources_step25.json',dict(synthetic_CPU_test=True,gpu={'stdout':'synthetic'}))
            rows=[dict(stage_step=i,update_seconds=1.,input_seconds=.5,peak_allocated_bytes=10,peak_reserved_bytes=20) for i in range(1,26)]
            (base/'training_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"partial":',encoding='utf-8')
            with patch.object(C,'OUT',out):result=C.overlap_evidence(dict(stage_step=25))
            self.assertEqual(result['formal_observed_updates'],25)
            self.assertIsNone(result['comparable_process_peak_mib'])
            self.assertEqual(result['process_peak_status'],'NA_WDDM_PROCESS_MEMORY_UNAVAILABLE')
            self.assertEqual(result['observed_update_seconds'],25.)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)
    for module in (R,C):ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    if 'torch' in sys.modules:raise RuntimeError('Launcher CPU tests imported model framework')
    report=dict(status='PASS',tests=result.testsRun,syntax='PASS',model_forwards=0,GPU_calls=0,
        neural_framework_imported=False,large_model_loads=0,subprocess_jobs_launched=0,
        source_hashes={str(Path(m.__file__)):sha(m.__file__) for m in (R,C)},
        fixes=['Dead/reused/invisible supervisor stops explicitly','Incomplete launch cannot relaunch',
               'Actual formal1..25 measurements retained; unavailable WDDM sampler process peak implies serial GPU'])
    write_new(OUT/'engineering/launcher_repair_cpu_receipt.json',report)
    print(json.dumps(report,indent=2))
