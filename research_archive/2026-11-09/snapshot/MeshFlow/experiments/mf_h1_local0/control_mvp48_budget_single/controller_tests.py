"""CPU-only controller contract tests; no model, CUDA or optimizer calls."""
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

from .common import atomic_json, digest, freeze_json, json_hash, LIMITS
from .run_all import (MODULE, stages, worker_status, orphan_budget_workers,
                      wait_managed_child, execution_budget_audit)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.out=Path(self.temp.name)
        self.protocol={'fixed':'identity','main_updates':12000,'probe_updates':1000}
        atomic_json(self.out/'protocol.json',self.protocol)

    def tearDown(self):
        self.temp.cleanup()

    def training(self,arm='T1',target=5000,status='SEGMENT_COMPLETE'):
        updates=target-(0 if arm=='T1_fit2_probe' else 1000)
        count=dict(optimizer_updates=updates,optimizer_step_attempts=updates,
                   forward_attempts=8*updates,forward_returns=8*updates,
                   backward_attempts=8*updates,backward_returns=8*updates)
        path=self.out/'checkpoints'/f'{arm}_step{target}.pt'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((arm+str(target)).encode())
        metadata=dict(variant=arm,logical_step=target,protocol_sha256=json_hash(self.protocol),
                      counts=count,actual_new_updates=updates)
        atomic_json(path.with_suffix('.json'),dict(path=str(path),checkpoint_sha256=digest(path),metadata=metadata))
        state=dict(status=status,arm=arm,logical_step=target,counts=count,protocol_sha256=json_hash(self.protocol))
        atomic_json(self.out/'training'/f'{arm}_to{target}_summary.json',state)
        return dict(key=f'train_{arm}_{target}',module='training',args=['--arm',arm,'--target',str(target)])

    def complete_fixture(self):
        for arm in ('T1','Geo','Full'):self.training(arm)
        self.training('T1_fit2_probe',1000)
        registered=stages()
        atomic_json(self.out/'run_state.json',dict(stages={r['key']:dict(status='PASS') for r in registered}))
        atomic_json(self.out/'registration/wiring_precheck.json',dict(status='PASS',optimizer_updates=0,
                    forwards_attempted=27,forwards_returned=27))
        sizes=dict(curve=(64,48),swap=(16,0),fit=(8,6),holdout=(32,0))
        for stage in registered:
            if stage['module']!='evaluation':continue
            args=stage['args'];phase=args[args.index('--phase')+1];model=args[args.index('--model-id')+1]
            gs,ts=sizes[phase]
            sampling=dict(complete_rollouts=gs,rollout_attempts=gs)
            teacher=dict(teacher_returns=ts,teacher_attempts=ts)
            for key in ('model_forward_attempts','model_forward_returns','backbone_forward_attempts','backbone_forward_returns'):
                sampling[key]=gs*50;teacher[key]=ts
            rows=[dict(case_id=f'case{i}',status='PASS',score_path=f'scores/{i}.json') for i in range(gs)]
            teachers=[dict(case_id=f'teacher{i}',status='PASS') for i in range(ts)]
            atomic_json(self.out/f'evaluation/{phase}/{model}/ledger.json',dict(status='PASS',sampling_counts=sampling,
                        teacher_counts=teacher,rows=rows,teacher_rows=teachers))

    def test_stage_dag_and_fixed_matrix(self):
        graph=stages();by_key={s['key']:s for s in graph}
        self.assertEqual(len(graph),len(by_key))
        seen=set()
        for stage in graph:
            self.assertTrue(set(stage['dependencies'])<=seen,stage['key'])
            seen.add(stage['key'])
        self.assertEqual(sum(s['key'].startswith('curve_') for s in graph),15)
        self.assertEqual(sum(s['key'].startswith('swap_') for s in graph),7)
        self.assertEqual(sum(s['key'].startswith('fit_') for s in graph),3)
        self.assertEqual(sum(s['key'].startswith('holdout_') for s in graph),3)
        for arm in ('T1','Geo','Full'):
            self.assertIn(f'train_{arm}_2000',by_key[f'train_{arm}_3500']['dependencies'])
            self.assertIn(f'train_{arm}_3500',by_key[f'train_{arm}_5000']['dependencies'])
            self.assertEqual(set(by_key[f'holdout_{arm}_5000']['dependencies']),
                             {f'train_{a}_5000' for a in ('T1','Geo','Full')})
        generations=15*64+7*16+3*8+3*32
        teachers=15*48+3*6
        self.assertEqual(generations,1192)
        self.assertEqual(teachers,738)
        self.assertEqual(13000*8+(generations+4)*50+teachers+80,164618)
        self.assertEqual(13000*8+generations*50+teachers+80,164418)
        self.assertEqual(LIMITS['model_forwards'],164618)

    def test_safe_interrupt_cannot_be_completed(self):
        stage=self.training(status='INTERRUPTED_SAFE')
        self.assertEqual(worker_status(self.out,stage)['status'],'STOPPED')

    def test_fixed_checkpoint_required_and_verified(self):
        stage=self.training()
        self.assertEqual(worker_status(self.out,stage)['status'],'PASS')
        cp=self.out/'checkpoints/T1_step5000.pt'
        cp.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA'):worker_status(self.out,stage)

    def test_status_fallback_and_target_gate(self):
        stage=self.training(target=3500)
        summary=self.out/'training/T1_to3500_summary.json'
        summary.rename(self.out/'training/T1_status.json')
        self.assertEqual(worker_status(self.out,stage)['status'],'PASS')
        sidecar=self.out/'checkpoints/T1_step3500.json'
        sidecar.unlink()
        self.assertEqual(worker_status(self.out,stage)['status'],'STOPPED')

    def test_frozen_identity_refuses_change(self):
        path=self.out/'frozen.json';freeze_json(path,{'seed':123,'revision':'repair_v2'})
        freeze_json(path,{'seed':123,'revision':'repair_v2'})
        with self.assertRaisesRegex(ValueError,'identity changed'):
            freeze_json(path,{'seed':124,'revision':'repair_v2'})
        stage=self.training()
        atomic_json(self.out/'protocol.json',dict(self.protocol,main_updates=12001))
        with self.assertRaisesRegex(ValueError,'protocol identity'):worker_status(self.out,stage)

    def test_orphan_scan_scoped_to_budget_workers(self):
        table='PID PPID COMMAND\n'
        table+=f'210 1 python -B -m {MODULE}.training --out {self.out} --arm T1 --target 2000\n'
        table+=f'211 1 python -B -m {MODULE}.evaluation --out /tmp/unrelated --phase curve\n'
        table+='212 1 python -m unrelated.training --out '+str(self.out)+'\n'
        found=orphan_budget_workers(self.out,table)
        self.assertEqual([r['pid'] for r in found],[210])

    def test_owned_child_receives_sigterm_and_is_reaped(self):
        child=mock.Mock()
        child.wait.side_effect=[KeyboardInterrupt('stop'),0]
        child.poll.side_effect=[None,None,0]
        with mock.patch('signal.signal'):
            with self.assertRaises(KeyboardInterrupt):wait_managed_child(child)
        child.send_signal.assert_called_once_with(signal.SIGTERM)
        self.assertEqual(child.wait.call_count,2)

    def test_actual_budget_audit_requires_exact_updates(self):
        self.complete_fixture()
        result=execution_budget_audit(self.out)
        self.assertEqual(result['status'],'PASS',result['errors'])
        self.assertEqual(result['optimizer_updates'],13000)
        self.assertEqual(result['complete_rollouts'],1192)
        self.assertEqual(result['teacher_returns'],738)
        self.assertEqual(result['model_forward_attempts'],104000+59600+738+27)
        path=self.out/'training/T1_to5000_summary.json'
        data=json.loads(path.read_text());data['counts']['optimizer_updates']=3999
        atomic_json(path,data)
        self.assertEqual(execution_budget_audit(self.out)['status'],'FAIL')

    def test_actual_budget_audit_rejects_counter_mismatch_and_partial_matrix(self):
        self.complete_fixture()
        path=self.out/'evaluation/curve/T1_5000/ledger.json'
        data=json.loads(path.read_text());data['sampling_counts']['complete_rollouts']=65
        atomic_json(path,data)
        result=execution_budget_audit(self.out)
        self.assertEqual(result['status'],'FAIL')
        self.assertTrue(any('budget' in e or 'counter' in e for e in result['errors']))
        path=self.out/'run_state.json';data=json.loads(path.read_text());data['stages'].pop('holdout_Full_5000')
        atomic_json(path,data)
        self.assertTrue(any('stage set' in e for e in execution_budget_audit(self.out)['errors']))


if __name__=='__main__':unittest.main(verbosity=2)