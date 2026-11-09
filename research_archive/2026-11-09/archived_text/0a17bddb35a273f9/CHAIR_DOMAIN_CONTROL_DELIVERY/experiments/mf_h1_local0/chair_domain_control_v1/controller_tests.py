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
        self.protocol={'fixed':'identity','main_updates':9000,'source':str(self.out/'source'),'source_checkpoints':{}}
        atomic_json(self.out/'protocol.json',self.protocol)

    def tearDown(self):
        self.temp.cleanup()

    def training(self,arm='T1',target=3000,status='SEGMENT_COMPLETE'):
        updates=target
        count=dict(optimizer_updates=updates,optimizer_step_attempts=updates,
                   forward_attempts=8*updates,forward_returns=8*updates,
                   backward_attempts=8*updates,backward_returns=8*updates)
        path=self.out/'checkpoints'/f'{arm}_phase{target}_total{1000+target}.pt'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((arm+str(target)).encode())
        metadata=dict(variant=arm,phase_step=target,conditional_total_step=1000+target,logical_step=1000+target,protocol_sha256=json_hash(self.protocol),
                      counts=count,actual_new_updates=updates,model_state_sha256='state'+arm+str(target))
        atomic_json(path.with_suffix('.json'),dict(path=str(path),checkpoint_sha256=digest(path),metadata=metadata))
        state=dict(status=status,arm=arm,phase_step=target,conditional_total_step=1000+target,logical_step=1000+target,counts=count,protocol_sha256=json_hash(self.protocol))
        atomic_json(self.out/'training'/f'{arm}_to{target}_summary.json',state)
        return dict(key=f'train_{arm}_{target}',module='training',args=['--arm',arm,'--target',str(target)])

    def complete_fixture(self):
        patcher=mock.patch('experiments.mf_h1_local0.chair_domain_control_v1.run_all.audit_training_events',return_value=dict(status='PASS',errors=[]))
        patcher.start();self.addCleanup(patcher.stop)
        for arm in ('T1','Geo','Full'):
            cp=self.out/'source/checkpoints'/f'{arm}_step1000.pt';cp.parent.mkdir(parents=True,exist_ok=True);cp.write_bytes(arm.encode())
            self.protocol['source_checkpoints'][arm]=dict(checkpoint_sha256=digest(cp))
        atomic_json(self.out/'protocol.json',self.protocol)
        for arm in ('T1','Geo','Full'):
            for phase in (1000,2500,3000):self.training(arm,phase)
        registered=stages()
        atomic_json(self.out/'run_state.json',dict(stages={r['key']:dict(status='PASS') for r in registered}))
        atomic_json(self.out/'registration/wiring_precheck.json',dict(status='PASS',optimizer_updates=0,forwards_attempted=27,forwards_returned=27))
        panels=dict(A=[],B=[],C=[])
        for i in range(144):
            phases=[3000]
            if i<40:phases.append(0)
            if i<8:phases.extend([1000,2500])
            panels['A'].append(dict(case_id=f'case{i}',status='READY',phase_steps=phases))
        atomic_json(self.out/'panels.json',panels)
        for stage in registered:
            if stage['module']!='evaluation':continue
            args=stage['args'];phase=int(args[-1]);arm=args[1]
            requests=[r for r in panels['A'] if phase in r['phase_steps']]
            gs,ts=len(requests),24
            sampling=dict(complete_rollouts=gs,rollout_attempts=gs)
            teacher=dict(teacher_returns=ts,teacher_attempts=ts)
            for key in ('model_forward_attempts','model_forward_returns','backbone_forward_attempts','backbone_forward_returns'):
                sampling[key]=gs*50;teacher[key]=ts
            score=self.out/'score.json';atomic_json(score,dict(status='PASS'))
            rows=[dict(case_id=r['case_id'],status='PASS',score_path='score.json',score_sha256=digest(score)) for r in requests]
            teachers=[dict(case_id=f'teacher{i}',status='PASS') for i in range(ts)]
            cp=self.out/'source/checkpoints'/f'{arm}_step1000.pt' if phase==0 else self.out/'checkpoints'/f'{arm}_phase{phase}_total{1000+phase}.pt'
            meta=dict(variant=arm,phase_step=phase,logical_step=1000+phase,model_state_sha256='state'+arm+str(phase))
            hashes=dict(checkpoint_sha256=digest(cp),model_state_sha256=meta['model_state_sha256'])
            identity=dict(arm=arm,phase_step=phase,protocol_sha256=digest(self.out/'protocol.json'),checkpoint=str(cp),checkpoint_sha256=digest(cp))
            atomic_json(self.out/f'evaluation/phase{phase}/{arm}/ledger.json',dict(status='PASS',identity=identity,model_metadata=meta,model_hashes=hashes,sampling_counts=sampling,teacher_counts=teacher,rows=rows,teacher_rows=teachers))

    def test_stage_dag_and_fixed_matrix(self):
        graph=stages();by_key={s['key']:s for s in graph}
        self.assertEqual(len(graph),23);self.assertEqual(len(graph),len(by_key))
        seen=set()
        for stage in graph:
            self.assertTrue(set(stage['dependencies'])<=seen,stage['key']);seen.add(stage['key'])
        self.assertEqual(sum(s['module']=='evaluation' for s in graph),12)
        self.assertEqual(sum(s['module']=='training' for s in graph),9)
        for arm in ('T1','Geo','Full'):
            self.assertEqual(set(by_key[f'train_{arm}_1000']['dependencies']),{f'eval_{a}_0' for a in ('T1','Geo','Full')})
            self.assertIn(f'eval_{arm}_1000',by_key[f'train_{arm}_2500']['dependencies'])
            self.assertEqual(set(by_key[f'eval_{arm}_3000']['dependencies']),{f'train_{a}_3000' for a in ('T1','Geo','Full')})
        self.assertEqual(len(by_key['reporting']['dependencies']),22)
        self.assertEqual(9000*8+602*50+288+32,102420)
        self.assertEqual(LIMITS['model_forwards'],102420)

    def test_safe_interrupt_cannot_be_completed(self):
        stage=self.training(status='INTERRUPTED_SAFE')
        self.assertEqual(worker_status(self.out,stage)['status'],'STOPPED')

    def test_fixed_checkpoint_required_and_verified(self):
        stage=self.training()
        self.assertEqual(worker_status(self.out,stage)['status'],'PASS')
        cp=self.out/'checkpoints/T1_phase3000_total4000.pt'
        cp.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA'):worker_status(self.out,stage)

    def test_exact_segment_record_required(self):
        stage=self.training(target=2500)
        summary=self.out/'training/T1_to2500_summary.json'
        summary.rename(self.out/'training/T1_status.json')
        self.assertEqual(worker_status(self.out,stage)['status'],'STOPPED')

    def test_frozen_identity_refuses_change(self):
        path=self.out/'frozen.json';freeze_json(path,{'seed':123,'revision':'repair_v2'})
        freeze_json(path,{'seed':123,'revision':'repair_v2'})
        with self.assertRaisesRegex(ValueError,'identity changed'):
            freeze_json(path,{'seed':124,'revision':'repair_v2'})
        stage=self.training()
        atomic_json(self.out/'protocol.json',dict(self.protocol,main_updates=9001))
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

    def test_evaluation_wrong_checkpoint_cannot_pass(self):
        self.complete_fixture()
        path=self.out/'evaluation/phase0/T1/ledger.json';row=json.loads(path.read_text())
        wrong=self.out/'checkpoints/T1_phase3000_total4000.pt'
        row['identity'].update(checkpoint=str(wrong),checkpoint_sha256=digest(wrong))
        atomic_json(path,row)
        stage=next(s for s in stages() if s['key']=='eval_T1_0')
        with self.assertRaisesRegex(ValueError,'wrong registered checkpoint'):worker_status(self.out,stage)

    def test_actual_budget_audit_requires_exact_updates(self):
        self.complete_fixture()
        result=execution_budget_audit(self.out)
        self.assertEqual(result['status'],'PASS',result['errors'])
        self.assertEqual(result['optimizer_updates'],9000)
        self.assertEqual(result['complete_rollouts'],600)
        self.assertEqual(result['teacher_returns'],288)
        self.assertEqual(result['model_forward_attempts'],72000+30000+288+27)
        path=self.out/'training/T1_to3000_summary.json'
        data=json.loads(path.read_text());data['counts']['optimizer_updates']=2999
        atomic_json(path,data)
        self.assertEqual(execution_budget_audit(self.out)['status'],'FAIL')

    def test_actual_budget_audit_rejects_counter_mismatch_and_partial_matrix(self):
        self.complete_fixture()
        path=self.out/'evaluation/phase3000/T1/ledger.json'
        data=json.loads(path.read_text());data['sampling_counts']['complete_rollouts']=145
        atomic_json(path,data)
        result=execution_budget_audit(self.out)
        self.assertEqual(result['status'],'FAIL')
        self.assertTrue(any('budget' in e or 'denominator' in e or 'attempt' in e for e in result['errors']))
        path=self.out/'run_state.json';data=json.loads(path.read_text());data['stages'].pop('eval_Full_3000')
        atomic_json(path,data)
        self.assertTrue(any('stage set' in e for e in execution_budget_audit(self.out)['errors']))


if __name__=='__main__':unittest.main(verbosity=2)