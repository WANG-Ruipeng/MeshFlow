"""CPU-only interface tests; no real generator, scorer, or CUDA calls."""
from __future__ import annotations
import unittest
from unittest.mock import patch
from . import evaluation as E


class EvaluationContracts(unittest.TestCase):
    def test_privileged_label_explicit(self):
        self.assertEqual(E.label_spec('LAYOUT1000')['input_scope'],'ORACLE_PRIVILEGED_SOURCE_REGIONS')
        self.assertEqual(E.label_spec('LAYOUT_NULL')['input_scope'],'C_ONLY')
        self.assertEqual(E.label_spec('LAYOUT_NULL')['generator_path'],E.label_spec('LAYOUT1000')['generator_path'])
        self.assertNotEqual(E.label_spec('PLAIN1000')['generator_path'],E.label_spec('LAYOUT1000')['generator_path'])

    def test_no_baseline_model_dispatch(self):
        with self.assertRaises(ValueError): E.label_spec('EXPOSURE')
        with self.assertRaises(ValueError): E.sample('EXPOSURE')

    def test_no_original_module_mutation(self):
        from r2.pilots.bench_four_routes_v1.exposure import evaluation as old
        before=(old.OUT,old.MANIFEST,old.LABELS)
        engine=E.engine()
        self.assertEqual((old.OUT,old.MANIFEST,old.LABELS),before)
        self.assertEqual(engine.raw_one.__globals__['OUT'],E.OUT)
        self.assertEqual(engine.raw_one.__globals__['root_for']('EXPOSURE'),E.OLD)
        self.assertEqual(engine.d0.__globals__['all_raw_barrier'],E.all_raw_barrier)
        self.assertEqual(engine.raw_one.__globals__['cpu_one'],E._cpu_worker)
        self.assertIsNot(engine.raw_one.__globals__,vars(old))

    def test_registered_dynamic_budget(self):
        qs=[{'parent_id':'a','request_id':'a0'},{'parent_id':'a','request_id':'a1'},
            {'parent_id':'b','request_id':'b0'},{'parent_id':'b','request_id':'b1'}]
        with patch.object(E,'requests',return_value=qs):
            b=E.budget()
        self.assertEqual((b['new_RAW'],b['new_D0'],b['total_score_rows']),(12,12,48))
        self.assertEqual(b['generator_forwards'],600)

    def test_reject_manifest_cardinality_change(self):
        def read(path):
            if path==E.MANIFEST: return {'requests':[{'request_id':'a'}]}
            return {'eval_requests':12}
        with patch.object(E.C,'read',side_effect=read):
            with self.assertRaises(ValueError): E.requests()

    def test_root_scope(self):
        self.assertEqual(E.root_for('EXPOSURE'),E.OLD)
        self.assertTrue(all(E.root_for(x)==E.OUT for x in E.NEW_LABELS))
        with self.assertRaises(ValueError): E.root_for('UNKNOWN')


if __name__=='__main__': unittest.main()
