"""Bounded CPU checks for the pinned daily baseline and historical routing.

These tests never read a real checkpoint or construct a model. Run separately
from data/metric suites so this validation adds no OT or autograd calls.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from native_t1 import baseline, checkpoint
from native_t1 import __main__ as cli


TEST_COUNTS = dict(metadata_validation_calls=0, mocked_checkpoint_load_calls=0,
                   mocked_file_hash_calls=0, profile_resolutions=0,
                   historical_dispatch_calls=0, forbidden_operation_attempts={})


def valid_metadata():
    return dict(schema='a_then_fc_branch_v1', arm='A_continue',
                base_cumulative_updates=1000, additional_updates=500,
                cumulative_updates=1500, loss_switches=dict(FC=False,FF=False),
                optimizer_step_in_progress=False, batch_in_progress=False)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.assertFalse(torch.cuda.is_initialized())

        def forbidden(name):
            def call(*args, **kwargs):
                counters = TEST_COUNTS['forbidden_operation_attempts']
                counters[name] = counters.get(name,0)+1
                raise AssertionError('CPU-only test attempted '+name)
            return call

        self.stack.enter_context(mock.patch.object(torch,'load',side_effect=forbidden('real_checkpoint_load')))
        self.stack.enter_context(mock.patch.object(torch.cuda,'_lazy_init',side_effect=forbidden('CUDA_initialization')))
        for module in (baseline,checkpoint):
            for name in ('DiT','NativeInpaintingModel'):
                self.stack.enter_context(mock.patch.object(module,name,side_effect=forbidden('model_construction')))

    def parse_sample(self, *extra):
        return cli.build_parser().parse_args([
            'sample','--condition','unread_condition.npy','--seed','9601',
            '--out','uncreated_cpu_test_output',*extra])

    def resolve(self, args):
        TEST_COUNTS['profile_resolutions'] += 1
        return cli.resolve_profile(args)

    def validate(self, payload):
        TEST_COUNTS['metadata_validation_calls'] += 1
        return baseline.validate_baseline_metadata(payload)

    def test_original_t1_pins_and_default_signature_remain_historical(self):
        self.assertEqual(checkpoint.T1_FILE_SHA256,
                         '7928d77507bf1fe610876f61b5b11e614263323b5fc0f595cc6740ffd24af634')
        self.assertEqual(checkpoint.T1_STATE_SHA256,
                         '37e31fe5925b8f870b9a4d3bbbbc84edc7bf30177946a6f67f4529a7f8e97172')
        self.assertEqual(checkpoint.DEFAULT_CHECKPOINT,
                         checkpoint.REPO/'experiments/mf_h1_local0/outputs/native_patch_20260929_010000/Native_correct/endpoint_step500.pt')
        signature = inspect.signature(checkpoint.load_t1)
        self.assertEqual(signature.parameters['checkpoint_path'].default,checkpoint.DEFAULT_CHECKPOINT)
        self.assertEqual(signature.parameters['config_path'].default,checkpoint.DEFAULT_CONFIG)
        self.assertNotEqual(checkpoint.DEFAULT_CHECKPOINT,baseline.BASELINE_CHECKPOINT)

    def test_sample_defaults_to_pinned_a_continue(self):
        args = self.parse_sample()
        self.assertEqual(args.profile,'a-continue')
        self.assertIsNone(args.checkpoint)
        loader, path = self.resolve(args)
        self.assertIs(loader,baseline.load_baseline)
        self.assertEqual(path,baseline.BASELINE_CHECKPOINT)
        self.assertEqual(baseline.BASELINE_FILE_SHA256,
                         'd8366d8c172407ec97ee4288143863825fae583d2c879d200f3862315747e400')
        self.assertEqual(baseline.BASELINE_STATE_SHA256,
                         'f162c79cc0d0d8dbd3ff208fecc6f433f348096d950652678a06cfaa328cc90b')
        self.assertEqual(baseline.BASELINE_BACKBONE_SHA256,
                         '25d5414e6a3af1cbf1f0f71f27a68010f28318578aaf092af150d80b00e4a1a2')

    def test_explicit_profiles_and_relocated_checkpoint_paths(self):
        for profile, expected_loader, default_path in (
                ('a-continue',baseline.load_baseline,baseline.BASELINE_CHECKPOINT),
                ('original-t1',checkpoint.load_t1,checkpoint.DEFAULT_CHECKPOINT)):
            with self.subTest(profile=profile):
                loader,path = self.resolve(self.parse_sample('--profile',profile))
                self.assertIs(loader,expected_loader)
                self.assertEqual(path,default_path)
                loader,path = self.resolve(self.parse_sample('--profile',profile,'--checkpoint','relocated.pt'))
                self.assertIs(loader,expected_loader)
                self.assertEqual(path,Path('relocated.pt'))
        with self.assertRaisesRegex(ValueError,'Unknown checkpoint profile'):
            self.resolve(SimpleNamespace(profile='unregistered',checkpoint=None))

    def test_both_verify_commands_dispatch_only_historical_regression(self):
        with mock.patch('native_t1.verify.run_regression',return_value='historical-only') as run, \
                mock.patch.object(cli,'resolve_profile',side_effect=AssertionError('Verify used sample profile')), \
                mock.patch.object(cli.np,'load',side_effect=AssertionError('Mock verify read an input')):
            for command in ('verify-historical','verify'):
                self.assertEqual(cli.main([command,'--out','uncreated_historical_test_output']),
                                 'historical-only')
                TEST_COUNTS['historical_dispatch_calls'] += 1
                run.assert_called_with(Path('uncreated_historical_test_output'))
            self.assertEqual(run.call_count,2)

    def test_valid_completed_pure_fm_metadata_is_accepted_without_io(self):
        self.assertIsNone(self.validate(valid_metadata()))

    def test_other_branches_schemas_updates_switches_and_incomplete_states_rejected(self):
        changes = (
            ('arm','A_then_FC'), ('arm','Native_correct'),
            ('schema','native_connected_patch_v1'),
            ('base_cumulative_updates',500), ('additional_updates',499),
            ('cumulative_updates',1000),
            ('loss_switches',dict(FC=True,FF=False)),
            ('loss_switches',dict(FC=False,FF=True)),
            ('loss_switches',dict(FC=False)),
            ('optimizer_step_in_progress',True), ('batch_in_progress',True),
        )
        for key,value in changes:
            payload = valid_metadata()
            payload[key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                self.validate(payload)
        missing = valid_metadata()
        del missing['schema']
        with self.assertRaises(ValueError):
            self.validate(missing)

    def test_wrong_file_or_config_hash_rejected_before_runtime_or_checkpoint_io(self):
        for hashes, expected_calls in ((['wrong-file'],1),
                                       ([baseline.BASELINE_FILE_SHA256,'wrong-config'],2)):
            with self.subTest(hashes=hashes), \
                    mock.patch.object(baseline,'file_sha256',side_effect=hashes) as hash_mock, \
                    mock.patch.object(baseline,'configure_stable_runtime',side_effect=AssertionError('Early reject configured runtime')), \
                    mock.patch.object(Path,'read_text',side_effect=AssertionError('Early reject read config')), \
                    self.assertRaises(ValueError):
                try:
                    baseline.load_baseline('unread.pt','unread.yaml',device='cpu')
                finally:
                    TEST_COUNTS['mocked_file_hash_calls'] += hash_mock.call_count
                    self.assertEqual(hash_mock.call_count,expected_calls)

    def test_bad_payload_rejected_before_any_model_is_constructed(self):
        payload = valid_metadata()
        payload['arm'] = 'A_then_FC'
        config_text = 'model:\n  model_type: equidit\n  pe_freq: 20\ntransport:\n  prediction: velocity\n'
        with mock.patch.object(baseline,'file_sha256',side_effect=[baseline.BASELINE_FILE_SHA256,baseline.BASELINE_CONFIG_SHA256]) as hash_mock, \
                mock.patch.object(baseline,'configure_stable_runtime') as runtime_mock, \
                mock.patch.object(Path,'read_text',return_value=config_text), \
                mock.patch.object(torch,'load',return_value=deepcopy(payload)) as load_mock, \
                mock.patch.object(baseline,'validate_baseline_metadata',wraps=baseline.validate_baseline_metadata) as metadata_mock:
            with self.assertRaisesRegex(ValueError,'A_continue cumulative1500 metadata'):
                baseline.load_baseline('unread.pt','unread.yaml',device='cpu')
            runtime_mock.assert_called_once_with()
            load_mock.assert_called_once_with(Path('unread.pt'),map_location='cpu',weights_only=True,mmap=True)
            metadata_mock.assert_called_once_with(payload)
            TEST_COUNTS['metadata_validation_calls'] += metadata_mock.call_count
            TEST_COUNTS['mocked_checkpoint_load_calls'] += load_mock.call_count
            TEST_COUNTS['mocked_file_hash_calls'] += hash_mock.call_count
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
