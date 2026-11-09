"""CPU contracts for pinned START selection; no real weights, forward or CUDA."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from native_t1 import __main__ as cli
from native_t1 import chair_checkpoint as cc



def tiny_start_payload():
    """Synthetic structure; fixed protocol/stream hashes are explicitly mocked."""
    return dict(schema=cc.SCHEMA, update_in_progress=False,
        metadata=deepcopy(cc.EXPECTED_METADATA), protocol={},
        model={'tiny':torch.zeros(1,dtype=torch.float32)},
        optimizer=dict(param_groups=[dict(lr=1e-5,betas=(.9,.95),weight_decay=0.,params=[0])],
            state={0:dict(step=torch.tensor(5000.),exp_avg=torch.zeros(1),exp_avg_sq=torch.zeros(1))}),
        rng=dict(python=None,numpy=None,torch_cpu=None,torch_cuda=None),
        stream_state=dict(arm='OT_HYBRID',seed=20261004,batch_index=1000,sample_index=8000,
            lambda_value=.25,RNG_mode='stateless_per_sample_frozen_seeds',
            coupling_contract='original_preOT_cost_scale__FP32_final_target_x2__raw_iid_noise_slots_v1'))


class ChairCheckpointContracts(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        for owner,name in ((torch,'load'),(torch.cuda,'_lazy_init'),
                           (cc,'DiT'),(cc,'ChairModel'),(cc,'ContextGeometryEncoder')):
            self.stack.enter_context(mock.patch.object(owner,name,
                side_effect=AssertionError('Forbidden real operation: '+name)))

    def tiny_validation(self):
        stack=ExitStack()
        stack.enter_context(mock.patch.object(cc,'MODEL_TENSOR_COUNT',1))
        stack.enter_context(mock.patch.object(cc,'MODEL_PARAMETER_COUNT',1))
        stack.enter_context(mock.patch.object(cc,'_json_sha256',side_effect=[
            cc.EXPECTED_METADATA['protocol_sha256'],cc.EXPECTED_METADATA['stream_state_sha256']]))
        return stack

    def test_exact_start_identity_is_distinct_from_moment_endpoint(self):
        self.assertEqual(cc.CHAIR_START_FILE_SHA256,
            'd640ec187291a114a961959848b542e7176202a82a5707afedc144177d013d80')
        self.assertEqual(cc.CHAIR_START_STATE_SHA256,
            '166c71fef888f26494fc4a0af1776b77a484efa72a6f3ce06552c604fbecc972')
        self.assertEqual(cc.CHAIR_START_CHECKPOINT.name,'chair_hybrid_start_total5000.pt')
        self.assertEqual(cc.EXPECTED_METADATA['arm'],'OT_HYBRID')
        self.assertEqual(cc.EXPECTED_METADATA['conditional_total_step'],5000)
        self.assertEqual(cc.EXPECTED_METADATA['loss_recipe'],'fm')

    def test_completed_start_metadata_passes_with_tiny_mocked_structure(self):
        payload=tiny_start_payload()
        with self.tiny_validation():
            meta=cc.validate_chair_start(payload,cc.EXPECTED_METADATA['config_sha256'])
        self.assertEqual(meta,cc.EXPECTED_METADATA)
        self.assertIsNot(meta,payload['metadata'])

    def test_wrong_schema_arm_step_recipe_or_coupling_is_rejected(self):
        for key,value in (
                ('arm','MOMENT_XY'),('arm','OT_BASE'),('pilot_step',999),
                ('conditional_total_step',6000),('completed_updates',999),
                ('loss_recipe','edge5'),('context_encoder','t1'),
                ('lambda_variance',0.),('model_state_sha256','wrong'),
                ('optimizer_reset',True),('full_recovery_state',False),
                ('pilot_step',1000.0)):
            payload=tiny_start_payload();payload['metadata'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                cc.validate_chair_start(payload)
        for field,value in (('schema','chair_moment_channel_v2'),('update_in_progress',True),
                            ('update_in_progress',None),('metadata',{})):
            payload=tiny_start_payload();payload[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                cc.validate_chair_start(payload)
        with self.assertRaises(ValueError):
            cc.validate_chair_start(tiny_start_payload(),'different-config-sha')

    def test_protocol_and_input_stream_identity_are_not_trusted_from_metadata(self):
        for hashes in (['wrong-protocol'],[cc.EXPECTED_METADATA['protocol_sha256'],'wrong-stream']):
            with self.subTest(hashes=hashes),mock.patch.object(cc,'_json_sha256',side_effect=hashes),self.assertRaises(ValueError):
                cc.validate_chair_start(tiny_start_payload())
        payload=tiny_start_payload();payload['stream_state']['sample_index']=7999
        with self.tiny_validation(),self.assertRaises(ValueError):
            cc.validate_chair_start(payload)

    def test_incomplete_or_relation_augmented_saved_state_is_rejected(self):
        for failure in ('half_model','relation_parameters','missing_rng','wrong_adam_step','missing_moment'):
            payload=tiny_start_payload()
            if failure=='half_model':payload['model']['tiny']=payload['model']['tiny'].half()
            elif failure=='relation_parameters':payload['model']={'relation_modules.layer_03.fc1.weight':torch.zeros(1)}
            elif failure=='missing_rng':del payload['rng']['torch_cpu']
            elif failure=='wrong_adam_step':payload['optimizer']['state'][0]['step']=torch.tensor(6000.)
            else:del payload['optimizer']['state'][0]['exp_avg_sq']
            with self.subTest(failure=failure),self.tiny_validation(),self.assertRaises(ValueError):
                cc.validate_chair_start(payload)

    def test_wrong_file_hash_rejected_before_config_or_checkpoint_load(self):
        with mock.patch.object(cc,'file_sha256',return_value='wrong-file') as hashed, \
                mock.patch.object(cc,'_read_config',side_effect=AssertionError('Early reject read config')), \
                mock.patch.object(cc,'configure_stable_runtime',side_effect=AssertionError('Early reject configured runtime')):
            with self.assertRaises(ValueError):
                cc.load_chair_start('unread.pt','unread.yaml',device='cpu')
            hashed.assert_called_once_with(Path('unread.pt'))

    def test_loader_checks_metadata_before_model_construction(self):
        payload=tiny_start_payload();payload['metadata']['loss_recipe']='surface'
        with mock.patch.object(cc,'file_sha256',return_value=cc.CHAIR_START_FILE_SHA256), \
                mock.patch.object(cc,'_read_config',return_value={'pe_freq':20}), \
                mock.patch.object(cc,'config_sha256',return_value=cc.EXPECTED_METADATA['config_sha256']), \
                mock.patch.object(torch,'load',return_value=payload) as load, \
                mock.patch.object(cc,'configure_stable_runtime',side_effect=AssertionError('Bad metadata reached runtime')):
            with self.assertRaisesRegex(ValueError,'loss_recipe'):
                cc.load_chair_start('unread.pt','unread.yaml',device='cpu')
            load.assert_called_once_with(Path('unread.pt'),map_location='cpu',weights_only=True,mmap=True)

    def test_actual_tensor_hash_is_checked_after_strict_mock_load(self):
        payload=tiny_start_payload()
        model=SimpleNamespace(backbone=SimpleNamespace(x_embedder=SimpleNamespace()))
        model.load_state_dict=mock.Mock(return_value=SimpleNamespace(missing_keys=[],unexpected_keys=[]))
        model.set_trainable=mock.Mock(return_value=model);model.eval=mock.Mock(return_value=model)
        with self.tiny_validation(), \
                mock.patch.object(cc,'file_sha256',return_value=cc.CHAIR_START_FILE_SHA256), \
                mock.patch.object(cc,'_read_config',return_value={'pe_freq':20}), \
                mock.patch.object(cc,'config_sha256',return_value=cc.EXPECTED_METADATA['config_sha256']), \
                mock.patch.object(torch,'load',return_value=payload), \
                mock.patch.object(Path,'exists',return_value=False), \
                mock.patch.object(cc,'configure_stable_runtime',return_value={}), \
                mock.patch.object(cc,'DiT',return_value=object()), \
                mock.patch.object(cc,'ChairModel',return_value=model), \
                mock.patch.object(cc,'ContextGeometryEncoder',return_value=object()), \
                mock.patch.object(cc,'get_embedder',return_value=(None,None)), \
                mock.patch.object(cc,'state_sha256',return_value='wrong-actual-state'):
            with self.assertRaisesRegex(ValueError,'actual model tensor hash'):
                cc.load_chair_start('unread.pt','unread.yaml',device='cpu')
        model.load_state_dict.assert_called_once_with(payload['model'],strict=True,assign=True)
        model.set_trainable.assert_called_once_with(False)
        self.assertFalse(torch.cuda.is_initialized())


class ChairSampleContracts(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for owner, name in ((torch, 'load'), (torch.cuda, '_lazy_init'),
                            (cc, 'DiT'), (cc, 'ChairModel'),
                            (cc, 'ContextGeometryEncoder')):
            self.stack.enter_context(mock.patch.object(
                owner, name, side_effect=AssertionError('Forbidden real operation: '+name)))

    def parse_sample(self, *extra):
        return cli.build_parser().parse_args([
            'sample', '--condition', 'unread_condition.npy', '--seed', '19',
            '--out', 'uncreated_chair_contract_test_output', *extra])

    def test_default_and_relocated_start_keep_pinned_loader(self):
        args = self.parse_sample()
        self.assertEqual(args.profile, 'chair-hybrid')
        self.assertIsNone(args.num_faces)
        self.assertIsNone(args.checkpoint)
        loader, path = cli.resolve_profile(args)
        self.assertIs(loader, cc.load_chair_start)
        self.assertEqual(path, cc.CHAIR_START_CHECKPOINT)
        args = self.parse_sample('--checkpoint', 'relocated_START.pt', '--num-faces', '128')
        loader, path = cli.resolve_profile(args)
        self.assertIs(loader, cc.load_chair_start)
        self.assertEqual(path, Path('relocated_START.pt'))
        self.assertEqual(args.num_faces, 128)

    def test_explicit_legacy_profiles_keep_their_loader_and_n112(self):
        from native_t1.working_model import load_fm_geo, load_edge5, FM_GEO_CHECKPOINT, EDGE5_CHECKPOINT
        for profile, loader, path in (
                ('fm-geo', load_fm_geo, FM_GEO_CHECKPOINT),
                ('edge5', load_edge5, EDGE5_CHECKPOINT)):
            with self.subTest(profile=profile):
                actual, checkpoint = cli.resolve_profile(self.parse_sample('--profile', profile))
                self.assertIs(actual, loader)
                self.assertEqual(checkpoint, path)
                self.assertEqual(cli.validate_sample_num_faces(profile), 112)
                self.assertEqual(cli.validate_sample_num_faces(profile, 112), 112)
                with self.assertRaises(ValueError):
                    cli.validate_sample_num_faces(profile, 128)

    def test_chair_requires_explicit_supported_total_faces(self):
        for total in (None, -1, 0, 112, 127, 257):
            with self.subTest(total=total), self.assertRaises(ValueError):
                cli.validate_sample_num_faces('chair-hybrid', total)
        for total in (128, 193, 256):
            self.assertEqual(cli.validate_sample_num_faces('chair-hybrid', total), total)

    def test_chair_accepts_condition_formats_and_open_k_range(self):
        for total, known in ((128, 1), (128, 127), (256, 1), (256, 255)):
            for shape in ((known, 9), (known, 3, 3)):
                with self.subTest(total=total, shape=shape):
                    C = np.zeros(shape, dtype=np.float32)
                    self.assertEqual(cli.validate_sample_condition(C, 'chair-hybrid', total), total)

    def test_chair_rejects_empty_full_oversized_or_malformed_condition(self):
        bad = [np.zeros((0,9),np.float32), np.zeros((128,9),np.float32),
               np.zeros((129,9),np.float32), np.zeros((2,3,2),np.float32),
               np.zeros((2,9),np.float64), np.zeros((9,),np.float32),
               np.full((2,9),np.nan,np.float32), np.full((2,9),np.inf,np.float32)]
        for C in bad:
            with self.subTest(shape=C.shape, dtype=C.dtype), self.assertRaises(ValueError):
                cli.validate_sample_condition(C, 'chair-hybrid', 128)

    def test_legacy_condition_sizes_are_still_restricted(self):
        for profile in ('fm-geo', 'edge5'):
            for known in (2,4,8,12):
                self.assertEqual(cli.validate_sample_condition(
                    np.zeros((known,3,3),np.float32), profile), 112)
            for known in (0,1,13,112):
                with self.subTest(profile=profile,known=known), self.assertRaises(ValueError):
                    cli.validate_sample_condition(np.zeros((known,9),np.float32),profile)

    def test_missing_chair_faces_rejected_before_condition_io(self):
        with mock.patch.object(Path, 'exists', return_value=False), \
                mock.patch.object(cli.np, 'load', side_effect=AssertionError('Early rejection read condition')), \
                mock.patch.object(Path, 'mkdir', side_effect=AssertionError('Rejected request created output')), \
                mock.patch.object(cli, 'configure_stable_runtime', side_effect=AssertionError('Rejected request configured runtime')):
            with self.assertRaises(ValueError):
                cli.main(['sample','--condition','unread.npy','--seed','19','--out','uncreated_output'])

    def test_bad_chair_k_rejected_before_output_or_model_loading(self):
        with mock.patch.object(Path, 'exists', return_value=False), \
                mock.patch.object(cli.np, 'load', return_value=np.zeros((128,9),np.float32)), \
                mock.patch.object(Path, 'mkdir', side_effect=AssertionError('Rejected request created output')), \
                mock.patch.object(cli, 'configure_stable_runtime', side_effect=AssertionError('Rejected request configured runtime')), \
                mock.patch.object(cli, 'resolve_profile', side_effect=AssertionError('Rejected request resolved loader')):
            with self.assertRaises(ValueError):
                cli.main(['sample','--condition','unread.npy','--seed','19','--out','uncreated_output','--num-faces','128'])



class ChairSamplingArithmetic(unittest.TestCase):
    """Synthetic CPU arrays only; these helpers never invoke a model."""
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_free_gaussian_draw_and_c_first_slots(self):
        from native_t1.chair_sampling import make_noise
        # Frozen first MT19937 seed-0 Gaussian face, before FP32 conversion.
        face=np.asarray([1.764052345967664, .4001572083672233, .9787379841057392,
            2.240893199201458, 1.8675579901499675, -.977277879876411,
            .9500884175255894, -.1513572082976979, -.10321885179355784],dtype=np.float32)
        numpy_before=np.random.get_state()
        torch_before=torch.random.get_rng_state().clone()
        with mock.patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CPU arithmetic initialized CUDA')):
            for total in (128,256):
                z=make_noise(0,total,total-1)
                self.assertEqual(tuple(z.shape),(1,total,9))
                self.assertEqual(z.dtype,torch.float32)
                self.assertEqual(z.device.type,'cpu')
                self.assertFalse(bool(z[:,:total-1].any()))
                np.testing.assert_array_equal(z[0,-1].numpy(),face)
        numpy_after=np.random.get_state()
        self.assertEqual(numpy_before[0],numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1],numpy_after[1])
        self.assertEqual(numpy_before[2:],numpy_after[2:])
        self.assertTrue(torch.equal(torch_before,torch.random.get_rng_state()))

    def test_noise_input_boundaries_reject_before_drawing(self):
        from native_t1.chair_sampling import make_noise
        invalid=[(-1,128,1),(2**32,128,1),(True,128,1),(0,112,1),
                 (0,257,1),(0,128,0),(0,128,128),(0,128,1.5)]
        with mock.patch.object(np.random,'RandomState',side_effect=AssertionError('Invalid request drew noise')):
            for seed,total,known in invalid:
                with self.subTest(seed=seed,total=total,known=known),self.assertRaises(ValueError):
                    make_noise(seed,total,known)

    def test_euler_preserves_c_and_original_fp32_free_update(self):
        from native_t1.chair_sampling import euler_step
        state=torch.arange(128*9,dtype=torch.float32).reshape(1,128,9)/13
        velocity=(torch.arange(128*9,dtype=torch.float32).reshape(1,128,9)-400)/7
        C=torch.full((3,9),-2.25,dtype=torch.float32)
        before=state.clone();velocity_before=velocity.clone();condition_before=C.clone()
        expected=state.clone()
        expected[:,3:]=state[:,3:]+velocity[:,3:]*(1/50)
        expected[:,:3]=C[None]
        result=euler_step(state,velocity,C,3)
        self.assertTrue(torch.equal(result,expected))
        self.assertTrue(torch.equal(result[0,:3],C))
        self.assertEqual(result.dtype,torch.float32)
        self.assertTrue(torch.equal(state,before))
        self.assertTrue(torch.equal(velocity,velocity_before))
        self.assertTrue(torch.equal(C,condition_before))
        self.assertNotEqual(result.data_ptr(),state.data_ptr())


if __name__ == '__main__':
    unittest.main()
