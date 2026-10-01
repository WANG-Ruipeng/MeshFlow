"""CPU contract tests: no real generator, CUDA, training or generation."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from native_t1 import geometry_checkpoint as gc, train_cli
from native_t1 import __main__ as cli
from native_t1.objectives import NAMES, GeometryObjective, objective_spec, validate_objective
from native_t1.tests.test_geometry_checkpoint import valid_payload


class RecipeTests(unittest.TestCase):
    def test_loss_menu_uses_registered_names_and_defaults_to_fm(self):
        options = ['--inputs', 'unread.npz', '--init-geo', 'unread.pt',
                   '--context-encoder', 'geo', '--out', 'uncreated', '--updates', '500']
        parser = train_cli.build_parser()
        self.assertEqual(parser.parse_args(options).loss_recipe, 'fm')
        menu = next(action for action in parser._actions if action.dest == 'loss_recipe')
        self.assertEqual(tuple(menu.choices), NAMES)
        for name in NAMES:
            self.assertEqual(parser.parse_args(options + ['--loss-recipe', name]).loss_recipe, name)
            self.assertEqual(validate_objective(objective_spec(name))['name'], name)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(options + ['--loss-recipe', 'unregistered_candidate'])

    def test_schema_preserves_loss_identity_and_rejects_changed_recipe(self):
        for recipe in ('surface', 'edge5'):
            payload = valid_payload()
            payload.update(schema=gc.OBJECTIVE_SCHEMA, loss='masked_free_fm_plus_geometry',
                           training_objective=objective_spec(recipe))
            self.assertEqual(gc.validate_geometry_checkpoint(payload), payload['context'])
            for field, value in [('lambda_edge', 1.), ('time_gate', 't>=0.4'),
                                 ('auxiliary_denominator', 'active_samples')]:
                wrong = copy.deepcopy(payload)
                wrong['training_objective'][field] = value
                with self.assertRaises(ValueError):
                    gc.validate_geometry_checkpoint(wrong)
            payload['schema'] = gc.SCHEMA
            with self.assertRaises(ValueError):
                gc.validate_geometry_checkpoint(payload)

    def test_pure_schema_rejects_forged_geometry_identity(self):
        payload = valid_payload()
        payload["training_objective"] = {"name": "edge5"}
        with self.assertRaisesRegex(ValueError, "Pure FM schema"):
            gc.validate_geometry_checkpoint(payload)

    def test_main_dispatch_chooses_one_fixed_recipe(self):
        with mock.patch('native_t1.train_cli.main', return_value='edge') as entry:
            self.assertEqual(cli.main(['train-edge5', '--check-only']), 'edge')
            entry.assert_called_once_with(['--check-only'], recipe='edge5')
        options=['--inputs','unread.npz','--init-geo','unread.pt','--out','uncreated', '--updates','500']
        with mock.patch.object(train_cli,'run',return_value='configured') as run:
            self.assertEqual(train_cli.main(options,recipe='edge5'),'configured')
            args=run.call_args.args[0]
            self.assertEqual((args.loss_recipe,args.context_encoder,args.updates),('edge5','geo',500))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train_cli.main(options+['--loss-recipe','fm'],recipe='edge5')

    def test_gate_ramp_all_eight_denominator_and_no_C_gradient(self):
        objective=GeometryObjective.__new__(GeometryObjective)
        objective.spec=objective_spec('edge5')
        v=torch.arange(18,dtype=torch.float32).reshape(1,2,9).requires_grad_()
        known=torch.tensor([[True,False]])
        batch=dict(t=torch.tensor([.5]),xt=torch.zeros_like(v),known_mask=known,
                   valid_mask=torch.ones_like(known))
        def evaluate(name,pred,target):
            return dict(loss=pred.square().mean(),counts={},applicable=True)
        with mock.patch('native_t1.losses.evaluate',side_effect=evaluate) as fn:
            value,diag=objective.auxiliary(v,batch,{'D_edge':None,'L5':None},additional_step=25)
            expected=.5*(objective.spec['lambda_edge']+objective.spec['lambda_surface'])*(.5*v[:,1]).square().mean()/8
            torch.testing.assert_close(value,expected,rtol=0,atol=0)
            gradient=torch.autograd.grad(value,v)[0]
            self.assertEqual(int(torch.count_nonzero(gradient[:,0])),0)
            self.assertGreater(int(torch.count_nonzero(gradient[:,1])),0)
            self.assertEqual(fn.call_count,2)
            batch['t']=torch.tensor([.499])
            off,diag=objective.auxiliary(v,batch,{},additional_step=500)
            self.assertEqual(float(off),0.)
            self.assertEqual(fn.call_count,2)
            self.assertEqual(int(torch.count_nonzero(torch.autograd.grad(off,v)[0])),0)

    def test_surface_ablation_only_calls_surface_term(self):
        objective=GeometryObjective.__new__(GeometryObjective)
        objective.spec=objective_spec('surface')
        v=torch.ones((1,2,9),requires_grad=True)
        batch=dict(t=torch.tensor([.75]),xt=torch.zeros_like(v),known_mask=torch.tensor([[True,False]]),valid_mask=torch.ones((1,2),dtype=torch.bool))
        with mock.patch('native_t1.losses.evaluate',return_value=dict(loss=v[:,1].square().mean(),counts={})) as fn:
            _,d=objective.auxiliary(v,batch,{'L5':None},additional_step=500)
            self.assertEqual(fn.call_count,1)
            self.assertEqual(fn.call_args.args[0],'L5')

    def test_working_profile_rejects_pure_checkpoint_before_model_construction(self):
        from native_t1.working_model import load_edge5
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'pure.pt';torch.save(valid_payload(),path)
            with mock.patch.object(gc,'NativeInpaintingModel',side_effect=AssertionError('model built')):
                with self.assertRaisesRegex(ValueError,'requested training recipe'):
                    load_edge5(path,device='cpu')


if __name__=='__main__':
    unittest.main()
