"""Bounded CPU implementation tests, explicitly synthetic; zero optimizer updates."""
import ast
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from native_t1 import runtime
import torch
from models.equidit import DiT
from native_t1.artifacts import atomic_json
from native_t1.model import native_execution
from native_t1.data import model_inputs
from native_t1.training import masked_fm_loss
from native_t1.context_geometry import attach_context_encoder
from native_t1.losses import prepare_targets
from native_t1.objectives import GeometryObjective, objective_spec
from .model_mvp import MVPModel, velocity_mvp, integrate_mvp_step, json_hash
from .train_stream import collate, FrozenStream, equal_bits
from .train_binding import data_api
from .train_contract import ROOT
from .train_io import tree_hash, capture_rng, restore_rng

def fixture(n,k):
    generator=np.random.RandomState(n+k)
    x0=generator.randn(n,9).astype(np.float32)
    x1=generator.randn(n,9).astype(np.float32)
    context=np.zeros_like(x0);context[:k]=x1[:k]
    return dict(x0=x0,x1=x1,u=x1-x0,context=context,t=np.asarray(.6,np.float32),
        y=np.asarray(n,np.int64),known_mask=np.arange(n)<k,valid_mask=np.ones(n,bool))

class Units(unittest.TestCase):
    def test_sources_parse(self):
        for path in Path(__file__).parent.glob('*.py'):
            if path.name.startswith(('train_','model_')): ast.parse(path.read_text())

    def test_variable_fm_denominator_and_gradient(self):
        samples=[fixture(128,26),fixture(256,51)]
        batch=collate(samples,pad_to=259)
        self.assertEqual(batch['y'].tolist(),[128,256])
        denominator=9*((128-26)+(256-51))
        velocity=torch.zeros_like(batch['u'],requires_grad=True)
        joint=masked_fm_loss(velocity,batch['u'],batch['valid_mask'],batch['known_mask'])
        pieces=[]
        for s in samples:
            b=collate([s])
            pieces.append(masked_fm_loss(torch.zeros_like(b['u']),b['u'],b['valid_mask'],b['known_mask'],denominator=denominator))
        torch.testing.assert_close(joint,sum(pieces),atol=1e-6,rtol=1e-6)
        joint.backward()
        expected=(torch.ones_like(velocity)/denominator)*2*(velocity.detach()-batch['u'])
        expected[~batch['loss_mask']]=0
        torch.testing.assert_close(velocity.grad,expected,atol=0,rtol=0)
        self.assertTrue(torch.equal(batch['xt'][batch['known_mask']],batch['context'][batch['known_mask']]))
        with self.assertRaises(ValueError): collate(samples,pad_to=128)

    def test_original_auxiliary_gate_and_output_gradient(self):
        gt=torch.tensor([[[1.,0,0],[1,1,0],[0,1,0]]])
        known=torch.tensor([[[0.,0,0],[1,0,0],[0,1,0]]])
        target=prepare_targets(dict(free_gt=gt,known=known,free_ids=np.array([[1,3,2]]),
            known_ids=np.array([[0,1,2]]),bbox_L=2**.5))
        obj=GeometryObjective.__new__(GeometryObjective);obj.spec=objective_spec('edge5')
        velocity=torch.full((1,2,9),.1,requires_grad=True)
        batch=dict(t=torch.tensor([.5]),xt=torch.cat((known,gt)).reshape(1,2,9),
            valid_mask=torch.ones(1,2,dtype=torch.bool),known_mask=torch.tensor([[True,False]]))
        loss,diagnostic=obj.auxiliary(velocity,batch,target,additional_step=1)
        self.assertTrue(diagnostic['active'])
        self.assertEqual(diagnostic['ramp'],.02)
        raw=diagnostic['raw_losses']
        expected=.02/8*(4.071385484299878*raw['D_edge']+.3266104383520167*raw['L5'])
        self.assertAlmostEqual(float(loss),expected,places=8)
        loss.backward()
        self.assertTrue(torch.isfinite(velocity.grad).all())
        self.assertGreater(float(velocity.grad.norm()),0)
        self.assertTrue(torch.equal(velocity.grad[:,:1],torch.zeros_like(velocity.grad[:,:1])))
        batch['t']=torch.tensor([.499])
        inactive,diag=obj.auxiliary(velocity,batch,target,additional_step=50)
        self.assertFalse(diag['active']);self.assertEqual(float(inactive),0.)

    def test_cpu_variable_model_masks_final_time_zero_geo(self):
        runtime.configure_stable_runtime()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(713)
            # Explicit unit fixture only: one layer, random weights; never a formal initialization.
            backbone=DiT(hidden_dim=768,num_layers=1,num_heads=12,max_length=800,
                version=3,gradient_checkpointing=False,face_cond=True,face_bin=20,
                use_dit_like_pe=False,use_qknorm=True,use_rmsnorm=True,pe_freq=20)
            with torch.no_grad():
                backbone.final_layer.linear.weight.normal_(0,.01)
                backbone.layers[0].adaLN_modulation[-1].weight.normal_(0,.005)
            model=MVPModel(backbone,trainable=True)
            samples=[fixture(128,26),fixture(144,29)]
            b=collate(samples,pad_to=147)
            final=[]
            def observe(record):
                if record['site']=='final':
                    known=b['known_mask'];times=record['face_time']
                    self.assertTrue(torch.equal(times[known],torch.ones_like(times[known])))
                    self.assertTrue(torch.equal(times[~known],b['t'][:,None].expand_as(times)[~known]))
                    final.append(True)
            with torch.no_grad(),native_execution(model,coordinates=b['xt'],autocast=False):
                result=model(*model_inputs(b),observer=observe)
            self.assertEqual(final,[True]);self.assertTrue(torch.isfinite(result).all())
            self.assertGreater(float(result[b['known_mask']].abs().max()),0)
            rng=torch.random.get_rng_state().clone()
            attach_context_encoder(model,'geo',seed=1010)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            with torch.no_grad(): zero=velocity_mvp(model,*model_inputs(b))
            self.assertTrue(torch.equal(result,zero))
            altered=copy.deepcopy(b);altered['xt'][~b['valid_mask']]=127.
            with torch.no_grad(): masked=velocity_mvp(model,*model_inputs(altered))
            torch.testing.assert_close(zero[b['valid_mask']],masked[b['valid_mask']],atol=0,rtol=0)
            with torch.no_grad():
                integrated=integrate_mvp_step(model,b['xt'],b['t'],.02,b['y'],b['valid_mask'],b['known_mask'],b['context'])
            self.assertTrue(torch.equal(integrated[b['known_mask']],b['context'][b['known_mask']]))
            self.assertTrue(torch.equal(integrated[~b['valid_mask']],b['xt'][~b['valid_mask']]))

    def test_stateless_ot_replay_and_source_corner_labels(self):
        api=data_api(ROOT/'data/load_sample.py')
        stream=FrozenStream.__new__(FrozenStream)
        stream.api=api;stream.seed=10;stream.batch_index=0
        stream.binding=dict(plan=dict(sha256='a'*64))
        stream.counts=dict(actual_OT_attempts=0,actual_OT_returns=0,OT_cache_hits=0)
        n=128;k=26;gen=np.random.RandomState(71)
        V=gen.randn(n*3,3).astype(np.float32);F=np.arange(n*3).reshape(n,3)
        pre=(V[F]*np.float32(.5));Y=(pre*np.float32(2)).astype(np.float32)
        stream.objects={'fixture':dict(V=V,F=F,pre=pre,Y=Y,bbox_L=float(np.linalg.norm(np.ptp(V,axis=0))))}
        stream.cases={('fixture','p0'):dict(object_id='fixture',patch_id='p0',known_ids=np.arange(k))}
        rec=dict(uid='fixture',patch_id='p0',gaussian_seed=803,t_seed=904,permutation_seed=1001)
        before=capture_rng()
        first=stream._sample(rec);state=stream.state_dict();second=stream._sample(rec)
        self.assertEqual(first['shared_input_sha256'],second['shared_input_sha256'])
        self.assertEqual(tree_hash(before),tree_hash(capture_rng()))
        stream.batch_index=200;stream.load_state_dict(state)
        self.assertEqual(stream.batch_index,0)
        label=stream.label(first)
        ids=np.take_along_axis(F[first['source_face_ids']],first['corner_permutations'],axis=1)
        self.assertTrue(equal_bits(V[ids][~first['known_mask']],label['free_gt']))
        self.assertEqual(stream.counts['actual_OT_returns'],2)
        self.assertEqual(stream.counts['OT_cache_hits'],0)

    def test_checkpoint_metadata_rejects_state_tampering(self):
        from .model_mvp import validate_metadata,SCHEMA
        from .train_contract import PROTOCOL
        from native_t1.portable_checkpoint import OFFICIAL_FILE_SHA256,OFFICIAL_BACKBONE_SHA256,config_sha256
        from native_t1.checkpoint import DEFAULT_CONFIG
        protocol=dict(PROTOCOL,data_ready_sha256='a'*64,binding_sha256='b'*64,plan_sha256='c'*64)
        opt=dict(param_groups=[dict(lr=1e-5,betas=(.9,.95),weight_decay=0)],state={0:dict(step=torch.tensor(500.),exp_avg=torch.zeros(2))})
        rng=capture_rng();stream=dict(batch_index=500)
        meta=dict(variant='Geo_shared',logical_step=500,context_encoder='geo',
            official_file_sha256=OFFICIAL_FILE_SHA256,official_backbone_sha256=OFFICIAL_BACKBONE_SHA256,
            protocol_sha256=json_hash(protocol),config_sha256=config_sha256(DEFAULT_CONFIG),
            optimizer_state_sha256=tree_hash(opt),rng_sha256=tree_hash(rng),stream_state_sha256=json_hash(stream),
            executed_phase_updates=dict(T1=1000,Geo_shared=500,Geo=0,Full=0),actual_updates_at_save=1500,
            loss_recipe='fm',optimizer_reset_at_phase_start=True)
        payload=dict(schema=SCHEMA,metadata=meta,protocol=protocol,model={},optimizer=opt,rng=rng,stream_state=stream,update_in_progress=False)
        self.assertEqual(validate_metadata(payload),meta)
        corrupt=copy.deepcopy(payload);corrupt['optimizer']['state'][0]['exp_avg'][0]=1
        with self.assertRaises(ValueError):validate_metadata(corrupt)
        corrupt=copy.deepcopy(payload);corrupt['metadata']['actual_updates_at_save']=2000
        with self.assertRaises(ValueError):validate_metadata(corrupt)
        corrupt=copy.deepcopy(payload);corrupt['metadata']['variant']='Full'
        with self.assertRaises(ValueError):validate_metadata(corrupt)

    def test_rng_roundtrip_no_cuda(self):
        before=capture_rng()
        torch.rand(3);np.random.randn(3)
        restore_rng(before)
        self.assertEqual(tree_hash(before),tree_hash(capture_rng()))
        self.assertFalse(torch.cuda.is_initialized())

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    ROOT.joinpath('training').mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Units)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    from .train_run import source_files
    from native_t1.artifacts import file_sha256
    from .train_run import REPO
    summary=dict(source_hashes={str(p.relative_to(REPO)):file_sha256(p) for p in source_files()},status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=[dict(test=str(test),traceback=text) for test,text in result.failures+result.errors],
        scope='CPU synthetic implementation fixtures only; no real-data or official-weight gate claimed',
        optimizer_updates=0,CUDA_initialized=torch.cuda.is_initialized(),
        actual_CPU_OT_calls=2,no_GPU_model_calls=True)
    atomic_json(ROOT/'training/cpu_tests.json',summary)
    if not result.wasSuccessful(): raise SystemExit(1)

if __name__=='__main__': main()
