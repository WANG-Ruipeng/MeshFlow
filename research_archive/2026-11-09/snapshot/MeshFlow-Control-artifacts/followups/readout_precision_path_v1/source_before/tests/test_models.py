"""Compact CPU-only model tests; no weights, CUDA, or external data."""
import itertools
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from meshflow_control.models.readout import (CornerReadout, make_readouts,
    apply_readout, gather_known_corners, corner_indices, READOUT_PARAMETERS)
from meshflow_control.models.native import NativeModel, build_model
from meshflow_control.models.backbone.equidit import XEmbedder

CONFIG = dict(hidden_dim=768, num_heads=12, max_length=800, num_layers=12,
    gradient_checkpointing=False, use_coord_encoding=True, version=3, pe_freq=20,
    mixed_precision="bf16", use_dit_like_pe=False, face_cond=True, face_bin=20,
    use_rmsnorm=True, use_qknorm=True)


def nonzero_module(dtype=torch.float64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(911)
        module = CornerReadout(12, 8, 2).to(dtype)
        nn.init.normal_(module.W_O.weight, std=.07)
    return module


def literal(module, query, memory):
    q0 = torch.nn.functional.layer_norm(query, (12,), eps=1e-5)
    m0 = torch.nn.functional.layer_norm(memory, (12,), eps=1e-5)
    q = (q0 @ module.W_Q.weight.T).reshape(-1, 2, 4)
    k = (m0 @ module.W_K.weight.T).reshape(-1, 2, 4)
    v = (m0 @ module.W_V.weight.T).reshape(-1, 2, 4)
    outputs=[]
    for qi in range(query.shape[0]):
        heads=[]
        for h in range(2):
            weights=(q[qi,h] @ k[:,h].T / 2).softmax(-1)
            heads.append(weights @ v[:,h])
        outputs.append(torch.cat(heads) @ module.W_O.weight.T)
    return torch.stack(outputs)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initialization_capacity_rng_and_modes(self):
        before=torch.get_rng_state().clone()
        b=make_readouts(20261007); c=make_readouts(20261007)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in b.parameters()),READOUT_PARAMETERS)
        self.assertEqual(len(list(b.parameters())),16)
        for (name,p),(_,q) in zip(b.named_parameters(),c.named_parameters()):
            self.assertTrue(torch.equal(p,q))
            self.assertEqual(bool(torch.count_nonzero(p)),not name.endswith('W_O.weight'))
        self.assertFalse(torch.cuda.is_initialized())

    def test_meta_factory_original_keys_and_counts(self):
        before=torch.get_rng_state().clone()
        with torch.device('meta'):
            a=build_model(CONFIG,readout_mode='none')
            b=build_model(CONFIG,readout_mode='mixed')
            c=build_model(CONFIG,readout_mode='clean')
        old={n:p for n,p in a.named_parameters()}
        self.assertEqual(len(old),194)
        self.assertEqual(sum(p.numel() for p in old.values()),130401155)
        self.assertEqual(list(old),[n for n,_ in b.named_parameters() if not n.startswith('readouts.')])
        self.assertEqual(sum(p.numel() for p in b.parameters()),132760451)
        self.assertEqual(list(b.state_dict()),list(c.state_dict()))
        for name,p in b.readouts.named_parameters():
            self.assertTrue(torch.equal(p,c.readouts.get_parameter(name)))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(a.backbone.x_embedder.embed_fn(torch.zeros(1,1,3)).device.type,'cpu')
        with self.assertRaises(ValueError):build_model(dict(CONFIG,num_layers=13))
        self.assertFalse(torch.cuda.is_initialized())

    def test_fp64_literal_values_and_all_gradients(self):
        m=nonzero_module(); gen=torch.Generator().manual_seed(22)
        q=torch.randn(5,12,generator=gen,dtype=torch.float64,requires_grad=True)
        c=torch.randn(6,12,generator=gen,dtype=torch.float64,requires_grad=True)
        target=torch.randn(5,12,generator=gen,dtype=torch.float64)
        actual=m.forward_reference(q,c); expected=literal(m,q,c)
        torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-12)
        tensors=(q,c,*m.parameters())
        ga=torch.autograd.grad((actual*target).sum(),tensors,retain_graph=True)
        ge=torch.autograd.grad((expected*target).sum(),tensors)
        for a,e in zip(ga,ge):torch.testing.assert_close(a,e,rtol=2e-11,atol=2e-11)

    def test_face_and_all_six_corner_permutations(self):
        m=nonzero_module(); gen=torch.Generator().manual_seed(33)
        h=torch.randn(1,21,12,generator=gen,dtype=torch.float64)
        known=torch.tensor([[1,0,1,0,0,0,0]],dtype=torch.bool)
        valid=torch.tensor([[1,1,1,1,1,0,0]],dtype=torch.bool)
        base=apply_readout(h,known,valid,m,reference=True).reshape(1,7,3,12)
        faceperm=torch.tensor([4,2,6,0,3,5,1])
        for order in itertools.permutations(range(3)):
            cp=torch.tensor(order)
            hp=h.reshape(1,7,3,12)[:,faceperm][:,:,cp].reshape(1,21,12)
            actual=apply_readout(hp,known[:,faceperm],valid[:,faceperm],m,reference=True)
            expected=base[:,faceperm][:,:,cp].reshape_as(actual)
            torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-12)

    def test_padding_content_length_and_gradient_isolation(self):
        m=nonzero_module(); h=torch.randn(1,21,12,dtype=torch.float64,requires_grad=True)
        known=torch.tensor([[1,0,1,0,0,0,0]],dtype=torch.bool)
        valid=torch.tensor([[1,1,1,1,1,0,0]],dtype=torch.bool)
        actual=apply_readout(h,known,valid,m,reference=True)
        other=h.detach().clone();other[:,15:]=1000
        changed=apply_readout(other,known,valid,m,reference=True)
        self.assertTrue(torch.equal(actual[:,:15],changed[:,:15]))
        longer=torch.cat((h.detach(),torch.randn(1,12,12,dtype=torch.float64)),dim=1)
        kp=torch.cat((known,torch.zeros(1,4,dtype=torch.bool)),1)
        vp=torch.cat((valid,torch.zeros(1,4,dtype=torch.bool)),1)
        long=apply_readout(longer,kp,vp,m,reference=True)
        self.assertTrue(torch.equal(actual,long[:,:21]))
        grad=torch.autograd.grad(actual[:,:15].square().sum(),h)[0]
        self.assertTrue(torch.equal(grad[:,15:],torch.zeros_like(grad[:,15:])))
        # Invalid known flags must not become context in the compact helper.
        invalid=known.clone();invalid[:,5:]=True
        self.assertTrue(torch.equal(actual,apply_readout(h,invalid,valid,m,reference=True)))
        untouched=corner_indices(~(valid[0]&~known[0]))
        self.assertTrue(torch.equal(actual[0,untouched],h[0,untouched]))

    def test_empty_sets_defined(self):
        m=nonzero_module();h=torch.randn(1,12,12,dtype=torch.float64)
        valid=torch.ones(1,4,dtype=torch.bool)
        for known in (torch.zeros_like(valid),valid):
            self.assertTrue(torch.equal(h,apply_readout(h,known,valid,m,reference=True)))
        self.assertTrue(torch.equal(m.forward_reference(h[0],h[0,:0]),torch.zeros_like(h[0])))
        self.assertEqual(m.forward_reference(h[0,:0],h[0]).shape,(0,12))

    def test_zero_identity_bf16_and_exactly_two_isolated_updates(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9)
            m=CornerReadout(12,8,2)
            q=torch.randn(9,12);c=torch.randn(6,12);target=torch.randn(9,12)
        self.assertTrue(torch.equal(q,q+m(q,c)))
        records=[]
        with torch.autocast('cpu',dtype=torch.bfloat16):
            r=m(q.bfloat16(),c.bfloat16(),observer=records.append)
        self.assertEqual(r.dtype,torch.float32)
        self.assertEqual(records[0]['branch_dtype'],'torch.float32')
        self.assertTrue(torch.equal(q.bfloat16(),q.bfloat16()+r.bfloat16()))
        optimizer=torch.optim.SGD(m.parameters(),lr=.1)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            ((m(q,c)-target).square().mean()).backward()
            self.assertGreater(float(m.W_O.weight.grad.norm()),0)
            for module in (m.W_Q,m.W_K,m.W_V):
                norm=float(module.weight.grad.norm())
                self.assertEqual(norm==0,step==0)
            optimizer.step()
        self.assertGreater(float(m(q,c).norm()),0)

    def test_clean_memory_values_independence_and_encoder_gradients(self):
        class SmallBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.version=3;self.hidden_size=768;self.use_dit_like_pe=False;self.face_cond=True
                self.x_embedder=XEmbedder(768,2,version=3)
                self.layers=nn.ModuleList([nn.Identity() for _ in range(12)])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(75)
            model=NativeModel(SmallBackbone(),mode='clean')
            with torch.no_grad():nn.init.normal_(model.context_encoder.W_out.weight,std=.02)
            x=torch.randn(1,6,9,requires_grad=True)
        known=torch.tensor([[1,0,1,0,0,0]],dtype=torch.bool)
        valid=torch.tensor([[1,1,1,1,1,0]],dtype=torch.bool)
        h,_=model._embed(x,known,valid);memory=gather_known_corners(h,known,valid)
        other=x.detach().clone();other[:,~known[0]]=100*other[:,~known[0]]
        h2,_=model._embed(other,known,valid);memory2=gather_known_corners(h2,known,valid)
        self.assertTrue(torch.equal(memory[0],memory2[0]))
        self.assertNotEqual(memory[0].data_ptr(),h.data_ptr())
        self.assertTrue(memory[0].requires_grad)
        memory[0].square().sum().backward()
        self.assertTrue(torch.equal(x.grad[:,~known[0]],torch.zeros_like(x.grad[:,~known[0]])))
        self.assertGreater(float(x.grad[:,known[0]].norm()),0)
        self.assertGreater(float(model.backbone.x_embedder.mlp[0].weight.grad.norm()),0)
        self.assertGreater(float(model.role_embedding.grad[1].norm()),0)
        self.assertEqual(float(model.role_embedding.grad[0].norm()),0)
        self.assertGreater(float(model.context_encoder.W_self.weight.grad.norm()),0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
