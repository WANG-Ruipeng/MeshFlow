"""Small CPU-only tests; no official model, OT, optimizer, or CUDA execution."""
import ast
import copy
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from native_t1.context_geometry import face_geometry, FORMULA_ATOL, FORMULA_RTOL
from native_t1.geometry_representation import (
    descriptor13, GeometryTargets, AuxiliaryHeads, auxiliary_losses, GROUP_SLICES,
)

COUNTS=dict(descriptor_calls=0,legacy_geometry_calls=0,target_fit_calls=0,target_lookup_calls=0,
            auxiliary_head_forwards=0,CPU_autograd_grad_calls=0,official_model_forwards=0,
            OT_calls=0,optimizer_updates=0)


def describe(x):
    COUNTS['descriptor_calls']+=1
    return descriptor13(x)


def fit(archive):
    COUNTS['target_fit_calls']+=1
    return GeometryTargets.from_archive(archive)


def lookup(targets,sample):
    COUNTS['target_lookup_calls']+=1
    return targets.for_sample(sample)


def heads_forward(heads,h,valid,known):
    COUNTS['auxiliary_head_forwards']+=1
    return heads(h,valid,known)


def grad(*args,**kwargs):
    COUNTS['CPU_autograd_grad_calls']+=1
    return torch.autograd.grad(*args,**kwargs)


def archive_fixture():
    cases={}
    for parent in (0,1):
        triangles=[]
        for face in range(16):
            size=np.float32(.25+face/32)
            origin=np.asarray([face/8,parent/2,face/16],np.float32)
            triangles.append(origin+np.asarray([[0,0,0],[size,0,0],[0,size*1.5,0]],np.float32))
        full=np.stack(triangles).reshape(16,9)
        for K in (4,8,12):
            free=np.arange(K,16,dtype=np.int64)
            cases[parent,K]=SimpleNamespace(full_target=full,free_target=full[free],free_source_ids=free)
    return SimpleNamespace(cases=cases,sha256='a'*64)


def sample_fixture(archive,parent=0):
    full=archive.cases[parent,4].full_target
    order=np.asarray([7,0,15,2,9,1,8,3,4,14,5,13,6,12,10,11])
    known=order<4
    triangles=full[order].reshape(16,3,3).copy()
    for i in range(16):triangles[i]=triangles[i,list(tuple(itertools.permutations(range(3)))[i%6])]
    y=triangles.reshape(16,9);y[known]=np.nan
    return dict(parent_index=parent,source_face_ids=order,x1=y,known_mask=known,valid_mask=np.ones(16,bool))


class GeometryRepresentationTests(unittest.TestCase):
    def test_single_face_formula_S3_face_order_and_no_adjacency(self):
        c=np.asarray([[[0,0,0],[1,.2,.1],[.1,1,.3]],[[2,1,0],[2.7,1.1,.2],[2.1,1.8,.7]]],np.float32)
        raw=describe(c)
        for order in itertools.permutations(range(3)):
            actual=describe(c[:,order,:])
            np.testing.assert_array_equal(actual['raw'],raw['raw'])
            np.testing.assert_array_equal(actual['transformed'],raw['transformed'])
        np.testing.assert_array_equal(describe(c[::-1].copy())['raw'],raw['raw'][::-1])
        COUNTS['legacy_geometry_calls']+=1
        original,_,_=face_geometry(torch.from_numpy(c))
        np.testing.assert_allclose(raw['raw'],original.numpy(),atol=FORMULA_ATOL,rtol=FORMULA_RTOL)
        independent=[]
        for tri in c.astype(np.float64):
            cross=np.cross(tri[1]-tri[0],tri[2]-tri[0]);area=np.linalg.norm(cross)/2
            nx,ny,nz=cross/(2*area)
            independent.append(np.r_[tri.mean(0),sorted(np.linalg.norm(tri[i]-tri[j]) for i,j in ((0,1),(0,2),(1,2))),area,
                                       nx*nx,ny*ny,nz*nz,nx*ny,nx*nz,ny*nz])
        np.testing.assert_allclose(raw['raw'],independent,atol=1e-12,rtol=1e-12)

    def test_true_degenerate_rejected_but_log_floor_not_geometry_repair(self):
        for c in (np.zeros((1,3,3),np.float32),np.full((1,3,3),np.nan,np.float32),
                  np.asarray([[[0,0,0],[1,0,0],[2,0,0]]],np.float32)):
            with self.assertRaises(ValueError):describe(c)
        small=describe(np.asarray([[[0,0,0],[1e-8,0,0],[0,1e-8,0]]],np.float32))
        self.assertEqual(small['area_floor_faces'],1)
        self.assertGreater(small['raw'][0,6],0)
        self.assertEqual(small['transformed'][0,6],np.log(1e-12))

    def test_deduplicated_FP64_statistics_and_metadata_roundtrip(self):
        archive=archive_fixture();targets=fit(archive);meta=targets.metadata()
        self.assertEqual(meta['unique_faces'],24)
        self.assertEqual(sum(v['free_faces'] for v in meta['fit_cases']),48)
        all_faces=np.concatenate([archive.cases[p,4].free_target for p in (0,1)])
        r=describe(all_faces)['transformed']
        for name,sl in GROUP_SLICES.items():
            mu=r[:,sl].mean(0);scale=np.sqrt(np.mean((r[:,sl]-mu)**2));width=sl.stop-sl.start
            np.testing.assert_array_equal(meta['groups'][name]['mean'],mu)
            self.assertEqual(meta['groups'][name]['centered_RMS'],scale)
            self.assertEqual(meta['groups'][name]['denominator'],max(scale,1e-6)*np.sqrt(width))
        self.assertTrue(meta['groups']['normal_outer']['scale_floor_active'])
        restored=GeometryTargets.from_metadata(meta)
        self.assertEqual(restored.metadata(),meta)
        bad=copy.deepcopy(meta);bad['groups']['centroid']['mean'][0]+=1
        with self.assertRaises(ValueError):GeometryTargets.from_metadata(bad)
        self.assertEqual(targets.metadata(),meta)

    def test_actual_slot_mapping_S3_known_placeholders_and_no_target_grad(self):
        archive=archive_fixture();targets=fit(archive);sample=sample_fixture(archive)
        q=lookup(targets,sample)
        self.assertEqual(q['free_count'],12);self.assertEqual(q['known_targets_evaluated'],0)
        self.assertFalse(q['q'].requires_grad)
        meta=targets.metadata();by_id={v['source_face_id']:v['q'] for v in meta['source_samples'] if v['parent']==0}
        np.testing.assert_array_equal(q['q'].numpy(),np.asarray([by_id[i] for i in sample['source_face_ids'][~sample['known_mask']]],np.float32))
        reordering=np.arange(16)[::-1]
        permuted={k:(v[reordering].copy() if isinstance(v,np.ndarray) else v) for k,v in sample.items()}
        actual=lookup(targets,permuted)
        np.testing.assert_array_equal(actual['q'].numpy(),q['q'].numpy()[::-1])
        bad=copy.deepcopy(sample);bad['x1'][0,0]+=.01
        with self.assertRaises(ValueError):lookup(targets,bad)
        padded=copy.deepcopy(sample);padded['valid_mask'][0]=False;padded['x1'][0]=np.nan
        self.assertEqual(lookup(targets,padded)['free_count'],11)

    def test_projector_initialization_matches_and_preserves_RNG(self):
        before=torch.get_rng_state().clone()
        angular=AuxiliaryHeads(False);both=AuxiliaryHeads(True)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertIsNone(angular.S)
        for name,value in angular.P.state_dict().items():self.assertTrue(torch.equal(value,both.P.state_dict()[name]))
        self.assertGreater(int(torch.count_nonzero(angular.P[-1].weight)),0)
        self.assertEqual(angular.metadata()['P_parameters'],201741)
        self.assertEqual(both.metadata()['S_parameters'],1741)
        self.assertEqual(both.metadata()['total_parameters'],203482)
        again=AuxiliaryHeads(True);again.load_state_dict(both.state_dict(),strict=True)
        for name,value in both.state_dict().items():self.assertTrue(torch.equal(value,again.state_dict()[name]))

    def test_FP32_pooling_scattered_masks_and_head_face_permutation(self):
        heads=AuxiliaryHeads(True)
        h=torch.linspace(-2,2,1*18*768).reshape(1,18,768).to(torch.bfloat16)
        known=torch.tensor([[True,False,True,False,False,True]])
        valid=torch.tensor([[True,True,True,True,False,True]])
        with torch.autocast('cpu',dtype=torch.bfloat16):actual=heads_forward(heads,h,valid,known)
        expected=h.float().reshape(1,6,3,768).mean(2)[valid&~known]
        self.assertTrue(torch.equal(actual['H_face'],expected))
        self.assertEqual(actual['z'].dtype,torch.float32);self.assertEqual(actual['q_hat'].dtype,torch.float32)
        order=torch.tensor([5,3,0,4,1,2]);hp=h.reshape(1,6,3,768)[:,order][:,:,[2,0,1]].reshape_as(h)
        other=heads_forward(heads,hp,valid[:,order],known[:,order])
        torch.testing.assert_close(other['z'],actual['z'].flip(0),atol=2e-6,rtol=2e-5)

    def test_angular_scale_gradients_and_detached_GT(self):
        heads=AuxiliaryHeads(True)
        h=torch.linspace(-1,1,18*768).reshape(1,18,768).clone().requires_grad_()
        known=torch.tensor([[True,False,True,False,False,False]]);valid=torch.ones_like(known)
        predictions=heads_forward(heads,h,valid,known)
        q=torch.linspace(-.8,.9,4*13).reshape(4,13).clone().requires_grad_()
        losses=auxiliary_losses(predictions,q)
        variables=(h,heads.P[1].weight,heads.S[0].weight,q)
        angular=grad(losses['angular'],variables,retain_graph=True,allow_unused=True)
        scale=grad(losses['scale'],variables,allow_unused=True)
        self.assertIsNone(angular[2]);self.assertIsNone(angular[3]);self.assertIsNone(scale[3])
        for collection in (angular[:2],scale[:3]):
            for value in collection:self.assertTrue(bool(torch.isfinite(value).all()));self.assertGreater(float(value.norm()),0)
        mask=known[:,:,None,None].expand(1,6,3,768).reshape_as(h)
        self.assertTrue(torch.equal(scale[0][mask],torch.zeros_like(scale[0][mask])))

    def test_loss_oracle_zero_direction_and_full_batch_denominators(self):
        q=torch.zeros(2,13);q[0,0]=1
        z=q.clone().requires_grad_();normalized=F.normalize(z,dim=-1,eps=1e-6)
        p=dict(z=z,z_norm=normalized,q_hat=q.clone())
        out=auxiliary_losses(p,q)
        self.assertEqual(out['angular_count'],1);self.assertEqual(out['invalid_angular_count'],1)
        self.assertEqual(float(out['angular']),0);self.assertEqual(float(out['scale']),0)
        zeros=torch.zeros_like(q)
        with self.assertRaises(ValueError):auxiliary_losses(p,zeros)
        local=auxiliary_losses(p,zeros,angular_denominator=3,free_denominator=8)
        g,=grad(local['angular'],(z,))
        self.assertTrue(torch.equal(g,torch.zeros_like(g)))
        z=torch.linspace(-1,2,36*13).reshape(36,13).clone().requires_grad_()
        q=torch.cos(torch.arange(36*13,dtype=torch.float32)).reshape(36,13);q[0]=0
        norm=F.normalize(z,dim=-1,eps=1e-6);p=dict(z=z,z_norm=norm,q_hat=norm*.7+.2)
        full=auxiliary_losses(p,q);terms=[];start=0
        for size in range(1,9):
            stop=start+size
            terms.append(auxiliary_losses({k:v[start:stop] for k,v in p.items()},q[start:stop],
                         angular_denominator=35,free_denominator=36))
            start=stop
        angular=sum(v['angular'] for v in terms);scale=sum(v['scale'] for v in terms)
        torch.testing.assert_close(angular,full['angular'],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(scale,full['scale'],atol=1e-7,rtol=1e-6)
        a,=grad(full['angular']+full['scale'],(z,),retain_graph=True)
        b,=grad(angular+scale,(z,))
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
        with self.assertRaises(ValueError):auxiliary_losses(p,q,angular_denominator=34,free_denominator=36)
        with self.assertRaises(ValueError):auxiliary_losses(p,q,angular_denominator=35,free_denominator=35)

    def test_feature_api_static_reference_capture_not_a_new_model_input(self):
        source=Path(__file__).parents[1]/'model.py'
        tree=ast.parse(source.read_text(encoding='utf-8-sig'))
        cls=next(v for v in tree.body if isinstance(v,ast.ClassDef) and v.name=='NativeInpaintingModel')
        forward=next(v for v in cls.body if isinstance(v,ast.FunctionDef) and v.name=='forward')
        self.assertEqual([v.arg for v in forward.args.kwonlyargs],['observer','return_feature_layer'])
        self.assertIsNone(forward.args.kw_defaults[-1].value)
        captures=[v for v in ast.walk(forward) if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='feature' for t in v.targets)]
        self.assertEqual(len(captures),2)
        self.assertTrue(any(isinstance(v.value,ast.Name) and v.value.id=='hidden' for v in captures))
        # Real-model bitwise/reference-pointer and upstream gradient gates are
        # deliberately owned by the separately counted GPU preflight.


if __name__=='__main__':unittest.main()
