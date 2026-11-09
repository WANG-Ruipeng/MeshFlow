"""CPU tensor/identity tests; no generator construction or CUDA operations."""
from __future__ import annotations
import unittest
import numpy as np
from . import patch as P


def fixture():
    # 64 disjoint, clean two-triangle unit squares. First complete square is C.
    v=[]; faces=[]
    for i in range(64):
        base=len(v); x=float(i*3)
        v.extend([[x,0,0],[x+1,0,0],[x+1,1,0],[x,1,0]])
        faces.extend([[base,base+1,base+2],[base+2,base+3,base]])
    v=np.asarray(v,np.float32); f=np.asarray(faces,np.int64)
    # Nontrivial canonical permutations deliberately differ from winding.
    cp=np.tile([2,0,1],(len(f),1)); canonical=f[np.arange(len(f))[:,None],cp]
    return dict(model_vertices=v,source_faces=f,source_face_vertex_ids=canonical,
                source_corner_permutations=cp,full_target=v[canonical],pre_ot_full=v[canonical]*np.float32(.5),
                known_ids=np.arange(2,dtype=np.int64),free_ids=np.arange(2,128,dtype=np.int64),
                C_vertices=v[:4],C_faces_original=f[:2],C_faces_model=canonical[:2],
                occupied_labels=np.zeros(4,bool))


class PatchTests(unittest.TestCase):
    def test_roundtrip_deterministic_matching(self):
        case=fixture(); plan=P.build_source_plan(case)
        self.assertEqual(plan,P.build_source_plan(case)); self.assertEqual(plan['status'],'PASS')
        row=dict(alpha=.93,noise_seed=100,epsilon_seed=101,time_seed=102,permutation_seed=103)
        sample=P.make_sample(case,row,plan)
        again=P.make_sample(case,row,plan)
        for name in ('x0','x1','xt','u','source_face_ids','source_corner_permutations'):
            self.assertTrue(np.array_equal(sample[name],again[name]))
        self.assertEqual(int(sample['free_coordinate_count']),9*126-6*(126//8))
        self.assertEqual(P.verify_sample(sample)['patch_count'],15)
        for name in ('x0','x1','xt','u'): P.assert_shared(sample[name],sample)
        # Each paired face is a cyclic source order, never a reflected winding.
        source=case['source_faces']
        moved=case['source_face_vertex_ids'][sample['source_face_ids'][:,None],sample['source_corner_permutations']]
        for fi in np.flatnonzero(sample['patch_ids']>=0):
            sf=source[sample['source_face_ids'][fi]]
            self.assertTrue(any(np.array_equal(np.roll(sf,j),moved[fi]) for j in range(3)))

    def test_unit_noise_and_50_euler_shared_steps(self):
        case=fixture(); layout=P.inference_layout(case['C_vertices'],case['C_faces_original'],case['C_faces_model'],128)
        z=np.random.RandomState(8).normal(size=(126,3,3)).astype(np.float32)
        x=P.initial_state(layout,z); p=layout['shared_corner_pairs']; flat=z.reshape(-1,3)
        want=(flat[p[:,0]-6]+flat[p[:,1]-6])*np.float32(1/np.sqrt(2.))
        self.assertTrue(np.array_equal(x.reshape(-1,3)[p[:,0]],want))
        original_C=x[:2].copy()
        for step in range(50):
            v=np.random.RandomState(20+step).normal(size=x.shape).astype(np.float32)
            v=P.project_velocity(v,layout)
            x=P.project_coordinates(x+np.float32(.02)*v,layout);x[:2]=original_C
            P.assert_shared(x,layout)
        result=P.decode_indices(x,layout)
        self.assertTrue(np.array_equal(result['vertices'][result['faces']],x))
        self.assertTrue(np.array_equal(x[:2],original_C))

    def test_raw_schema_original_C_order(self):
        case=fixture(); layout=P.inference_layout(case['C_vertices'],case['C_faces_original'],case['C_faces_model'],128)
        z=np.random.RandomState(80).normal(size=(126,3,3)).astype(np.float32)
        x=P.initial_state(layout,z)
        source=dict(C_original_source_vertex_ids=np.arange(4,dtype=np.int64),
                    C_source_corner_permutations=case['source_corner_permutations'][:2])
        raw=P.raw_arrays(x,layout,np.zeros(len(layout['occupied']),np.float32),source)
        self.assertTrue(np.array_equal(raw['C'],case['model_vertices'][case['source_faces'][:2]]))
        self.assertTrue(np.array_equal(raw['vertices'][raw['faces']],raw['output']))
        pairs=raw['raw_shared_corner_pairs']
        ff=raw['faces'].reshape(-1)
        self.assertTrue(np.array_equal(ff[pairs[:,0]],ff[pairs[:,1]]))
        self.assertEqual(len(raw['free']),126)

    def test_loss_gradients_average_and_independent_count(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        case=fixture(); layout=P.inference_layout(case['C_vertices'],case['C_faces_original'],case['C_faces_model'],128)
        p=torch.arange(128*9,dtype=torch.float32).reshape(1,128,9).requires_grad_()
        projected=P.project_velocity(p,layout)
        target=torch.zeros_like(projected)
        loss=P.weighted_fm_loss(projected,target,layout,layout['free_coordinate_count'])
        loss.backward()
        self.assertEqual(p.grad[0,:2].abs().sum().item(),0.)
        pair=layout['shared_corner_pairs'][0]
        grad=p.grad.reshape(-1,3)
        self.assertTrue(torch.equal(grad[pair[0]],grad[pair[1]]))
        P.assert_shared(projected.detach().numpy(),layout)
        self.assertFalse(torch.cuda.is_initialized())

    def test_blocked_not_silently_reduced(self):
        case=fixture()
        deny=lambda a,b,L:dict(verified=True,classification='coplanar_positive_area_overlap')
        plan=P.build_source_plan(case,verified_pair=deny)
        self.assertEqual(plan['status'],'BLOCKED')
        self.assertEqual(plan['required_patch_count'],15)
        self.assertEqual(plan['source_pairs'],[])
        with self.assertRaises(ValueError):
            P.make_sample(case,dict(alpha=1.,noise_seed=1,epsilon_seed=2,time_seed=3,permutation_seed=4),plan)


if __name__=='__main__':unittest.main()
