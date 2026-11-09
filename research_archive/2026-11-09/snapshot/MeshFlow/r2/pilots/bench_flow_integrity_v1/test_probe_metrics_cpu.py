"""Small no-model/no-OT fixtures for the diagnostic's mathematical contract."""
import unittest
import numpy as np
from . import probe_metrics as M


def fixture(fixed=False):
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
    faces=np.array([[0,1,2],[1,3,2]])
    full=vertices[faces]; ids=np.array([1,0]); cp=np.array([[2,0,1],[1,2,0]])
    y=full[ids[:,None],cp]; x0=np.zeros_like(y)
    anchors=np.full((2,3),-1)
    known=np.zeros(2,bool)
    if fixed:
        anchors[1]=[1,2,0]; known[1]=True; x0[1]=y[1]
    t=np.float32(.3)
    return dict(full_target=full,source_face_vertex_ids=faces),dict(N=2,x0=x0,x1=y,
        xt=(1-t)*x0+t*y,u=y-x0,t=t,source_face_ids=ids,source_corner_permutations=cp,
        anchor_ids=anchors,known_mask=known,valid_mask=np.ones(2,bool))


class Contract(unittest.TestCase):
    def test_permutation_and_identity(self):
        case,sample=fixture(); pairs=M.prepare_pairs(sample,case)
        pred=sample['u']+np.arange(18,dtype=np.float32).reshape(2,3,3)/100
        rows,detail=M.compute_metrics(sample,pred,pairs=pairs)
        self.assertEqual(pairs['source']['legal_edge_count'],1)
        self.assertEqual(detail['pair_count'],2)
        self.assertLess(detail['shared_identity_max_abs'],1e-7)
        self.assertEqual(next(r for r in rows if r['stratum']=='bulk--seam_tip')['status'],'NA_NO_LEGAL_SOURCE_PAIRS')

    def test_fixed_projection_is_explicit(self):
        case,sample=fixture(True); pairs=M.prepare_pairs(sample,case)
        pred=sample['u']+1
        rows,detail=M.compute_metrics(sample,pred,pairs=pairs)
        self.assertEqual(next(r for r in rows if r['stratum']=='all_one_fixed')['count'],2)
        self.assertEqual(detail['raw_fixed_prediction_squared_sum'],9)
        self.assertLess(detail['shared_identity_max_abs'],1e-7)

    def test_corrupt_corner_map_rejected(self):
        case,sample=fixture(); sample['source_corner_permutations'][0]=[0,0,1]
        with self.assertRaises(ValueError): M.prepare_pairs(sample,case)

    def test_wrong_source_cache_rejected(self):
        case,sample=fixture(); source=M.source_pairs(case); source['source_coordinate_sha256']='bad'
        with self.assertRaises(ValueError): M.prepare_pairs(sample,case,source=source)

    def test_opposite_duplicate_not_a_legal_shared_pair(self):
        case,_=fixture(); case['full_target']=np.r_[case['full_target'],case['full_target'][:1,::-1]]
        case['source_face_vertex_ids']=np.r_[case['source_face_vertex_ids'],case['source_face_vertex_ids'][:1,::-1]]
        source=M.source_pairs(case)
        self.assertEqual(source['duplicate_faces'],[0,2])
        self.assertEqual(source['pair_count'],0)

    def test_altered_bridge_rejected(self):
        case,sample=fixture(); pairs=M.prepare_pairs(sample,case); sample['xt'][0,0,0]+=1
        with self.assertRaises(ValueError): M.compute_metrics(sample,sample['u'],pairs=pairs)


if __name__=='__main__': unittest.main()
