import unittest
import numpy as np
from .planar_rebuild import retriangulate,triangulate_loop
from r2.data_tools.bench_user_v2 import build as b


class GeometricContracts(unittest.TestCase):
    def test_planar_interior_removed_boundary_preserved(self):
        v=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,0]],float)
        f=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        r,lineage,ledger=retriangulate(dict(model_vertices=v,source_faces=f))
        self.assertEqual(len(r['source_faces']),2)
        np.testing.assert_array_equal(r['model_vertices'],v)
        self.assertAlmostEqual(b.area2(v,r['source_faces']).sum(),2.)
        self.assertEqual({e for e,x in b.Q.edgebook(r['source_faces']).items() if len(x)==1},{(0,1),(1,2),(2,3),(0,3)})
        self.assertEqual(lineage,[[0,1,2,3],[0,1,2,3]])

    def test_nonplanar_fold_preserved(self):
        v=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[.5,.5,.1]],float)
        f=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        r,_,_=retriangulate(dict(model_vertices=v,source_faces=f))
        self.assertEqual({tuple(x) for x in r['source_faces']},{tuple(x) for x in f})

    def test_concave_polygon_area(self):
        v=np.array([[0,0,0],[2,0,0],[2,1,0],[1,1,0],[1,2,0],[0,2,0]],float)
        f=triangulate_loop(v,list(range(6)),np.array([0.,0.,1.]))
        self.assertEqual(len(f),4)
        self.assertAlmostEqual(b.area2(v,f).sum()/2,3.)

    def test_hole_not_filled(self):
        v=np.array([[0,0,0],[3,0,0],[3,3,0],[0,3,0],[1,1,0],[2,1,0],[2,2,0],[1,2,0]],float)
        f=np.array([[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
        r,_,ledger=retriangulate(dict(model_vertices=v,source_faces=f))
        self.assertEqual(len(r['source_faces']),8)
        self.assertAlmostEqual(b.area2(v,r['source_faces']).sum()/2,8.)
        self.assertTrue(any(x['action']=='RETAINED_MULTI_LOOP_OR_HOLE' for x in ledger))


if __name__=='__main__':unittest.main()
