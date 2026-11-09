"""Small CPU numeric contracts for the offline layer explanation, no real cells."""
import json
import unittest
import numpy as np
from . import geometry as G


def quad(z=0.):
    return np.asarray([[[0,0,z],[1,0,z],[1,1,z]],[[0,0,z],[1,1,z],[0,1,z]]],np.float64)


def fixture(free,points=None,owners=None,C=None,source=None,old=None):
    C=quad() if C is None else np.asarray(C,np.float64);F=np.asarray(free,np.float64).reshape(-1,3,3)
    points=np.asarray([[.5,.5,0.]] if points is None else points,np.float64);owners=np.asarray([0]*len(points) if owners is None else owners,np.int64)
    tri=np.concatenate((C,F));vertices,inverse=np.unique(tri.reshape(-1,3),axis=0,return_inverse=True);faces=inverse.reshape(-1,3)
    return dict(C=C,F=F,C_original=C.copy() if source is None else source,vertices=vertices,faces=faces,known_mask=np.arange(len(tri))<len(C),L=1.,old_conflict_dup=old,
                query=dict(C_points=points,C_point_faces=owners,C_weights=np.ones(len(points))/len(points),C_border_distance=np.full(len(points),.25)))


class GeometryContracts(unittest.TestCase):
    def test_finite_triangle_projection_and_edge_clamp(self):
        tri=np.asarray([[[0,0,0],[1,0,0],[0,1,0]]],float)
        d,p,b=G.finite_distances([[.2,.3,1.],[2.,0.,0.]],tri)
        np.testing.assert_allclose(d[:,0],[1.,1.],atol=1e-14)
        np.testing.assert_allclose(p[:,0],[[.2,.3,0.],[1.,0.,0.]],atol=1e-14)
        np.testing.assert_allclose(b[:,0].sum(-1),1.)

    def test_coplanar_quad_is_one_group_not_two_layers(self):
        result=G.analyze_geometry(fixture(quad(),old=[True]))
        self.assertEqual(len(result['hits']['query_id']),2)
        self.assertEqual(len(result['layers']),1)
        self.assertEqual(result['layers'][0]['free_face_ids'],[0,1])
        self.assertEqual(result['summary']['old_dup_comparison']['status'],'PASS')
        self.assertEqual(result['summary']['near_parallel_free_indices'],[0,1])

    def test_parallel_underside_and_sidewall_adjacency(self):
        z=-.003
        wall=np.asarray([[[0,0,0],[0,1,0],[0,1,z]],[[0,0,0],[0,1,z],[0,0,z]]],float)
        result=G.analyze_geometry(fixture(np.concatenate((quad(z),wall)),old=[True]))
        self.assertEqual(len(result['layers']),1)
        self.assertAlmostEqual(result['layers'][0]['signed_t_mean'],z)
        self.assertTrue(any(row['candidate_C_to_parallel'] for row in result['graph']['sidewall_adjacency']))
        self.assertEqual(result['summary']['near_parallel_free_indices'],[0,1])
        self.assertGreater(result['summary']['graph']['exact_CF_edges'],0)

    def test_extra_sheet_keeps_all_projected_hits_beyond_nearest(self):
        result=G.analyze_geometry(fixture(np.concatenate((quad(-.003),quad(.04),quad(.2))),old=[True]))
        self.assertEqual(len(result['hits']['query_id']),6)
        self.assertEqual(result['queries'][0]['corridor_layer_groups'],2)
        self.assertEqual(result['queries'][0]['all_isolated_normal_hits'],6)
        self.assertEqual(result['queries'][0]['corridor_isolated_normal_hits'],4)
        self.assertEqual(set(result['hits']['free_face_id']),set(range(6)))
        self.assertAlmostEqual(result['queries'][0]['nearest_F_distance'],.003)

    def test_folded_crossing_is_not_parallel_duplicate(self):
        slope=np.asarray([[[0,0,-.2],[1,0,.2],[0,1,.2]]],float)
        result=G.analyze_geometry(fixture(slope,points=[[.25,.25,0.]],old=[False]))
        self.assertAlmostEqual(result['queries'][0]['nearest_F_distance'],0.)
        self.assertEqual(result['queries'][0]['corridor_layer_groups'],1)
        self.assertFalse(result['queries'][0]['reconstructed_dup'])
        self.assertEqual(result['summary']['old_dup_comparison']['status'],'PASS')

    def test_reversed_source_normals_reverse_only_signed_offsets(self):
        a=G.analyze_geometry(fixture(quad(.003),old=[True]))
        b=G.analyze_geometry(fixture(quad(.003),source=quad()[:,[0,2,1]],old=[True]))
        self.assertAlmostEqual(a['queries'][0]['signed_nearest_point_offset'],.003)
        self.assertAlmostEqual(b['queries'][0]['signed_nearest_point_offset'],-.003)
        np.testing.assert_allclose(a['hits']['signed_t'],-b['hits']['signed_t'])
        np.testing.assert_array_equal(a['query_vectors']['reconstructed_dup'],b['query_vectors']['reconstructed_dup'])

    def test_reversed_free_normals_preserve_bidirectional_parallel_proxy(self):
        a=G.analyze_geometry(fixture(quad(.003),old=[True]))
        b=G.analyze_geometry(fixture(quad(.003)[:,[0,2,1]],old=[True]))
        self.assertEqual(a['summary']['near_parallel_free_indices'],b['summary']['near_parallel_free_indices'])
        np.testing.assert_allclose(a['hits']['signed_t'],b['hits']['signed_t'])

    def test_coplanar_normal_line_saved_as_interval(self):
        # A vertical F triangle contains the source-normal line. It is an interval,
        # not a fabricated isolated layer at t=0.
        f=np.asarray([[[.5,0,-.2],[.5,1,-.2],[.5,.5,.2]]],float)
        hits,intervals=G.normal_line_hits([[.5,.5,0]],[[0,0,1]],f,1.)
        self.assertEqual(len(hits['query_id']),0)
        self.assertEqual(len(intervals['query_id']),1)
        self.assertLess(intervals['t_min'][0],0);self.assertGreater(intervals['t_max'][0],0)
        self.assertTrue(intervals['intersects_corridor'][0])

    def test_missing_old_mask_and_degenerate_free_are_explicit(self):
        f=np.asarray([[[0,0,.01],[1,0,.01],[1,0,.01]]],float)
        result=G.analyze_geometry(fixture(f,old=None))
        self.assertEqual(result['summary']['old_dup_comparison']['status'],'NA_MISSING_OLD_MASK')
        self.assertFalse(result['queries'][0]['reconstructed_dup'])
        self.assertEqual(result['queries'][0]['all_isolated_normal_hits'],0)
        self.assertTrue(np.isfinite(result['queries'][0]['nearest_F_distance']))

    def test_old_mask_disagreement_not_silently_accepted(self):
        result=G.analyze_geometry(fixture(quad(.02),old=[True]))
        self.assertEqual(result['summary']['status'],'MISMATCH_OLD_DUP')
        self.assertEqual(result['summary']['old_dup_comparison']['mismatched_query_ids'],[0])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(GeometryContracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),real_cells=0,model_calls=0,old_scorer_calls=0)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
