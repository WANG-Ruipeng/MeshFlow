"""CPU-only tests for the isolated numerical evaluation contract."""
import unittest
import numpy as np
from native_t1.metrics import (
    array_sha, symmetric_surface_rms, prepare_common_reference,
    extract_prediction, evaluate_output, binding_four_grid, shape_quality,
)
from native_t1.metric_geometry import boundary_metrics


class MetricContractTests(unittest.TestCase):
    @staticmethod
    def targets():
        face = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=np.float32)
        first = np.stack([face + np.array([2*(i%14),2*(i//14),.1*(i%2)],dtype=np.float32) for i in range(112)])
        second = first + np.array([.2,-.1,.4],dtype=np.float32)
        return first, second

    def test_symmetric_rms_original_direction_identity(self):
        first, second = self.targets()
        value = symmetric_surface_rms(first[12:],second[12:],30.)
        self.assertEqual(value['points_per_surface'],512)
        self.assertEqual(value['seed'],9911)
        expected = np.sqrt((value['predicted_to_target_mean_squared']+value['target_to_predicted_mean_squared'])/2)
        self.assertEqual(value['symmetric_rms'],expected)
        repeated = symmetric_surface_rms(first[12:],second[12:],30.)
        self.assertEqual(value,repeated)

    def test_fixed_mask_and_bitwise_context(self):
        first, _ = self.targets(); before=first.tobytes()
        package = extract_prediction(first,first[:12])
        self.assertEqual(package['free'].shape,(100,3,3))
        self.assertEqual(package['audit']['extraction_count'],1)
        self.assertEqual(package['free'].tobytes(),first[12:].tobytes())
        self.assertEqual(first.tobytes(),before)
        bad=first.copy();bad[0,0,0]+=1
        with self.assertRaises(ValueError):extract_prediction(bad,first[:12])

    def test_fixed_references_cross_scores_and_unimplemented_status(self):
        first,second=self.targets();lengths=[30.,30.]
        common=prepare_common_reference([first[12:],second[12:]],lengths)
        audit=repr(common['audit'])
        metrics=[]
        for p,target in enumerate((first,second)):
            score=evaluate_output(target,target[:12],target,[first[12:],second[12:]],lengths,p,common_reference=common)
            self.assertLess(score['remainder']['own']['symmetric_rms_bbox_pct'],1e-10)
            self.assertEqual(score['remainder']['cross'][0]['predicted_free_sha256'],score['remainder']['cross'][1]['predicted_free_sha256'])
            self.assertEqual(score['independent_full_geometry_audit']['status'],'NOT_RUN')
            self.assertEqual(score['geometry_postprocessing'],'NONE')
            metrics.append(score)
        binding=binding_four_grid(*metrics)
        self.assertTrue(binding['both_directions_positive'])
        self.assertEqual(binding['B'],(binding['Delta0']+binding['Delta1'])/2)
        self.assertEqual(repr(common['audit']),audit)
        self.assertEqual(metrics[0]['common_unknown_R12']['cross'][0]['target_point_sha256'],metrics[1]['common_unknown_R12']['cross'][0]['target_point_sha256'])
        wrong=prepare_common_reference([second[12:],first[12:]],lengths)
        with self.assertRaises(ValueError):evaluate_output(first,first[:12],first,[first[12:],second[12:]],lengths,0,common_reference=wrong)

    def test_boundary_and_degeneracy_are_distinct(self):
        face=np.array([[[0.,0.,0.],[1.,0.,0.],[.5,np.sqrt(3)/2,0.]]],dtype=np.float32)
        quality=shape_quality(face,2.)
        self.assertAlmostEqual(quality['q_actual']['mean'],1.,places=12)
        zero=shape_quality(np.zeros((1,3,3),dtype=np.float32),2.)
        self.assertEqual(zero['exact_zero_area_count'],1)
        self.assertEqual(zero['q_actual']['nan_count'],1)
        boundary=boundary_metrics(face,face,2.)
        self.assertLess(boundary['rms'],1e-12)
        self.assertGreater(boundary['coverage_fraction'],.999999)
        self.assertTrue(boundary['length_weighted'])


if __name__=='__main__':
    unittest.main()
