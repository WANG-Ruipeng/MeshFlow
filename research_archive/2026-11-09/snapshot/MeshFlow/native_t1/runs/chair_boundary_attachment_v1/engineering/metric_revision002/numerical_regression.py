"""Revision-only CPU regressions; no model, generation, or model raw."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from experiments.mf_h1_local0.chair_boundary_attachment_v1 import control_metrics as current
from experiments.mf_h1_local0.chair_boundary_attachment_v1.metric_tests import _box

folder = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('preserved_control_metrics_v1', folder/'original_control_metrics.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
q = np.array([[0., 0., 0.]])
normal = np.array([[0., 0., 1.]])
d = normal.copy()
triangle = np.array([[[-1e6, -2e-7, 1.], [1e6, -2e-7, 1.], [0., 2e-7, 1.]]])
# Analytic interior hit: barycentrics (1/4,1/4,1/2), positive t=1.
hit = triangle[0].T @ np.array([.25,.25,.5])
assert np.array_equal(hit, np.array([0.,0.,1.]))
old_ray = previous.ray_visibility(q, normal, triangle, d)
new_ray = current.ray_visibility(q, normal, triangle, d)
assert old_ray[0,0] == previous.VISIBLE
assert new_ray[0,0] == current.UNKNOWN
# Reordering cannot convert the unresolved rejected determinant to visibility.
assert current.ray_visibility(q, normal, triangle[:,::-1], d)[0,0] == current.UNKNOWN
reference = dict(arrays=dict(C_points=np.array([[0.,0.,0.],[1.,0.,0.]]),
    C_point_normals=np.array([[0.,0.,1.],[0.,0.,0.]]), C_point_normal_valid=np.array([True,False]),
    directions=d, C_visibility=np.array([[current.VISIBLE],[current.EXCLUDED]],np.uint8),
    GT_visibility=np.array([[current.VISIBLE],[current.EXCLUDED]],np.uint8), C_weights=np.array([.5,.5])))
free = _box((5.,5.,5.),(6.,6.,6.))
old_bound,_ = previous.exposure_metrics(free,dict(L0=1.),reference)
new_bound,_ = current.exposure_metrics(free,dict(L0=1.),reference)
assert old_bound['ExposedRetain_global_lower']==old_bound['ExposedRetain_global_upper']==1.
assert new_bound['ExposedRetain_global_lower']==0. and new_bound['ExposedRetain_global_upper']==1.
assert old_bound['ExposedRetain']==new_bound['ExposedRetain']==1.
assert new_bound['ExposedRetain_status']=='PARTIAL'
# No valid reference domain remains NA, while the auxiliary global interval is broad.
reference['arrays']['C_point_normal_valid'][:]=False
reference['arrays']['C_visibility'][:]=current.EXCLUDED
reference['arrays']['GT_visibility'][:]=current.EXCLUDED
none_bound,_ = current.exposure_metrics(free,dict(L0=1.),reference)
assert none_bound['ExposedRetain'] is None
assert none_bound['ExposedRetain_global_lower']==0. and none_bound['ExposedRetain_global_upper']==1.
constants=['TAUS','THETA','RHO_CROSS','RHO_DUP','R_BORDER','NORMAL_ANGLE','LOW_SUPPORT_N','AREA_FLOOR','RAY_EPS','RAY_DET_REL','RAY_BARY_EPS','TANGENT_COS']
assert all(getattr(previous,k)==getattr(current,k) for k in constants)
assert np.array_equal(previous.directions(),current.directions())
assert np.array_equal(previous.directions(True),current.directions(True))
result=dict(status='PASS',regression_count=4,model_calls=0,backwards=0,optimizer_updates=0,model_raw_read=0,
    source_version=current.VERSION,scientific_constants_unchanged=constants,
    skinny_triangle=dict(analytic_positive_hit_t=1.,analytic_barycentric=[.25,.25,.5],area=.4,
        before_visibility_code=int(old_ray[0,0]),after_visibility_code=int(new_ray[0,0]),
        meaning='Previously false VISIBLE; now conservative UNKNOWN for rejected determinant'),
    free_corner_order_invariance='PASS',
    invalid_C_normal=dict(before_global=[old_bound['ExposedRetain_global_lower'],old_bound['ExposedRetain_global_upper']],
        after_global=[new_bound['ExposedRetain_global_lower'],new_bound['ExposedRetain_global_upper']],
        definite_reference_primary_unchanged=True,primary_status=new_bound['ExposedRetain_status']),
    no_definite_reference=dict(primary=None,primary_status=none_bound['ExposedRetain_status'],global_interval=[0.,1.]))
(folder/'numerical_regression_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
