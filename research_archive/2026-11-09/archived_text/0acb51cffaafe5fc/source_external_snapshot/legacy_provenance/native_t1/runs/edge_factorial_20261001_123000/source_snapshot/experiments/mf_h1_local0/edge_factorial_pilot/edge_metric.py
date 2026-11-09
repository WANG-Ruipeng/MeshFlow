"""Independent boundary-query distance to all finite generated free edges.

No GT face correspondence, welding, duplicate-edge removal or coordinate repair.
Low distance is neither shared topology nor a watertightness certificate.
Queries/weights are the unchanged registered Gamma-to-surface inputs.
"""
import numpy as np

COVERAGE_FRACTIONS = (.0005, .001, .002, .005, .01)
SURFACE_INEQUALITY_TOLERANCE_FRACTION = 1e-10
METRIC_SPEC = dict(schema="Gamma_to_free_edges_v1", precision="CPU_float64",
    query="unchanged_registered_Gamma_points_and_length_weights",
    prediction="all_three_finite_edges_per_free_triangle_with_duplicates_retained",
    zero_length="point_at_original_endpoint", coverage_comparison="strict_less_than",
    coverage_bbox_fractions=list(COVERAGE_FRACTIONS),
    p95="stable_distance_argsort_cumulative_weight_searchsorted_left_0.95",
    surface_inequality_absolute_tolerance_bbox_fraction=SURFACE_INEQUALITY_TOLERANCE_FRACTION,
    topology_or_watertightness_certificate=False)


def _inc(ledger,key,value=1):
    if ledger is not None:ledger[key]=ledger.get(key,0)+int(value)


def gamma_to_free_edges(free_triangles,gamma_points,gamma_weights,bbox_L,*,ledger=None):
    """Return JSON metrics and per-query distances in original coordinate units.

    Bbox normalization is applied only to reported percentages and thresholds.
    No distance epsilon changes a nonzero segment into a point.
    """
    _inc(ledger,'gamma_to_free_edges_calls_attempted')
    triangles=np.asarray(free_triangles,dtype=np.float64)
    points=np.asarray(gamma_points,dtype=np.float64)
    weights=np.asarray(gamma_weights,dtype=np.float64)
    L=float(bbox_L)
    if triangles.ndim!=3 or triangles.shape[1:]!=(3,3) or len(triangles)==0:
        raise ValueError('Nonempty free triangles[F,3,3] required')
    if points.ndim!=2 or points.shape[1]!=3 or len(points)==0 or weights.shape!=(len(points),):
        raise ValueError('Matching nonempty original Gamma points[Q,3] and weights[Q] required')
    if not all(np.isfinite(x).all() for x in (triangles,points,weights)) or not np.isfinite(L) or L<=0:
        raise ValueError('Finite geometry/weights and positive bbox required')
    if np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Nonnegative weights with positive total mass required')
    starts=triangles[:,[0,1,2]].reshape(-1,3)
    ends=triangles[:,[1,2,0]].reshape(-1,3)
    direction=ends-starts
    length2=(direction*direction).sum(axis=1)
    zero=length2==0
    divisor=np.where(zero,1.,length2)
    distances=np.empty(len(points),dtype=np.float64)
    for first in range(0,len(points),128):
        query=points[first:first+128]
        difference=query[:,None,:]-starts[None,:,:]
        parameter=(difference*direction[None,:,:]).sum(axis=2)/divisor[None,:]
        closest=starts[None,:,:]+np.clip(parameter,0.,1.)[:,:,None]*direction[None,:,:]
        squared=((query[:,None,:]-closest)**2).sum(axis=2)
        distances[first:first+len(query)]=np.sqrt(squared.min(axis=1))
        _inc(ledger,'gamma_to_free_edges_point_segment_pairs',len(query)*len(starts))
    order=np.argsort(distances,kind='stable')
    cumulative=np.cumsum(weights[order])
    p95=float(distances[order[min(int(np.searchsorted(cumulative,.95*cumulative[-1],side='left')),len(order)-1)]])
    rms=float(np.sqrt(np.sum(weights*distances*distances)/weights.sum()))
    _inc(ledger,'gamma_to_free_edges_zero_length_occurrences',int(zero.sum()))
    result=dict(status='PASS',RMS=rms,p95=p95,RMS_bbox_pct=100*rms/L,p95_bbox_pct=100*p95/L,
        coverage_by_bbox_fraction={str(v):float(weights[distances<v*L].sum()/weights.sum()) for v in COVERAGE_FRACTIONS},
        per_point_distances=distances.tolist(),point_count=len(points),edge_count=len(starts),
        zero_length_edge_count=int(zero.sum()),weight_sum=float(weights.sum()),bbox_L=L,
        coordinate_units='original_model_coordinates',duplicate_edges_retained=True,
        predicted_GT_edge_correspondence_used=False,geometry_modified=False,
        topology_connection_rate=False,metric_spec=METRIC_SPEC)
    _inc(ledger,'gamma_to_free_edges_calls_returned')
    return result


def check_surface_lower_bound(edge_distances,surface_distances,bbox_L):
    """Compare existing distances only; never compute a second surface metric."""
    edge=np.asarray(edge_distances,dtype=np.float64)
    surface=np.asarray(surface_distances,dtype=np.float64)
    L=float(bbox_L)
    if edge.shape!=surface.shape or edge.ndim!=1 or not len(edge) or not np.isfinite(L) or L<=0:
        raise ValueError('Matching nonempty per-query distances and positive bbox required')
    if not np.isfinite(edge).all() or not np.isfinite(surface).all() or np.any(edge<0) or np.any(surface<0):
        raise ValueError('Distances must be nonnegative and finite')
    gap=edge-surface;tolerance=SURFACE_INEQUALITY_TOLERANCE_FRACTION*L
    violations=int((gap < -tolerance).sum())
    return dict(status='PASS' if violations==0 else 'FAIL',point_count=len(edge),violations=violations,
        minimum_edge_minus_surface=float(gap.min()),absolute_tolerance=tolerance,
        inequality='distance_to_free_edges >= distance_to_free_triangle_surface')
