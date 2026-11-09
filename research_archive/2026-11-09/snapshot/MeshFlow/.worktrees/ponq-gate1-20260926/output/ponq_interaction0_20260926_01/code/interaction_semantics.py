"""Frozen evaluation-only diagnostics for SINGLE_STATE_INTERACTION0.

These metrics never alter states, normals, matching, model loss, or extraction.
Target point coverage is not coverage of a generated shape distribution.
"""
import numpy as np
from scipy.spatial import cKDTree
from single_state_common import semantic

NORMAL_NEARZERO = 1e-8
NEAR_DUPLICATE_BBOX_FRACTION = .001
COVERAGE_BBOX_FRACTIONS = (.001,.005,.01)
COVERAGE_LABELS = ('0p1pct','0p5pct','1p0pct')

METRIC_DEFINITIONS = dict(
    version='INTERACTION_SEMANTICS_V1',diagnostic_only=True,changes_training_loss=False,
    inherited='single_state_common.semantic: aligned P RMS, raw n errors, F/C and spectral-normalized C, world Q',
    distance_units='Physical distances; *_pct_bbox is 100*distance/target bbox diagonal',
    bidirectional_NN='P_to_target and target_to_P are independent nearest-neighbor distances; no rematching of n/F',
    coverage_thresholds_bbox_fraction=list(COVERAGE_BBOX_FRACTIONS),
    target_coverage='Fraction of target vertices with nearest predicted vertex distance <= threshold; point-set coverage only, not generated distribution coverage',
    predicted_proximity='Fraction of predicted vertices with nearest target vertex distance <= the same threshold',
    near_duplicate_threshold_bbox_fraction=NEAR_DUPLICATE_BBOX_FRACTION,
    near_duplicate_pairs='Unordered distinct-index predicted pairs at Euclidean distance <= 0.001*target bbox diagonal',
    near_duplicate_points='Number/fraction of predicted vertices participating in at least one near-duplicate pair; not a count of removed vertices',
    normal_nearzero_threshold=NORMAL_NEARZERO,
    normal_nearzero_rule='raw normal length <= 1e-8; no normalization or replacement of the stored normal',
    normal_direction_usable='Both predicted and corresponding target raw normal lengths > 1e-8',
    normal_reverse_rate='Fraction with oriented cosine < 0 among direction-usable pairs; denominator reported explicitly',
    normal_angle_above_nearzero='Oriented arccos angle in degrees over direction-usable pairs; diagnostic normalization only',
    normal_length_errors='Raw predicted versus aligned target lengths; signed mean error, MAE, and inherited RMS')


def extended_semantic(state, aligned_target, bbox_diagonal):
    """Return one flat JSON-safe dict, including all original semantic fields."""
    state=np.asarray(state,dtype=np.float64)
    target=np.asarray(aligned_target,dtype=np.float64)
    if state.ndim!=2 or state.shape[1]!=22 or state.shape!=target.shape or len(state)<2:
        raise ValueError('Expected matched (N,22) states, N>=2')
    if not np.isfinite(bbox_diagonal) or bbox_diagonal<=0:
        raise ValueError('Target bbox diagonal must be finite and positive')
    if not np.isfinite(target).all():
        raise ValueError('Target must be finite')
    row=semantic(state,target,bbox_diagonal)
    row.update(P_element_count=len(state),metric_version=METRIC_DEFINITIONS['version'],
               P_near_duplicate_threshold=NEAR_DUPLICATE_BBOX_FRACTION*bbox_diagonal,
               n_nearzero_threshold=NORMAL_NEARZERO)
    if not row.get('finite',False):
        row['nonfinite_state_entries']=int((~np.isfinite(state)).sum())
        return row
    P=state[:,:3];T=target[:,:3]
    tree_P=cKDTree(P);tree_T=cKDTree(T)
    pred_to_target=tree_T.query(P,k=1)[0]
    target_to_pred=tree_P.query(T,k=1)[0]
    for name,distances in [('P_to_target',pred_to_target),('target_to_P',target_to_pred)]:
        rms=float(np.sqrt(np.mean(distances**2)))
        mean=float(np.mean(distances));p95=float(np.quantile(distances,.95))
        row.update({name+'_nn_rms':rms,name+'_nn_rms_pct_bbox':100*rms/bbox_diagonal,
                    name+'_nn_mean':mean,name+'_nn_mean_pct_bbox':100*mean/bbox_diagonal,
                    name+'_nn_p95':p95,name+'_nn_p95_pct_bbox':100*p95/bbox_diagonal})
    symmetric=float(np.sqrt(.5*(np.mean(pred_to_target**2)+np.mean(target_to_pred**2))))
    row['P_symmetric_nn_rms']=symmetric
    row['P_symmetric_nn_rms_pct_bbox']=100*symmetric/bbox_diagonal
    for fraction,label in zip(COVERAGE_BBOX_FRACTIONS,COVERAGE_LABELS):
        covered=int(np.count_nonzero(target_to_pred<=fraction*bbox_diagonal))
        nearby=int(np.count_nonzero(pred_to_target<=fraction*bbox_diagonal))
        row['P_target_coverage_'+label+'_count']=covered
        row['P_target_coverage_'+label+'_fraction']=covered/len(T)
        row['P_pred_target_proximity_'+label+'_count']=nearby
        row['P_pred_target_proximity_'+label+'_fraction']=nearby/len(P)
    pairs=tree_P.query_pairs(NEAR_DUPLICATE_BBOX_FRACTION*bbox_diagonal,output_type='ndarray')
    participating=np.unique(pairs.reshape(-1)) if pairs.size else np.empty(0,dtype=np.int64)
    row['P_near_duplicate_pair_count']=int(len(pairs))
    row['P_near_duplicate_point_count']=int(len(participating))
    row['P_near_duplicate_point_fraction']=len(participating)/len(P)
    row['P_exact_duplicate_excess_count']=int(len(P)-len(np.unique(P,axis=0)))
    row['P_nearest_other_distance_min']=float(tree_P.query(P,k=2)[0][:,1].min())
    n=state[:,3:6];nt=target[:,3:6]
    lengths=np.linalg.norm(n,axis=1);target_lengths=np.linalg.norm(nt,axis=1)
    usable=(lengths>NORMAL_NEARZERO)&(target_lengths>NORMAL_NEARZERO)
    row['n_nearzero_count']=int((lengths<=NORMAL_NEARZERO).sum())
    row['n_target_nearzero_count']=int((target_lengths<=NORMAL_NEARZERO).sum())
    row['n_direction_usable_count']=int(usable.sum())
    row['n_length_mae']=float(np.mean(abs(lengths-target_lengths)))
    row['n_length_signed_mean_error']=float(np.mean(lengths-target_lengths))
    if usable.any():
        cosines=np.clip(np.sum(n[usable]*nt[usable],axis=1)/(lengths[usable]*target_lengths[usable]),-1.,1.)
        angles=np.degrees(np.arccos(cosines));reverse=int((cosines<0).sum())
        row.update(n_reverse_count=reverse,n_reverse_rate=reverse/int(usable.sum()),
                   n_oriented_angle_mean_deg_above_nearzero=float(angles.mean()),
                   n_oriented_angle_p95_deg_above_nearzero=float(np.quantile(angles,.95)))
    else:
        row.update(n_reverse_count=0,n_reverse_rate=None,n_oriented_angle_mean_deg_above_nearzero=None,
                   n_oriented_angle_p95_deg_above_nearzero=None)
    return row
