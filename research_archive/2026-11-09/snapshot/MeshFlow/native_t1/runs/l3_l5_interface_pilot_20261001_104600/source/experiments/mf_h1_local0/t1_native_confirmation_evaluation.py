"""Read-only endpoint evaluation for pure Native0 / T1 / T2 confirmation.

Original metric keys, sample streams and denominators remain unchanged.  This
module never calls a model, computes an autograd energy, or changes geometry.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import time

import numpy as np

from .guided_native_geometry_audit import score_endpoint
from .gpu_guidance_evaluation import (
    audit_context as _original_audit_context, shape_quality, _detector_view,
)
from .native_patch_metrics import patch_binding_four_grid
from .native_patch_seams import extract_boundary_segments
from .native_inpainting_metrics import NEAR_CONTEXT_FRACTION, SURFACE_POINTS, array_sha
from .evaluate import sample_mesh_surface, surface_distance_squared

MODELS = ("Native0", "T1", "T2")
PROPER = "proper_transverse_penetration"
COPLANAR = "coplanar_positive_area_overlap"
NEAR_C_SAMPLE_SEED = 9951
BOUNDARY_BAND_FRACTION = 1e-3
CONCENTRATION_CANCELLATION_FRACTION = .1
CONCENTRATION_ABSOLUTE_FLOOR = 1e-8
PROTOCOL = dict(
    source_metrics="Unchanged guided_native_geometry_audit.score_endpoint / native_patch_metrics",
    fixed_common_R12_points=512, fixed_common_R12_seed_base=9973,
    near_C=dict(points=SURFACE_POINTS, seed=NEAR_C_SAMPLE_SEED,
        area_uniform=True, surface_threshold_bbox_fraction=NEAR_CONTEXT_FRACTION,
        physical_outer_boundary_band_bbox_fraction=BOUNDARY_BAND_FRACTION,
        rule="Near C surface split by sample distance to the physical outer boundary, at the same fixed 0.001L scale",
        interpretation="Area-sample proximity only; legal seams and thin structures can be near C"),
    concentration=dict(relative_net_floor=CONCENTRATION_CANCELLATION_FRACTION,
        absolute_net_floor_bbox_percentage_points=CONCENTRATION_ABSOLUTE_FLOOR,
        rule="Single-seed/net percentage displayed only for positive net >0.1*sum(abs(B)) and >1e-8; otherwise NOT_DEFINED",
        always_report="All seed B, positive/negative sums, net, largest positive B, and leave-largest-positive-out net"),
    summaries="Object-wise descriptive distributions; seeds are repeated noise trials, not independent objects",
    model_calls=0, geometry_autograd_calls=0, repairs=0)


def _inc(ledger, key, amount=1):
    if ledger is not None:
        ledger[key] = ledger.get(key, 0) + amount


def _tri(value):
    value = np.asarray(value)
    if value.ndim == 2 and value.shape[-1] == 9:
        value = value.reshape(-1, 3, 3)
    if value.ndim != 3 or value.shape[1:] != (3, 3):
        raise ValueError("Expected [N,3,3] or [N,9]")
    return value


def audit_context(C, L, ledger=None):
    value = _original_audit_context(C, L, ledger)
    value["schema"] = "t1_native_confirmation_static_context_v1"
    return value


def _finite_distribution(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    f = a[np.isfinite(a)]
    return dict(count=len(a), finite_count=len(f), nonfinite_count=len(a)-len(f),
        minimum=float(f.min()) if len(f) else None,
        q05=float(np.quantile(f,.05)) if len(f) else None,
        q25=float(np.quantile(f,.25)) if len(f) else None,
        median=float(np.median(f)) if len(f) else None,
        q75=float(np.quantile(f,.75)) if len(f) else None,
        q95=float(np.quantile(f,.95)) if len(f) else None,
        maximum=float(f.max()) if len(f) else None,
        mean=float(f.mean()) if len(f) else None)


def near_C_area_diagnostic(free, C, L, old_near_context=None, ledger=None):
    """Fixed area samples, finite C surface, and physical outer-edge distance.

    Classification uses sample-to-boundary distance, not an inferred semantic
    interior or a signed solid test. Nothing is projected or assigned away.
    """
    free, C = _tri(free), _tri(C)
    points = sample_mesh_surface(free, SURFACE_POINTS, np.random.RandomState(NEAR_C_SAMPLE_SEED))
    if points is None:
        return dict(status="ZERO_OR_INVALID_SURFACE_AREA", protocol=PROTOCOL["near_C"])
    L = float(L)
    d2_C = surface_distance_squared(points, C)
    segments, boundary = extract_boundary_segments(C)
    if len(segments):
        p = np.asarray(points, dtype=np.float64)[:,None,:]
        a = np.asarray(segments, dtype=np.float64)[None,:,0,:]
        edge = np.asarray(segments, dtype=np.float64)[None,:,1,:] - a
        denom = np.sum(edge*edge, axis=-1)
        t = np.divide(np.sum((p-a)*edge, axis=-1), denom,
            out=np.zeros((len(points),len(segments))), where=denom>0)
        projected = a + np.clip(t,0.,1.)[...,None]*edge
        d2_boundary = np.sum((p-projected)**2,axis=-1).min(axis=1)
        in_boundary_band = d2_boundary <= (BOUNDARY_BAND_FRACTION*L)**2
        boundary_distance = _finite_distribution(100*np.sqrt(d2_boundary)/L)
    else:
        in_boundary_band = np.zeros(len(points), dtype=bool)
        boundary_distance = dict(status="NO_OUTER_BOUNDARY", count=0)
    near = d2_C <= (NEAR_CONTEXT_FRACTION*L)**2
    fraction = float(near.mean())
    if old_near_context is not None:
        old_fraction = old_near_context["free_surface_sample_points_near_context_fraction"]
        if fraction != old_fraction:
            raise AssertionError("Fixed near-C sampling no longer matches the original diagnostic")
    _inc(ledger, "near_C_fixed_area_sample_diagnostics")
    return dict(status="PASS", point_count=len(points), point_seed=NEAR_C_SAMPLE_SEED,
        point_sha256=array_sha(points), sampling="Original area-uniform free samples, unchanged seed9951",
        context_distance_bbox_pct=_finite_distribution(100*np.sqrt(d2_C)/L),
        physical_outer_boundary_distance_bbox_pct=boundary_distance,
        near_C_surface_sample_fraction=fraction,
        near_C_and_boundary_band_sample_fraction=float((near & in_boundary_band).mean()),
        near_C_outside_boundary_band_sample_fraction=float((near & ~in_boundary_band).mean()),
        far_from_C_sample_fraction=float((~near).mean()),
        fraction_denominator="All512 area-uniform free-surface samples",
        all_samples_near_boundary_band_fraction=float(in_boundary_band.mean()),
        near_C_count=int(near.sum()), near_C_boundary_band_count=int((near & in_boundary_band).sum()),
        near_C_outside_boundary_band_count=int((near & ~in_boundary_band).sum()),
        protocol=PROTOCOL["near_C"], outer_boundary=boundary,
        boundary_classification="Sample-to-physical-outer-edge Euclidean distance; not a semantic part label",
        false_error_warning="Near C is not automatically incorrect, repeated C, an intersection, or an invalid seam",
        geometry_modified=False, model_calls=0, geometry_autograd_calls=0)


def score_output(tokens, bank, parent, ledger=None, topology=None):
    tick = time.perf_counter()
    full = _tri(tokens)
    case = bank["cases_by_key"][(int(parent),12)]
    C = _tri(case["constraints"])
    if full.dtype != np.float32 or full.shape != (112,3,3):
        raise ValueError("Require actual FP32112 triangles")
    if full[:12].tobytes() != C.tobytes():
        raise AssertionError("C must preserve original FP32 bit patterns and corner order")
    before = full.tobytes()
    metrics = score_endpoint(full, bank, int(parent), ledger=ledger)
    L = float(case["bbox_diagonal"])
    quality = shape_quality(full[12:], L)
    # q/angles are geometry diagnostics. Do not expose the earlier experiment's
    # incidental CPU energy-reference label as if this run evaluated guidance.
    quality.pop("guarded_EA_CPU_formula_reference", None)
    quality.pop("EA_scope", None)
    quality["schema"] = "t1_native_confirmation_shape_quality_v1"
    scans = metrics["guided_geometry_audit"]
    detectors = {scope:{which:_detector_view(scans[key],which)
        for which in ("primary","independent","verified")}
        for scope,key in (("CF","C_free"),("FF","free_free"))}
    if detectors["CF"]["verified"]["tested_pairs"] != 1200 or detectors["FF"]["verified"]["tested_pairs"] != 4950:
        raise AssertionError("Complete pair-scan denominator changed")
    static = bank.get("context_audits",{}).get(int(parent))
    if static is not None:
        if static["context_sha256"] != hashlib.sha256(np.ascontiguousarray(C).tobytes()).hexdigest() or static["bbox_diagonal"] != L:
            raise AssertionError("Static C/C audit belongs to different immutable C")
        detectors["CC"] = deepcopy(static["detector_views"])
        _inc(ledger, "static_context_audit_reuses")
    near = near_C_area_diagnostic(full[12:], C, L, metrics["near_context"], ledger)
    input_inheritance = None
    parent_audits = bank.get("input_topology",{}).get("parents",{}) if isinstance(bank.get("input_topology"),dict) else {}
    if isinstance(parent_audits,dict):
        input_inheritance = parent_audits.get(str(parent),parent_audits.get(int(parent)))
    elif isinstance(parent_audits,list):
        input_inheritance = next((row for row in parent_audits if int(row["parent"])==int(parent)),None)
    if topology is None:
        from .t1_native_confirmation_topology import generated_attribution
        topology = generated_attribution(full, C, input_inheritance)
        for key, value in topology.get("ledger",{}).items():
            _inc(ledger, key, value)
        _inc(ledger, "generated_topology_attributions")
    any_illegal = any(view[which]["illegal_pair_count"] for view in detectors.values() for which in ("primary","independent"))
    unresolved = sum(view["verified"]["unresolved_count"] for view in detectors.values())
    extra = dict(schema="t1_native_confirmation_evaluation_v1", free_shape_quality=quality,
        detector_views=detectors, near_C_area=near,
        exact_coordinate_topology=deepcopy(topology) if topology is not None else dict(status="NOT_RUN"),
        input_topology_parent_reference=int(parent) if input_inheritance is not None else None,
        input_inheritance_scope="Use separate source/C input topology audit plus generated attribution; current winding inconsistency is not itself geometric penetration",
        C_PRESERVED="PASS", total_faces=112, free_faces=100,
        all_coordinates_finite=bool(np.isfinite(full).all()),
        nan_coordinate_count=int(np.isnan(full).sum()), inf_coordinate_count=int(np.isinf(full).sum()),
        EMBEDDED_GEOMETRY_VALIDITY_STATUS="FAIL_DETECTED_ILLEGAL_INTERSECTIONS" if any_illegal else "UNRESOLVED" if unresolved else "NO_ILLEGAL_TRIANGLE_INTERSECTIONS_DETECTED" if static is not None else "C_C_NOT_RUN",
        complete_pair_scope="CF1200 + FF4950 + fixed CC66 reused" if static is not None else "CF1200 + FF4950; CC NOT_RUN",
        exact_predicate_certification="NOT_RUN", continuous_collision_safety="NOT_RUN",
        diagnosis_caveat="Numerical static tests at existing tolerances; exact shared edges and geometric seam distances are distinct",
        geometry_modified=False, geometry_guidance=0, geometry_autograd_calls=0, model_calls=0,
        scoring_seconds=time.perf_counter()-tick)
    metrics["t1_native_confirmation"] = extra
    if full.tobytes() != before:
        raise AssertionError("Read-only evaluation mutated generated coordinates")
    _inc(ledger, "t1_native_confirmation_endpoint_scores")
    json.dumps(metrics, allow_nan=False)
    return metrics


def cross_binding(context0_metrics, context1_metrics):
    out = patch_binding_four_grid(context0_metrics, context1_metrics)
    out.update(E00=out["E_R0_F_C0"], E01=out["E_R0_F_C1"], E11=out["E_R1_F_C1"], E10=out["E_R1_F_C0"],
        Delta0=out["direction0_bbox_percentage_points"], Delta1=out["direction1_bbox_percentage_points"],
        B=out["mean_binding_bbox_percentage_points"])
    out["direction0_positive"] = out["Delta0"] > 0
    out["direction1_positive"] = out["Delta1"] > 0
    out["caveat"] = "Native models use the same fixed free slots; changing C changes actual model input. No re-extraction, face reassignment, or geometry selection."
    return out


def _get(mapping,*keys):
    for key in keys:
        if not isinstance(mapping,dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def scalar_metrics(metrics):
    """Old archived metrics remain usable; missing new diagnostics stay null."""
    own = metrics["remainder"]["own"]
    whole = metrics["whole"]["surface"]
    common = metrics.get("common_unknown_R12",{}).get("own",{})
    extra = metrics.get("t1_native_confirmation",{})
    quality = extra.get("free_shape_quality",{})
    scan = metrics.get("guided_geometry_audit",{})
    boundary = scan.get("boundary",{})
    legacy_cf = metrics.get("seam",{}).get("context_free_intersections",{})
    cf = scan.get("C_free",{})
    ff = scan.get("free_free",{})
    # Archived seam fields differ from later verified scan. Do not relabel an
    # old primary-only count as a new independently verified result.
    row = dict(
        remainder_symmetric_RMS_bbox_pct=own.get("symmetric_rms_bbox_pct"),
        free_to_R12_RMS_bbox_pct=own.get("predicted_to_target_rms_bbox_pct"),
        R12_to_free_RMS_bbox_pct=own.get("target_to_predicted_rms_bbox_pct"),
        whole_symmetric_RMS_bbox_pct=whole.get("symmetric_rms_bbox_pct"),
        whole_generated_to_GT_RMS_bbox_pct=whole.get("predicted_to_target_rms_bbox_pct"),
        whole_GT_to_generated_RMS_bbox_pct=whole.get("target_to_predicted_rms_bbox_pct"),
        fixed_R12_coverage_RMS_bbox_pct=common.get("rms_bbox_pct"),
        fixed_R12_coverage_within_1pct_fraction=common.get("fraction_target_points_within_1pct_bbox"),
        boundary_RMS_bbox_pct=boundary.get("rms_bbox_pct"),
        boundary_p95_bbox_pct=boundary.get("p95_bbox_pct"),
        boundary_coverage_fraction=boundary.get("coverage_fraction"),
        CF_proper_affected_free_fraction=cf.get("proper_affected_face_fraction"),
        CF_proper_pair_count=cf.get("proper_pair_count"),
        CF_coplanar_pair_count=cf.get("coplanar_positive_area_pairs"),
        CF_unresolved_or_degenerate_pair_count=cf.get("unresolved_or_degenerate_pairs"),
        FF_proper_pair_count=ff.get("proper_pair_count"),
        FF_coplanar_pair_count=ff.get("coplanar_positive_area_pairs"),
        FF_illegal_pair_count=(ff["proper_pair_count"]+ff["coplanar_positive_area_pairs"]) if ff else None,
        FF_unresolved_or_degenerate_pair_count=ff.get("unresolved_or_degenerate_pairs"),
        free_minimum_angle_lt5_fraction=quality.get("min_angle_lt5_fraction"),
        free_angle_p05_degrees=quality.get("angle_p05_degrees"),
        free_q_gt10_fraction=quality.get("q_actual_gt10_fraction"),
        free_q_mean=_get(quality,"q_actual","mean"),
        free_near_degenerate_count=quality.get("near_degenerate_area_count"),
        free_exact_zero_area_count=quality.get("exact_zero_area_count"),
        near_C_area_fraction=_get(extra,"near_C_area","near_C_surface_sample_fraction"),
        near_C_boundary_band_area_fraction=_get(extra,"near_C_area","near_C_and_boundary_band_sample_fraction"),
        near_C_outside_boundary_band_area_fraction=_get(extra,"near_C_area","near_C_outside_boundary_band_sample_fraction"),
        exact_C_free_shared_edge_count=_get(extra,"exact_coordinate_topology","attribution","exact_C_free_shared_edge_count"),
        new_free_related_nonmanifold_vertex_count=None,
        C_surface_area_fraction=_get(metrics,"context_coverage","surface_area_fraction"))
    new_ids = _get(extra,"exact_coordinate_topology","attribution","new_free_related_nonmanifold_vertex_ids")
    if new_ids is not None:
        row["new_free_related_nonmanifold_vertex_count"] = len(new_ids)
    return row


def summarize_values(rows, key, higher_better=False):
    vals = [(int(row["seed"]),row.get(key)) for row in rows]
    valid = [(seed,float(value)) for seed,value in vals if value is not None and math.isfinite(float(value))]
    out = _finite_distribution([value for _,value in valid])
    out.update(expected_count=len(vals), missing_count=len(vals)-len(valid),
        values_by_seed=[dict(seed=seed,value=value) for seed,value in vals],
        status="PASS" if len(valid)==len(vals) else "PARTIAL_OR_NOT_RUN",
        worst=None, higher_is_better=higher_better)
    if valid:
        worst = sorted(valid,key=lambda x:(x[1] if higher_better else -x[1],x[0]))[0]
        out["worst"] = dict(seed=worst[0],value=worst[1])
    return out


def concentration(rows, key="B"):
    pairs = [(int(x["seed"]),float(x[key])) for x in rows]
    net = sum(v for _,v in pairs)
    absolute = sum(abs(v) for _,v in pairs)
    positive = [(s,v) for s,v in pairs if v>0]
    largest = sorted(positive,key=lambda x:(-x[1],x[0]))[0] if positive else None
    stable = net>CONCENTRATION_ABSOLUTE_FLOOR and net>CONCENTRATION_CANCELLATION_FRACTION*absolute
    leave_out = net-(largest[1] if largest else 0.)
    max_abs = sorted(pairs,key=lambda x:(-abs(x[1]),x[0]))[0] if pairs else None
    return dict(net_sum=net, mean=net/len(pairs) if pairs else None,
        maximum_absolute_seed=max_abs[0] if max_abs else None, maximum_absolute_value=max_abs[1] if max_abs else None,
        leave_one_out_means=[dict(omitted_seed=s,mean=(net-v)/(len(pairs)-1)) for s,v in pairs] if len(pairs)>1 else [],
        sum_positive=sum(v for _,v in pairs if v>0), sum_negative=sum(v for _,v in pairs if v<0),
        sum_absolute=absolute, largest_positive_seed=largest[0] if largest else None,
        largest_positive_value=largest[1] if largest else None,
        net_without_largest_positive_seed=leave_out,
        net_becomes_nonpositive_without_largest_positive=bool(largest and net>0 and leave_out<=0),
        largest_positive_exceeds_half_net=bool(largest and net>0 and largest[1]>.5*net),
        largest_seed_net_contribution_percent=100*largest[1]/net if stable and largest else None,
        contribution_percent_status="DEFINED_POSITIVE_NET_WITH_LIMITED_CANCELLATION" if stable and largest else "NOT_DEFINED",
        policy=PROTOCOL["concentration"], values_by_seed=[dict(seed=s,value=v) for s,v in pairs])


def aggregate_confirmation(rows, seeds, phase="confirmation"):
    """Descriptive aggregation only. It never rescans meshes or loads models."""
    seeds = tuple(int(x) for x in seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    selected = [r for r in rows if r.get("model") in MODELS and int(r.get("K",12))==12 and int(r["seed"]) in seeds]
    index = {(r["model"],int(r["parent"]),int(r["seed"])):r for r in selected}
    expected = {(m,p,s) for m in MODELS for p in (0,1) for s in seeds}
    if set(index)!=expected or len(selected)!=len(expected):
        raise ValueError("Require exactly three models x two parents x every fixed seed")
    result = dict(schema="t1_native_confirmation_summary_v1", phase=phase, seeds=list(seeds),
        number_of_training_parents=2, seed_trials_per_parent=len(seeds), models={}, paired_T1_vs={}, bindings=[],
        statistical_scope="Descriptive paired noise trials on two training parents; no independent-object generalization inference",
        protocol=PROTOCOL)
    for model in MODELS:
        group = dict(parents={},binding={})
        for parent in (0,1):
            scalar = [dict(seed=s,**scalar_metrics(index[(model,parent,s)]["metrics"])) for s in seeds]
            names = list(scalar[0].keys())[1:]
            summaries = {key:summarize_values(scalar,key,higher_better=("coverage" in key and "RMS" not in key) or key=="free_angle_p05_degrees") for key in names}
            below = [x["seed"] for x in scalar if x["remainder_symmetric_RMS_bbox_pct"]<2.]
            group["parents"][str(parent)] = dict(per_seed=scalar, metrics=summaries,
                worst_RMS=summaries["remainder_symmetric_RMS_bbox_pct"]["worst"],
                RMS_below_historical_2pct_count=len(below) if parent==0 else None,
                RMS_below_historical_2pct_seeds=below if parent==0 else None,
                historical_2pct_scope="P0 descriptive historical reference only; not a universal success threshold")
        binding = [dict(seed=s,**cross_binding(index[(model,0,s)]["metrics"],index[(model,1,s)]["metrics"])) for s in seeds]
        result["bindings"].extend(dict(phase=phase,model=model,seed=row["seed"],K=12,metrics=row) for row in binding)
        group["binding"] = dict(per_seed=binding,
            metrics={key:summarize_values(binding,key,higher_better=True) for key in ("Delta0","Delta1","B")},
            both_positive_count=sum(x["both_directions_positive"] for x in binding),
            direction0_positive_count=sum(x["Delta0"]>0 for x in binding),
            direction1_positive_count=sum(x["Delta1"]>0 for x in binding),
            concentration=concentration(binding))
        result["models"][model] = group
    for baseline in ("Native0","T2"):
        compare = dict(parents={},binding={})
        for parent in (0,1):
            a = result["models"][baseline]["parents"][str(parent)]["per_seed"]
            b = result["models"]["T1"]["parents"][str(parent)]["per_seed"]
            paired = []
            for old,new in zip(a,b):
                values = {key:(old[key]-new[key] if old[key] is not None and new[key] is not None else None)
                          for key in old if key!="seed"}
                # Most fields use baseline-T1; coverage/angles retain explicitly
                # labeled arithmetic and are not all treated as improvement.
                paired.append(dict(seed=old["seed"],**values))
            key = "remainder_symmetric_RMS_bbox_pct"
            compare["parents"][str(parent)] = dict(per_seed_baseline_minus_T1=paired,
                metrics={name:summarize_values(paired,name,higher_better=True) for name in paired[0] if name!="seed"},
                remainder_T1_better_count=sum(x[key]>0 for x in paired),
                remainder_T1_tie_count=sum(x[key]==0 for x in paired),
                remainder_T1_worse_count=sum(x[key]<0 for x in paired),
                sign="baseline minus T1; positive means lower T1 only, not universally better for coverage/angle fields")
        old = result["models"][baseline]["binding"]["per_seed"]
        new = result["models"]["T1"]["binding"]["per_seed"]
        paired = [dict(seed=a["seed"],**{key:b[key]-a[key] for key in ("Delta0","Delta1","B")}) for a,b in zip(old,new)]
        compare["binding"] = dict(per_seed_T1_minus_baseline=paired,
            metrics={key:summarize_values(paired,key,higher_better=True) for key in ("Delta0","Delta1","B")},
            T1_higher_B_count=sum(x["B"]>0 for x in paired),
            T1_both_direction_deltas_higher_count=sum(x["Delta0"]>0 and x["Delta1"]>0 for x in paired),
            concentration_of_paired_B_increment=concentration(paired), sign="T1 minus baseline; positive is stronger binding")
        result["paired_T1_vs"][baseline] = compare
    json.dumps(result,allow_nan=False)
    return result


def run_cpu_tests():
    """Small evaluator fixtures only; no actual generated output is inspected."""
    base = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],dtype=np.float32)
    quality = shape_quality(base,2.)
    assert quality["exact_zero_area_count"]==0
    from .native_inpainting_metrics import near_context_diagnostic
    old = near_context_diagnostic(base,base,2.)
    near = near_C_area_diagnostic(base,base,2.,old)
    assert near["near_C_surface_sample_fraction"]==1.
    assert near["near_C_and_boundary_band_sample_fraction"]+near["near_C_outside_boundary_band_sample_fraction"]==1.
    vals = [dict(seed=1,x=1.),dict(seed=2,x=3.),dict(seed=3,x=5.),dict(seed=4,x=7.)]
    dist = summarize_values(vals,"x")
    assert dist["mean"]==4. and dist["median"]==4. and dist["q25"]==2.5 and dist["q75"]==5.5 and dist["worst"]["seed"]==4
    cancel = concentration([dict(seed=1,B=1.),dict(seed=2,B=-.999999999999)])
    assert cancel["contribution_percent_status"]=="NOT_DEFINED"
    stable = concentration([dict(seed=1,B=3.),dict(seed=2,B=1.)])
    assert stable["largest_seed_net_contribution_percent"]==75. and stable["net_without_largest_positive_seed"]==1.
    negative = concentration([dict(seed=1,B=-1.),dict(seed=2,B=.1)])
    assert negative["contribution_percent_status"]=="NOT_DEFINED"
    rows=[]
    for model in MODELS:
        for parent in (0,1):
            for seed in (1,2):
                own = {"Native0":3.,"T1":1.,"T2":2.}[model]+.1*seed
                cross=[dict(symmetric_rms_bbox_pct=own+(0 if p==parent else 1.),predicted_to_target_rms_bbox_pct=own,target_to_predicted_rms_bbox_pct=own) for p in (0,1)]
                common=[dict(rms_bbox_pct=own+(0 if p==parent else 1.),fraction_target_points_within_1pct_bbox=.5) for p in (0,1)]
                metric=dict(K=12,variant="raw",remainder=dict(own=cross[parent],cross=cross),whole=dict(surface=cross[parent]),common_unknown_R12=dict(own=common[parent],cross=common))
                rows.append(dict(model=model,parent=parent,seed=seed,K=12,metrics=metric))
    summary=aggregate_confirmation(rows,(1,2),phase="FIXTURE")
    assert summary["models"]["T1"]["binding"]["both_positive_count"]==2
    assert summary["paired_T1_vs"]["Native0"]["parents"]["0"]["remainder_T1_better_count"]==2
    assert summary["models"]["T1"]["parents"]["0"]["RMS_below_historical_2pct_count"]==2
    assert summary["models"]["T1"]["parents"]["0"]["metrics"]["CF_proper_pair_count"]["missing_count"]==2
    json.dumps([near,summary],allow_nan=False)
    return dict(status="PASS",fixture_families=7,tests=["actual_q_triangle","fixed_near_C_area_sampling_and_partition","distribution_quartiles_worst","near_zero_net_ratio_suppressed","stable_positive_net_ratio","negative_net_ratio_suppressed","legacy_missing_auxiliary_and_paired_binding_summary"],
        fixed_area_sample_diagnostics=1, original_near_context_diagnostics=1,
        model_calls=0,GPU_calls=0,geometry_autograd_calls=0,optimizer_updates=0,generated_outputs_scored=0)


