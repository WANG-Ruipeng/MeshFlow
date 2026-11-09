"""Fixed full offline evaluation and descriptive comparisons for T1/A/B/C.

The historical complete CPU evaluator is reused verbatim. Nothing in this
module trains, samples, repairs, computes guidance, or chooses a checkpoint.
"""
from __future__ import annotations

import json
import numpy as np

from .t1_native_confirmation_evaluation import (
    score_output, audit_context, cross_binding, scalar_metrics as _core_scalars,
    summarize_values, concentration, PROTOCOL as ORIGINAL_METRIC_PROTOCOL,
)

MODELS = ("T1", "A", "B", "C")
PAIRINGS = (("A", "T1"), ("B", "A"), ("C", "A"))
HIGHER_BETTER = frozenset(("fixed_R12_coverage_within_1pct_fraction", "boundary_coverage_fraction", "free_angle_p05_degrees"))
DESCRIPTIVE_ONLY = frozenset(("near_C_area_fraction", "near_C_boundary_band_area_fraction", "near_C_outside_boundary_band_area_fraction", "C_surface_area_fraction", "exact_C_free_shared_edge_count", "exact_C_free_shared_vertex_count", "C_inherited_nonmanifold_vertex_count", "C_current_winding_conflict_count", "shared_edge_component_count"))
ERROR_METRIC = "remainder_symmetric_RMS_bbox_pct"
QUALITY_METRICS = ("CF_proper_affected_free_fraction", "CF_proper_pair_count", "FF_illegal_pair_count", "CF_coplanar_pair_count", "FF_coplanar_pair_count", "free_minimum_angle_lt5_fraction", "free_q_gt10_fraction", "free_near_degenerate_count", "free_exact_zero_area_count")
PROTOCOL = dict(original_metric_protocol=ORIGINAL_METRIC_PROTOCOL,
    model_order=list(MODELS), paired_comparisons=[f"{candidate}_minus_{reference}" for candidate,reference in PAIRINGS],
    pair_sign="candidate minus reference; negative lowers an error/count, positive increases coverage/angle/binding",
    statistics="Per training parent, paired fixed Gaussian seeds; mean/median/Q25/Q75/worst; no independent-object inference",
    interpretation="B and C must be compared with continued-training A. Merely beating old T1 is not isolated mechanism evidence. No composite score or automatic winner.",
    binding="Baseline binding was already established; 8/8 positive is descriptive, not a newly invented success gate")


def _get(value, *keys):
    for key in keys:
        if not isinstance(value,dict) or key not in value:
            return None
        value=value[key]
    return value


def scalar_metrics(metrics):
    values=_core_scalars(metrics)
    extra=metrics.get("t1_native_confirmation",{})
    topo=extra.get("exact_coordinate_topology",{})
    attribution=topo.get("attribution",{})
    values.update(
        nan_coordinate_count=extra.get("nan_coordinate_count"),
        inf_coordinate_count=extra.get("inf_coordinate_count"),
        exact_C_free_shared_vertex_count=attribution.get("exact_C_free_shared_vertex_count"),
        C_inherited_nonmanifold_vertex_count=len(attribution["inherited_C_only_unmodified_link_vertex_ids"]) if "inherited_C_only_unmodified_link_vertex_ids" in attribution else None,
        C_current_winding_conflict_count=len(attribution["orientation_conflicts_C_only"]) if "orientation_conflicts_C_only" in attribution else None,
        free_related_current_winding_conflict_count=len(attribution["orientation_conflicts_free_related"]) if "orientation_conflicts_free_related" in attribution else None,
        shared_edge_component_count=_get(topo,"topology","shared_edge_components","count"),
        exact_index_boundary_edge_count=_get(topo,"topology","edge_incidence","one"),
        exact_index_overincident_edge_count=_get(topo,"topology","edge_incidence","more_than_two"))
    return values


def _counts(values, name):
    finite=[float(row[name]) for row in values if row.get(name) is not None and np.isfinite(float(row[name]))]
    if name in DESCRIPTIVE_ONLY:
        return dict(available=len(finite),better=None,tie=None,worse=None,scope="Descriptive proximity/connectivity count; no universal improvement direction assigned")
    sign=1 if name in HIGHER_BETTER else -1
    return dict(available=len(finite),better=sum(sign*x>0 for x in finite),tie=sum(x==0 for x in finite),worse=sum(sign*x<0 for x in finite),
        improvement_direction="positive candidate-reference" if sign==1 else "negative candidate-reference")


def aggregate_results(rows, seeds, phase="confirmation"):
    """Only saved scalar arithmetic. Complete four-model fixed matrix required."""
    seeds=tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("Require a nonempty fixed list of distinct seeds")
    selected=[row for row in rows if row.get("model") in MODELS and int(row.get("K",12))==12 and int(row["seed"]) in seeds]
    index={(row["model"],int(row["parent"]),int(row["seed"])):row for row in selected}
    expected={(model,parent,seed) for model in MODELS for parent in (0,1) for seed in seeds}
    if set(index)!=expected or len(selected)!=len(expected):
        raise ValueError("Require exactly T1/A/B/C x P0/P1 x every registered seed, without duplicates")
    result=dict(schema="t1_interface_summary_v1",phase=phase,seeds=list(seeds),models={},paired={},bindings=[],
        training_parents=2,noise_trials_per_parent=len(seeds),protocol=PROTOCOL,
        automatic_winner="NOT_DEFINED",checkpoint_selection="Fixed step500 only",geometry_calls=0,model_calls=0)
    for model in MODELS:
        group=dict(parents={},binding={})
        for parent in (0,1):
            values=[dict(seed=seed,**scalar_metrics(index[model,parent,seed]["metrics"])) for seed in seeds]
            names=[key for key in values[0] if key!="seed"]
            summaries={key:summarize_values(values,key,higher_better=key in HIGHER_BETTER) for key in names}
            group["parents"][str(parent)]=dict(per_seed=values,metrics=summaries,worst_RMS=summaries[ERROR_METRIC]["worst"],
                C_all_bitwise_preserved=all(_get(index[model,parent,seed]["metrics"],"t1_native_confirmation","C_PRESERVED")=="PASS" for seed in seeds),
                N_all_112_free_all_100=all(index[model,parent,seed]["metrics"].get("total_faces")==112 and index[model,parent,seed]["metrics"].get("free_faces")==100 for seed in seeds))
        binding=[dict(seed=seed,**cross_binding(index[model,0,seed]["metrics"],index[model,1,seed]["metrics"])) for seed in seeds]
        result["bindings"].extend(dict(phase=phase,model=model,seed=row["seed"],K=12,metrics=row) for row in binding)
        group["binding"]=dict(per_seed=binding,metrics={key:summarize_values(binding,key,higher_better=True) for key in ("Delta0","Delta1","B")},
            both_positive_count=sum(row["both_directions_positive"] for row in binding),
            direction0_positive_count=sum(row["Delta0"]>0 for row in binding),direction1_positive_count=sum(row["Delta1"]>0 for row in binding),
            concentration=concentration(binding),criterion="Descriptive; baseline binding is already established")
        result["models"][model]=group
    for candidate,reference in PAIRINGS:
        pair=dict(candidate=candidate,reference=reference,parents={},binding={})
        for parent in (0,1):
            old=result["models"][reference]["parents"][str(parent)]["per_seed"]
            new=result["models"][candidate]["parents"][str(parent)]["per_seed"]
            values=[dict(seed=a["seed"],**{key:b[key]-a[key] if b[key] is not None and a[key] is not None else None for key in a if key!="seed"}) for a,b in zip(old,new)]
            names=[key for key in values[0] if key!="seed"]
            improvements={key:_counts(values,key) for key in names}
            tradeoffs=[]
            for row in values:
                if row[ERROR_METRIC] is None:
                    continue
                worse_quality=[key for key in QUALITY_METRICS if row.get(key) is not None and row[key]>0]
                tradeoffs.append(dict(seed=row["seed"],remainder_RMS_change=row[ERROR_METRIC],
                    error_lower=row[ERROR_METRIC]<0,quality_metrics_increased=worse_quality,
                    lower_error_with_quality_cost=bool(row[ERROR_METRIC]<0 and worse_quality)))
            pair["parents"][str(parent)]=dict(per_seed_candidate_minus_reference=values,
                metrics={key:summarize_values(values,key,higher_better=key in HIGHER_BETTER) for key in names},
                improvement_counts=improvements,remainder_better_count=improvements[ERROR_METRIC]["better"],
                remainder_tie_count=improvements[ERROR_METRIC]["tie"],remainder_worse_count=improvements[ERROR_METRIC]["worse"],
                tradeoffs=tradeoffs,sign=PROTOCOL["pair_sign"])
        old=result["models"][reference]["binding"]["per_seed"]
        new=result["models"][candidate]["binding"]["per_seed"]
        binding=[dict(seed=a["seed"],**{key:b[key]-a[key] for key in ("Delta0","Delta1","B")}) for a,b in zip(old,new)]
        pair["binding"]=dict(per_seed_candidate_minus_reference=binding,
            metrics={key:summarize_values(binding,key,higher_better=True) for key in ("Delta0","Delta1","B")},
            higher_B_count=sum(row["B"]>0 for row in binding),both_direction_increments_positive_count=sum(row["Delta0"]>0 and row["Delta1"]>0 for row in binding),
            concentration=concentration(binding),sign="candidate minus reference; positive means stronger binding")
        result["paired"][f"{candidate}_minus_{reference}"]=pair
    json.dumps(result,allow_nan=False)
    return result


aggregate_confirmation=aggregate_results


def run_cpu_tests():
    """Synthetic saved-score arithmetic only; no evaluator/model invocation."""
    rows=[]
    seeds=(1,2,3)
    for model in MODELS:
        for parent in (0,1):
            for seed in seeds:
                own={"T1":4.,"A":3.,"B":2.,"C":2.5}[model]+.01*seed
                cross=[dict(symmetric_rms_bbox_pct=own+(0 if p==parent else 2.),predicted_to_target_rms_bbox_pct=own,target_to_predicted_rms_bbox_pct=own) for p in (0,1)]
                common=[dict(rms_bbox_pct=own+(0 if p==parent else 2.),fraction_target_points_within_1pct_bbox=.5) for p in (0,1)]
                metric=dict(K=12,variant="raw",parent_index=parent,total_faces=112,free_faces=100,
                    remainder=dict(own=cross[parent],cross=cross),whole=dict(surface=cross[parent]),common_unknown_R12=dict(own=common[parent],cross=common),
                    guided_geometry_audit=dict(C_free=dict(proper_pair_count=2 if model=="B" else 1,proper_affected_face_fraction=.02 if model=="B" else .01),free_free=dict(proper_pair_count=1,coplanar_positive_area_pairs=0),boundary=dict(rms_bbox_pct=own,p95_bbox_pct=own,coverage_fraction=.5)),
                    t1_native_confirmation=dict(C_PRESERVED="PASS"))
                rows.append(dict(model=model,parent=parent,seed=seed,K=12,metrics=metric))
    summary=aggregate_results(rows,seeds,phase="FIXTURE")
    assert len(summary["bindings"])==12
    assert summary["paired"]["A_minus_T1"]["parents"]["0"]["remainder_better_count"]==3
    assert summary["paired"]["B_minus_A"]["parents"]["0"]["metrics"][ERROR_METRIC]["mean"]==-1
    assert all(row["lower_error_with_quality_cost"] for row in summary["paired"]["B_minus_A"]["parents"]["0"]["tradeoffs"])
    assert summary["paired"]["C_minus_A"]["binding"]["higher_B_count"]==0
    assert summary["models"]["A"]["parents"]["0"]["metrics"][ERROR_METRIC]["worst"]["seed"]==3
    try:aggregate_results(rows+[rows[0]],seeds)
    except ValueError:pass
    else:raise AssertionError("Duplicate fixed matrix row accepted")
    return dict(status="PASS",fixture_checks=7,model_calls=0,geometry_calls=0,real_output_scores=0,training_updates=0)
