"""Phase-separated full evaluation for the gated FC/FF 2x2 factorial.

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

MODELS = ("M00", "M10", "M01", "M11")
PAIRINGS = (("M10", "M00"), ("M11", "M01"), ("M01", "M00"), ("M11", "M10"), ("M11", "M00"))
HIGHER_BETTER = frozenset(("fixed_R12_coverage_within_1pct_fraction", "boundary_coverage_fraction", "free_angle_p05_degrees"))
DESCRIPTIVE_ONLY = frozenset(("near_C_area_fraction", "near_C_boundary_band_area_fraction", "near_C_outside_boundary_band_area_fraction", "C_surface_area_fraction", "exact_C_free_shared_edge_count", "exact_C_free_shared_vertex_count", "C_inherited_nonmanifold_vertex_count", "C_current_winding_conflict_count", "shared_edge_component_count"))
ERROR_METRIC = "remainder_symmetric_RMS_bbox_pct"
QUALITY_METRICS = ("CF_proper_affected_free_fraction", "CF_proper_pair_count", "FF_illegal_pair_count", "CF_coplanar_pair_count", "FF_coplanar_pair_count", "free_minimum_angle_lt5_fraction", "free_q_gt10_fraction", "free_near_degenerate_count", "free_exact_zero_area_count")
PROTOCOL = dict(original_metric_protocol=ORIGINAL_METRIC_PROTOCOL,
    model_order=list(MODELS), paired_comparisons=[f"{candidate}_minus_{reference}" for candidate,reference in PAIRINGS],
    pair_sign="candidate minus reference; negative lowers an error/count, positive increases coverage/angle/binding",
    statistics="Per training parent, paired fixed Gaussian seeds; mean/median/Q25/Q75/worst; no independent-object inference",
    interpretation="M00=old A, M10=gated FC only, M01=gated FF only, M11=old C_gate. Development and confirmation are separate phases. Form all main/conditional effects and interaction per parent/seed before aggregation; no composite score, automatic winner or causal claim beyond the fixed controlled setting.",
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


def aggregate_results(rows, phase, seeds):
    """Only saved scalar arithmetic. Each phase requires exactly 4 x 2 x 8 rows."""
    seeds=tuple(int(seed) for seed in seeds)
    if phase not in ("development","confirmation"):raise ValueError("Require an explicit independent phase")
    if len(seeds)!=8 or len(set(seeds))!=8:
        raise ValueError("Require exactly eight fixed distinct seeds per phase")
    selected=[row for row in rows if row.get("phase")==phase and row.get("model") in MODELS and int(row.get("K",12))==12 and int(row["seed"]) in seeds]
    index={(row["model"],int(row["parent"]),int(row["seed"])):row for row in selected}
    expected={(model,parent,seed) for model in MODELS for parent in (0,1) for seed in seeds}
    if set(index)!=expected or len(selected)!=len(expected):
        raise ValueError("Require exactly M00/M10/M01/M11 x P0/P1 x every registered seed within this phase, without duplicates")
    result=dict(schema="t1_gated_factorial_summary_v1",phase=phase,seeds=list(seeds),models={},paired={},bindings=[],
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
    result["factorial"]=_factorial_tables(result,seeds)
    json.dumps(result,allow_nan=False)
    return result


EFFECT_FORMULAS={
    'D_FC_off':'S10-S00',
    'D_FC_on':'S11-S01',
    'D_FF_off':'S01-S00',
    'D_FF_on':'S11-S10',
    'I':'S11-S10-S01+S00',
}


def _effects(values):
    if any(values.get(m) is None for m in MODELS):return {key:None for key in EFFECT_FORMULAS}
    s00,s10,s01,s11=(float(values[m]) for m in MODELS)
    return dict(D_FC_off=s10-s00,D_FC_on=s11-s01,D_FF_off=s01-s00,D_FF_on=s11-s10,I=s11-s10-s01+s00)


def _effect_distribution(values,seeds):
    finite=[(seed,float(value)) for seed,value in zip(seeds,values) if value is not None and np.isfinite(float(value))]
    a=np.asarray([value for seed,value in finite],dtype=np.float64)
    low=float(a.min()) if len(a) else None;high=float(a.max()) if len(a) else None
    return dict(expected_count=len(seeds),finite_count=len(finite),missing_or_nonfinite_count=len(seeds)-len(finite),
        mean=float(a.mean()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
        q25=float(np.quantile(a,.25)) if len(a) else None,q75=float(np.quantile(a,.75)) if len(a) else None,
        minimum=low,maximum=high,positive_count=int((a>0).sum()),negative_count=int((a<0).sum()),zero_count=int((a==0).sum()),
        minimum_seeds=[seed for seed,value in finite if value==low],maximum_seeds=[seed for seed,value in finite if value==high],
        maximum_absolute_seeds=[seed for seed,value in finite if abs(value)==float(np.abs(a).max())] if len(a) else [],
        values_by_seed=[dict(seed=seed,value=value) for seed,value in zip(seeds,values)],
        sign_counts='Exact saved scalar arithmetic; no post-outcome threshold; positive/negative are not automatic improvement labels')


def _factorial_block(by_model,seeds,names):
    per_seed=[]
    for index,seed in enumerate(seeds):
        observed={model:by_model[model][index] for model in MODELS}
        if any(int(row['seed'])!=seed for row in observed.values()):raise ValueError('Unaligned factorial seed rows')
        effects={name:_effects({model:observed[model].get(name) for model in MODELS}) for name in names}
        per_seed.append(dict(seed=seed,metrics=effects))
    metrics={name:{effect:_effect_distribution([row['metrics'][name][effect] for row in per_seed],seeds)
             for effect in EFFECT_FORMULAS} for name in names}
    return dict(per_seed_effects=per_seed,metrics=metrics)


def _factorial_tables(summary,seeds):
    parents={}
    for parent in ('0','1'):
        values={model:summary['models'][model]['parents'][parent]['per_seed'] for model in MODELS}
        names=[name for name in values['M00'][0] if name!='seed']
        parents[parent]=_factorial_block(values,seeds,names)
    binding={model:summary['models'][model]['binding']['per_seed'] for model in MODELS}
    return dict(formulas=EFFECT_FORMULAS,parents=parents,binding=_factorial_block(binding,seeds,('Delta0','Delta1','B')),
        reduction='Compute each contrast within the same parent/seed first, then summarize those contrasts. Never subtract marginal medians or pool phases.',
        normalization='All original metric denominators and thresholds unchanged',
        interaction='I is nonadditivity in the displayed metric; its sign alone is not mechanistic synergy or a causal mechanism',
        geometry_calls=0,model_calls=0)


def run_cpu_tests():
    """Minimal scalar-only fixture. Not run on import; no model/autograd."""
    rows=[];seeds=tuple(range(1,9))
    for phase in ('development','confirmation'):
        for model in MODELS:
            for parent in (0,1):
                for seed in seeds:
                    own={'M00':10.,'M10':8.,'M01':9.,'M11':6.}[model]+seed*.125
                    cross=[dict(symmetric_rms_bbox_pct=own+(0 if p==parent else 2.),predicted_to_target_rms_bbox_pct=own,target_to_predicted_rms_bbox_pct=own) for p in (0,1)]
                    common=[dict(rms_bbox_pct=own+(0 if p==parent else 2.),fraction_target_points_within_1pct_bbox=.5) for p in (0,1)]
                    metric=dict(K=12,variant='raw',parent_index=parent,total_faces=112,free_faces=100,
                        remainder=dict(own=cross[parent],cross=cross),whole=dict(surface=cross[parent]),common_unknown_R12=dict(own=common[parent],cross=common),
                        guided_geometry_audit=dict(C_free=dict(proper_pair_count=1,proper_affected_face_fraction=.01),free_free=dict(proper_pair_count=2,coplanar_positive_area_pairs=0),boundary=dict(rms_bbox_pct=own,p95_bbox_pct=own,coverage_fraction=.5)),
                        t1_native_confirmation=dict(C_PRESERVED='PASS'))
                    rows.append(dict(phase=phase,model=model,parent=parent,seed=seed,K=12,metrics=metric))
    summary=aggregate_results(rows,'development',seeds)
    assert len(summary['bindings'])==32
    actual=summary['factorial']['parents']['0']['metrics'][ERROR_METRIC]
    expected=dict(D_FC_off=-2.,D_FC_on=-3.,D_FF_off=-1.,D_FF_on=-2.,I=-1.)
    assert all(np.isclose(actual[k]['mean'],v,rtol=0,atol=1e-12) and actual[k]['negative_count']==8 for k,v in expected.items())
    assert summary['factorial']['binding']['metrics']['B']['I']['zero_count']==8
    a=np.asarray([0,0,0,0,100,100,100,100]);b=np.asarray([0,0,100,100,100,100,100,100])
    distribution=_effect_distribution((b-a).tolist(),seeds)
    assert distribution['median']==0 and np.median(b)-np.median(a)==50
    try:aggregate_results(rows+[rows[0]],'development',seeds)
    except ValueError:pass
    else:raise AssertionError('Duplicate phase row accepted')
    try:aggregate_results([row for row in rows if row['phase']=='confirmation'],'development',seeds)
    except ValueError:pass
    else:raise AssertionError('Other phase substituted for missing scores')
    return dict(status='PASS',checks=6,model_calls=0,geometry_calls=0,autograd_calls=0,optimizer_updates=0)
