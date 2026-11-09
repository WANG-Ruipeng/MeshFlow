"""Read-only complete evaluation for the A continuation / late FC study.

The historical full scorer is reused unchanged. Aggregation consumes saved
scalars only; it never calls a model, OT, autograd, or geometry repair.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .t1_native_confirmation_evaluation import (
    score_output, audit_context, cross_binding, summarize_values, concentration,
    PROTOCOL as ORIGINAL_METRIC_PROTOCOL,
)
from .t1_gated_factorial_evaluation import (
    scalar_metrics as _original_scalars, HIGHER_BETTER,
    DESCRIPTIVE_ONLY as _original_descriptive, QUALITY_METRICS,
)

MODELS = ('A_old', 'FC_old', 'A_continue', 'A_then_FC')
PHASE = 'development_followup'
ERROR_METRIC = 'remainder_symmetric_RMS_bbox_pct'
AREA_METRIC = 'free_total_surface_area'
PRIMARY_PAIRINGS = (('A_continue', 'A_old'), ('A_then_FC', 'A_continue'))
PAIRINGS = PRIMARY_PAIRINGS + (('A_old', 'A_continue'), ('FC_old', 'A_continue'),
    ('A_continue', 'A_continue'), ('FC_old', 'A_old'))
DESCRIPTIVE_ONLY = _original_descriptive | frozenset((AREA_METRIC,))
PROTOCOL = dict(original_metric_protocol=ORIGINAL_METRIC_PROTOCOL,
    phase=PHASE, model_order=list(MODELS),
    primary_pairings=[f'{a}_minus_{b}' for a,b in PRIMARY_PAIRINGS],
    pair_sign='candidate minus reference; negative lowers error/count; positive increases coverage/angle/binding',
    free_area='Total geometric area of all fixed 100 free triangles; original model coordinate squared. Overlap counted per triangle; no universal better direction.',
    area_source='Read unchanged free_shape_quality.total_area. If missing, explicitly supplement all four models by the same FP64 raw-area formula; never silently mix formulas.',
    phase_scope='Previously observed 9601..9608 noises on the same two training parents; development follow-up, not independent confirmation or new-object generalization.',
    comparison_scope='A_continue and A_then_FC have equal cumulative1500 updates from the same A_old. A_old/FC_old have cumulative1000; FC_old is a different-budget route reference, not an equal-budget early-FC control.',
    summaries='Object-wise paired fixed-noise descriptive distributions; form differences before medians; no composite score or automatic winner',
    binding='Already established positive binding is descriptive, not a new success criterion')


def _get(value,*keys):
    for key in keys:
        if not isinstance(value,dict) or key not in value:return None
        value=value[key]
    return value


def scalar_metrics(metrics):
    values=_original_scalars(metrics)
    supplement=metrics.get('a_then_fc_area_supplement',{})
    values[AREA_METRIC]=supplement.get('total_area') if supplement else _get(metrics,'t1_native_confirmation','free_shape_quality','total_area')
    return values


def _path(value):
    p=Path(value)
    if p.is_file():return p
    text=str(value).replace('\\','/')
    if len(text)>2 and text[1]==':':return Path('/mnt/'+text[0].lower()+text[2:])
    if text.startswith('/mnt/') and len(text)>7:return Path(text[5].upper()+':/'+text[7:])
    return p


def complete_area_metrics(rows):
    """Explicit optional supplement; returns copied rows without changing inputs.

    With complete historical quality fields, no raw file is opened. If any
    field is absent, every row is recomputed using the same pure NumPy formula.
    This helper never runs distance, intersection, model, or repair code.
    """
    rows=list(rows)
    missing=[(r['model'],r['parent'],r['seed']) for r in rows if scalar_metrics(r['metrics'])[AREA_METRIC] is None]
    if not missing:
        return rows,dict(status='PASS',source='Unchanged saved free_shape_quality.total_area',rows=len(rows),raw_area_calculations=0,geometry_distance_calls=0,intersection_calls=0)
    result=[]
    for row in rows:
        with np.load(_path(row['raw_path']),allow_pickle=False) as archive:full=archive['output'].copy().reshape(112,3,3)
        if full.dtype!=np.float32 or not np.isfinite(full).all():raise ValueError('Area supplement requires unchanged finite FP32 raw112')
        free=full[12:].astype(np.float64)
        area=float((.5*np.linalg.norm(np.cross(free[:,1]-free[:,0],free[:,2]-free[:,0]),axis=-1)).sum())
        metrics=dict(row['metrics']);metrics['a_then_fc_area_supplement']=dict(total_area=area,
            source='FP64 direct cross-product area of raw[12:112], all models supplemented with identical formula',
            coordinate_units='original model coordinate squared',triangles=100,geometry_modified=False)
        result.append(dict(row,metrics=metrics))
    return result,dict(status='PASS',source='Uniform raw area supplementation for every model',missing_saved_area=missing,
        rows=len(rows),raw_area_calculations=len(rows),geometry_distance_calls=0,intersection_calls=0)


def _counts(rows,key):
    a=[float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(row[key])]
    if key in DESCRIPTIVE_ONLY:return dict(available=len(a),better=None,tie=None,worse=None,scope='Descriptive, no universal improvement direction')
    sign=1 if key in HIGHER_BETTER else -1
    return dict(available=len(a),better=sum(sign*x>0 for x in a),tie=sum(x==0 for x in a),worse=sum(sign*x<0 for x in a),
        improvement_direction='positive candidate-reference' if sign==1 else 'negative candidate-reference')


def _summary(rows,key):
    result=summarize_values(rows,key,higher_better=key in HIGHER_BETTER)
    if key in DESCRIPTIVE_ONLY:
        result['worst']=None;result['higher_is_better']=None
        result['direction']='Descriptive; minimum and maximum are reported without a quality ranking'
    return result


def aggregate_results(rows,seeds,phase=PHASE):
    """Require one complete 4x2x8 matrix; no geometry/file/model calls."""
    seeds=tuple(map(int,seeds));rows=list(rows)
    if phase!=PHASE:raise ValueError('Only previously observed development_followup is supported')
    if len(seeds)!=8 or len(set(seeds))!=8:raise ValueError('Require exactly eight registered distinct seeds')
    if any(row.get('phase')!=phase or row.get('model') not in MODELS or int(row.get('K',12))!=12 or int(row['seed']) not in seeds for row in rows):
        raise ValueError('Unexpected phase/model/K/seed; do not silently pool or discard rows')
    index={(row['model'],int(row['parent']),int(row['seed'])):row for row in rows}
    expected={(m,p,s) for m in MODELS for p in (0,1) for s in seeds}
    if set(index)!=expected or len(rows)!=len(expected):raise ValueError('Require all64 rows, no duplicates or missing endpoints')
    if any(scalar_metrics(row['metrics'])[AREA_METRIC] is None for row in rows):
        raise ValueError('Missing free area; root must explicitly run complete_area_metrics on the entire matrix')
    result=dict(schema='a_then_fc_summary_v1',phase=phase,seeds=list(seeds),models={},paired={},bindings=[],
        protocol=PROTOCOL,training_parents=2,noise_trials_per_parent=8,automatic_winner='NOT_DEFINED',
        checkpoint_selection='Fixed additional500 / cumulative1500 for new arms; old endpoints cumulative1000',geometry_calls=0,model_calls=0)
    for model in MODELS:
        group=dict(parents={},binding={})
        for parent in (0,1):
            values=[dict(seed=seed,**scalar_metrics(index[model,parent,seed]['metrics'])) for seed in seeds]
            names=[key for key in values[0] if key!='seed'];summaries={key:_summary(values,key) for key in names}
            group['parents'][str(parent)]=dict(per_seed=values,metrics=summaries,worst_RMS=summaries[ERROR_METRIC]['worst'],
                C_all_bitwise_preserved=all(_get(index[model,parent,seed]['metrics'],'t1_native_confirmation','C_PRESERVED')=='PASS' for seed in seeds),
                N_all_112_free_all_100=all(index[model,parent,seed]['metrics'].get('total_faces')==112 and index[model,parent,seed]['metrics'].get('free_faces')==100 for seed in seeds))
        binding=[dict(seed=seed,**cross_binding(index[model,0,seed]['metrics'],index[model,1,seed]['metrics'])) for seed in seeds]
        result['bindings'].extend(dict(phase=phase,model=model,seed=row['seed'],K=12,metrics=row) for row in binding)
        group['binding']=dict(per_seed=binding,metrics={key:summarize_values(binding,key,higher_better=True) for key in ('Delta0','Delta1','B')},
            both_positive_count=sum(row['both_directions_positive'] for row in binding),
            direction0_positive_count=sum(row['Delta0']>0 for row in binding),direction1_positive_count=sum(row['Delta1']>0 for row in binding),
            concentration=concentration(binding),criterion='Baseline binding already established; descriptive only')
        result['models'][model]=group
    for candidate,reference in PAIRINGS:
        name=f'{candidate}_minus_{reference}'
        pair=dict(candidate=candidate,reference=reference,parents={},binding={},primary=(candidate,reference) in PRIMARY_PAIRINGS,
            equal_cumulative_budget=candidate in ('A_continue','A_then_FC') and reference in ('A_continue','A_then_FC'),
            scope='Old FC is a historical route/budget reference; only late-FC versus A_continue isolates the new enabled loss')
        for parent in ('0','1'):
            old=result['models'][reference]['parents'][parent]['per_seed'];new=result['models'][candidate]['parents'][parent]['per_seed']
            values=[dict(seed=a['seed'],**{key:b[key]-a[key] if b[key] is not None and a[key] is not None else None for key in a if key!='seed'}) for a,b in zip(old,new)]
            names=[key for key in values[0] if key!='seed'];counts={key:_counts(values,key) for key in names}
            tradeoffs=[]
            for row in values:
                cost=[key for key in QUALITY_METRICS if row.get(key) is not None and row[key]>0]
                tradeoffs.append(dict(seed=row['seed'],remainder_RMS_change=row[ERROR_METRIC],quality_metrics_increased=cost,
                    lower_error_with_quality_cost=bool(row[ERROR_METRIC] is not None and row[ERROR_METRIC]<0 and cost)))
            pair['parents'][parent]=dict(per_seed_candidate_minus_reference=values,metrics={key:_summary(values,key) for key in names},
                improvement_counts=counts,remainder_better_count=counts[ERROR_METRIC]['better'],remainder_tie_count=counts[ERROR_METRIC]['tie'],
                remainder_worse_count=counts[ERROR_METRIC]['worse'],tradeoffs=tradeoffs,sign=PROTOCOL['pair_sign'])
        old=result['models'][reference]['binding']['per_seed'];new=result['models'][candidate]['binding']['per_seed']
        binding=[dict(seed=a['seed'],**{key:b[key]-a[key] for key in ('Delta0','Delta1','B')}) for a,b in zip(old,new)]
        pair['binding']=dict(per_seed_candidate_minus_reference=binding,metrics={key:summarize_values(binding,key,higher_better=True) for key in ('Delta0','Delta1','B')},
            higher_B_count=sum(row['B']>0 for row in binding),both_direction_increments_positive_count=sum(row['Delta0']>0 and row['Delta1']>0 for row in binding),
            concentration=concentration(binding),sign='candidate-reference; positive means stronger binding')
        result['paired'][name]=pair
    json.dumps(result,allow_nan=False)
    return result


def run_cpu_tests():
    """Small synthetic saved-scalar fixture; root may call once, never on import."""
    seeds=tuple(range(9601,9609));rows=[]
    for model in MODELS:
        for parent in (0,1):
            for j,seed in enumerate(seeds):
                own={'A_old':10.,'FC_old':8.,'A_continue':9.,'A_then_FC':7.}[model]+j*.125
                cross=[dict(symmetric_rms_bbox_pct=own+(0 if p==parent else 2.),predicted_to_target_rms_bbox_pct=own,target_to_predicted_rms_bbox_pct=own) for p in (0,1)]
                common=[dict(rms_bbox_pct=own+(0 if p==parent else 2.),fraction_target_points_within_1pct_bbox=.5) for p in (0,1)]
                metrics=dict(K=12,variant='raw',parent_index=parent,total_faces=112,free_faces=100,
                    remainder=dict(own=cross[parent],cross=cross),whole=dict(surface=cross[parent]),common_unknown_R12=dict(own=common[parent],cross=common),
                    guided_geometry_audit=dict(C_free=dict(proper_pair_count=1,proper_affected_face_fraction=.01),free_free=dict(proper_pair_count=2,coplanar_positive_area_pairs=0),boundary=dict(rms_bbox_pct=own,p95_bbox_pct=own,coverage_fraction=.5)),
                    t1_native_confirmation=dict(C_PRESERVED='PASS',free_shape_quality=dict(total_area=3.+j*.25)))
                rows.append(dict(phase=PHASE,model=model,parent=parent,seed=seed,K=12,metrics=metrics))
    summary=aggregate_results(rows,seeds)
    assert len(summary['bindings'])==32 and len(summary['paired'])==6
    for name,expected in (('A_continue_minus_A_old',-1.),('A_then_FC_minus_A_continue',-2.)):
        group=summary['paired'][name]['parents']['0']
        assert np.isclose(group['metrics'][ERROR_METRIC]['mean'],expected,rtol=0,atol=1e-12) and group['remainder_better_count']==8
    assert summary['paired']['A_continue_minus_A_continue']['parents']['0']['remainder_tie_count']==8
    assert summary['paired']['A_then_FC_minus_A_continue']['parents']['0']['improvement_counts'][AREA_METRIC]['better'] is None
    a=np.asarray([0,0,0,0,100,100,100,100]);b=np.asarray([0,0,100,100,100,100,100,100])
    paired=[dict(seed=s,value=float(v)) for s,v in zip(seeds,b-a)]
    assert summarize_values(paired,'value')['median']==0 and np.median(b)-np.median(a)==50
    for bad in (rows+[rows[0]],[dict(row,phase='confirmation') for row in rows]):
        try:aggregate_results(bad,seeds)
        except ValueError:pass
        else:raise AssertionError('Duplicate or wrong-phase matrix accepted')
    return dict(status='PASS',checks=6,model_calls=0,geometry_calls=0,autograd_calls=0,optimizer_updates=0)
