"""Read-only reuse of saved A/C generation evidence and full training t logs.

No model, OT, autograd, distance matrix, intersection classifier, output-to-GT
matching, or geometry modification is called. New measurements are only cheap
triangle areas and coordinate moments/bounds of the archived raw free faces.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

SEEDS=tuple(range(9501,9509))
CHECKPOINT_SHA={
    ('A',500):'43e56e107b0c6d28766aebd8cf8a0a764d8d52ef62ba2ca6e756d881634f876b',
    ('C',500):'8a40651bcf54aebb2c48cd19a2433fb4e0679370c0da6e79e916cc727f067168',
    ('A',250):'aea34110ae1f75c108298a366e8b2668c28aa09ad8fc7d683890f5caa415e305',
    ('C',250):'a7b558451a3e733a5173f7a7a7a180df7358340418e4fe28b11d35f05a0f1ec2',
}
TIME_EDGES=(0.,.2,.4,.6,.8,1.)


def _get(value,*keys,default=None):
    for key in keys:
        if not isinstance(value,dict) or key not in value:return default
        value=value[key]
    return value


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _path(value):
    path=Path(value)
    if path.exists():return path.resolve()
    text=str(value).replace('\\','/')
    if text.startswith('/mnt/') and len(text)>7:
        alternative=Path(text[5].upper()+':/'+text[7:])
        if alternative.exists():return alternative.resolve()
    if len(text)>2 and text[1]==':':
        alternative=Path('/mnt/'+text[0].lower()+text[2:])
        if alternative.exists():return alternative.resolve()
    return path.resolve()


def locate_source_run(outputs_root):
    """Discover by actual report/results/checkpoint metadata, never timestamp.

    This does not rehash multi-GB checkpoints. The root asset audit must verify
    their actual bytes once; identities here are checked against saved records.
    """
    matches=[]
    for report in sorted(Path(outputs_root).rglob('THREE_ARM_INTERFACE_REPORT.md')):
        result_path=report.parent/'results.json'
        if not result_path.is_file():continue
        data=json.loads(result_path.read_text(encoding='utf-8-sig'))
        valid=True
        for (arm,step),expected in CHECKPOINT_SHA.items():
            entry=_get(data,'arms',arm,'checkpoints',str(step))
            filename=f'additional_step{step}_cumulative_step{500+step}.pt'
            if not entry or entry.get('sha256')!=expected or not (report.parent/arm/filename).is_file():
                # Intermediate weights are optional; endpoints identify a run.
                if step==500:valid=False;break
        if valid:matches.append(report.parent.resolve())
    if len(matches)!=1:raise RuntimeError(f'Expected one report/checkpoint-identified run, found {len(matches)}: {matches}')
    return matches[0]


def _distribution(value):
    a=np.asarray(value,dtype=np.float64).reshape(-1);finite=a[np.isfinite(a)]
    return dict(count=len(a),finite_count=len(finite),minimum=float(finite.min()) if len(finite) else None,
        q01=float(np.quantile(finite,.01)) if len(finite) else None,
        q05=float(np.quantile(finite,.05)) if len(finite) else None,
        q25=float(np.quantile(finite,.25)) if len(finite) else None,
        median=float(np.median(finite)) if len(finite) else None,
        q75=float(np.quantile(finite,.75)) if len(finite) else None,
        q95=float(np.quantile(finite,.95)) if len(finite) else None,
        q99=float(np.quantile(finite,.99)) if len(finite) else None,
        maximum=float(finite.max()) if len(finite) else None,mean=float(finite.mean()) if len(finite) else None)


def _cheap_geometry(raw,length):
    full=np.asarray(raw)
    if full.dtype!=np.float32 or full.shape not in ((112,9),(112,3,3)) or not np.isfinite(full).all():
        raise ValueError('Require unchanged finite FP32 N112 output')
    free=full.reshape(112,3,3)[12:].astype(np.float64)
    corners=free.reshape(-1,3);centroid=corners.mean(axis=0);centered=corners-centroid
    areas=.5*np.linalg.norm(np.cross(free[:,1]-free[:,0],free[:,2]-free[:,0]),axis=-1)
    low,high=corners.min(axis=0),corners.max(axis=0);extent=high-low
    return dict(status='PASS_CHEAP_READONLY_MOMENTS',total_faces=112,free_faces=100,
        free_total_area=float(areas.sum()),free_total_area_over_target_L2=float(areas.sum()/(length*length)),
        per_free_face_area=areas.tolist(),free_face_area_distribution=_distribution(areas),
        normalized_face_area_distribution=_distribution(areas/(length*length)),
        exact_zero_area_faces=int((areas==0).sum()),
        free_corner_centroid=centroid.tolist(),centroid_definition='Arithmetic mean of all300 free corner copies; not surface-area weighted',
        centered_coordinate_RMS=float(np.sqrt(np.mean(centered*centered))),
        centered_coordinate_RMS_over_target_L=float(np.sqrt(np.mean(centered*centered))/length),
        centered_radial_RMS=float(np.sqrt(np.mean(np.sum(centered*centered,axis=1)))),
        centered_axis_RMS=np.sqrt(np.mean(centered*centered,axis=0)).tolist(),
        bbox_min=low.tolist(),bbox_max=high.tolist(),bbox_extent=extent.tolist(),
        bbox_diagonal=float(np.linalg.norm(extent)),bbox_diagonal_over_target_L=float(np.linalg.norm(extent)/length),
        bbox_max_extent=float(extent.max()),target_bbox_L=float(length),
        face_identity='Local archived output indices only; no cross-model or GT face/corner correspondence is assigned',
        caveat='Area or scale decreases alone do not establish collapse; unchanged scale does not exclude local errors')


def _saved_metrics(metrics):
    g=metrics.get('guided_geometry_audit',{});own=_get(metrics,'remainder','own',default={})
    cf=g.get('C_free',{});ff=g.get('free_free',{});boundary=g.get('boundary',{})
    extra=metrics.get('t1_native_confirmation',{});quality=extra.get('free_shape_quality',{})
    near=extra.get('near_C_area',{})
    values=dict(
        remaining_symmetric_RMS_bbox_pct=own.get('symmetric_rms_bbox_pct'),
        free_to_R12_RMS_bbox_pct=own.get('predicted_to_target_rms_bbox_pct'),
        R12_to_free_RMS_bbox_pct=own.get('target_to_predicted_rms_bbox_pct'),
        whole_symmetric_RMS_bbox_pct=_get(metrics,'whole','surface','symmetric_rms_bbox_pct'),
        fixed_R12_coverage_RMS_bbox_pct=_get(metrics,'common_unknown_R12','own','rms_bbox_pct'),
        fixed_R12_coverage_within_1pct_fraction=_get(metrics,'common_unknown_R12','own','fraction_target_points_within_1pct_bbox'),
        Gamma_RMS_bbox_pct=boundary.get('rms_bbox_pct'),Gamma_p95_bbox_pct=boundary.get('p95_bbox_pct'),Gamma_coverage_fraction=boundary.get('coverage_fraction'),
        CF_affected_free_fraction=cf.get('proper_affected_face_fraction'),CF_proper_pairs=cf.get('proper_pair_count'),
        FF_intersections=None if not ff else ff.get('proper_pair_count',0)+ff.get('coplanar_positive_area_pairs',0),
        FF_proper_pairs=ff.get('proper_pair_count'),FF_coplanar_pairs=ff.get('coplanar_positive_area_pairs'),
        CF_coplanar_pairs=cf.get('coplanar_positive_area_pairs'),
        CF_unresolved_or_degenerate=cf.get('unresolved_or_degenerate_pairs'),FF_unresolved_or_degenerate=ff.get('unresolved_or_degenerate_pairs'),
        narrow_fraction=quality.get('min_angle_lt5_fraction'),q_gt10_fraction=quality.get('q_actual_gt10_fraction'),
        near_degenerate_count=quality.get('near_degenerate_area_count'),exact_zero_area_count=quality.get('exact_zero_area_count'),
        nan_count=extra.get('nan_coordinate_count'),inf_count=extra.get('inf_coordinate_count'),
        stored_free_total_area=quality.get('total_area'),
        near_C_area_fraction=near.get('near_C_surface_sample_fraction'),
        near_C_boundary_band_area_fraction=near.get('near_C_and_boundary_band_sample_fraction'),
        near_C_outside_boundary_band_area_fraction=near.get('near_C_outside_boundary_band_sample_fraction'))
    return dict(status='REUSED_SAVED_METRICS',values=values,
        unavailable_fields=[key for key,value in values.items() if value is None],
        near_C_distance_distribution=near.get('context_distance_bbox_pct',dict(status='NOT_AVAILABLE')),
        near_C_sampling=dict(point_count=near.get('point_count'),seed=near.get('point_seed'),point_sha256=near.get('point_sha256'),protocol=near.get('protocol')),
        distances_and_intersections_recomputed=False,FF_names='FF_intersections is a free-face collision count; L_FF is a different training consistency loss')


def _sign(value):
    return 'NOT_AVAILABLE' if value is None else 'DECREASED' if value<0 else 'INCREASED' if value>0 else 'UNCHANGED'


def _difference(a,b):return None if a is None or b is None else b-a


def _ratio(a,b):return None if a is None or b is None or a==0 else b/a


def _training_time_coverage(results):
    records=_get(results,'arms','A','stream_audits',default=[])
    if len(records)!=500 or any(len(row.get('times',[]))!=8 for row in records):
        return dict(status='INSUFFICIENT_FULL_TIME_LOG',available_batches=len(records),expected_batches=500,
            training_expectation_from_five_teacher_times=False)
    rows=[]
    for index,row in enumerate(records):
        if row.get('step',index+1)!=index+1:raise ValueError('Training time log update order differs')
        for t,parent,K in zip(row['times'],row['parent_indices'],row['K']):
            if not 0<=float(t)<=1:raise ValueError('Recorded training t is outside [0,1]')
            rows.append(dict(t=float(t),parent=int(parent),K=int(K),explicit_endpoint_weight=(1-float(t))**2))
    if len(rows)!=4000:raise AssertionError('Expected original4000 effective samples')
    same={}
    for arm in ('B','C'):
        other=_get(results,'arms',arm,'stream_audits',default=[])
        same[arm]=dict(compared_batches=min(len(records),len(other)),total_batches=len(other),
            full_stream_identical=len(other)==len(records) and all(a==b for a,b in zip(records,other)))
    def summarize(subset):
        times=np.asarray([x['t'] for x in subset]);weights=np.asarray([x['explicit_endpoint_weight'] for x in subset])
        counts,_=np.histogram(times,bins=TIME_EDGES)
        return dict(samples=len(subset),times=_distribution(times),explicit_weight=_distribution(weights),
            histogram_edges=list(TIME_EDGES),histogram_counts=counts.tolist(),histogram_fractions=(counts/max(len(subset),1)).tolist())
    return dict(status='PASS_FULL500_UPDATE_TIME_LOG',source_arm='A',updates=500,samples=4000,
        all_samples=summarize(rows),by_parent={str(p):summarize([x for x in rows if x['parent']==p]) for p in (0,1)},
        by_K={str(K):summarize([x for x in rows if x['K']==K]) for K in (4,8,12)},
        by_parent_K={f'P{p}_K{K}':summarize([x for x in rows if x['parent']==p and x['K']==K]) for p in (0,1) for K in (4,8,12)},
        other_arm_stream_checks=same,distribution='Original actual logged logit-normal draws, not reconstructed or resampled',
        histogram_rule='Fixed width0.2 bins [left,right), final bin includes1; descriptive coverage only, no weight modification',
        fixed_teacher_times=[.1,.3,.5,.7,.9],fixed_teacher_explicit_weights=[.81,.49,.25,.09,.01],
        equal_teacher_time_mean_is_training_expectation=False,
        warning='Explicit(1-t)^2 is not the measured gradient norm; the five equal-time teacher states do not estimate this actual training distribution by an unweighted mean')


def analyze_saved(source_run):
    """Read the identified run once, return compact old-output evidence to root."""
    started=time.perf_counter();run=Path(source_run).resolve();report=run/'THREE_ARM_INTERFACE_REPORT.md';result_path=run/'results.json'
    if not report.is_file() or not result_path.is_file():raise FileNotFoundError('Source requires both the named report and results.json')
    before={str(path):_sha(path) for path in (report,result_path)}
    results=json.loads(result_path.read_text(encoding='utf-8-sig'));checkpoint_records=[]
    for (arm,step),expected in CHECKPOINT_SHA.items():
        info=_get(results,'arms',arm,'checkpoints',str(step));path=run/arm/f'additional_step{step}_cumulative_step{500+step}.pt'
        if step==500 and (not info or not path.is_file()):raise FileNotFoundError(f'Missing required {arm} endpoint')
        if info is not None and info.get('sha256')!=expected:raise RuntimeError(f'{arm}step{step} recorded identity mismatch')
        checkpoint_records.append(dict(arm=arm,additional_step=step,path=str(path),expected_sha256=expected,
            recorded_sha256=info.get('sha256') if info else None,available=path.is_file(),
            identity_scope='Metadata and required endpoint file presence checked here; actual large checkpoint bytes must be verified in root asset audit'))
    saved=[row for row in results.get('scores',[]) if row.get('model') in ('A','C') and row.get('K')==12 and int(row['seed']) in SEEDS]
    index={(row['model'],int(row['parent']),int(row['seed'])):row for row in saved}
    expected={(model,p,seed) for model in ('A','C') for p in (0,1) for seed in SEEDS}
    if len(saved)!=32 or set(index)!=expected:raise ValueError('Require the existing complete32 A/C endpoint scoring matrix')
    outputs=[];raw_index={};raw_bytes=0;raw_files=0
    for model in ('A','C'):
        for parent in (0,1):
            for seed in SEEDS:
                source=index[model,parent,seed];path=_path(source['raw_path']);metric=_saved_metrics(source['metrics'])
                length=_get(source['metrics'],'remainder','own','target_full_bbox_diagonal')
                geometry=dict(status='NOT_AVAILABLE',reason='Original raw endpoint array missing')
                raw_record=dict(path=str(path),available=path.is_file())
                if path.is_file():
                    digest=_sha(path);matching=next((row for row in results.get('generated',[]) if row.get('model')==model and row.get('parent')==parent and row.get('seed')==seed),None)
                    expected_hash=(matching or {}).get('raw_sha256')
                    if expected_hash is not None and digest!=expected_hash:raise RuntimeError('Archived raw bytes changed')
                    with np.load(path,allow_pickle=False) as archive:
                        raw=archive['output'].copy() if 'output' in archive else None
                    if raw is not None:
                        if length is None:raise ValueError('Target L metadata missing; no inferred scale')
                        geometry=_cheap_geometry(raw,float(length))
                        raw_files+=1;raw_bytes+=path.stat().st_size
                    raw_record.update(sha256=digest,matched_saved_hash=expected_hash is not None,bytes=path.stat().st_size)
                row=dict(model=model,parent=parent,K=12,seed=seed,raw=raw_record,saved=metric,cheap_geometry=geometry)
                outputs.append(row);raw_index[model,parent,seed]=row
    pairs=[]
    for parent in (0,1):
        for seed in SEEDS:
            a,c=raw_index['A',parent,seed],raw_index['C',parent,seed]
            av,cv=a['saved']['values'],c['saved']['values']
            delta={key:_difference(av[key],cv[key]) for key in av}
            area_a,area_c=a['cheap_geometry'],c['cheap_geometry']
            ratios={key:_ratio(area_a.get(key),area_c.get(key)) for key in ('free_total_area','centered_coordinate_RMS','bbox_diagonal','bbox_max_extent')}
            area_quantile_ratios={key:_ratio(_get(area_a,'free_face_area_distribution',key),_get(area_c,'free_face_area_distribution',key)) for key in ('minimum','q01','q05','q25','median','q75','q95','maximum')}
            ff,rms=_sign(delta['FF_intersections']),_sign(delta['remaining_symmetric_RMS_bbox_pct'])
            pairs.append(dict(parent=parent,seed=seed,C_minus_A=delta,
                FF_intersections_vs_remaining_RMS=f'FF_INTERSECTIONS_{ff}__REMAINING_RMS_{rms}',
                intersection_decreased_and_RMS_improved=ff=='DECREASED' and rms=='DECREASED',
                intersection_decreased_and_RMS_worsened=ff=='DECREASED' and rms=='INCREASED',
                both_intersections_and_RMS_worsened=ff=='INCREASED' and rms=='INCREASED',
                Gamma=dict(RMS_change=delta['Gamma_RMS_bbox_pct'],RMS_direction=_sign(delta['Gamma_RMS_bbox_pct']),
                    p95_change=delta['Gamma_p95_bbox_pct'],p95_direction=_sign(delta['Gamma_p95_bbox_pct']),
                    coverage_change=delta['Gamma_coverage_fraction'],coverage_direction=_sign(delta['Gamma_coverage_fraction'])),
                C_over_A_scale_ratios=ratios,C_over_A_independent_area_quantile_ratios=area_quantile_ratios,
                area_quantile_scope='Compare distributions, not paired face identities; no prediction-to-GT correspondence',
                interpretation='Descriptive cross-classification only; no output excluded and no claim that L_FF equals FF_intersections'))
    parents={}
    for parent in (0,1):
        subset=[row for row in pairs if row['parent']==parent]
        groups={label:[row['seed'] for row in subset if row['FF_intersections_vs_remaining_RMS']==label] for label in sorted({row['FF_intersections_vs_remaining_RMS'] for row in subset})}
        parents[str(parent)]=dict(seeds=list(SEEDS),cross_classification=groups,
            C_minus_A_distributions={key:_distribution([row['C_minus_A'][key] for row in subset if row['C_minus_A'][key] is not None]) for key in subset[0]['C_minus_A']},
            scale_ratio_distributions={key:_distribution([row['C_over_A_scale_ratios'][key] for row in subset if row['C_over_A_scale_ratios'][key] is not None]) for key in subset[0]['C_over_A_scale_ratios']},
            Gamma_improved_count=sum(row['Gamma']['RMS_direction']=='DECREASED' for row in subset),
            Gamma_worsened_count=sum(row['Gamma']['RMS_direction']=='INCREASED' for row in subset),
            FF_intersections_decreased_count=sum(row['C_minus_A']['FF_intersections'] is not None and row['C_minus_A']['FF_intersections']<0 for row in subset))
    after={str(path):_sha(path) for path in (report,result_path)}
    if before!=after:raise RuntimeError('Source report or results changed during read-only analysis')
    result=dict(schema='c_relation_saved_evidence_v1',status='PASS_SAVED_EVIDENCE_REUSE',source_run=str(run),
        source_files=before,checkpoint_identity_records=checkpoint_records,
        outputs=outputs,paired=pairs,parents=parents,training_time_coverage=_training_time_coverage(results),
        scope=dict(models=['A','C'],seeds=list(SEEDS),parents=[0,1],K=12,old_generation_outputs=32,
            FF_names='L_FF is training copy-consistency loss; FF_intersections is saved free/free illegal pair count',
            no_GT_ids_assigned_to_generated_outputs=True,no_nearest_neighbor_assignment=True,
            saved_distance_and_intersection_metrics_reused=True,all_seeds_retained=True),
        ledger=dict(raw_endpoints_read=raw_files,raw_archive_compressed_bytes=raw_bytes,
            cheap_area_and_coordinate_statistics=raw_files,old_saved_metrics_reused=32,
            new_distance_matrix_evaluations=0,new_intersection_tests=0,model_forwards=0,
            model_backwards=0,OT_calls=0,autograd_calls=0,optimizer_updates=0,geometry_changes=0),
        seconds=time.perf_counter()-started)
    json.dumps(result,allow_nan=False)
    return result
