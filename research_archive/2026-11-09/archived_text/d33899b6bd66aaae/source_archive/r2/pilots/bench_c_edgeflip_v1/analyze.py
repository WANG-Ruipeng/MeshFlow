"""Read-only collection of registered RAW scores for the frozen C edge-flip test.

All original 177 fields and status records are retained.  New rows always use
the independently computed original-C diagnostic, even though K is unchanged.
Area and angle supplements are reductions of the existing per-face scorer
output, not additional quality scans or new point queries.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import defaultdict
import json
import math
import sys
import numpy as np
from r2.pilots.seam_abc_v1 import eval_common as C
from .evaluate import OUT, require_all_raw

DEST = OUT/'analysis'
OLD = C.ROOT/'experiments/r2_bench_four_routes_v1/exposure'
VERSION = 'bench_C_edgeflip_RAW_full177_commonC_cached_shape_analysis_v1'
COMMON_METRICS = (
    'ConflictCov_C', 'ConflictCov_C_upper', 'ConflictCov_C_cross',
    'ConflictCov_C_dup', 'ConflictCov_C_cross_and_dup', 'ConflictCov_C_status',
    'ConflictCov_C_interior_samples', 'ConflictCov_C_interior_weight_fraction',
    'ConflictCov_C_uncertain_fraction', 'ConflictCov_C_unresolved_pairs',
    'ConflictCov_C_coplanar_illegal_pairs', 'ConflictCov_C_proper_pairs',
    'ConflictCov_C_segment_extraction_failures', 'CF_illegal_pair_count',
    'CF_illegal_affected_free_fraction', 'CF_proper_pair_count', 'CF_coplanar_pair_count',
)


def _numeric(value):
    return isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)) and math.isfinite(float(value))


def _metadata(cell):
    q = cell['q']
    return dict(cell_id=cell['cell_id'], label=cell['label'], stage='RAW',
        parent_id=q['parent_id'], parent_prefix=q['parent_id'][:4],
        request_id=q['request_id'], condition_id=q['condition_id'],
        noise_index=int(q['noise_index']), split=q['split'],
        N=int(q['N']), K=int(q['K']), free_count=int(q['N'])-int(q['K']),
        native_count_bin=int(q['N'])//20, origin=cell['origin'])


def cells():
    data = C.read(OUT/'inputs/cells.json')
    registration = C.read(OUT/'registration.json')
    if len(data) != registration.get('expected_cells',len(data)) or len({c['cell_id'] for c in data}) != len(data):
        raise ValueError('Expected the registered unique cells')
    if {c['label'] for c in data} != {'ORIGINAL','EDGE_FLIP'}:
        raise ValueError('Unexpected edge-flip arm')
    pairs = defaultdict(dict)
    for cell in data:
        rid,label=cell['q']['request_id'],cell['label']
        if label in pairs[rid]:
            raise ValueError('Duplicate request/arm')
        pairs[rid][label] = cell
    if len(pairs) != 6 or any('ORIGINAL' not in v for v in pairs.values()):
        raise ValueError('Expected the six frozen original requests')
    new_count=sum('EDGE_FLIP' in v for v in pairs.values())
    if new_count != registration.get('expected_new_raw',new_count):
        raise ValueError('Feasible EDGE_FLIP count differs from registration')
    if len({c['q']['parent_id'] for c in data}) != 3:
        raise ValueError('Expected the registered three parents')
    for pair in pairs.values():
        if 'EDGE_FLIP' not in pair:
            continue
        a,b = pair['ORIGINAL']['q'],pair['EDGE_FLIP']['q']
        for key in ('parent_id','condition_id','noise_index','N','K','split'):
            if a[key] != b[key]:
                raise ValueError('Pair differs in '+key)
        if C.sha(a['noise_npz']) != C.sha(b['noise_npz']):
            raise ValueError('Pair noise files differ')
    return data


def _score_path(cell):
    if cell['label'] == 'ORIGINAL':
        return C.resolve(cell['score_path'])
    return OUT/'scores'/cell['cell_id']


def _verify(book):
    for path,digest in book.items():
        if C.sha(path) != digest:
            raise ValueError('Protected input changed: '+str(path))


def _aggregate(rows, groups, fields):
    grouped=defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in groups)].append(row)
    result=[]
    for group,values in sorted(grouped.items(),key=lambda item:str(item[0])):
        out=dict(zip(groups,group), outputs=len(values), aggregation='MACRO_MEAN_PER_OUTPUT')
        for key in fields:
            numeric=[float(r[key]) for r in values if _numeric(r.get(key))]
            present=[r.get(key) for r in values if r.get(key) is not None]
            out[key]=float(np.mean(numeric)) if numeric else (present[0] if present and all(v==present[0] for v in present) else None)
            out[key+'__numeric_n']=len(numeric)
            out[key+'__present_n']=len(present)
            out[key+'__missing_n']=len(values)-len(present)
            if present and not numeric and not all(v==present[0] for v in present):
                out[key+'__distinct_values']=list(dict.fromkeys(str(v) for v in present))
        result.append(out)
    return result


def _pairs(data, rows, fields):
    by={row['cell_id']:row for row in rows}
    requests=defaultdict(dict)
    for cell in data:
        requests[cell['q']['request_id']][cell['label']]=cell
    result=[]
    for rid,pair in sorted(requests.items()):
        if 'EDGE_FLIP' not in pair:
            continue
        a,b=pair['ORIGINAL'],pair['EDGE_FLIP']
        aa,bb=by[a['cell_id']],by[b['cell_id']]
        meta=_metadata(b)
        out={key:meta[key] for key in ('stage','parent_id','parent_prefix','request_id','condition_id','noise_index','split','N','K','free_count','native_count_bin')}
        out.update(reference='ORIGINAL', candidate='EDGE_FLIP', contrast='INTERNAL_C_EDGE_FLIP_SAME_K_N_BIN_NOISE',
            reference_cell_id=a['cell_id'],candidate_cell_id=b['cell_id'],delta_convention='EDGE_FLIP_MINUS_ORIGINAL')
        for key in fields:
            av,bv=aa.get(key),bb.get(key)
            out[key+'__reference']=av
            out[key+'__candidate']=bv
            out[key+'__delta']=bv-av if _numeric(av) and _numeric(bv) else None
        result.append(out)
    if len(result) != sum(c['label']=='EDGE_FLIP' for c in data):
        raise ValueError('Expected one metric pair per feasible EDGE_FLIP')
    return result


def _common_row(cell, score, common):
    q,old=cell['q'],cell['original_q']
    nfree=int(q['N'])-int(q['K'])
    result=dict(_metadata(cell), metric_scope='COMMON_ORIGINAL_PHYSICAL_C',
        original_K=int(old['K']), fixed_free_face_denominator=nfree,
        fixed_CF_tested_pairs=int(old['K'])*nfree, fixed_C_query_count=2048)
    if cell['label'] == 'EDGE_FLIP':
        if common is None or common['status'] != 'PASS':
            raise ValueError('Every EDGE_FLIP needs its original-C scan, including unchanged K')
        if common['fixed_free_face_denominator'] != nfree or common['original_K'] != int(old['K']):
            raise ValueError('Common-C denominator differs')
        if common['original_C_query_sha256'] != old['query_descriptor']['sha256']:
            raise ValueError('Common-C query identity differs')
        if common['original_C_input_sha256'] != C.sha(old['inference_npz']):
            raise ValueError('Common-C input identity differs')
        metrics,boundary=common['metrics'],common['boundary']
        result['common_C_origin']='EXTRA_FIXED_ORIGINAL_C_SCAN'
    else:
        # Original input is itself the common reference.  Reuse the cached
        # metrics and boundary arrays; no additional original quality scan.
        metrics,boundary=score['metrics'],score['boundary']
        result['common_C_origin']='REUSED_ORIGINAL_C_FULL_SCORE'
    for key in COMMON_METRICS:
        if key not in metrics:
            raise ValueError('Missing common-C field: '+key)
        result[key]=metrics[key]
    for row in boundary:
        setting=row['setting']
        for key in ('status','boundary_vertex_count','matched_count','VMR','perfect','threshold_model','scale','tau'):
            result[f'boundary__{setting}__{key}']=row[key]
    result['scope']='Fixed original C for CF/I and boundary diagnostics; this does not certify generated connectivity or a required interface.'
    return result


def _shape_row(cell, shape, stage='RAW'):
    free=shape['free']
    per=free['per_face']
    nfree=int(free['faces'])
    if len(per) != nfree or (stage=='RAW' and nfree != int(cell['q']['N'])-int(cell['q']['K'])):
        raise ValueError('Saved shape free-face denominator differs')
    out=dict(_metadata(cell), shape_schema=free['schema'],
        shape_origin='REDUCED_EXISTING_SHAPE_PER_FACE_JSON', free_face_denominator=nfree,
        fixed_L_model=float(free['bbox_diagonal']),
        existing_normalized_area_floor=float(free['normalized_area_floor']),
        existing_near_degenerate_area_threshold=float(free['near_degenerate_area_threshold']))
    out['stage']=stage
    for field in ('area','area_normalized','minimum_angle_degrees'):
        values=np.asarray([r[field] for r in per],np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite cached shape values')
        out['free_'+field+'_minimum']=float(values.min())
        for label,quantile in (('p01',.01),('p05',.05),('median',.5),('p95',.95)):
            out['free_'+field+'_'+label]=float(np.quantile(values,quantile))
        out['free_'+field+'_mean']=float(values.mean())
    for field in ('area_exact_zero','area_near_degenerate','area_floor_triggered','minimum_angle_lt5','q_actual_gt10'):
        count=sum(bool(r[field]) for r in per)
        out['free_'+field+'_count']=count
        out['free_'+field+'_fraction']=count/nfree
    out['interpretation']='Small minimum angles, existing tiny-area flags and area quantiles describe free triangles. RAW soup/index components do not directly measure visible fragments or completeness.'
    return out


def _historical_background(data, fields, protected):
    """Read only the six original RAW and historical CLEAN_FINAL_POST caches."""
    rows=[];shapes=[];paired=[]
    for cell in data:
        if cell['label']!='ORIGINAL':
            continue
        rid=cell['q']['request_id'];raw_record=None
        folders=(('RAW',_score_path(cell)),
                 ('CLEAN_FINAL_POST',OLD/'cpu/EXPOSURE'/rid/'CLEAN_FINAL_POST/scores'))
        for stage,folder in folders:
            head=dict(_metadata(cell),background_only=True,new_candidate_POST=False,
                historical_model='EXPOSURE_ORIGINAL',source_score_path=str(folder))
            head['stage']=stage
            head['label']='ORIGINAL_'+stage+'_BACKGROUND'
            paths=[folder/'row.json',folder/'score.json',folder/'shape_per_face.json',folder/'complete.json']
            absent=[str(path) for path in paths if not path.is_file()]
            if absent:
                rows.append(dict(head,status='NA_MISSING_HISTORICAL_CACHE',NA_reason='Missing existing score files',missing_paths=absent))
                continue
            row,score,shape=C.read(paths[0]),C.read(paths[1]),C.read(paths[2])
            for key in ('request_id','parent_id','N','K','noise_index'):
                if row[key]!=cell['q'][key]:
                    raise ValueError('Historical background identity differs: '+rid)
            _verify(C.read(paths[3]).get('files',{}))
            for path in paths:
                protected[str(path)]=C.sha(path)
            out=dict(head,status='REUSED_HISTORICAL_SCORE',source_stage_field=row.get('stage'),
                source_label_field=row.get('label'),score_version=score['version'])
            out.update({key:row.get(key) for key in fields})
            out['missing_original_metric_fields']=[key for key in fields if key not in row]
            rows.append(out)
            shape_row=_shape_row(cell,shape,stage=stage)
            shape_row.update(label=head['label'],background_only=True,new_candidate_POST=False)
            shapes.append(shape_row)
            if stage=='RAW':
                raw_record=out
            elif raw_record is not None:
                pair=dict(head,contrast='ORIGINAL_RAW_TO_ITS_HISTORICAL_FINAL_ONLY',
                    reference_stage='RAW',candidate_stage='CLEAN_FINAL_POST',
                    scope='Stage background of ORIGINAL only. No EDGE_FLIP postprocessing or success is implied.')
                for key in fields:
                    av,bv=raw_record.get(key),out.get(key)
                    pair[key+'__RAW']=av;pair[key+'__FINAL']=bv
                    pair[key+'__FINAL_minus_RAW']=bv-av if _numeric(av) and _numeric(bv) else None
                for key in ('free_total_surface_area','whole_total_surface_area','free_face_denominator'):
                    av,bv=raw_record.get(key),out.get(key)
                    pair[key+'__FINAL_over_RAW']=bv/av if _numeric(av) and _numeric(bv) and av>0 else None
                paired.append(pair)
    return rows,shapes,paired


def collect():
    """No score calls, geometry-detector calls, model calls or random draws."""
    if 'torch' in sys.modules:
        raise RuntimeError('Collection must run in an isolated CPU process')
    complete=DEST/'metrics_complete.json'
    if complete.exists():
        saved=C.read(complete)
        if saved['source_sha256'] != C.sha(__file__) or saved['input_cells_sha256'] != C.sha(OUT/'inputs/cells.json'):
            raise ValueError('Completed analysis identity changed')
        _verify(saved['protected']); _verify(saved['files'])
        return saved
    data=cells()
    fields=C.read(OLD/'data/eval_manifest.json')['metric_fields']
    if len(fields) != 177 or len(set(fields)) != 177:
        raise ValueError('Original 177-field dictionary differs')
    rows=[]; common_rows=[]; shape_rows=[]; statuses=[]; missing=[]; protected={}
    for cell in data:
        folder=_score_path(cell)
        receipt=folder/('complete.json' if cell['label']=='ORIGINAL' else 'audit_receipt.json')
        paths=[folder/'row.json', folder/'score.json', folder/'shape_per_face.json', receipt]
        if cell['label']=='EDGE_FLIP':
            paths.append(folder/'common_C.json')
        absent=[str(p) for p in paths if not p.is_file()]
        if absent:
            missing.append(dict(cell_id=cell['cell_id'],status='PENDING_NOT_RECOMPUTED',missing=absent))
            continue
        row,score,shape=C.read(paths[0]),C.read(paths[1]),C.read(paths[2])
        for key in ('request_id','parent_id','N','K','noise_index'):
            if row[key] != cell['q'][key]:
                raise ValueError('Score identity differs for '+cell['cell_id']+': '+key)
        if any(key not in row for key in fields):
            raise ValueError('Original metric column missing')
        if score['C_query_count'] != 2048 or int(row['free_face_denominator']) != int(cell['q']['N'])-int(cell['q']['K']):
            raise ValueError('Score denominator differs')
        if score['status'] != 'PASS' or not score['C_preserved_exact'] or not score['C_original_order_preserved']:
            raise ValueError('Score execution/input-preservation check did not pass')
        _verify(C.read(receipt).get('files',{}))
        for path in paths:
            protected[str(path)]=C.sha(path)
        saved=dict(_metadata(cell), source_stage_field=row.get('stage'), source_label_field=row.get('label'),
            score_path=str(folder), metric_scope='NATIVE_INPUT_C')
        saved.update({key:row[key] for key in fields})
        rows.append(saved)
        common_rows.append(_common_row(cell,score,C.read(folder/'common_C.json') if cell['label']=='EDGE_FLIP' else None))
        shape_rows.append(_shape_row(cell,shape))
        statuses.append(dict(_metadata(cell),source_status=score['status'],source_version=score['version'],
            status_by_metric=score.get('status_by_metric',{}),C_preserved_exact=score['C_preserved_exact'],
            C_original_order_preserved=score['C_original_order_preserved'],source_score_path=str(folder/'score.json')))
    if missing:
        return dict(status='PENDING_SCORES',expected=len(data),collected=len(rows),missing=missing,model_calls=0,scoring_calls=0)
    require_all_raw()
    paths=[]
    def csv(name,values):
        path=DEST/name
        C.csvwrite(path,values)
        paths.append(path)
    group=('label','stage','parent_id','parent_prefix','split')
    csv('metrics_per_output_full177.csv',rows)
    csv('metrics_per_parent_full177.csv',_aggregate(rows,group,fields))
    native_pairs=_pairs(data,rows,fields)
    csv('metrics_paired_full177.csv',native_pairs)
    pair_fields=[key for key in native_pairs[0] if key.endswith('__delta')]
    csv('metrics_paired_per_parent_full177.csv',_aggregate(native_pairs,('stage','parent_id','parent_prefix','split'),pair_fields))
    common_fields=list(COMMON_METRICS)+[key for key in common_rows[0] if key.startswith('boundary__')]
    if any(set(key for key in row if key.startswith('boundary__')) != set(key for key in common_rows[0] if key.startswith('boundary__')) for row in common_rows):
        raise ValueError('Boundary setting columns differ between original and candidate')
    csv('common_C_per_output.csv',common_rows)
    csv('common_C_per_parent.csv',_aggregate(common_rows,group,common_fields))
    common_pairs=_pairs(data,common_rows,common_fields)
    csv('common_C_paired.csv',common_pairs)
    csv('common_C_paired_per_parent.csv',_aggregate(common_pairs,('stage','parent_id','parent_prefix','split'),[key for key in common_pairs[0] if key.endswith('__delta')]))
    shape_fields=[key for key in shape_rows[0] if key.startswith('free_') and key not in ('free_count',)]
    csv('cached_shape_per_output.csv',shape_rows)
    csv('cached_shape_per_parent.csv',_aggregate(shape_rows,group,shape_fields))
    shape_pairs=_pairs(data,shape_rows,shape_fields)
    csv('cached_shape_paired.csv',shape_pairs)
    csv('cached_shape_paired_per_parent.csv',_aggregate(shape_pairs,('stage','parent_id','parent_prefix','split'),[key for key in shape_pairs[0] if key.endswith('__delta')]))
    status_path=DEST/'metric_status_per_output.json'
    C.write(status_path,statuses);paths.append(status_path)
    background,bg_shapes,bg_pairs=_historical_background(data,fields,protected)
    csv('historical_original_RAW_FINAL_background_full177.csv',background)
    csv('historical_original_RAW_FINAL_background_per_parent.csv',_aggregate(background,('label','stage','parent_id','parent_prefix','split'),fields))
    csv('historical_original_RAW_FINAL_cached_shape.csv',bg_shapes)
    csv('historical_original_RAW_to_FINAL_background_paired.csv',bg_pairs)
    _verify(protected)
    registration=C.read(OUT/'registration.json')
    summary=dict(status='COMPLETE',version=VERSION,expected=len(data),collected=len(rows),original_cached_scores=6,new_scores=len(native_pairs),
        original_metric_fields=177,paired_rows=len(native_pairs),parents=3,paired_parents=len({r['parent_id'] for r in native_pairs}),
        blocked_inputs=registration.get('blocked_inputs',[]),logical_stage='RAW',model_calls=0,scoring_calls=0,
        new_random_draws=0,new_point_queries=0,new_geometry_detector_calls=0,
        common_C_policy='Every feasible EDGE_FLIP uses an extra original-C diagnostic despite identical K. Original six common-C rows reuse their cached full scores. Blocked conditions are not generated or scored as failures.',
        stage_label_adapter='Logical stage is RAW. Original source stage/label strings remain in source_stage_field/source_label_field.',
        shape_scope='Existing per-face scorer values are reduced to quantiles and pre-existing flags; no new tiny-area threshold, re-scoring or scan.',
        shape_interpretation='RAW soup topology commonly has one index component per face. Such counts alone do not establish visible fragments, missing supports or surface holes.',
        semantic_scope='Shape completeness, adoption and semantic validity require actual fixed-view review; no unique-parent reconstruction rank or GT success is inferred.',
        historical_FINAL_scope='Any old FINAL-stage evidence is background for the original baseline only; this table contains no FINAL candidate and no new D0/POST.',
        historical_background=dict(rows=len(background),paired_rows=len(bg_pairs),
            missing=[dict(request_id=r['request_id'],stage=r['stage'],reason=r.get('NA_reason')) for r in background if r['status'].startswith('NA_')],
            source_stage='CLEAN_FINAL_POST',source_root=str(OLD/'cpu/EXPOSURE'),
            scope='Separate baseline-only RAW/FINAL tables retain small angles, existing tiny-area flags, isolated-index-face counts, deleted faces and retained area.'),
        protected=protected,files={str(path):C.sha(path) for path in paths},
        source_sha256=C.sha(__file__),input_cells_sha256=C.sha(OUT/'inputs/cells.json'))
    C.write(complete,summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('collect',))
    parser.parse_args()
    result=collect()
    print(json.dumps({key:value for key,value in result.items() if key not in ('files','protected')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
