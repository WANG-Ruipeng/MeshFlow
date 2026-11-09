"""CPU trajectory and original-score collection for the frozen C probe.

No models, new geometry detection, source-GT matching, or random draws.  The
paired distance uses saved free-local slot/corner identities, not nearest-face
or OT rematching.  Numeric differences establish sensitivity, not improvement.
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
from pathlib import Path
import sys
import numpy as np
from . import common as C

VERSION = 'bench_C_tessellation_paired_free_trajectory_analysis_v1'
DEST = C.OUT / 'analysis'
METRIC_DEST = DEST / 'metrics_v2'


def cells():
    rows = C.read(C.OUT/'inputs/cells.json')
    if len(rows) != 16 or len({r['cell_id'] for r in rows}) != 16:
        raise ValueError('Expected the frozen sixteen cells')
    return rows


def _metadata(cell):
    q = cell['q']
    return dict(cell_id=cell['cell_id'], label=cell['label'], parent_id=q['parent_id'],
        parent_prefix=q['parent_id'][:4], request_id=q['request_id'],
        condition_id=q['condition_id'], noise_index=int(q['noise_index']),
        split=q['split'], N=int(q['N']), K=int(q['K']), free_count=int(q['N'])-int(q['K']),
        native_count_bin=int(q['N'])//20, effective_count_bin=int(cell['count_bin']),
        diagnostic_count_intervention=bool(cell.get('diagnostic_count_intervention', False)),
        origin=cell['origin'])


def _contrasts(rows):
    by = {(r['q']['request_id'], r['label']):r for r in rows}
    result=[]
    for r in rows:
        if r['label'] != 'ORIGINAL':
            continue
        rid=r['q']['request_id']
        result.append((r, by[rid,'REFINED_NATIVE'], 'NATURAL_CONTRACT'))
        if r['q']['parent_id'].startswith('459'):
            result.extend(((r,by[rid,'REFINED_OLD_BIN'],'TESSELLATION_FIXED_BIN9'),
                (r,by[rid,'ORIGINAL_NEW_BIN'],'COUNT_ONLY_ORIGINAL_C'),
                (by[rid,'REFINED_OLD_BIN'],by[rid,'REFINED_NATIVE'],'COUNT_ONLY_REFINED_C')))
    if len(result)!=12:
        raise ValueError('Expected twelve registered paired curves')
    return result


def _pair_metadata(a,b,contrast):
    ma,mb=_metadata(a),_metadata(b)
    if (ma['request_id'],ma['free_count']) != (mb['request_id'],mb['free_count']):
        raise ValueError('Pair must preserve request and free-face count')
    return dict(contrast=contrast, reference=ma['label'], candidate=mb['label'],
        reference_cell_id=ma['cell_id'], candidate_cell_id=mb['cell_id'],
        request_id=ma['request_id'], parent_id=ma['parent_id'], parent_prefix=ma['parent_prefix'],
        noise_index=ma['noise_index'], split=ma['split'], free_face_denominator=ma['free_count'],
        free_corner_denominator=3*ma['free_count'], free_coordinate_denominator=9*ma['free_count'],
        reference_K=ma['K'], candidate_K=mb['K'], reference_N=ma['N'], candidate_N=mb['N'],
        reference_effective_bin=ma['effective_count_bin'],candidate_effective_bin=mb['effective_count_bin'],
        count_bin_equal=ma['effective_count_bin']==mb['effective_count_bin'],
        involves_diagnostic_count_intervention=ma['diagnostic_count_intervention'] or mb['diagnostic_count_intervention'])


def _trajectory(cell):
    raw_path=C.resolve(cell['raw_path']); path=raw_path.parent/'trajectory.npz'
    saved=C.arrays(path)
    if set(saved)!= {'state'}:
        raise ValueError('Unexpected trajectory fields: '+str(path))
    state=saved['state']; q=cell['q']; n,k=int(q['N']),int(q['K'])
    if state.shape!=(51,n,3,3) or state.dtype!=np.float32 or not np.isfinite(state).all():
        raise ValueError('Malformed saved Euler50 trajectory')
    raw=C.arrays(raw_path)
    if not np.array_equal(state[-1],raw['raw']):
        raise ValueError('Trajectory terminal state differs from saved RAW')
    noise=C.arrays(q['noise_npz'])['z']
    if not np.array_equal(state[0,k:],noise):
        raise ValueError('Saved starting free noise changed')
    if any(not np.array_equal(s[:k],state[0,:k]) for s in state):
        raise ValueError('Known C is not constant along all 51 states')
    return state[:,k:].astype(np.float64), path


def _free_geometry(free):
    tri=np.asarray(free,np.float64)
    area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    flat=tri.reshape(-1,3); low,high=flat.min(0),flat.max(0)
    mean=flat.mean(0)
    ac=(tri.mean(1)*area[:,None]).sum(0)/area.sum() if area.sum()>0 else np.full(3,np.nan)
    result=dict(free_surface_area=float(area.sum()), free_face_area_min=float(area.min()),
        free_face_area_median=float(np.median(area)), free_face_area_p05=float(np.quantile(area,.05)),
        free_exact_zero_area_count=int((area==0).sum()))
    for i,axis in enumerate('xyz'):
        result.update({f'free_bbox_min_{axis}':float(low[i]),f'free_bbox_max_{axis}':float(high[i]),
            f'free_bbox_span_{axis}':float(high[i]-low[i]),f'free_corner_centroid_{axis}':float(mean[i]),
            f'free_area_centroid_{axis}':None if not np.isfinite(ac[i]) else float(ac[i])})
    return result


def _numeric(value):
    return isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)) and math.isfinite(float(value))


def _aggregate(rows,group_keys,fields):
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in group_keys)].append(r)
    results=[]
    for key,values in sorted(grouped.items(),key=lambda kv:str(kv[0])):
        out=dict(zip(group_keys,key));out['outputs']=len(values)
        for field in fields:
            vals=[float(r[field]) for r in values if _numeric(r.get(field))]
            nonnumeric=[r.get(field) for r in values if r.get(field) is not None and not _numeric(r.get(field))]
            out[field]=float(np.mean(vals)) if vals else (nonnumeric[0] if nonnumeric and all(v==nonnumeric[0] for v in nonnumeric) else None)
            out[field+'__valid_n']=len(vals) if vals else len(nonnumeric)
        results.append(out)
    return results


def _plots(per_step,endpoints):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by=defaultdict(list)
    for r in per_step:by[r['request_id'],r['contrast']].append(r)
    references=sorted({(r['parent_prefix'],r['request_id'],r['noise_index']) for r in endpoints},key=lambda a:(a[0],a[2]))
    fig,axes=plt.subplots(3,2,figsize=(13,11),sharex=True,constrained_layout=True)
    for ax,(prefix,rid,noise) in zip(axes.flat,references):
        for contrast in ('NATURAL_CONTRACT','TESSELLATION_FIXED_BIN9','COUNT_ONLY_ORIGINAL_C','COUNT_ONLY_REFINED_C'):
            series=by.get((rid,contrast))
            if series:
                ax.plot([r['t'] for r in series],[100*r['corner_l2_mean_over_L'] for r in series],label=contrast,linewidth=1.6)
        ax.set_title(f'{prefix} / n{noise}')
        ax.set_xlabel('Euler time');ax.set_ylabel('Mean paired free-corner displacement (%L)');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Frozen EXPOSURE: C tessellation and count-bin sensitivity\nSame free-local identities and saved noise; differences are not success scores',fontsize=12)
    path=DEST/'trajectory_pair_curves.png';fig.savefig(path,dpi=160);plt.close(fig)
    return path


def curves():
    if 'torch' in sys.modules:raise RuntimeError('CPU analysis must not import torch')
    data=cells(); trajectories={};protected={}; geometry=[]
    for cell in data:
        free,path=_trajectory(cell);trajectories[cell['cell_id']]=free
        protected[str(path)]=C.sha(path);protected[str(C.resolve(cell['raw_path']))]=C.sha(cell['raw_path'])
        q=cell['q'];L=float(q['query_descriptor']['identity']['L0'])
        row=dict(_metadata(cell),L_model=L,**_free_geometry(free[-1]))
        row['free_surface_area_over_L2']=row['free_surface_area']/L**2
        geometry.append(row)
    steps=[];ends=[]
    for a,b,contrast in _contrasts(data):
        fa,fb=trajectories[a['cell_id']],trajectories[b['cell_id']]
        if not np.array_equal(fa[0],fb[0]):raise ValueError('Pair starts from different free noise')
        L=float(a['q']['query_descriptor']['identity']['L0'])
        if L!=float(b['q']['query_descriptor']['identity']['L0']):raise ValueError('Scale mismatch')
        common=_pair_metadata(a,b,contrast)
        for i,(xa,xb) in enumerate(zip(fa,fb)):
            diff=xb-xa;distance=np.linalg.norm(diff,axis=2).reshape(-1);centroid=diff.mean((0,1))
            row=dict(common,step=i,t=i/50.,L_model=L,
                coordinate_rms_model=float(np.sqrt(np.mean(diff**2))),
                corner_l2_mean_model=float(distance.mean()),corner_l2_p95_model=float(np.quantile(distance,.95)),
                corner_l2_max_model=float(distance.max()),
                free_centroid_shift_x=float(centroid[0]),free_centroid_shift_y=float(centroid[1]),
                free_centroid_shift_z=float(centroid[2]),free_centroid_shift_norm_model=float(np.linalg.norm(centroid)))
            for key in ('coordinate_rms','corner_l2_mean','corner_l2_p95','corner_l2_max','free_centroid_shift_norm'):
                row[key+'_over_L']=row[key+'_model']/L
            steps.append(row)
            if i==50:
                end=dict(row)
                ga,gb=_free_geometry(xa),_free_geometry(xb)
                for key in ga:
                    end[key+'__reference']=ga[key];end[key+'__candidate']=gb[key]
                    end[key+'__delta']=gb[key]-ga[key] if _numeric(ga[key]) and _numeric(gb[key]) else None
                end['free_surface_area_ratio']=gb['free_surface_area']/ga['free_surface_area']
                ends.append(end)
    if len(steps)!=612 or len(ends)!=12:raise ValueError('Pairwise trajectory budget differs')
    C.csvwrite(DEST/'trajectory_pairs_per_step.csv',steps)
    C.csvwrite(DEST/'trajectory_pair_endpoints.csv',ends)
    C.csvwrite(DEST/'free_geometry_per_output.csv',geometry)
    fields=[k for k,v in ends[0].items() if _numeric(v) and (k.endswith('_model') or k.endswith('_over_L') or k.endswith('__delta') or k=='free_surface_area_ratio')]
    grouped=_aggregate(ends,('contrast','reference','candidate','parent_id','parent_prefix'),fields)
    C.csvwrite(DEST/'trajectory_endpoints_per_parent.csv',grouped)
    png=_plots(steps,ends)
    paths=[DEST/n for n in ('trajectory_pairs_per_step.csv','trajectory_pair_endpoints.csv','free_geometry_per_output.csv','trajectory_endpoints_per_parent.csv')]+[png]
    result=dict(status='COMPLETE',version=VERSION,trajectories=16,paired_curves=12,per_step_rows=612,
        model_calls=0,geometry_detector_calls=0,new_random_draws=0,source_GT_matching=False,
        identity='Free-local slot and corner index, unchanged from each saved initial Gaussian; no nearest-neighbor rematching.',
        interpretation='Coordinate differences measure representation/count-bin sensitivity, not geometric quality or adoption.',
        count_interpretation='459 natural comparison changes tessellation and N-bin; fixed-bin and count-only rows separate these interventions. Off-diagonal bins are diagnostic, not production.',
        files={str(p):C.sha(p) for p in paths},protected=protected,
        source_sha256=C.sha(__file__),input_cells_sha256=C.sha(C.OUT/'inputs/cells.json'))
    C.write(DEST/'curves_complete.json',result)
    return result


def _score_path(cell):
    return C.resolve(cell['score_path']) if cell['origin']=='REUSED_CACHE' else C.OUT/'scores'/cell['cell_id']


def _common_row(cell, row, path):
    meta=_metadata(cell); old=cell['original_q'];nfree=int(cell['q']['N'])-int(cell['q']['K'])
    result=dict(meta,stage='RAW',source_stage_field=row.get('stage'),source_label_field=row.get('label'),
        common_original_K=int(old['K']),fixed_free_face_denominator=nfree,
        fixed_CF_tested_pairs=int(old['K'])*nfree,physical_query_count=2048)
    if path.exists():
        saved=C.read(path)
        if saved['status']!='PASS' or saved['fixed_free_face_denominator']!=nfree or saved['original_K']!=int(old['K']):
            raise ValueError('Common-C denominator/status differs')
        result.update(common_C_origin='EXTRA_FIXED_ORIGINAL_C_SCAN',**saved['metrics'])
        for b in saved.get('boundary',[]):
            setting=b.get('setting')
            for key,value in b.items():
                if key in ('setting','domain','scale','tau','threshold_model'):continue
                if isinstance(value,(str,int,float,bool)) or value is None:result[f'boundary__{setting}__{key}']=value
    elif int(cell['q']['K'])==int(old['K']):
        result['common_C_origin']='REUSED_ORIGINAL_C_FULL_SCORE'
        for key,value in row.items():
            if key.startswith(('CF_','ConflictCov_C','boundary_distance_','A_VMR__','PMR__')) or key=='I_status':result[key]=value
    else:
        raise FileNotFoundError('Missing required common-C comparison: '+str(path))
    result['scope']='Fixed original physical C for geometric CF/I only; not subdivided topology. Native C counts remain in separate full177 table.'
    return result


def collect():
    """Collect completed original scores only; no score execution or retries."""
    data=cells();rows=[];common_rows=[];missing=[];protected={}
    fields=C.read(C.OLD/'data/eval_manifest.json')['metric_fields']
    if len(fields)!=177 or len(set(fields))!=177:raise ValueError('Original metric dictionary changed')
    for cell in data:
        folder=_score_path(cell);rp=folder/'row.json';sp=folder/'score.json'
        cp=folder/'common_C.json';receipt=folder/('complete.json' if cell['origin']=='REUSED_CACHE' else 'audit_receipt.json')
        required=[rp,sp,receipt]
        if int(cell['q']['K'])!=int(cell['original_q']['K']):required.append(cp)
        absent=[str(p) for p in required if not p.exists()]
        if absent:
            missing.append(dict(cell_id=cell['cell_id'],status='PENDING_NOT_RECOMPUTED',missing=absent));continue
        saved=C.read(rp)
        if any(k not in saved for k in fields):raise ValueError('Original metric column missing')
        q=cell['q']
        if any(saved[k]!=q[k] for k in ('request_id','parent_id','N','K','noise_index')):
            raise ValueError('Saved score identity differs: '+cell['cell_id'])
        if int(saved['free_face_denominator'])!=int(q['N'])-int(q['K']):raise ValueError('Free score denominator differs')
        C.verify(C.read(receipt).get('files',{}))
        out=dict(_metadata(cell),stage='RAW',source_stage_field=saved.get('stage'),
            source_label_field=saved.get('label'),score_path=str(folder),metric_scope='NATIVE_INPUT_C')
        out.update({k:saved[k] for k in fields});rows.append(out)
        common_rows.append(_common_row(cell,saved,cp))
        for path in required:protected[str(path)]=C.sha(path)
    status=dict(status='COMPLETE' if not missing else 'PENDING_SCORES',expected=16,collected=len(rows),missing=missing,
        model_calls=0,scoring_calls=0)
    if missing:
        # Do not seal a changing partial table as a final result.
        return status
    C.csvwrite(METRIC_DEST/'metrics_per_output_full177.csv',rows)
    C.csvwrite(METRIC_DEST/'metrics_per_parent_full177.csv',_aggregate(rows,('label','parent_id','split'),fields))
    common_fields=list(dict.fromkeys(k for r in common_rows for k in r if k not in _metadata(data[0]) and k not in ('scope','common_C_origin')))
    C.csvwrite(METRIC_DEST/'common_C_per_output.csv',common_rows)
    C.csvwrite(METRIC_DEST/'common_C_per_parent.csv',_aggregate(common_rows,('label','parent_id','split'),common_fields))
    by={r['cell_id']:r for r in rows}; cb={r['cell_id']:r for r in common_rows};pairs=[];common_pairs=[]
    for a,b,contrast in _contrasts(data):
        head=_pair_metadata(a,b,contrast)
        for dest, source, names in ((pairs,by,fields),(common_pairs,cb,common_fields)):
            aa,bb=source[a['cell_id']],source[b['cell_id']];out=dict(head)
            for name in names:
                av,bv=aa.get(name),bb.get(name)
                out[name+'__reference']=av;out[name+'__candidate']=bv
                out[name+'__delta']=bv-av if _numeric(av) and _numeric(bv) else None
            dest.append(out)
    C.csvwrite(METRIC_DEST/'metrics_paired_full177.csv',pairs)
    C.csvwrite(METRIC_DEST/'common_C_paired.csv',common_pairs)
    paths=[METRIC_DEST/n for n in ('metrics_per_output_full177.csv','metrics_per_parent_full177.csv','common_C_per_output.csv','common_C_per_parent.csv','metrics_paired_full177.csv','common_C_paired.csv')]
    status.update(version=VERSION+'_metrics_v2_source_stage',fields=177,paired_rows=12,protected=protected,source_sha256=C.sha(__file__),
        logical_stage='RAW',stage_label_adapter='Original scorer wrote method labels into some row.stage fields. Derived v2 uses logical RAW and retains source_stage_field/source_label_field; original score files are unchanged.',
        no_native_CF_count_comparability_claim=True,COMMON_C_no_topology_claim=True,
        files={str(p):C.sha(p) for p in paths})
    C.write(METRIC_DEST/'metrics_complete.json',status)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('curves','collect','all'))
    args=p.parse_args();results={}
    if args.action in ('curves','all'):results['curves']=curves()
    if args.action in ('collect','all'):results['collect']=collect()
    print(json.dumps({k:{x:y for x,y in r.items() if x not in ('files','protected')} for k,r in results.items()},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
