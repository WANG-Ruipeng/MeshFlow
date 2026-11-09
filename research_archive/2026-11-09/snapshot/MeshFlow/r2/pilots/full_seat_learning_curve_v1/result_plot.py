"""Plot fixed-checkpoint scientific results from existing final/partial summaries.

CPU only: no model, scorer, geometry reconstruction or new image judgments.
Missing/NA endpoints stay blank and matplotlib lines break at every missing point.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ.setdefault(key,'2')
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as C

VERSION='full_seat_fixed_checkpoint_result_plot_v1'
MAIN=('FULL_S0000','FULL_S1000','FULL_S3000','FULL_S6000')
STEPS=(0,1000,3000,6000)
STRATA=('TRAIN','DEV')
EV=C.OUT/'evaluation'
DEST=EV/'result_plots'
SPECS=(
 dict(metric='RAW_FINITE_EDGE_MEAN',stage='RAW',family='comparison_boundary',column='FULL_BOUNDARY__boundary_distances__edge__length_weighted_mean_over_L',scale=1.,aggregation='equal_parent_macro_mean',panel=0),
 dict(metric='RAW_EDGE_TANGENT_MEAN',stage='RAW',family='boundary_components',column='edge__tangent_plane_mean_over_L',scale=1.,aggregation='equal_parent_macro_mean',panel=1),
 dict(metric='POST_VALID_INTERFACE_OUTPUTS',stage='POST',family='comparison_score',column='C_F_has_valid_edge',scale=1.,aggregation='count_of_true_per_output',panel=2),
 dict(metric='POST_VALID_INTERFACE_LENGTH',stage='POST',family='comparison_score',column='C_F_valid_edge_total_length',scale=1./C.L,aggregation='equal_parent_macro_mean',panel=3),
 dict(metric='POST_CF_ILLEGAL',stage='POST',family='comparison_score',column='CF_illegal_affected_free_fraction',scale=100.,aggregation='equal_parent_macro_mean',panel=4),
 dict(metric='POST_FF_ILLEGAL',stage='POST',family='comparison_score',column='FF_illegal_affected_free_fraction',scale=100.,aggregation='equal_parent_macro_mean',panel=4),
)


def _rows(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def _num(value):
    if value in (None,''): return None
    if value in (True,'True','true'): return 1.
    if value in (False,'False','false'): return 0.
    try: result=float(value)
    except (ValueError,TypeError): return None
    return result if math.isfinite(result) else None


def _true(value): return value in (True,'True','true',1,'1')


def _snapshot(folder, filename, sources):
    pointer=C.OUT/folder/'latest.json'
    if not pointer.is_file(): return None,[],dict(status='MISSING_LATEST',folder=folder)
    value=C.read(pointer); target=C.resolve(value['path'])
    if not target.is_file() or C.sha(target)!=value['sha256']: raise ValueError('Snapshot pointer changed: '+str(pointer))
    manifest=C.read(target)
    sources[str(pointer)]=C.sha(pointer); sources[str(target)]=C.sha(target)
    table=target.parent/filename
    if not table.is_file(): return manifest,[],dict(status='MISSING_TABLE',path=str(table))
    expected=manifest.get('output_hashes',manifest.get('files',{})).get(str(table))
    actual=C.sha(table)
    if expected!=actual: raise ValueError('Source table hash differs: '+str(table))
    sources[str(table)]=actual
    return manifest,_rows(table),dict(status=manifest.get('status'),path=str(table),sha256=actual,manifest_path=str(target))


def _index(rows,requests,allowed):
    result={}
    for row in rows:
        label=row.get('label')
        if label not in allowed: continue
        key=(label,row.get('request_id'),row.get('stage'))
        q=requests.get(row.get('request_id'))
        if q is None or row.get('stage') not in ('RAW','POST') or key in result: raise ValueError('Unexpected/duplicate plotting identity')
        if row.get('parent_id')!=q['parent_id'] or int(row['candidate'])!=int(q['candidate']): raise ValueError('Plot source parent/noise identity differs')
        result[key]=row
    return result


def _aggregate(records, spec, requests):
    output=[]; parents=[]
    for label,step in zip(MAIN,STEPS):
        for split in STRATA:
            qs=[q for q in requests.values() if q['split']==split]
            observed=[row for row in records if row['label']==label and row['split']==split]
            present=[row for row in observed if row['display_value'] is not None]
            pmeans=[]
            for parent in sorted({q['parent_id'] for q in qs}):
                selected=[row for row in observed if row['parent_id']==parent]
                values=[r['display_value'] for r in selected if r['display_value'] is not None]
                mean=float(np.mean(values)) if values else None
                if mean is not None: pmeans.append(mean)
                parents.append(dict(metric=spec['metric'],label=label,local_step=step,stage=spec['stage'],split=split,parent_id=parent,
                    observed_noises=len(values),expected_noises=4,parent_observed_mean=mean,
                    parent_complete_mean=mean if len(values)==4 else None,status='COMPLETE' if len(values)==4 else 'PARTIAL_OR_NA'))
            complete=len(present)==len(qs)
            if spec['aggregation']=='count_of_true_per_output':
                value=float(sum(r['display_value'] for r in present)) if complete else None
                observed_value=float(sum(r['display_value'] for r in present)) if present else None
            else:
                observed_value=float(np.mean(pmeans)) if pmeans else None
                value=observed_value if complete else None
            output.append(dict(metric=spec['metric'],label=label,local_step=step,stage=spec['stage'],split=split,
                plotted_value=value,observed_aggregate_not_plotted_if_partial=observed_value,observed_requests=len(present),expected_requests=len(qs),
                expected_parents=len({q['parent_id'] for q in qs}),noises_per_parent=4,aggregation=spec['aggregation'],
                source_column=spec['column'],display_scale=spec['scale'],status='COMPLETE' if complete else 'PARTIAL_OR_NA',
                missing_reason=';'.join(sorted({r['missing_reason'] for r in observed if r['missing_reason']}))))
    return output,parents


def run():
    regpath=C.OUT/'registration.json'; reg=C.read(regpath); requests={q['request_id']:q for q in reg['requests']}
    if len(requests)!=24 or tuple(reg['evaluation_labels'])!=MAIN: raise ValueError('Plot requires frozen four-checkpoint24request panel')
    if {q['split'] for q in requests.values()}!=set(STRATA): raise ValueError('Plot requires registered TRAIN and DEV strata')
    sources={str(regpath):C.sha(regpath),str(C.CODE/'result_plot.py'):C.sha(C.CODE/'result_plot.py')}
    comparison,crows,cmeta=_snapshot('evaluation/comparison','main_per_output.csv',sources)
    components,brows,bmeta=_snapshot('evaluation/boundary_components_v1','per_output.csv',sources)
    review,rrows,rmeta=_snapshot('evaluation/review/summary','main_review_rows.csv',sources)
    ci=_index(crows,requests,MAIN); bi=_index(brows,requests,MAIN); ri=_index(rrows,requests,MAIN)
    review_complete=bool(review and review.get('status')=='COMPLETE' and review.get('main',{}).get('validated_rows')==192 and len(ri)==192)
    metrics=list(SPECS)
    for category in ('YES','NO','UNCERTAIN'):
        metrics.append(dict(metric='RAW_AI_PROXY_C_ADOPTION_'+category,stage='RAW',family='review',column='C_adoption',category=category,scale=1.,aggregation='count_of_true_per_output',panel=5))
    per_output=[]; curves=[]; parent_rows=[]
    for spec in metrics:
        records=[]
        for label,step in zip(MAIN,STEPS):
            for q in requests.values():
                key=(label,q['request_id'],spec['stage']); family=spec['family']; reason=''; value=None; raw=None
                if family.startswith('comparison'):
                    row=ci.get(key); source=cmeta.get('path')
                    required='boundary_present' if family=='comparison_boundary' else 'score_present'
                    if row is None or not _true(row.get(required)): reason='SOURCE_'+required.upper()+'_NOT_AVAILABLE'
                    else: raw=row.get(spec['column']); value=_num(raw)
                elif family=='boundary_components':
                    row=bi.get(key); source=bmeta.get('path')
                    if row is None or row.get('status')!='AVAILABLE': reason='BOUNDARY_COMPONENT_CELL_NOT_AVAILABLE'
                    else: raw=row.get(spec['column']); value=_num(raw)
                else:
                    row=ri.get(key); source=rmeta.get('path')
                    if not review_complete: reason='FORMAL_192_REVIEW_SUMMARY_NOT_COMPLETE'
                    elif row is None or row.get('summary_validation_status')!='VALID_ACTUAL': raise ValueError('Complete formal review summary has invalid actual row')
                    else:
                        raw=row.get('C_adoption')
                        if raw not in ('YES','NO','UNCERTAIN'): raise ValueError('Review label outside preserved three-state definition')
                        value=float(raw==spec['category'])
                if value is None and not reason: reason='SOURCE_METRIC_NA'
                if spec['metric']=='POST_VALID_INTERFACE_OUTPUTS' and value is not None and value not in (0.,1.): raise ValueError('Original has-valid-edge indicator is not boolean')
                records.append(dict(metric=spec['metric'],label=label,local_step=step,stage=spec['stage'],request_id=q['request_id'],parent_id=q['parent_id'],candidate=q['candidate'],split=q['split'],
                    source_table=source,source_column=spec['column'],source_value=raw,category=spec.get('category'),display_scale=spec['scale'],
                    display_value=value*spec['scale'] if value is not None else None,missing_reason=reason,status='OBSERVED' if value is not None else 'NA'))
        per_output.extend(records); c,p=_aggregate(records,spec,requests); curves.extend(c); parent_rows.extend(p)
    parameters=dict(version=VERSION,fixed_labels=list(MAIN),fixed_updates=list(STEPS),strata=list(STRATA),requests=24,parents=6,noises_per_parent=4,
        paired_identity='The same frozen C and four noise arrays per parent across all four checkpoints; TRAIN/DEV separate.',
        aggregation='Four observed noise values per parent, then equal-parent macro mean. A point is plotted only when every registered request in that stratum has a finite value.',
        count_panels='Sum of the existing per-output valid-edge boolean, or actual review category counts; denominators are the registered stratum request counts.',
        tangent_definition='Source-C two-dimensional tangent-plane residual to the saved nearest finite F edge. Not one-dimensional motion along a boundary tangent.',
        review_policy='Draw only when the formal192-row actual AI_PROXY summary is COMPLETE. Three categories remain visible; inherited reviews excluded.',
        legal_interface='Existing C_F_has_valid_edge and C_F_valid_edge_total_length, unchanged. Apparent image contact and nearest-distance proximity are different.',
        reference_length_L=C.L,metrics=metrics,figure_size_inches=[16,10],dpi=170,
        missing_policy='NaN in plotted arrays breaks every missing endpoint; no interpolation or future-value filling.',
        no_best_checkpoint_selection=True,no_confidence_interval=True,no_significance_or_independent_noise_claim=True)
    tag=hashlib.sha256(json.dumps(dict(sources=sources,parameters=parameters),sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
    dest=DEST/('snapshot_'+tag); final=dest/'manifest.json'
    if final.exists():
        old=C.read(final)
        for p,h in old['files'].items():
            if C.sha(p)!=h: raise ValueError('Saved figure artifact changed')
        C.write(DEST/'latest.json',dict(path=str(final),sha256=C.sha(final),status=old['status']),replace=True); return old
    index={(r['metric'],r['split'],r['local_step']):r for r in curves}
    colors={'TRAIN':'#2466A8','DEV':'#D3651E'}
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(2,3,figsize=(16,10)); axes=axes.ravel()
        def line(panel,metric,style='-',suffix=''):
            for split in STRATA:
                y=[index[metric,split,step]['plotted_value'] for step in STEPS]
                values=np.asarray([np.nan if value is None else value for value in y])
                axes[panel].plot(STEPS,values,linestyle=style,marker='o',markersize=5,color=colors[split],label=split+suffix)
        line(0,'RAW_FINITE_EDGE_MEAN'); line(1,'RAW_EDGE_TANGENT_MEAN'); line(2,'POST_VALID_INTERFACE_OUTPUTS'); line(3,'POST_VALID_INTERFACE_LENGTH')
        line(4,'POST_CF_ILLEGAL','-',' CF'); line(4,'POST_FF_ILLEGAL','--',' FF')
        titles=['RAW distance to finite F edges','RAW finite-edge tangent-plane offset','POST outputs with a legal C/F edge','POST legal C/F shared length','POST illegal affected free faces','RAW C adoption: actual AI_PROXY']
        ys=['Length-weighted mean / L','Length-weighted mean / L','Output count / 12 requests per stratum','Mean total legal length / L','Fraction of free faces (%)','Counts / 12 requests per stratum']
        for i,ax in enumerate(axes):
            ax.set_title(titles[i]); ax.set_xlabel('Additional updates (fixed checkpoints)'); ax.set_ylabel(ys[i]); ax.set_xticks(STEPS); ax.grid(alpha=.22,linewidth=.6); ax.set_axisbelow(True)
            if i<5:
                ax.legend(fontsize=8,loc='best'); ax.set_xlim(-150,6150)
        for i in (2,5): axes[i].set_ylim(0,12.8); axes[i].set_yticks([0,3,6,9,12])
        if review_complete:
            xpos=np.arange(4); width=.33; category_colors={'YES':'#50A878','NO':'#BC5E63','UNCERTAIN':'#B8B8B8'}
            for split,offset,hatch in [('TRAIN',-.18,''),('DEV',.18,'//')]:
                bottom=np.zeros(4)
                for category in ('YES','NO','UNCERTAIN'):
                    values=np.asarray([index['RAW_AI_PROXY_C_ADOPTION_'+category,split,step]['plotted_value'] for step in STEPS],float)
                    axes[5].bar(xpos+offset,values,width,bottom=bottom,color=category_colors[category],edgecolor='white',linewidth=.7,hatch=hatch,label=category if split=='TRAIN' else '_nolegend_')
                    bottom+=values
            axes[5].set_xticks(xpos,labels=[str(x) for x in STEPS]); axes[5].set_xlim(-.65,3.65); axes[5].set_ylim(0,15.4)
            axes[5].legend(fontsize=8,ncol=3,loc='upper center')
            axes[5].text(.5,.89,'Left: TRAIN | Hatched: DEV',ha='center',transform=axes[5].transAxes,fontsize=8)
        else:
            axes[5].text(.5,.5,'Not drawn: formal 192-row review\nsummary is not COMPLETE',transform=axes[5].transAxes,ha='center',va='center',color='#666666')
        for i in range(5):
            related=[r for r in curves if next(s['panel'] for s in metrics if s['metric']==r['metric'])==i]
            if not any(r['plotted_value'] is not None for r in related):
                axes[i].text(.5,.5,'No complete observed stratum yet',transform=axes[i].transAxes,ha='center',va='center',color='#666666')
        fig.suptitle('FULL seat learning curve | same fixed 24 requests across four checkpoints',fontsize=15,y=.978)
        fig.text(.5,.028,'TRAIN: 3 parents x 4 fixed noises | DEV: 3 parents x 4 fixed noises. Equal-parent means; counts retain /12 denominators.',ha='center',fontsize=9)
        fig.text(.5,.01,'Missing or NA endpoints remain blank. No best-checkpoint selection, confidence intervals, or independent-noise/generalization claim.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.055,1,.95))
        png=io.BytesIO(); svg=io.BytesIO(); fig.savefig(png,format='png',dpi=170); fig.savefig(svg,format='svg'); plt.close(fig)
    C._write_bytes(dest/'fixed_checkpoint_results.png',png.getvalue()); C._write_bytes(dest/'fixed_checkpoint_results.svg',svg.getvalue())
    C.csvwrite(dest/'plot_values.csv',curves); C.csvwrite(dest/'per_output_original_values.csv',per_output); C.csvwrite(dest/'per_parent_values.csv',parent_rows)
    C.write(dest/'parameters.json',parameters)
    for p,h in sources.items():
        if C.sha(p)!=h: raise ValueError('Source snapshot changed while plotting')
    complete=all(r['status']=='COMPLETE' for r in curves)
    manifest=dict(status='COMPLETE' if complete else 'PARTIAL',version=VERSION,path=str(dest),source_hashes=sources,
        source_status=dict(comparison=cmeta,boundary_components=bmeta,formal_review=rmeta),review_complete=review_complete,
        curve_points=len(curves),plotted_points=sum(r['plotted_value'] is not None for r in curves),missing_points=sum(r['plotted_value'] is None for r in curves),
        original_value_rows=len(per_output),model_calls=0,new_quality_scans=0,new_queries=0,new_image_reviews=0,new_labels=0,
        actual_figure_visual_inspection='PENDING_CALLER_VIEW',
        files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
    C.write(final,manifest); C.write(DEST/'latest.json',dict(path=str(final),sha256=C.sha(final),status=manifest['status']),replace=True)
    return manifest


if __name__=='__main__':
    r=run(); print(json.dumps({key:r[key] for key in ('status','path','plotted_points','missing_points','review_complete','actual_figure_visual_inspection','model_calls')},sort_keys=True))