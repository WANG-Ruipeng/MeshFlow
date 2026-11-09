"""Per-parent paired interpretation of the fixed CPU margin-target diagnosis."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .prepare import UIDS,read,require
from .evaluate import ENDPOINTS,UPDATES
from ..io_utils import file_hash,write_json


def fractional_change(before,after):return (before-after)/before if before>0 else None


def summarize(run):
    run=Path(run).resolve();p=read(run/'RUN_PROTOCOL.json');results=read(run/'results.json');diag=read(run/'TRAIN_DIAGNOSTICS.json')
    protocol_sha=file_hash(run/'RUN_PROTOCOL.json')
    require(results['protocol_sha256']==diag['protocol_sha256']==protocol_sha,'Report protocol mismatch')
    rows={(r['uid'],r['endpoint']):r for r in results['rows']}
    observations={(r['job'],r['update']):r for r in diag['rows']}
    paired=[]
    for uid in UIDS:
        record={'uid':uid,'endpoints':{},'observations':{},'missing_or_failed_endpoints':[]}
        fixed=rows.get((uid,'fixed'))
        for name in ENDPOINTS:
            row=rows.get((uid,name))
            if row is None:record['missing_or_failed_endpoints'].append(name);continue
            rms=[m['rms_bbox_percent'] for m in row['metrics']]
            fixed_rms=[m['rms_bbox_percent'] for m in fixed['metrics']] if fixed else None
            record['endpoints'][name]={'rms_bbox_percent':rms,'rms_over_fixed':[a/b for a,b in zip(rms,fixed_rms)] if fixed_rms else None,
                'qualified_original':row['qualified_original'],'qualified_strict':row['qualified_strict'],
                'actual_margin_vs_fixed_deg':row['actual_margin_vs_fixed_deg'],'direction_errors':row['direction_errors'],
                'face_count':row['quality']['face_count'],'exact_valid':row['audit']['valid']}
        for kind in ('old','margin'):
            job=kind+'_'+uid[:8]
            record['observations'][kind]=[{'update':update,'status':'success' if (job,update) in observations else 'missing_or_failed',
                'velocity':observations[(job,update)]['velocity'] if (job,update) in observations else None,
                'endpoint':observations[(job,update)]['endpoint'] if (job,update) in observations else None} for update in UPDATES]
        a=record['endpoints'].get('old_final');b=record['endpoints'].get('margin_final')
        record['margin_target_final_RMS_improvement_over_old_target_final']=[fractional_change(x,y) for x,y in zip(a['rms_bbox_percent'],b['rms_bbox_percent'])] if a and b else None
        record['old_teacher_zero_motion']=p['target_displacement_energy'][uid]['old']==0
        paired.append(record)
    value={'protocol_sha256':protocol_sha,'results_sha256':file_hash(run/'results.json'),
        'mechanism_sha256':file_hash(run/'TRAIN_DIAGNOSTICS.json'),'paired_time_sequence_audit':read(run/'PAIRING_AUDIT.json'),
        'scope':'one training seed10; four previously seen parents; eight paired single-parent fits',
        'reference_surface':p['evaluation']['reference_surface'],'reference_raw_asset_claim':False,
        'planned_endpoints':20,'successful_endpoints':results['success_count'],'errors':results['errors'],
        'missing_success_endpoints':results['missing_success_slots'],'planned_observations':24,
        'successful_observations':diag['success_count'],'missing_success_observations':diag['missing_success_slots'],
        'paired':paired,'pure_margin_scalar_causal_claim':False,'generalization_claim':False,
        'checkpoint_selection':False,'fallback':False,'new_target_changes_quality_margin_and_optimization_trajectory':True,
        'reference_semantics_correction':'output/REFERENCE_SEMANTICS_CORRECTION_20260925.md; sealed historical wording is not changed'}
    correction=run.parent/'REFERENCE_SEMANTICS_CORRECTION_20260925.md'
    if correction.exists():value['reference_semantics_correction_sha256']=file_hash(correction)
    write_json(run/'SUMMARY.json',value,replace=(run/'SUMMARY.json').exists());return value


def render(run,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    run=Path(run);p=read(run/'RUN_PROTOCOL.json');figdir=run/'figures';figdir.mkdir(exist_ok=True)
    generated=[];colors={'old':'#3265a8','margin':'#cd6134'}
    def finish(fig,name,title,subtitle):
        fig.suptitle(title,y=.99,fontsize=14)
        fig.text(.5,.947,subtitle,ha='center',fontsize=9)
        fig.tight_layout(rect=(.02,.025,.98,.91))
        for suffix in ('png','pdf'):
            path=figdir/(name+'.'+suffix);require(not path.exists(),'Do not overwrite scientific figure')
            fig.savefig(path,dpi=170);generated.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,uid in zip(axes.flat,UIDS):
        for kind in ('old','margin'):
            trace=read(run/'training'/f'{kind}_{uid[:8]}'/'TRAIN_TRACE.json')
            for start in range(0,len(trace),100):
                chunk=trace[start:start+100]
                ax.plot(np.mean([r['update'] for r in chunk]),np.mean([r['loss'] for r in chunk]),'.',color=colors[kind],ms=3)
        ax.set(title=uid[:8]+(' | old target zero motion' if p['target_displacement_energy'][uid]['old']==0 else ''),xlabel='Actual AdamW update',ylabel='C0 velocity MSE',yscale='log');ax.grid(alpha=.2)
    fig.legend(handles=[Line2D([],[],color=colors[k],marker='.',label=k+' target') for k in colors],loc='upper center',bbox_to_anchor=(.5,.929),ncol=2)
    finish(fig,'training_loss','Paired CPU single-parent C0 fits','Seed10 for every job; non-overlapping100-update means; different target magnitudes, no checkpoint selection')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,record in zip(axes.flat,summary['paired']):
        for kind in ('old','margin'):
            values=record['observations'][kind];xs=[x['update'] for x in values]
            ys=[x['endpoint']['endpoint_mse_over_zero_motion'] if x['endpoint'] else None for x in values]
            available=[(x,y) for x,y in zip(xs,ys) if y is not None]
            if available:ax.plot(*zip(*available),marker='o',color=colors[kind],label=kind+' target')
            if kind=='old' and record['old_teacher_zero_motion']:
                final=values[-1]['endpoint'];label=('null ratio: zero-motion target\nabsolute final MSE='+format(final['endpoint_direction_mse'],'.3g')) if final else 'null ratio: zero-motion target'
                ax.text(.03,.08,label,transform=ax.transAxes,fontsize=9,color=colors[kind])
        ax.axhline(1,color='.5',ls=':',lw=1);ax.set(title=record['uid'][:8],xlabel='Fixed observation update',ylabel='Endpoint direction MSE / target energy',yscale='log');ax.legend();ax.grid(alpha=.2)
    finish(fig,'direction_fit','Raw original32-step CPU deployment versus its assigned target','All0/5k/20k observations retained; zero target denominator is undefined; only final20k is the result')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    labels=['Fixed','Old teacher','Margin teacher','Old fit','Margin fit']
    for ax,record in zip(axes.flat,summary['paired']):
        for x,name in enumerate(ENDPOINTS):
            item=record['endpoints'].get(name)
            if item is None:ax.text(x,1,'missing',rotation=90,ha='center');continue
            ratios=item['rms_over_fixed']
            if ratios is None:continue
            face='#3265a8' if item['qualified_original']['qualified'] else 'none'
            ax.scatter([x-.07,x+.07],ratios,facecolors=face,edgecolors='#3265a8',s=48)
            if item['qualified_strict']['qualified']:ax.scatter([x-.07,x+.07],ratios,marker='+',c='black',s=65)
        ax.axhline(1,color='.5',ls=':');ax.set(title=record['uid'][:8],ylabel='Same-field surface RMS / Fixed')
        ax.set_xticks(range(5),labels,rotation=20);ax.grid(axis='y',alpha=.2)
    finish(fig,'final_surface','All20 raw endpoints; same frozen full-field surface reference','Two independent8192-per-direction MC evaluations; hollow=original floor fails; +=both angles at least Fixed+0.1degree')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,record in zip(axes.flat,summary['paired']):
        for x,name in enumerate(ENDPOINTS):
            item=record['endpoints'].get(name)
            if item is None:continue
            for key,marker in [('min_triangle_angle_deg','o'),('triangle_min_angle_p05_deg','x')]:
                ax.scatter(x,item['actual_margin_vs_fixed_deg'][key],marker=marker,c='#3265a8' if marker=='o' else '#cd6134')
        ax.axhline(0,color='.4',ls=':');ax.axhline(.1,color='.3',ls='--');ax.set(title=record['uid'][:8],ylabel='Angle difference from Fixed (degree)')
        ax.set_xticks(range(5),labels,rotation=20);ax.grid(axis='y',alpha=.2)
    finish(fig,'final_quality','Actual angle margins of every retained raw endpoint','Circle=minimum angle; cross=5th percentile of per-triangle minimum angles; gates use tolerance1e-6degree')
    manifest={'files':{str(path.relative_to(run)):file_hash(path) for path in generated},
        'summary_sha256':file_hash(run/'SUMMARY.json'),'report_source_sha256':file_hash(__file__),
        'all_four_parents_retained':True,'training_seeds':[10],'reference_original_asset_claim':False,
        'fixed_observations':[0,5000,20000],'missing_values_not_imputed':True}
    write_json(figdir/'FIGURE_MANIFEST.json',manifest)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--run',required=True);a.add_argument('--render',action='store_true');args=a.parse_args()
    value=summarize(args.run)
    if args.render:render(args.run,value)
    print(json.dumps({'successful_endpoints':value['successful_endpoints'],'successful_observations':value['successful_observations']}))


if __name__=='__main__':main()
