"""Balanced continuation figures with explicit objective transition and paired controls."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ..io_utils import file_hash,write_json
from .diagnose import CHECKPOINTS


def curve_series(groups, job, job_protocol):
    """Retain all five fixed observations, including initial and failed slots."""
    selected = [g for g in groups if g['job'] == job]
    lookup = {g['checkpoint']: g for g in selected}
    if len(lookup) != len(selected) or not set(lookup) <= set(CHECKPOINTS):
        raise ValueError('Duplicate or unregistered curve observation')
    expected = len(job_protocol['diagnostic_uids'])
    result = {'updates': [], 'velocity': [], 'endpoint': [], 'macro_endpoint_median': [], 'macro_endpoint_mean': [], 'observations': []}
    def safe_ratio(numerator, denominator):
        if numerator is None or denominator is None or not np.isfinite([numerator, denominator]).all() or denominator <= 0:
            return np.nan
        return float(numerator / denominator)
    for checkpoint in CHECKPOINTS:
        update = job_protocol['start_update'] if checkpoint == 'initial' else int(checkpoint)
        group = lookup.get(checkpoint)
        if group is None:
            recorded, velocity, endpoint = 0, np.nan, np.nan
            macro_median=macro_mean=np.nan
        else:
            if group['expected_parents'] != expected or not 0 <= group['successful_parents'] <= expected:
                raise ValueError('Curve parent denominator differs from protocol')
            recorded = group['successful_parents']
            velocity = safe_ratio(group['velocity']['mse']['mean'], group['velocity']['zero_velocity_mse']['mean'])
            endpoint = safe_ratio(group['endpoint']['endpoint_direction_mse_normalized']['mean'], group['endpoint']['zero_motion_direction_mse_normalized']['mean'])
            macro=group['endpoint']['endpoint_mse_over_zero_motion']
            macro_median=macro['median'] if macro['median'] is not None else np.nan
            macro_mean=macro['mean'] if macro['mean'] is not None else np.nan
        result['updates'].append(update)
        result['velocity'].append(velocity)
        result['endpoint'].append(endpoint)
        result['macro_endpoint_median'].append(macro_median)
        result['macro_endpoint_mean'].append(macro_mean)
        result['observations'].append({'checkpoint': checkpoint, 'registered_update': update,
            'successful_parents': recorded, 'expected_parents': expected,
            'missing_or_failed_parents': expected - recorded,
            'velocity_ratio_missing': not np.isfinite(velocity),
            'endpoint_ratio_missing': not np.isfinite(endpoint)})
    return result


def training_loss_segments(trace, transition=20000, block=1000):
    """Never smooth or join losses across their different objective definitions."""
    if not trace or [r['update'] for r in trace]!=list(range(1,len(trace)+1)):
        raise ValueError('Training trace updates missing or reordered')
    if not 0<transition<len(trace):raise ValueError('Both inherited and weighted trace segments required')
    historical,new=trace[:transition],trace[transition:]
    if any(r.get('loss_is_parent_weighted',False) for r in historical):raise ValueError('Imported C0 history relabeled as weighted')
    if not all(r.get('loss_is_parent_weighted') is True for r in new):raise ValueError('New weighted trace objective marker missing')
    def segment(rows,objective):
        values=[float(r['loss']) for r in rows]
        if not np.isfinite(values).all():raise ValueError('Nonfinite training loss')
        x,y=[],[]
        for i in range(0,len(rows),block):
            batch=rows[i:i+block];x.append(batch[-1]['update']);y.append(float(np.mean([r['loss'] for r in batch])))
        return {'objective':objective,'update_first':rows[0]['update'],'update_last':rows[-1]['update'],
            'updates':x,'block_mean_loss':y,'block_size':block,'rows':len(rows)}
    return {'historical':segment(historical,'original_unweighted_C0'),
            'new':segment(new,'c0_parent_weighted'),'smoothing_crosses_objective_boundary':False}


def plot_training_losses(run,out,protocol):
    fig,axes=plt.subplots(1,2,figsize=(13,5.3));observations={}
    for job in protocol['jobs']:
        trace=json.loads((run/'training'/job/'TRAIN_TRACE.json').read_text())
        series=training_loss_segments(trace,protocol['jobs'][job]['start_update'])
        observations[job]={key:{k:v for k,v in value.items() if k not in ('updates','block_mean_loss')} for key,value in series.items() if isinstance(value,dict)}
        for ax,key in zip(axes,('historical','new')):
            curve=series[key];ax.plot(curve['updates'],curve['block_mean_loss'],label=job)
    for ax,title,ylabel in zip(axes,('Inherited source history: updates1-20000','New continuation: updates20001-240000'),
            ('Original unweighted C0 loss','Parent-weighted C0 loss')):
        ax.set_title(title);ax.set_xlabel('Global optimizer update');ax.set_ylabel(ylabel)
        ax.set_yscale('log');ax.grid(alpha=.2);ax.legend()
    fig.subplots_adjust(bottom=.25,wspace=.3)
    fig.text(.02,.035,'DIFFERENT OBJECTIVES: panel heights and historical-to-new loss drops are not comparable. Block means stay within each segment.\nOriginal Adam/RNG/sampler history is intentionally inherited at20k; this is continuation, not training from scratch.\nWeights use endpoint energy (not time-dependent velocity energy); mean1 is over train12, not each microbatch. Compare arms using common unweighted diagnostics.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'training_loss_segments.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig);return observations


def plot_paired_controls(out,summary,protocol):
    records=summary['paired_weighted_vs_unweighted_controls']['records'];uids=sorted(protocol['uids'])
    failed={(row['uid'],row['model']) for row in summary['errors']}
    fig,axes=plt.subplots(1,2,figsize=(12.5,8.5),sharey=True)
    for ax,seed in zip(axes,(10,11)):
        lookup={r['uid']:r for r in records if r['seed']==seed and r['policy']=='raw'}
        for i,uid in enumerate(uids):
            row=lookup[uid]
            if row['status']!='success':
                label='deployment\nfailed' if any((uid,name) in failed for name in (f'joint_{seed}',f'unweighted_joint_{seed}')) else 'missing'
                ax.text(0,i,label,color='crimson',ha='center',va='center',fontsize=8);continue
            values=row['weighted_rms_gain_over_control_percent_two_measurements']
            qualified=row['weighted_qualified'] and row['control_qualified']
            for offset,value,marker in zip((-.10,.10),values,('o','s')):
                if value is None:ax.text(0,i+offset,'undefined',fontsize=7);continue
                ax.scatter(value,i+offset,marker=marker,s=45,facecolors='#2176ae' if qualified else 'none',edgecolors='#2176ae')
            if all(v is not None for v in values):ax.plot(values,[i-.10,i+.10],color='#2176ae',alpha=.4,lw=.8)
        ax.axvline(0,color='gray',ls='--');ax.set_title(f'Seed {seed}: weighted vs unweighted raw240k')
        ax.set_xlabel('Surface RMS reduction vs same-seed control (%)');ax.grid(axis='x',alpha=.2)
        ax.set_yticks(range(16),[u[:8]+(' [train]' if u in protocol['train_uids'] else ' [check]') for u in uids])
    axes[0].invert_yaxis();fig.subplots_adjust(left=.17,bottom=.17,wspace=.2)
    fig.text(.02,.02,'All16 previously used development parents, no filtering. Circle/square: two shared new MC draws (8192 points per direction).\nFilled: both raw outputs qualified; hollow: either raw output unqualified. Positive gain alone is not a valid mesh improvement.\nFixed fallback is a separate policy in the full paired report. Internal checks are not held-out generalization.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(out/f'paired_control_surface.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)


def plot(run):
    run=Path(run).resolve();p=json.loads((run/'RUN_PROTOCOL.json').read_text());d=json.loads((run/'TRAIN_DIAGNOSTICS.json').read_text());s=json.loads((run/'SUMMARY.json').read_text())
    out=run/'figures';out.mkdir(exist_ok=False)
    jobs=list(p['jobs'])
    if set(jobs)!={'joint_10','joint_11'}:raise ValueError('Only two new joint training curves are allowed')
    if s['results_sha256']!=file_hash(run/'results.json') or s['mechanism_sha256']!=file_hash(run/'TRAIN_DIAGNOSTICS.json'):
        raise ValueError('Summary refers to another result or mechanism ledger')
    loss_observations=plot_training_losses(run,out,p)
    fig,axes=plt.subplots(1,2,figsize=(13,5.5));observations={}
    for ax,job in zip(axes.flat,jobs):
        series=curve_series(d['groups'],job,p['jobs'][job]);observations[job]=series['observations']
        x,v,e=series['updates'],series['velocity'],series['endpoint']
        ax.plot(x,v,'o-',label='Offline bridge MSE / zero velocity')
        ax.plot(x,e,'s-',label='Free endpoint: ratio of global means')
        ax.plot(x,series['macro_endpoint_median'],'d--',label='Free endpoint: median parent ratio')
        ax.plot(x,series['macro_endpoint_mean'],'x:',label='Free endpoint: mean parent ratio')
        ax.scatter([x[0],x[0]],[v[0],e[0]],marker='^',s=90,facecolors='none',edgecolors='black',label='Initial snapshot',zorder=5)
        ax.axhline(1,color='gray',ls='--',lw=.8);ax.set_yscale('log');ax.set_title(job)
        ax.set_xlabel('Registered optimizer update');ax.set_ylabel('Relative error (global and macro shown separately)')
        missing=sum(item['missing_or_failed_parents'] for item in series['observations'])
        total=sum(item['expected_parents'] for item in series['observations'])
        overlap=' (same snapshot as 20000)' if p['jobs'][job]['start_update']==20000 else ''
        ax.text(.02,.02,f'Initial @ {x[0]}{overlap}; missing slots {missing}/{total}',transform=ax.transAxes,fontsize=7)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.subplots_adjust(bottom=.22,wspace=.28)
    fig.text(.02,.02,'TRAIN PARENTS ONLY; every fixed checkpoint shown, no weight selection.\nOnly parent loss weights change after frozen20k; final240k is fixed. All displayed diagnostic errors are UNWEIGHTED. No checkpoint selection.\nTeacher-conditioned bridge probes are offline diagnostics; free CPU deployment uses only original proxy256 input.',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(out/f'fit_curves.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    analysis=json.loads((run/'ANALYSIS_ROWS.json').read_text())['rows'];lookup={(r['uid'],r['model'],r['policy']):r for r in analysis}
    names=['parent_joint_10','unweighted_joint_10','joint_10','parent_joint_11','unweighted_joint_11','joint_11'];uids=sorted(p['uids'])
    colors=np.full((16,6),3);texts={}
    failed={(row['uid'],row['model']) for row in s['errors']}
    for i,uid in enumerate(uids):
        for j,name in enumerate(names):
            r=lookup.get((uid,name,'raw'))
            if r is None:texts[i,j]='deployment\nfailed' if (uid,name) in failed else 'missing';continue
            colors[i,j]=2 if r['qualified_ge2pct_both'] else 1 if r['qualified'] else 0
            gain=100*(1-r['rms_over_fixed_two_seeds'][0]) if r['paired_available'] else float('nan')
            texts[i,j]=f'{gain:+.2f}%\n'+('qualified' if r['qualified'] else 'unqualified')
    fig,ax=plt.subplots(figsize=(14,10))
    ax.imshow(colors,cmap=ListedColormap(['#e1e3e8','#fae5b2','#abdbbf','#e4a4a4']),vmin=0,vmax=3,aspect='auto')
    for (i,j),text in texts.items():ax.text(j,i,text,ha='center',va='center',fontsize=9)
    ax.set_xticks(range(6),['Source20k s10','Unweighted240k s10','Weighted240k s10','Source20k s11','Unweighted240k s11','Weighted240k s11'])
    ax.set_yticks(range(16),[u[:8]+(' [train]' if u in p['train_uids'] else ' [check]') for u in uids])
    ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,16,1),minor=True);ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',length=0)
    ax.set_title('Final raw paired surface RMS improvement vs Fixed')
    fig.subplots_adjust(left=.19,bottom=.12)
    fig.text(.01,.015,'16 previously used development parents; internal check is not held-out generalization.\nText: first new MC measurement. Green requires quality AND >=2% gain in BOTH measurements.\nGray: every unqualified raw result retained. Fixed fallback is reported separately.',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(out/f'final_parent_outcomes.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    plot_paired_controls(out,s,p)
    write_json(out/'FIGURE_MANIFEST.json',{'files':{f.name:file_hash(f) for f in sorted(out.iterdir()) if f.is_file()},
        'source_sha256':file_hash(__file__),'final_results_sha256':file_hash(run/'results.json'),
        'diagnostics_sha256':file_hash(run/'TRAIN_DIAGNOSTICS.json'),'all_fixed_checkpoints_shown':True,'fixed_observations':observations,
        'joint_initial_and_20000_are_separate_observations_at_same_x':True,
        'single_references_retrained_or_given_new_training_curves':False,
        'macro_and_global_error_curves_separate':True,
        'training_loss_objectives_plotted_in_separate_panels':True,'training_loss_segments':loss_observations,
        'paired_controls_use_same_new_MC_measurements':True,'all_16_parent_rows_include_unqualified_and_missing':True,
        'actual_deployment_failures_distinguished_from_unattempted':True,
        'missing_curve_values_rendered_as_NaN_with_explicit_missing_counts':True})
    print(json.dumps({'figures':8,'path':str(out)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();plot(args.run)