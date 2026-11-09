"""Standalone long-budget figures; frozen single references get no new training curve."""
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


def plot(run):
    run=Path(run);p=json.loads((run/'RUN_PROTOCOL.json').read_text());d=json.loads((run/'TRAIN_DIAGNOSTICS.json').read_text());s=json.loads((run/'SUMMARY.json').read_text())
    out=run/'figures';out.mkdir(exist_ok=False)
    jobs=list(p['jobs'])
    if set(jobs)!={'joint_10','joint_11'}:raise ValueError('Only two new joint training curves are allowed')
    if s['results_sha256']!=file_hash(run/'results.json') or s['mechanism_sha256']!=file_hash(run/'TRAIN_DIAGNOSTICS.json'):
        raise ValueError('Summary refers to another result or mechanism ledger')
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
    fig.text(.02,.02,'TRAIN PARENTS ONLY; every fixed checkpoint shown, no weight selection.\nJoint models resume frozen20k and end at240k; frozen single20k references are not retrained. Parent exposure matches80k, not Adam steps/history or compute.\nTeacher-conditioned bridge probes are offline diagnostics; free CPU deployment uses only original proxy256 input.',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(out/f'fit_curves.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    analysis=json.loads((run/'ANALYSIS_ROWS.json').read_text())['rows'];lookup={(r['uid'],r['model'],r['policy']):r for r in analysis}
    names=['parent_joint_10','joint_10','parent_joint_11','joint_11'];uids=sorted(p['uids'])
    colors=np.full((16,4),3);texts={}
    for i,uid in enumerate(uids):
        for j,name in enumerate(names):
            r=lookup.get((uid,name,'raw'))
            if r is None:texts[i,j]='missing';continue
            colors[i,j]=2 if r['qualified_ge2pct_both'] else 1 if r['qualified'] else 0
            gain=100*(1-r['rms_over_fixed_two_seeds'][0]) if r['paired_available'] else float('nan')
            texts[i,j]=f'{gain:+.2f}%\n'+('qualified' if r['qualified'] else 'unqualified')
    fig,ax=plt.subplots(figsize=(10,10))
    ax.imshow(colors,cmap=ListedColormap(['#e1e3e8','#fae5b2','#abdbbf','#e4a4a4']),vmin=0,vmax=3,aspect='auto')
    for (i,j),text in texts.items():ax.text(j,i,text,ha='center',va='center',fontsize=9)
    ax.set_xticks(range(4),['Source20k seed10','Joint240k seed10','Source20k seed11','Joint240k seed11'])
    ax.set_yticks(range(16),[u[:8]+(' [train]' if u in p['train_uids'] else ' [check]') for u in uids])
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,16,1),minor=True);ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',length=0)
    ax.set_title('Final raw paired surface RMS improvement vs Fixed')
    fig.subplots_adjust(left=.19,bottom=.12)
    fig.text(.01,.015,'16 previously used development parents; internal check is not held-out generalization.\nText: first new MC measurement. Green requires quality AND >=2% gain in BOTH measurements.\nGray: every unqualified raw result retained. Fixed fallback is reported separately.',fontsize=9)
    for ext in ['png','pdf']:fig.savefig(out/f'final_parent_outcomes.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    write_json(out/'FIGURE_MANIFEST.json',{'files':{f.name:file_hash(f) for f in sorted(out.iterdir()) if f.is_file()},
        'source_sha256':file_hash(__file__),'final_results_sha256':file_hash(run/'results.json'),
        'diagnostics_sha256':file_hash(run/'TRAIN_DIAGNOSTICS.json'),'all_fixed_checkpoints_shown':True,'fixed_observations':observations,
        'joint_initial_and_20000_are_separate_observations_at_same_x':True,
        'single_references_retrained_or_given_new_training_curves':False,
        'macro_and_global_error_curves_separate':True,
        'missing_curve_values_rendered_as_NaN_with_explicit_missing_counts':True})
    print(json.dumps({'figures':4,'path':str(out)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();plot(args.run)