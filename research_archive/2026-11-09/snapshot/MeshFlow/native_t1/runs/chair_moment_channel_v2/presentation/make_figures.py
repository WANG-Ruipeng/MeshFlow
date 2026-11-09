"""Render final V2 evidence only. No model imports, training, or score changes.

Usage (after all three fixed scoring stages finish):
    python -B presentation/make_figures.py --out <V2 output directory>

PNG/PDF and their exact input rows are written under presentation/figures/.
Missing or incomplete registered evidence produces a BLOCKED rendering receipt;
it never creates a quality plot from an incomplete denominator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter


LABELS=('START','H_CONT_END','MOMENT_XY_END')
NAMES={'START':'START','H_CONT_END':'H_CONT','MOMENT_XY_END':'MOMENT'}
COLORS={'START':'#566573','H_CONT_END':'#D97732','MOMENT_XY_END':'#237BB7'}
COMPARISONS=('MOMENT_XY_END-H_CONT_END','H_CONT_END-START','MOMENT_XY_END-START')
COMPARE_NAMES=('MOMENT - H_CONT','H_CONT - START','MOMENT - START')
DEFAULT_OUT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_rows(path,rows):
    fields=sorted({k for r in rows for k in r})
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:'NA' if v is None else json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def assert_finite(value,context):
    if not isinstance(value,(int,float,bool)) or not np.isfinite(value):
        raise ValueError('Missing/nonfinite registered number: '+context)
    return float(value)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,
        'savefig.dpi':240,'pdf.fonttype':42,'ps.fonttype':42})


def load_evidence(out):
    folder=out/'paired_results'
    names=('results','per_output','per_parent','primary_paired_parents','primary_summary','width_per_curve','control_curve_counts')
    files={name:folder/(name+'.json') for name in names}
    absent=[str(p) for p in files.values() if not p.exists()]
    if absent:raise ValueError('Final scoring files not available: '+str(absent))
    data={name:read(path) for name,path in files.items()};result=data['results']
    if result['stages']['TRAINED']!='DONE' or result['stages']['EVALUATED']!='DONE' or result['decision']['status']!='COMPLETE':
        raise ValueError('Training/evaluation is incomplete; no quality figure is rendered')
    if result['gate_results']['status']!='PASS':raise ValueError('V2 gate has not passed')
    for label in LABELS:
        rows=[r for r in data['per_output'] if r['checkpoint_label']==label]
        if len(rows)!=144 or sum(r['status']=='PASS' for r in rows)!=138 or sum(r['status']=='NOT_APPLICABLE' for r in rows)!=6:
            raise ValueError('Expected complete original 138 applicable + 6 NA for '+label)
    ac=lambda r:r['table']=='A' and r['split']=='validation' and r['ratio']==.2
    parents=[r for r in data['per_parent'] if ac(r)]
    cases=[r for r in data['per_output'] if ac(r)]
    uids=sorted({r['uid'] for r in parents})
    if len(uids)!=8 or len(parents)!=24 or len(cases)!=48:raise ValueError('Primary denominator changed')
    pi={(r['checkpoint_label'],r['uid']):r for r in parents}
    ci={(r['checkpoint_label'],r['uid'],r['noise_index']):r for r in cases}
    if len(pi)!=24 or len(ci)!=48:raise ValueError('Duplicate primary identity')
    for label in LABELS:
        for uid in uids:
            parent=pi[(label,uid)]
            if parent['status']!='PASS' or parent['requested_cases']!=2 or parent['noise_indices']!=[0,1]:
                raise ValueError('Incomplete two-noise parent')
            mean=assert_finite(parent['fixed_query_MSE'],uid)
            noise=[assert_finite(ci[(label,uid,n)]['fixed_query_MSE'],uid) for n in (0,1)]
            if not np.isclose(mean,np.mean(noise),atol=1e-16,rtol=1e-13):raise ValueError('Existing parent reduction does not match saved cases')
    pairs=data['primary_paired_parents'];summaries=data['primary_summary']
    if len(pairs)!=24 or len(summaries)!=3:raise ValueError('Expected three comparisons of eight parents')
    for comparison in COMPARISONS:
        selected=[r for r in pairs if r['comparison']==comparison]
        summary=next(r for r in summaries if r['comparison']==comparison)
        if {r['uid'] for r in selected}!=set(uids) or not summary['complete_8_parents']:raise ValueError('Incomplete registered paired comparison')
        delta=[assert_finite(r['fixed_query_MSE_delta'],comparison) for r in selected]
        if not np.isclose(np.mean(delta),summary['delta_mean'],atol=1e-16,rtol=1e-13):raise ValueError('Existing paired mean does not match saved pairs')
        if not np.isclose(np.median(delta),summary['paired_delta_median'],atol=1e-16,rtol=1e-13):raise ValueError('Existing paired median does not match saved pairs')
    curves=[r for r in data['width_per_curve'] if r['candidate']=='MODEL' and r['split']=='validation']
    wi={(r['checkpoint_label'],r['uid'],r['noise_index']):r for r in curves}
    widthuids=sorted({r['uid'] for r in curves})
    if len(curves)!=24 or len(wi)!=24 or len(widthuids)!=4:raise ValueError('Control denominator changed')
    if not set(widthuids).issubset(uids):raise ValueError('Control UID cannot be matched to registered validation parent')
    for label in LABELS:
        for uid in widthuids:
            for noise in (0,1):
                row=wi[(label,uid,noise)]
                if row['status']!='PASS':raise ValueError('Incomplete actual width curve')
                for metric in ('b_self_abs_error','b_GT_abs_error'):
                    value=assert_finite(row[metric],metric)
                    if value<0:raise ValueError('Absolute error may not be negative')
    counts=[r for r in data['control_curve_counts'] if r['candidate']=='MODEL' and r['split']=='validation' and r['noise_index']=='all']
    if len(counts)!=3:raise ValueError('Missing registered control count summary')
    for row in counts:
        if row['status']!='COMPLETE' or row['registered_curves']!=8 or row['registered_columns']!=24:
            raise ValueError('Control count denominator is incomplete')
    return dict(data=data,files=files,uids=uids,parent_index=pi,case_index=ci,parents=parents,cases=cases,
        pairs=pairs,summaries=summaries,width_index=wi,width_uids=widthuids,curves=curves,counts=counts)


def save_figure(fig,folder,stem):
    paths=[]
    for extension in ('png','pdf'):
        path=folder/(stem+'.'+extension)
        kwargs={'metadata':{'Creator':'V2 CPU presentation; saved evidence only'}} if extension=='pdf' else {}
        fig.savefig(path,facecolor='white',bbox_inches='tight',**kwargs);paths.append(path)
    plt.close(fig)
    return [dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in paths]


def primary_figure(e,folder,aliases):
    fig,axes=plt.subplots(1,2,figsize=(13.4,5.5),gridspec_kw={'width_ratios':[1.35,1.]})
    fig.subplots_adjust(left=.065,right=.985,bottom=.22,top=.78,wspace=.28)
    fig.suptitle('V2 fixed primary endpoint: A20 validation, 8 paired parents',fontsize=14,y=.98)
    ax=axes[0];offsets=(-.24,0,.24)
    for label,offset in zip(LABELS,offsets):
        for j,uid in enumerate(e['uids']):
            x=j+offset;parent=e['parent_index'][(label,uid)]
            noise=[e['case_index'][(label,uid,n)]['fixed_query_MSE'] for n in (0,1)]
            ax.plot([x,x],noise,color=COLORS[label],alpha=.48,linewidth=.8,zorder=1)
            ax.scatter([x-.026,x+.026],noise,s=17,facecolors='none',edgecolors=COLORS[label],linewidths=.8,zorder=2)
            ax.scatter(x,parent['fixed_query_MSE'],s=38,color=COLORS[label],marker='D',edgecolors='white',linewidths=.5,zorder=3)
    ax.set_xticks(range(8),[aliases[u] for u in e['uids']]);ax.set_xlim(-.65,7.65)
    ax.set_ylabel('fixed_query_MSE  (distance / L0) squared')
    ax.set_xlabel('Parent ID (same two frozen noises per checkpoint)')
    ax.set_title('a   Parent means and both saved noise outputs',loc='left',pad=12)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(axis='y',alpha=.18)
    checkpoint_handles=[Line2D([],[],color=COLORS[l],marker='D',linestyle='none',markersize=6,label=NAMES[l]) for l in LABELS]
    fig.legend(handles=checkpoint_handles,loc='upper center',bbox_to_anchor=(.36,.915),ncol=3,frameon=False)
    ax=axes[1]
    for j,comparison in enumerate(COMPARISONS):
        byuid={r['uid']:r for r in e['pairs'] if r['comparison']==comparison}
        summary=next(r for r in e['summaries'] if r['comparison']==comparison)
        for k,uid in enumerate(e['uids']):
            x=j+np.linspace(-.16,.16,8)[k];y=byuid[uid]['fixed_query_MSE_delta']
            ax.scatter(x,y,s=26,facecolors='#AAB3BD',edgecolors='#5F6872',linewidths=.6,zorder=3)
        ax.scatter(j,summary['delta_mean'],marker='D',s=66,color='#17212D',edgecolor='white',linewidth=.7,zorder=4)
        ax.plot([j-.21,j+.21],[summary['paired_delta_median']]*2,color='#A42B48',linewidth=2,zorder=5)
        ax.text(j,1.018,str(summary['improved_count'])+'/8 improved',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=8.5)
    ax.axhline(0,color='#3B4249',linewidth=1,linestyle='--',zorder=1)
    ax.set_xticks(range(3),['MOMENT\n- H_CONT','H_CONT\n- START','MOMENT\n- START']);ax.set_xlim(-.55,2.55)
    ax.set_ylabel('Paired parent difference in fixed_query_MSE')
    ax.set_title('b   Three pre-registered paired comparisons',loc='left',pad=25)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(axis='y',alpha=.18)
    ax.legend(handles=[Line2D([],[],color='#5F6872',marker='o',linestyle='none',markersize=5,label='One parent'),
        Line2D([],[],color='#17212D',marker='D',linestyle='none',markersize=6,label='Mean delta'),
        Line2D([],[],color='#A42B48',linewidth=2,label='Median delta')],loc='best',fontsize=8,framealpha=.94)
    fig.text(.065,.064,'Left: diamonds = within-parent mean of two noises; open circles = individual saved outputs.\n'
        'Right: all 8 paired parents, equal weight; negative differences favor the first named checkpoint. No confidence intervals are inferred.',fontsize=9,va='bottom')
    return save_figure(fig,folder,'primary_parent_pairs')


def control_figure(e,folder,aliases):
    fig,axes=plt.subplots(1,3,figsize=(15.6,5.8),gridspec_kw={'width_ratios':[1.05,1.1,1.1]})
    fig.subplots_adjust(left=.055,right=.985,bottom=.24,top=.77,wspace=.30)
    fig.suptitle('V2 condition response: actual target columns and per-curve absolute errors',fontsize=14,y=.98)
    handles=[Line2D([],[],color=COLORS[l],marker='o',linestyle='none',markersize=6,label=NAMES[l]) for l in LABELS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.50,.925),ncol=3,frameon=False)
    metrics=(('strict_column_win_count',24),('all_three_strict_win_curve_count',8),('one_input_strictly_wins_all_targets_curve_count',8))
    ax=axes[0]
    for label,offset in zip(LABELS,(-.24,0,.24)):
        row=next(r for r in e['counts'] if r['checkpoint_label']==label)
        for j,(metric,denom) in enumerate(metrics):
            value=row[metric];x=j+offset;fraction=value/denom
            ax.bar(x,fraction,width=.205,color=COLORS[label],alpha=.94,zorder=2)
            ax.annotate(str(value)+'/'+str(denom),(x,fraction),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=8.5,rotation=0)
    ax.set_xticks(range(3),['Correct input\nstrict column wins','All 3 targets\ncorrect together','One input wins\nall 3 targets'])
    ax.set_ylim(0,1.20);ax.set_yticks([0,.25,.5,.75,1.]);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Fraction of the registered denominator')
    ax.set_title('a   Validation counts (4 parents x 2 noises)',loc='left',pad=12)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    for axis,metric,name,panel in zip(axes[1:],('b_self_abs_error','b_GT_abs_error'),('|b_self - 1|','|b_GT - 1|'),('b','c')):
        for noise,base in ((0,0),(1,4)):
            for uid in e['width_uids']:
                values=[e['width_index'][(label,uid,noise)][metric] for label in LABELS]
                axis.plot(np.arange(3)+base,values,color='#ABB0B7',linewidth=.85,alpha=.8,zorder=1)
                for j,(label,value) in enumerate(zip(LABELS,values)):
                    axis.scatter(j+base,value,color=COLORS[label],s=35,marker='o',edgecolors='white',linewidths=.45,zorder=3)
            axis.text(base+1,1.03,'noise '+str(noise),transform=axis.get_xaxis_transform(),ha='center',fontsize=9)
        axis.axvline(3,color='#C2C6CB',linestyle=':',linewidth=1)
        axis.set_xlim(-.45,6.45);axis.set_xticks([0,1,2,4,5,6],['START','H','M','START','H','M'])
        axis.set_ylim(bottom=0);axis.set_ylabel(name+'  (computed separately per curve)')
        axis.set_xlabel('Checkpoint within each fixed-noise group')
        axis.set_title(panel+'   '+name+': four paired parents per noise',loc='left',pad=28)
        axis.grid(axis='y',alpha=.18);axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.text(.055,.074,'Counts: 24 fixed target columns; 8 complete curves. Ties do not count as strict wins.\n'
        'Gray lines: the same parent and noise across checkpoints. Errors are taken before averaging; under- and over-response cannot cancel.\n'
        'Correct-column / all-three wins favor larger values; one-input-all-target wins are disclosed separately. Absolute slope errors favor smaller values.',fontsize=9,va='bottom')
    return save_figure(fig,folder,'control_columns_and_curve_errors')


def main(out):
    out=Path(out).resolve();destination=out/'presentation';destination.mkdir(parents=True,exist_ok=True)
    status_path=destination/'figure_status.json'
    try:e=load_evidence(out)
    except (ValueError,KeyError,FileNotFoundError,StopIteration) as exc:
        value=dict(status='BLOCKED',reason=str(exc),plots_written=0,model_calls=0,score_changes=0)
        write(status_path,value);return value
    folder=destination/'figures';folder.mkdir(parents=True,exist_ok=True);style()
    aliases={uid:'P'+str(j+1) for j,uid in enumerate(e['uids'])}
    audit_inputs=dict(parent_key=[dict(parent_alias=aliases[uid],uid=uid,is_B_control_parent=uid in e['width_uids']) for uid in e['uids']],
        primary_parent_values=e['parents'],primary_noise_values=e['cases'],primary_paired_deltas=e['pairs'],
        primary_summary=e['summaries'],width_curves=e['curves'],control_counts=e['counts'])
    for name,rows in audit_inputs.items():
        write(folder/(name+'.json'),rows);csv_rows(folder/(name+'.csv'),rows)
    artifacts=primary_figure(e,folder,aliases)+control_figure(e,folder,aliases)
    value=dict(status='COMPLETE',plots_written=2,formats=['PNG','PDF'],model_calls=0,score_changes=0,
        training_and_scoring_modules_imported=False,source_files={str(p.relative_to(out)):digest(p) for p in e['files'].values()},
        script_sha256=digest(__file__),figures=artifacts,parent_count=8,noises_per_parent=2,control_parent_count=4,
        correct_column_denominator=24,all_three_and_universal_denominator=8,
        aggregate_rule='Read existing per-curve absolute errors and two-noise parent means; check their consistency; do not redefine scores.',
        quality_result=e['data']['results']['decision']['QUALITY_RESULT'],visual_QA='PENDING_LOCAL_IMAGE_INSPECTION')
    write(status_path,value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();result=main(args.out);print(json.dumps(result,ensure_ascii=False,indent=2))
