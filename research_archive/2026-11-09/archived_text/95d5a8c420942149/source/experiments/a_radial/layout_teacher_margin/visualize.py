"""Fixed four-parent plots after completed stricter-teacher optimization."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..io_utils import file_hash, new_dir, write_json


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def render(run):
    run=Path(run).resolve();p=read(run/'RUN_PROTOCOL.json');noise=read(run/'SENSITIVITY.json');results=read(run/'RESULTS.json')
    review=read(run/'REVIEW.json');assert review['status']=='PASS'
    assert review['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json')
    assert review['results_sha256']==file_hash(run/'RESULTS.json') and review['sensitivity_sha256']==file_hash(run/'SENSITIVITY.json')
    out=new_dir(run/'figures');uids=p['uids'];fractions=p['sensitivity']['fractions'];x=np.arange(len(fractions))
    names=['teacher','opt_fixed','opt_teacher'];colors={'teacher':'#606b77','opt_fixed':'#007eaa','opt_teacher':'#d56512'}
    labels={'teacher':'Original teacher','opt_fixed':'Fixed start + strict floor','opt_teacher':'Teacher start + strict floor'}
    fig,axes=plt.subplots(2,2,figsize=(12,8.2),sharex=True,sharey=True)
    for ax,uid in zip(axes.flat,uids):
        energy=p['parent_evidence'][uid]['original_teacher_energy'];zero=energy==0
        for name in names:
            records=[next(r for r in noise['per_parent'] if r['uid']==uid and r['endpoint']==name and r['fraction']==f) for f in fractions]
            rates=np.array([r['angular_pass']/r['planned'] for r in records])
            ax.plot(x,rates,marker='o',markersize=4,color=colors[name],label=labels[name],
                alpha=.5 if zero else 1.,linestyle='--' if zero else '-')
            for xx,record in zip(x,records):
                if record['missing_or_errors']:
                    ax.annotate(f"{record['missing_or_errors']} unavailable",(xx,rates[xx]),fontsize=7)
        ax.set_title(uid+'\n'+f"Original teacher component RMS = {np.sqrt(energy):.6g}",fontsize=9,pad=10)
        ax.set_xticks(x,['0','0.0001','0.001','0.01','0.1','0.5','1']);ax.set_ylim(-.04,1.08)
        ax.set_yticks([0,.25,.5,.75,1]);ax.grid(alpha=.25)
        if zero:
            ax.text(.5,.42,'ZERO-ENERGY CONTROL\nAll fractions mean zero perturbation.\nNo noise-robustness inference.',
                transform=ax.transAxes,ha='center',va='center',fontsize=10,
                bbox={'boxstyle':'round,pad=.5','facecolor':'#fff3cf','edgecolor':'#b18e44'})
        ax.set_xlabel('Fraction of original teacher displacement RMS')
        ax.set_ylabel('Both original Fixed angle gates passed / 16 trials')
    handles,legendlabels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legendlabels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.925),frameon=False)
    fig.suptitle('Stricter-teacher feasibility: angular neighborhood sensitivity\n4 predeclared training parents; same ambient noise and absolute scale per parent',fontsize=14,y=.985)
    fig.text(.5,.02,'Fixed fraction grid, including zero. Independent tangent noise is not learned error; no exact intersection audit of perturbations.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.835,bottom=.10,hspace=.48,wspace=.20)
    for ext in ('png','pdf'):fig.savefig(out/f'sensitivity.{ext}',dpi=180)
    plt.close(fig)
    lookup={(r['uid'],r['endpoint']):r for r in results['rows']}
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.2));markers={'fixed':'s','teacher':'o','opt_fixed':'^','opt_teacher':'D'}
    all_names=['fixed']+names;all_colors={'fixed':'#222222',**colors};all_labels={'fixed':'Fixed',**labels}
    for ax,uid in zip(axes.flat,uids):
        fixed=lookup[(uid,'fixed')];base=np.array([r['rms_bbox_percent'] for r in fixed['metrics']])
        for name in all_names:
            row=lookup[(uid,name)]
            if row['status']!='success':
                ax.text(.05,.05+.08*all_names.index(name),name+': unavailable',transform=ax.transAxes);continue
            ratio=np.array([r['rms_bbox_percent'] for r in row['metrics']])/base
            margin=min(row['actual_margin_vs_fixed_deg'].values())
            ax.scatter([margin]*2,ratio,s=44,marker=markers[name],color=all_colors[name],
                alpha=.8,label=all_labels[name],facecolors=all_colors[name] if row['qualified_original']['qualified'] else 'none')
            ax.plot([margin,margin],[min(ratio),max(ratio)],color=all_colors[name],alpha=.7)
        ax.axvline(.1,color='#b32339',linestyle='--',lw=1,label='New +0.1 degree floor')
        ax.axhline(1,color='#555555',linestyle=':',lw=1)
        ax.set_title(uid,fontsize=9,pad=9);ax.grid(alpha=.2)
        ax.set_xlabel('Smaller of two angle margins vs Fixed (degrees)')
        ax.set_ylabel('GT RMS / Fixed RMS (two fixed MC seeds)')
    handles,legendlabels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legendlabels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.91),frameon=False,fontsize=9)
    fig.suptitle('Endpoint quality and approximation error\nWarm start + 64 further steps + stricter floor vary jointly; not a causal margin effect',fontsize=14,y=.985)
    fig.text(.5,.02,'Both starts and all 4 parents retained. Original teacher is offline; no GT-based endpoint selection or label replacement.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.81,bottom=.10,hspace=.42,wspace=.30)
    for ext in ('png','pdf'):fig.savefig(out/f'endpoint_tradeoff.{ext}',dpi=180)
    plt.close(fig)
    manifest={'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'results_sha256':file_hash(run/'RESULTS.json'),
        'sensitivity_sha256':file_hash(run/'SENSITIVITY.json'),'review_sha256':file_hash(run/'REVIEW.json'),
        'source_sha256':file_hash(__file__),'files':{f.name:file_hash(f) for f in out.iterdir() if f.is_file()},
        'all_four_registered_UIDs':uids,'all_fractions':fractions,'zero_energy_not_robustness':True,
        'noise_does_not_represent_learned_error':True,'MC_seeds_not_independent_object_replicates':True}
    write_json(out/'FIGURE_MANIFEST.json',manifest)
    print(json.dumps({'figures':str(out),'files':list(manifest['files'])}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);args=parser.parse_args();render(args.run)
