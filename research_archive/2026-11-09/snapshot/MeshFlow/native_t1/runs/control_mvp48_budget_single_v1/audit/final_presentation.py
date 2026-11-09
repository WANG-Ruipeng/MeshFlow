"""CPU-only presentation of existing final CSVs; no model, OT or scoring."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
def read(name):
    with (ROOT/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

colors = {'T1':'#2869ad', 'Geo':'#d18324', 'Full':'#258453'}
styles = {'train':'-', 'validation':'--'}
generation = read('train_validation_learning_curves.csv')
metrics = [
    ('remainder_symmetric_RMS_bbox_pct','Free-surface symmetric RMS','% L',1),
    ('fixed_R12_coverage_within_1pct_fraction','GT coverage at 1% L','%',100),
    ('Gamma_to_free_edges_RMS_bbox_pct','Interface to free finite edges RMS','% L',1),
    ('CF_proper_affected_free_fraction','CF proper: affected free faces','%',100),
    ('FF_illegal_affected_free_fraction','FF illegal: affected free faces','%',100),
]
fig, axes = plt.subplots(2,3,figsize=(15,8), constrained_layout=True)
for ax,(metric,title,unit,scale) in zip(axes.flat,metrics):
    for arm,color in colors.items():
        for split,style in styles.items():
            selected=[r for r in generation if r['split']==split and (r['arm']==arm or (arm in ('Geo','Full') and r['model_id']=='Geo_shared_500'))]
            selected=sorted(selected,key=lambda r:int(r['step']))
            ax.plot([int(r['step']) for r in selected],[float(r[metric])*scale for r in selected],color=color,linestyle=style,marker='o',markersize=4,label=arm+' '+split)
    for split,style in [('train',':'),('validation','-.')]:
        native=next(r for r in generation if r['model_id']=='Native0' and r['split']==split)
        ax.axhline(float(native[metric])*scale,color='#6c6c6c',linestyle=style,linewidth=1,label='Native0 '+split)
    ax.set(title=title,xlabel='MVP48 conditional fine-tuning step',ylabel=unit)
    ax.set_xticks([500,1000,2000,3500,5000]);ax.grid(alpha=.18)
axes.flat[-1].axis('off')
handles,labels=axes.flat[0].get_legend_handles_labels()
axes.flat[-1].legend(handles,labels,loc='center',frameon=False)
axes.flat[-1].text(.05,.05,'8 parents per split; equal parent weights\n2 patches x 2 fixed noises per parent\nFinal comparison fixed at step 5000\nCF/FF definitions follow the original scorer',transform=axes.flat[-1].transAxes,fontsize=10)
fig.suptitle('Fixed generation panel: training gains and validation transfer',fontsize=15)
fig.savefig(FIG/'summary_curves.png',dpi=160);plt.close(fig)

teacher=read('teacher_curve_summary.csv')
fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
for i,t in enumerate([.2,.5,.8]):
    for j,metric in enumerate(['FM_error','D_edge','L5']):
        ax=axes[i,j]
        for arm,color in colors.items():
            for split,style in styles.items():
                selected=[]
                for r in teacher:
                    if r['phase']!='curve' or r['split']!=split or float(r['t'])!=t:continue
                    model=r['model_id']
                    if model.startswith(arm+'_') or (arm in ('Geo','Full') and model=='Geo_shared_500'):
                        selected.append((int(model.rsplit('_',1)[1]),float(r[metric])))
                selected.sort()
                ax.plot([v[0] for v in selected],[v[1] for v in selected],color=color,linestyle=style,marker='o',markersize=3,label=arm+' '+split)
        ax.set(title=metric+' | t='+str(t),xlabel='MVP48 conditional fine-tuning step',ylabel='Original scalar (log scale)',yscale='log')
        ax.set_xticks([500,1000,2000,3500,5000]);ax.grid(alpha=.18)
axes[0,0].legend(fontsize=8)
fig.suptitle('Fixed teacher diagnostics: compare train and validation at the same t\n8 parents per split; shared OT/noise/permutations; generation remains the quality evidence',fontsize=14)
fig.savefig(FIG/'teacher_curves.png',dpi=160);plt.close(fig)

fit=read('fit2_probe.csv')
for row in fit:
    row['source_main_step']='1000'
    row['probe_new_updates']='0' if row['model_id']=='T1_1000' else row['model_id'].rsplit('_',1)[1]
    row['conditional_cumulative_step']=str(1000+int(row['probe_new_updates']))
with (ROOT/'fit2_probe.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(fit[0]));w.writeheader();w.writerows(fit)

manifest=json.loads((FIG/'manifest.json').read_text())
manifest['readable_generation_summary']='summary_curves.png'
manifest['fixed_teacher_curves']='teacher_curves.png'
manifest['presentation_note']='CPU-only plots of existing final CSVs. Native split references labelled separately. No model, OT, scoring, sampling or new updates.'
(FIG/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','new_model_calls':0,'new_OT_calls':0,'new_score_calls':0,'figures':['summary_curves.png','teacher_curves.png'],'fit_step_mapping_added':True}))
