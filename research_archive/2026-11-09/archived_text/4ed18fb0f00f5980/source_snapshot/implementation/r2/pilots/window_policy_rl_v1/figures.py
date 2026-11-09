"""Static scientific summaries from completed, unchanged score tables."""
from .bootstrap import bind,OUT
D,E=bind()
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def calibration():
    source=OUT/'analysis/calibration_per_condition_stage.csv'
    rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
    quant=D.read(OUT/'analysis/calibration_quantitative.json')
    labels=['FULL_DET','S005','S010','S020']
    cids=sorted({r['condition_id'] for r in rows})
    lookup={(r['label'],r['condition_id']):int(r['legal_CF_outputs']) for r in rows if r['stage']=='POST'}
    values=np.asarray([[lookup[(label,cid)] for label in labels] for cid in cids])
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),gridspec_kw={'width_ratios':[1.15,1,1.1]},layout='constrained')
    ax=axes[0];ax.imshow(values,cmap='Blues',vmin=0,vmax=8,aspect='auto')
    ax.set_xticks(range(4),['DET','.005','.010','.020']);ax.set_yticks(range(8),[c.split('__')[0][:8] for c in cids])
    ax.set_title('Legal C/F interface outputs / 8')
    for i in range(8):
        for j in range(4):ax.text(j,i,str(values[i,j]),ha='center',va='center',color='white' if values[i,j]>=5 else '#13243a')
    ax=axes[1];cs=quant['candidates'];x=np.arange(3)
    kept=[c['DET_positive_retained_count'] for c in cs]
    gained=[c['candidate_legal_CF_outputs']-c['DET_positive_retained_count'] for c in cs]
    lost=[c['DET_legal_CF_outputs']-c['DET_positive_retained_count'] for c in cs]
    ax.bar(x,kept,label='Retained DET positive',color='#346a9a');ax.bar(x,gained,bottom=kept,label='New positive',color='#479968')
    ax.bar(x,-np.asarray(lost),label='Lost DET positive',color='#cb5b51');ax.axhline(0,color='#333333',lw=.7)
    ax.set_ylim(-4,34);ax.set_xticks(x,['.005','.010','.020']);ax.set_ylabel('Paired requests');ax.set_title('Paired connection changes (64 requests)');ax.legend(fontsize=8,loc='upper right')
    ax=axes[2]
    for j,(field,name,color) in enumerate([('CF_illegal_fraction','CF','#557fa5'),('FF_illegal_fraction','FF','#c47d3b'),('I_lower','I','#8771a4')]):
        y=[c['deltas'][field+'_mean_delta_pp'] for c in cs]
        ax.bar(x+(j-1)*.24,y,width=.23,label=name,color=color)
    ax.axhline(0,color='#333333',lw=.7);ax.set_xticks(x,['.005','.010','.020']);ax.set_ylabel('Percentage points vs DET; positive = worse')
    ax.set_title('POST geometry mean changes');ax.legend(fontsize=8)
    fig.suptitle('Frozen FULL: window-noise calibration (not an RL training result)',fontsize=13)
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    files=[]
    for suffix in ('png','svg'):
        p=dest/('calibration_quantitative_v2.'+suffix)
        if p.exists():raise FileExistsError(p)
        fig.savefig(p,dpi=170,bbox_inches='tight');files.append(p)
    plt.close(fig)
    D.write(dest/'calibration_quantitative_v2.json',dict(status='COMPLETE',source_tables={str(source):D.sha(source),str(OUT/'analysis/calibration_quantitative.json'):D.sha(OUT/'analysis/calibration_quantitative.json')},
        files={str(p):D.sha(p) for p in files},model_calls=0,scope='All eight conditions and all three registered stochastic settings; actual body review reported separately'))

if __name__=='__main__':calibration()
