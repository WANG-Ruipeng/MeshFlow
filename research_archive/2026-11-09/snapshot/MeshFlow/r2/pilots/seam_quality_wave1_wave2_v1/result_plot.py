"""Plot saved metrics only; no generated or selected extra examples."""
import argparse,csv
from .protocol import OUT,WAVE2

def run(scope):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (OUT/'results'/scope/'key_results.csv').open(encoding='utf-8-sig') as f:
        data=list(csv.DictReader(f))
    lookup={(r['label'],r['stage'],r['split']):r for r in data}
    order=['B_LONG_LAYOUT','PLAIN_S1000','PLAIN_S2000','PLAIN3000']
    candidates=['WISE_B_PLAIN_050']+list(WAVE2 if scope=='final' else [])
    short={'WISE_B_PLAIN_050':'WISE 0.5','CLEAN_BULK_FM_S1000':'CLEAN +1000','SURGICAL_LATE4_S1000':'SURGICAL +1000'}
    fields=[('FF_illegal_affected_free_fraction','Illegal FF affected faces (%)',100),
            ('ConflictCov_C','Conflict coverage on C (%)',100),
            ('EXACT_COORD_face_components','Exact-coordinate face components',1),
            ('free_only_reached_bulk_area_fraction','Legal graph: reachable bulk area (%)',100)]
    fig,axs=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    colors=['#0072B2','#D55E00','#009E73']
    for i,split in enumerate(('TRAIN','DEV')):
        for ax,(field,title,mult) in zip(axs[i],fields):
            ax.plot([0,1000,2000,3000],[mult*float(lookup[a,'POST',split][field]) for a in order],'-o',
                    color='#555555',label='Actual PLAIN trajectory')
            for a,color in zip(candidates,colors):
                # WISE is a weight-space chord midpoint, not an actual update1500 checkpoint.
                x=1500 if a=='WISE_B_PLAIN_050' else 1000
                ax.scatter([x],[mult*float(lookup[a,'POST',split][field])],marker='D' if a=='WISE_B_PLAIN_050' else 's',
                    color=color,s=42,zorder=4,label=short[a])
            if field=='FF_illegal_affected_free_fraction' and split=='DEV':ax.axhline(16,color='#aa3333',ls=':',label='DEV FF16% registered line')
            ax.set_title(split+' / POST: '+title,fontsize=10);ax.set_xlabel('Continuation updates (WISE at1500 only for display)',fontsize=8)
            ax.grid(alpha=.2);ax.set_xticks([0,1000,2000,3000])
    handles,labels=axs[1,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=9)
    dest=OUT/'results'/scope
    fig.savefig(dest/'quality_trajectory.png',dpi=170);fig.savefig(dest/'quality_trajectory.svg');plt.close(fig)
    print(dest/'quality_trajectory.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope',choices=['wave1','final']);run(p.parse_args().scope)
