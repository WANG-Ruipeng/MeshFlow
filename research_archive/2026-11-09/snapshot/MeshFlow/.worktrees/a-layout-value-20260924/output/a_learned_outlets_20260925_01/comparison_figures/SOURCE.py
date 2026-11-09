"""Descriptive all-sample paired plot; never modifies models or scientific results."""
from pathlib import Path
import argparse,hashlib,json,shutil,traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ARMS=('original_a','longfit','balanced')
NAMES={'original_a':'Original A','longfit':'Longfit 240k','balanced':'Balanced 240k'}
COLORS={'original_a':'#b45f06','longfit':'#0072b2','balanced':'#8b3fa3'}
MARKERS={'original_a':'D','longfit':'o','balanced':'^'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(run,out):
 run=Path(run).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 shutil.copyfile(__file__,out/'SOURCE.py')
 try:
  p=read(run/'RUN_PROTOCOL.json');new=read(run/'RESULTS.json')['rows'];old=read(run/'REFERENCE_METRICS.json')['rows']
  assert len(new)==256 and len(old)==384
  maps={'original_a':{(r['seed'],r['sample_id']):r for r in old if r['arm']=='a'}}
  for arm in ['longfit','balanced']:maps[arm]={(r['seed'],r['sample_id']):r for r in new if r['arm']==arm}
  expected={(s,f'{i:05d}') for s in [10,11] for i in range(64)}
  assert all(set(m)==expected for m in maps.values())
  points=[]
  for seed in [10,11]:
   for arm in ARMS:
    for i in range(64):
     r=maps[arm][(seed,f'{i:05d}')];ok=r['metric_status']=='ok'
     gains=r['rms_gain_percent_two_measurements'] if ok else None
     points.append({'seed':seed,'arm':arm,'sample_id':f'{i:05d}','sample_index':i,'status':r['metric_status'],
      'gains_two_MC':gains,'display_mean_of_two_MC':float(np.mean(gains)) if ok else None,
      'qualified':r.get('qualified'),'numeric_stats_include':ok,'failure':r.get('metric_error')})
  finite=[x for r in points if r['gains_two_MC'] is not None for x in r['gains_two_MC']]
  lo,hi=min(finite),max(finite);span=hi-lo;ylim=(lo-.13*span,hi+.08*span)
  plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42})
  fig,axes=plt.subplots(2,2,figsize=(13.4,7.0),gridspec_kw={'height_ratios':[3.5,1.15]},sharex='col')
  for col,seed in enumerate([10,11]):
   ax=axes[0,col];qa=axes[1,col];quality=[]
   for ai,arm in enumerate(ARMS):
    pp=[r for r in points if r['seed']==seed and r['arm']==arm];ok=[r for r in pp if r['numeric_stats_include']];bad=[r for r in pp if not r['numeric_stats_include']]
    x=np.array([r['sample_index'] for r in ok],float)+(ai-1)*.18;y=np.array([r['display_mean_of_two_MC'] for r in ok]);g=np.array([r['gains_two_MC'] for r in ok])
    ax.vlines(x,g.min(1),g.max(1),color=COLORS[arm],alpha=.65,lw=.7)
    ax.scatter(x,y,s=15 if ai else 12,marker=MARKERS[arm],color=COLORS[arm],alpha=.8,linewidths=0,zorder=3)
    for r in bad:ax.scatter(r['sample_index']+(ai-1)*.18,.025,transform=ax.get_xaxis_transform(),marker='x',s=48,color=COLORS[arm],lw=1.8,zorder=4)
    quality.append([1 if r['qualified'] is True else 0 if r['numeric_stats_include'] else -1 for r in pp])
   ax.axhline(0,color='#333333',lw=.9);ax.set_ylim(*ylim);ax.set_xlim(-1,64);ax.grid(axis='y',alpha=.2);ax.set_title(f'Training seed {seed} | all 64 registered fields')
   ax.set_ylabel('RMS gain relative to Fixed (%)' if col==0 else '')
   qa.imshow(np.array(quality),aspect='auto',interpolation='none',cmap=ListedColormap(['#111111','#dedede','#288263']),norm=BoundaryNorm([-1.5,-.5,.5,1.5],3),extent=[-.5,63.5,2.5,-.5])
   qa.set_yticks(range(3),[NAMES[a] for a in ARMS]);qa.set_xticks([0,16,32,48,63],['00000','00016','00032','00048','00063']);qa.set_xlabel('Registered sample ID');qa.set_xlim(-1,64)
   counts=[sum(r['qualified'] is True for r in points if r['seed']==seed and r['arm']==a) for a in ARMS]
   qa.set_title('Quality pass / 64: '+', '.join(f'{NAMES[a]} {n}' for a,n in zip(ARMS,counts)),fontsize=10)
  handles=[Line2D([],[],color=COLORS[a],marker=MARKERS[a],lw=0,label=NAMES[a],markersize=5) for a in ARMS]
  handles += [Line2D([],[],color='#333333',marker='x',lw=0,label='Runtime failure (bottom marker; no value)',markersize=6)]
  fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.935),ncol=4,frameon=False,fontsize=10)
  fig.suptitle('Original AE3200 / FM8000 fields: learned outlets, paired against the same Fixed mesh',y=.984,fontsize=14)
  fig.legend(handles=[Patch(color='#288263',label='Quality qualified'),Patch(color='#dedede',label='Unqualified'),Patch(color='#111111',label='Runtime failure')],loc='lower center',bbox_to_anchor=(.5,.07),ncol=3,frameon=False)
  fig.text(.5,.027,'Each point is the mean of two predeclared MC gains; vertical segments show both values (not confidence intervals).\nFull denominators: 128 per arm. New-arm continuous statistics use 127 readable outputs each. No fallback or sample selection.',ha='center',va='center',fontsize=9)
  fig.subplots_adjust(top=.835,bottom=.20,left=.09,right=.985,hspace=.28,wspace=.18)
  fig.savefig(out/'all_samples_paired.png',dpi=170);fig.savefig(out/'all_samples_paired.pdf');plt.close(fig)
  (out/'POINTS.json').write_text(json.dumps(points,indent=2)+'\n')
  manifest={'status':'COMPLETE_DESCRIPTIVE_NOT_SELECTION','source_sha256':sha(__file__),'protocol_sha256':sha(run/'RUN_PROTOCOL.json'),
   'results_sha256':sha(run/'RESULTS.json'),'reference_metrics_sha256':sha(run/'REFERENCE_METRICS.json'),
   'samples_per_arm':128,'arms':list(ARMS),'points':384,'all_samples_preserved':True,
   'two_MC_values_not_confidence_intervals':True,'display_statistic':'mean of the two per-sample percentage gains; both observations retained as vertical segments',
   'runtime_failures_not_plotted_as_zero':True,'reference':'same saved generated mainfield support mesh; not raw asset',
   'files':{n:sha(out/n) for n in ['SOURCE.py','POINTS.json','all_samples_paired.png','all_samples_paired.pdf']}}
  (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
 except Exception as exc:
  (out/'FAILED.json').write_text(json.dumps({'error':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--out',required=True);v=a.parse_args();build(v.run,v.out)
