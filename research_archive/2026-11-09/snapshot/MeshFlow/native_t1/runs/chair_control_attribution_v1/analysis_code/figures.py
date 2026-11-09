from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def make_figures():
 D=OUT/'figures';D.mkdir(exist_ok=True);manifest=[]
 plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.dpi':120,'savefig.dpi':150})
 def save(fig,name,definition,sources):
  fig.savefig(D/name,bbox_inches='tight');plt.close(fig)
  manifest.append(dict(file='figures/'+name,definition=definition,source_tables=sources,sha256=digest(D/name)))
 w=readcsv(OUT/'width_per_output.csv');c=readcsv(OUT/'width_per_curve.csv');ana=readcsv(OUT/'analytic_source_audit.csv')
 colors=['#1679b7','#d95821']
 # Six figures, every registered endpoint curve shown, never select by observed score.
 wr=[r for r in w if r['phase_step']==3000]
 ymax=max(max(r['p_alpha'],r['g_alpha'])/r['L0'] for r in wr)*1.05
 for arm in ARMS:
  for split in ('train','validation'):
   rs=[r for r in wr if r['arm']==arm and r['split']==split];uids=sorted({r['uid'] for r in rs})
   fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
   for ax,u in zip(axes.flat,uids):
    gt=sorted([r for r in rs if r['uid']==u and r['noise_index']==0],key=lambda r:r['alpha'])
    ax.plot(ALPHAS,[r['g_alpha']/r['L0'] for r in gt],color='black',marker='s',label='GT')
    for noise in (0,1):
     sub=sorted([r for r in rs if r['uid']==u and r['noise_index']==noise],key=lambda r:r['alpha'])
     ax.plot(ALPHAS,[r['p_alpha']/r['L0'] for r in sub],color=colors[noise],marker='o',label=f'model noise{noise}')
     for candidate,style in [('No-response',':'),('Post-scale','--')]:
      aa=sorted([r for r in ana if r['phase_step']==3000 and r['arm']==arm and r['uid']==u and r['noise_index']==noise and r['candidate']==candidate],key=lambda r:r['alpha'])
      ax.plot(ALPHAS,[r['width']/r['L0'] for r in aa],color=colors[noise],linestyle=style,alpha=.7,label=f'{candidate} n{noise}')
    ax.set(title=u[:10],ylim=(0,ymax),xticks=ALPHAS,xlabel='input alpha',ylabel='free q95-q05 width / L0');ax.grid(alpha=.2)
   axes[0,0].legend(fontsize=7,ncol=2);fig.suptitle(f'{NAMES[arm]} | {split} | phase3000 | four parents, two saved noises',y=1.01);fig.tight_layout()
   save(fig,f'width_{arm}_{split}.png','All actual raw widths; fixed y axis across six figures; analytic lines use saved candidates',['width_per_output.csv','analytic_source_audit.csv'])
 fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True)
 for ax,(split,arm) in zip(axes.flat,[(s,a) for s in ('train','validation') for a in ARMS]):
  for r in wr:
   if r['arm']!=arm or r['split']!=split or r['alpha']==1:continue
   ax.scatter(r['base'],r['response'],color=colors[r['noise_index']],marker='<' if r['alpha']==.9 else '>',s=32)
   ax.annotate(r['uid'][:4]+'n'+str(r['noise_index']), (r['base'],r['response']),fontsize=5,xytext=(2,2),textcoords='offset points')
  ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set(title=f'{NAMES[arm]} / {split}',xlabel='alpha*(p-g)/L0',ylabel='(p_alpha-alpha*p)/L0');ax.grid(alpha=.15)
 fig.suptitle('Signed base and response terms | < alpha=.9, > alpha=1.1 | color = saved noise');fig.tight_layout()
 save(fig,'base_response_scatter.png','Every endpoint off-center request, labeled parent/noise; no absolute-value additivity',['width_per_output.csv'])
 cross=readcsv(OUT/'cross_matrix.csv');cg=groups(cross,['curve_id']);vmax=max(r['E_RMS_L0']*100 for r in cross)
 for page,start in enumerate(range(0,len(cg),12),1):
  fig,axes=plt.subplots(3,4,figsize=(13,10));items=list(cg.items())[start:start+12]
  for ax,((cid,),rs) in zip(axes.flat,items):
   M=np.zeros((3,3))
   for r in rs:M[ALPHAS.index(r['alpha_in']),ALPHAS.index(r['beta_target'])]=r['E_RMS_L0']*100
   im=ax.imshow(M,vmin=0,vmax=vmax,cmap='viridis',aspect='equal')
   for i in range(3):
    for j in range(3):ax.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if M[i,j]<vmax*.6 else 'black')
   r=rs[0];ax.set(title=f"{r['arm']} p{r['phase_step']} {r['uid'][:7]} n{r['noise_index']}",xticks=range(3),xticklabels=ALPHAS,yticks=range(3),yticklabels=ALPHAS,xlabel='target beta (column)',ylabel='input alpha (row)')
  fig.subplots_adjust(hspace=.65,wspace=.5,right=.9,top=.92);cb=fig.add_axes([.93,.2,.015,.6]);fig.colorbar(im,cax=cb,label='one-way RMS (% L0)')
  fig.suptitle(f'All saved 3 x 3 matrices | page {page}/6 | common color scale')
  save(fig,f'cross_matrices_{page:02}.png','All 72 groups paginated, labels map explicit row/column; common color range',['cross_matrix.csv'])
 parts=readcsv(OUT/'coverage_partition_summary.csv');paired=readcsv(OUT/'coverage_paired.csv');parent=readcsv(OUT/'coverage_per_parent.csv')
 fig,axes=plt.subplots(2,4,figsize=(17,8),sharex=False,sharey=True)
 for i,comp in enumerate(('Geo-Native','Joint-Geo')):
  for j,(split,ratio) in enumerate((('train',.2),('train',.4),('validation',.2),('validation',.4))):
   ax=axes[i,j];rs=[r for r in parts if r['comparison']==comp and r['split']==split and r['ratio']==ratio]
   order=['both_in','enter','leave','both_out'];idx={r['group']:r for r in rs}
   ax.bar(range(4),[idx[k]['fixed_query_MSE_contribution']*1e4 for k in order],color=['#6699aa','#3ca868','#dd874a','#a56699'])
   ax.axhline(0,color='black',lw=.7);ax.set(xticks=range(4),xticklabels=['both in','enter','leave','both out'],title=f'{comp} {split} {ratio:.0%}',ylabel='MSE contribution x 10,000');ax.tick_params(axis='x',rotation=25)
 fig.suptitle('Coverage partitions: signed contribution to fixed-query MSE | parent equal weight');fig.tight_layout()
 save(fig,'coverage_partition_cost.png','Distance vector partitions at unchanged 1%L threshold, positive means higher squared distance',['coverage_partition_summary.csv'])
 fig,axes=plt.subplots(2,4,figsize=(17,8))
 for i,comp in enumerate(('Geo-Native','Joint-Geo')):
  for j,(split,ratio) in enumerate((('train',.2),('train',.4),('validation',.2),('validation',.4))):
   ax=axes[i,j];rs=[r for r in paired if r['comparison']==comp and r['split']==split and r['ratio']==ratio]
   for r in rs:ax.scatter(100*r['coverage_fraction_delta'],100*r['d_p95_L_delta'],color=colors[r['noise_index']],s=15,alpha=.5)
   ps=[r for r in parent if r['comparison']==comp and r['split']==split and r['ratio']==ratio]
   for r in ps:
    ax.scatter(100*r['coverage_fraction_delta'],100*r['d_p95_L_delta'],color='black',marker='x',s=32);ax.annotate(r['uid'][:4],(100*r['coverage_fraction_delta'],100*r['d_p95_L_delta']),fontsize=6)
   ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6);ax.set(title=f'{comp} {split} {ratio:.0%}',xlabel='coverage change (pp)',ylabel='p95 distance change (%L)')
 endpoint_pairs=[r for r in paired if r['comparison'] in ('Geo-Native','Joint-Geo')]
 for ax in axes.flat:
  ax.set_xlim(min(100*r['coverage_fraction_delta'] for r in endpoint_pairs)-2,max(100*r['coverage_fraction_delta'] for r in endpoint_pairs)+2)
  ax.set_ylim(min(100*r['d_p95_L_delta'] for r in endpoint_pairs)-.2,max(100*r['d_p95_L_delta'] for r in endpoint_pairs)+.2)
 fig.suptitle('Coverage gains and distance tails | colored dots: saved noise; black x: parent mean');fig.tight_layout()
 save(fig,'coverage_tail_cost.png','Per-case and per-parent actual fixed-query p95, not original R12 quantile',['coverage_paired.csv','coverage_per_parent.csv'])
 # Fixed examples registered by lexicographic UID before scores: first train and first validation, n0.
 gen=[r for r in generated() if r['table']=='B' and r['phase_step']==3000 and r['noise_index']==0]
 for split in ('train','validation'):
  uid=min(r['uid'] for r in gen if r['split']==split)
  plate_points=[]
  for rr in [r for r in gen if r['uid']==uid]:
   ff,bb,_=raw_for(rr);LL=float(bb['L0']);pp=ff.astype(float).copy();pp[...,0]/=float(np.float32(rr['alpha']))
   plate_points.extend([ff.reshape(-1,3)/LL,pp.reshape(-1,3)/LL,bb['C'].reshape(-1,3)/LL,bb['gt_free'].reshape(-1,3)/LL])
  fixed_points=np.concatenate(plate_points);fixed_mid=(fixed_points.min(0)+fixed_points.max(0))/2;fixed_half=max(fixed_points.max(0)-fixed_points.min(0))*.55
  for arm in ARMS:
   ix={r['alpha']:r for r in gen if r['uid']==uid and r['arm']==arm};f1,b1,_=raw_for(ix[1.]);L=float(b1['L0']);C1=b1['C'];gt=b1['gt_free'].reshape(-1,3,3)
   geoms=[]
   for a in (.9,1.1):
    f,b,_=raw_for(ix[a]);pull=f.astype(float).copy();pull[...,0]/=float(np.float32(a));geoms.append([(f,b['C']), (pull,C1),(f1,C1),(gt,C1)])
   pointsall=np.concatenate([x.reshape(-1,3)/L for row in geoms for f,cc in row for x in (f,cc)])
   mid=fixed_mid;half=fixed_half
   fig=plt.figure(figsize=(16,11))
   for i,a in enumerate((.9,1.1)):
    for j,title in enumerate(('raw F_alpha','offline pullback','saved F_1','GT Y_1')):
     ax=fig.add_subplot(2,4,4*i+j+1,projection='3d');f,cc=geoms[i][j]
     ax.add_collection3d(Poly3DCollection(f/L,facecolors='#7ba3be',edgecolors='#355970',linewidths=.18,alpha=.55))
     ax.add_collection3d(Poly3DCollection(cc/L,facecolors='#e8a147',edgecolors='#765925',linewidths=.2,alpha=.8))
     ax.set(xlim=(mid[0]-half,mid[0]+half),ylim=(mid[1]-half,mid[1]+half),zlim=(mid[2]-half,mid[2]+half),title=f'a={a}: {title}',xlabel='X/L0',ylabel='Y/L0',zlabel='Z/L0');ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-60);ax.tick_params(labelsize=6)
   fig.suptitle(f'{NAMES[arm]} | {split} {uid[:10]} noise0 | offline geometry diagnostic',y=.97)
   fig.subplots_adjust(left=.02,right=.98,bottom=.06,top=.89,wspace=.12,hspace=.32)
   save(fig,f'geometry_{arm}_{split}.png','Preregistered UID/noise; fixed camera and shared limits across all three arms for same parent; no ICP/slot match; actual arrays only',['pullback_per_case.csv','source_manifest.json'])
 savej(OUT/'figures/manifest.json',manifest)
 return dict(figures=len(manifest)),['figures/manifest.json']+[r['file'] for r in manifest]
