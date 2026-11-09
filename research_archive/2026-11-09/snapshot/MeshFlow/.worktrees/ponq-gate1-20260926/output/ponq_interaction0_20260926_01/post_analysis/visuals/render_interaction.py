"""Static, complete-ledger-only visuals for SINGLE_STATE_INTERACTION0.

No model execution, sample selection, remeshing, result editing or geometry
reinterpretation. Writes only a new post_analysis/visuals supplementary bundle.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ARMS=['B0','B1','B2']
LABELS={'B0':'B0: original continuation','B1':'B1: pointwise residual','B2':'B2: self-attention'}
COLORS={'B0':'#526b83','B1':'#cd862b','B2':'#008b82'}
DISPLAY_IDS=[32,33,34,35]
QUALITY_THRESHOLD=.20
CAMERA=dict(elevation=58,azimuth=-90,projection='orthographic')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root.resolve()
    sources={}
    def read(path):
        path=Path(path);sources[str(path)]=sha(path);return json.loads(path.read_text())
    execution=read(root/'EXECUTION.json')
    if execution['status']!='COMPLETE':raise RuntimeError('Only a COMPLETE execution may be rendered')
    parent_manifest_hash=sha(root/'MANIFEST.json');sources[str(root/'MANIFEST.json')]=parent_manifest_hash
    protocol=read(root/'RUN_PROTOCOL.json');read(root/'RUN_PROTOCOL_SHA256.json')
    assert protocol['engineering_gate']['RMS_threshold_pct_bbox']==QUALITY_THRESHOLD
    records={};training={};summaries={}
    for arm in ARMS:
        ledger=read(root/'evaluation'/arm/'FREE_LEDGER.json');summaries[arm]=read(root/'evaluation'/arm/'SUMMARY.json')
        if len(ledger)!=64 or {r['sample_id'] for r in ledger}!=set(range(64)):raise AssertionError('Missing or duplicate free samples: '+arm)
        records[arm]={r['sample_id']:r for r in ledger}
        for i,r in records[arm].items():
            assert r['arm']==arm and r['split']==('old' if i<32 else 'new') and r['split_id']==i%32
            assert r['generated'] and not r['oracle_assisted'] and r['combination']=='PPP' and r['domain']=='fixed'
            threshold_pass=bool(r['valid'] and r.get('rms_pct_bboxdiag',float('inf'))<=QUALITY_THRESHOLD)
            assert r['quality_pass']==threshold_pass
        result=read(root/'training'/arm/'RESULT.json')
        assert result['status']=='COMPLETED' and result['completed_steps']==9000 and result['new_updates']==6000
        trace_path=root/'training'/arm/'TRAIN_TRACE.jsonl';sources[str(trace_path)]=sha(trace_path)
        training[arm]=[json.loads(line) for line in trace_path.read_text().splitlines() if line]
        assert len(training[arm])==6000 and [r['step'] for r in training[arm]]==list(range(3001,9001))
    out=root/'post_analysis/visuals';out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'render_interaction.py')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    plots=[]
    def save(fig,name):
        path=out/(name+'.png');fig.savefig(path,dpi=150,facecolor='white');plt.close(fig);plots.append(path.name)
    counts={}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True);fig.subplots_adjust(left=.07,right=.985,bottom=.17,top=.76,wspace=.10)
    for ax,split in zip(axes,['old','new']):
        validity=[];quality=[]
        for arm in ARMS:
            rr=[r for r in records[arm].values() if r['split']==split]
            assert len(rr)==32
            valid=sum(bool(r['valid']) for r in rr);good=sum(bool(r['quality_pass']) for r in rr)
            assert valid==summaries[arm]['splits'][split]['valid'] and good==summaries[arm]['splits'][split]['quality_pass']
            counts[arm+'_'+split]=dict(valid=valid,valid_and_quality=good,denominator=32)
            validity.append(valid);quality.append(good)
        x=np.arange(3);width=.33
        first=ax.bar(x-width/2,validity,width,color='#8ba7b8',edgecolor='#355669',label='Geometry valid')
        second=ax.bar(x+width/2,quality,width,color='#087d73',label='Valid + RMS <= 0.20%')
        for bars in [first,second]:
            for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.65,f'{int(bar.get_height())}/32',ha='center',fontsize=10)
        ax.set_xticks(x,['B0\nContinuation','B1\nPoint MLP','B2\nAttention']);ax.set_ylim(0,36);ax.set_yticks([0,8,16,24,32]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_title(('Old: paired development noise' if split=='old' else 'New: frozen confirmation noise'),pad=12)
        if split=='new':
            ax.axhline(24,color='#a4453b',linestyle='--',linewidth=1.4)
            ax.text(2.43,24.6,'24/32 advance line',ha='right',va='bottom',fontsize=9,color='#a4453b')
    axes[0].set_ylabel('Samples / 32 (all failures retained)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.89),ncol=2,frameon=False)
    fig.suptitle('One fixed target | common 3000-step checkpoint + 6000 matched updates per arm',fontsize=14,y=.98)
    fig.text(.5,.025,'Direct FM, 64 Euler steps, fixed official +/-1 corners. Old and new noise are not pooled.\nQuality line is a preregistered single-object engineering criterion, not a generalization claim.',ha='center',fontsize=9)
    save(fig,'validity_and_quality_old_vs_new')
    def mesh(path):
        path=Path(path)
        if not path.exists():return None
        sources[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as z:return z['vertices'].copy(),z['faces'].copy()
    target=mesh(root/'target/canonical_mesh.npz')
    if target is None:raise ValueError('Missing canonical target mesh')
    meshes={(a,i):mesh(root/'evaluation'/a/f'{i:03d}_PPP_fixed/mesh.npz') for a in ARMS for i in DISPLAY_IDS}
    vv=np.concatenate([target[0]]+[m[0] for m in meshes.values() if m is not None and len(m[0])]);lo=vv.min(0);hi=vv.max(0)
    center=(lo+hi)/2;span=float(np.max(hi-lo))*1.10;limits=np.c_[center-span/2,center+span/2]
    if not np.isfinite(limits).all() or span<=0:raise ValueError('Nonfinite plotting bounds')
    def panel(ax,m,title,row=None):
        ax.set_proj_type('ortho');ax.view_init(elev=CAMERA['elevation'],azim=CAMERA['azimuth']);ax.set_box_aspect((1,1,1))
        ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1]);ax.set_zlim(*limits[2]);ax.set_axis_off();ax.set_title(title,fontsize=10,pad=-2)
        if m is not None and len(m[1]):
            v,f=m;ax.add_collection3d(Poly3DCollection(v[f],facecolors='#b9c5cc' if row is None else '#70aabf',edgecolors='#244554',linewidths=.20,alpha=1.))
            if row is None:caption=f'{len(v)} vertices | {len(f)} faces';color='#49656a'
            else:
                state='QUALITY PASS' if row['quality_pass'] else 'GEOMETRY VALID' if row['valid'] else 'INVALID'
                rms=row.get('rms_pct_bboxdiag');rms_text='NA' if rms is None or not np.isfinite(rms) else f'{rms:.3f}%'
                caption=f'{state} | {len(f)} faces\nRMS {rms_text} bbox diagonal'
                color='#087d73' if row['quality_pass'] else '#9a680c' if row['valid'] else '#a32222'
        else:
            ax.text2D(.5,.5,'NO SAVED MESH\nFailure retained',transform=ax.transAxes,ha='center',fontsize=9,color='#a32222')
            caption=str(row.get('status','missing'));color='#a32222'
        ax.text2D(.5,.035,caption,transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color=color)
    fig=plt.figure(figsize=(16,10.3));fig.subplots_adjust(left=.01,right=.99,bottom=.08,top=.87,wspace=-.025,hspace=.13)
    for j,arm in enumerate(ARMS):
        panel(fig.add_subplot(3,5,j*5+1,projection='3d'),target,LABELS[arm]+'\nTarget reference')
        for k,i in enumerate(DISPLAY_IDS):panel(fig.add_subplot(3,5,j*5+k+2,projection='3d'),meshes[arm,i],f'New {i-32:02d} | global {i:02d}',records[arm][i])
    fig.suptitle('Preselected new sample IDs 0–3 | same noise across B0 / B1 / B2',fontsize=15,y=.975)
    fig.text(.5,.92,'All meshes are free PPP generations. Shared orthographic camera and world scale; no selected best examples.\nThe target is a visual reference only. No positioning, normalization, geometry repair or domain switch.',ha='center',fontsize=10)
    fig.text(.5,.027,'Global IDs 32–35 were selected before results. Validity labels come from full geometry audits; images alone do not certify validity.\nAll panels use the same physical limits, including the target; actual output face counts are shown.',ha='center',fontsize=9)
    save(fig,'new_samples_000_to_003_all_arms')
    fig,ax=plt.subplots(figsize=(13,5.5));fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.80)
    missing={};rms_values=[]
    for index,arm in enumerate(ARMS):
        rr=[records[arm][i] for i in range(32,64)];missing[arm]=[]
        for valid,marker in [(False,'x'),(True,'o')]:
            usable=[r for r in rr if bool(r['valid'])==valid and r.get('rms_pct_bboxdiag') is not None and np.isfinite(r['rms_pct_bboxdiag'])]
            x=np.array([r['split_id'] for r in usable])+(.15*(index-1));y=np.array([r['rms_pct_bboxdiag'] for r in usable]);rms_values.extend(y.tolist())
            ax.scatter(x,y,s=42 if valid else 48,c=COLORS[arm],marker=marker,linewidths=1.25,alpha=.9)
        absent=[r['split_id'] for r in rr if r.get('rms_pct_bboxdiag') is None or not np.isfinite(r.get('rms_pct_bboxdiag'))]
        missing[arm]=absent
        if absent:ax.scatter(np.array(absent)+(.15*(index-1)),np.full(len(absent),.025),transform=ax.get_xaxis_transform(),marker='v',facecolors='none',edgecolors=COLORS[arm],s=50,clip_on=False)
    ax.set_yscale('symlog',linthresh=.02,linscale=.7);ax.axhline(QUALITY_THRESHOLD,color='#a4453b',linestyle='--',linewidth=1.4)
    ymax=max([.5]+rms_values)*1.18;ax.set_ylim(0,ymax);ax.set_xlim(-.7,31.7);ax.set_xticks(range(0,32,2))
    yticks=[x for x in [0,.02,.05,.1,.2,.5,1,2,5,10,20,50] if x<=ymax]
    ax.set_yticks(yticks,[str(x) for x in yticks]);ax.grid(alpha=.20);ax.set_xlabel('Paired new noise ID (global ID = new ID + 32)')
    ax.set_ylabel('Surface RMS / target bbox diagonal (%)\nsymlog scale, linear below 0.02%')
    ax.text(31.4,QUALITY_THRESHOLD*1.09,'0.20% quality line',ha='right',va='bottom',fontsize=9,color='#a4453b')
    handles=[Line2D([],[],color=COLORS[a],marker='o',linestyle='',label=LABELS[a]) for a in ARMS]
    handles += [Line2D([],[],color='#444',marker='o',linestyle='',label='Geometry valid'),Line2D([],[],color='#444',marker='x',linestyle='',label='Invalid')]
    if any(missing.values()):handles.append(Line2D([],[],color='#444',marker='v',linestyle='',fillstyle='none',label='RMS unavailable'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.93),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('All 32 new noise samples | invalid meshes remain in the surface-error plot',fontsize=14,y=.995)
    fig.text(.5,.03,'Same target and fixed evaluation protocol for all arms. A low-RMS invalid mesh still fails the joint contract.\nMissing RMS (if any) is shown as an open triangle on a separate NA row; it is never replaced with zero.',ha='center',fontsize=9)
    save(fig,'paired_new_sample_surface_rms')
    fig,axes=plt.subplots(1,4,figsize=(16,4.3));fig.subplots_adjust(left=.055,right=.99,top=.80,bottom=.19,wspace=.32)
    for ax,key,title in zip(axes,['loss','x_mse','n_mse','F_mse'],['Total 22-channel MSE','Position velocity MSE','Raw normal velocity MSE','Factor velocity MSE']):
        for arm in ARMS:
            values=np.array([r[key] for r in training[arm]]);blocks=values.reshape(60,100).mean(axis=1);steps=np.arange(3100,9001,100)
            ax.plot(steps,blocks,color=COLORS[arm],label=arm,linewidth=1.5)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Cumulative optimizer updates');ax.set_xlim(3000,9000);ax.set_xticks([3000,6000,9000]);ax.grid(alpha=.2)
    axes[0].set_ylabel('100-update block mean')
    fig.suptitle('Matched continuation | all 6000 added updates, unchanged training objective',fontsize=14,y=.97)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.90),ncol=3,frameon=False)
    fig.text(.5,.035,'Non-overlapping 100-update block means, no endpoint selection. Total = (3 position + 3 normal + 16 factor) / 22.\nTraining loss is a diagnostic, not the geometry-validity or 0.20% delivery criterion.',ha='center',fontsize=9)
    save(fig,'training_loss_by_field')
    report=dict(status='COMPLETE',plots=plots,counts=counts,display_global_new_ids=DISPLAY_IDS,
        camera={**CAMERA,'axis_limits':limits.tolist(),'shared_across_mesh_panels':True},
        paired_scatter_requested_per_arm=32,paired_scatter_missing_RMS_ids=missing,loss_aggregation='60 consecutive blocks of100 added updates',
        source_files=[dict(path=p,sha256=h) for p,h in sorted(sources.items())],
        original_manifest_unmodified=sha(root/'MANIFEST.json')==parent_manifest_hash,
        note='Read-only static visualization of full free-generation results. No oracle-assisted field rows enter these charts.')
    assert report['original_manifest_unmodified']
    dump(out/'VISUALIZATION.json',report)
    dump(out/'SUPPLEMENTAL_MANIFEST.json',dict(parent_manifest_sha256=parent_manifest_hash,
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name!='SUPPLEMENTAL_MANIFEST.json']))
    print(json.dumps(dict(output=str(out),plots=plots,counts=counts,total_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())),indent=2))
if __name__=='__main__':main()