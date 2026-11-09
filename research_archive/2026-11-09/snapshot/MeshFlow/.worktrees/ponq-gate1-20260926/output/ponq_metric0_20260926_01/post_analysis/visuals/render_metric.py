"""Complete-ledger-only scientific figures for SINGLE_STATE_METRIC0.

Read-only analysis; output exclusively a new post_analysis/visuals bundle.
No gradients, model calls, mesh extraction, training or sample selection occur.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ARMS=['B0','M0']
FIELDS=['x','n','F']
COLORS={'B0':'#526b83','M0':'#00877f'}
FIELD_COLORS={'x':'#168abd','n':'#c8832f','F':'#76599b'}
FIELD_LABELS={'x':'Position','n':'Raw normal','F':'Centered factor'}
DISPLAY_IDS=[0,1,2,3]
QUALITY_THRESHOLD=.20
ELEVATION=58
AZIMUTH=-90


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();root=a.root.resolve();sources={}
    def read(path):
        path=Path(path);sources[str(path)]=sha(path);return json.loads(path.read_text())
    execution=read(root/'EXECUTION.json')
    if execution['status']!='COMPLETE':raise RuntimeError('Wait for COMPLETE; never render partial result selection')
    manifest_hash=sha(root/'MANIFEST.json');sources[str(root/'MANIFEST.json')]=manifest_hash
    protocol=read(root/'RUN_PROTOCOL.json');read(root/'RUN_PROTOCOL_SHA256.json')
    assert protocol['metrics']['RMS_threshold_pct_bbox']==QUALITY_THRESHOLD
    records={};counts={}
    for arm in ARMS:
        rows=read(root/'evaluation'/arm/'FREE_LEDGER.json');summary=read(root/'evaluation'/arm/'SUMMARY.json')
        if len(rows)!=32 or {r['sample_id'] for r in rows}!=set(range(32)):raise AssertionError('Need complete32 unique samples for '+arm)
        records[arm]={r['sample_id']:r for r in rows}
        for r in rows:
            assert r['arm']==arm and r['split']=='fresh' and r['generated'] and not r['oracle_assisted'] and r['combination']=='PPP' and r['domain']=='fixed'
            assert r['quality_pass']==bool(r['valid'] and r.get('rms_pct_bboxdiag',float('inf'))<=QUALITY_THRESHOLD)
        valid=sum(r['valid'] for r in rows);quality=sum(r['quality_pass'] for r in rows)
        assert valid==summary['valid'] and quality==summary['quality_pass']
        counts[arm]=dict(valid=valid,quality_pass=quality,denominator=32)
    calibration=read(root/'calibration/WEIGHTS.json');assert calibration['passed'] and calibration['status']=='PASS'
    parent=read(root/'calibration/weighted_verification/RESULT.json')
    probes={'parent3000':parent}
    for label in ['M0_5000','M0_7000','M0_9000','B0_9000']:probes[label]=read(root/'probes'/label/'RESULT.json')
    for label,record in probes.items():
        assert record['passed'] and record['optimizer_updates']==0 and record['mesh_extractions']==0
        assert record['shared_parameter_count']==85376 and record['parameter_grad_buffers_remain_none'] and record['model_unchanged']
        if label!='parent3000':assert record['weights_sha256']==sha(root/'calibration/WEIGHTS.json')
    out=root/'post_analysis/visuals';out.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),out/'render_metric.py')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    plots=[]
    def save(fig,name):
        path=out/(name+'.png');fig.savefig(path,dpi=150,facecolor='white');plt.close(fig);plots.append(path.name)
    fig,ax=plt.subplots(figsize=(13,5.5));fig.subplots_adjust(left=.08,right=.985,bottom=.19,top=.81)
    missing={};values=[]
    for j,arm in enumerate(ARMS):
        rr=list(records[arm].values());missing[arm]=[]
        for valid,marker in [(False,'x'),(True,'o')]:
            rows=[r for r in rr if bool(r['valid'])==valid and r.get('rms_pct_bboxdiag') is not None and np.isfinite(r['rms_pct_bboxdiag'])]
            xs=np.array([r['sample_id'] for r in rows])+(j-.5)*.15;ys=np.array([r['rms_pct_bboxdiag'] for r in rows]);values.extend(ys.tolist())
            ax.scatter(xs,ys,c=COLORS[arm],marker=marker,s=48,linewidths=1.3,alpha=.9)
        absent=[r['sample_id'] for r in rr if r.get('rms_pct_bboxdiag') is None or not np.isfinite(r['rms_pct_bboxdiag'])]
        missing[arm]=absent
        if absent:ax.scatter(np.array(absent)+(j-.5)*.15,np.full(len(absent),.025),transform=ax.get_xaxis_transform(),marker='v',facecolors='none',edgecolors=COLORS[arm],s=50,clip_on=False)
    ax.axhline(QUALITY_THRESHOLD,color='#a4453b',linestyle='--',linewidth=1.4);ax.set_yscale('symlog',linthresh=.02,linscale=.7)
    ymax=max([.5]+values)*1.18;ax.set_ylim(0,ymax);ax.set_xlim(-.7,31.7);ax.set_xticks(range(0,32,2))
    ticks=[x for x in [0,.02,.05,.1,.2,.5,1,2,5,10,20,50] if x<=ymax];ax.set_yticks(ticks,[str(x) for x in ticks])
    ax.grid(alpha=.20);ax.set_xlabel('Paired fresh noise ID');ax.set_ylabel('Surface RMS / target bbox diagonal (%)\nsymlog scale, linear below 0.02%')
    ax.text(31.4,QUALITY_THRESHOLD*1.1,'0.20% quality line',ha='right',fontsize=9,color='#a4453b')
    handles=[Line2D([],[],color=COLORS[a],marker='o',linestyle='',label=f'{a}: '+('old metric' if a=='B0' else 'frozen calibrated metric')+f' | valid {counts[a]["valid"]}/32; quality {counts[a]["quality_pass"]}/32') for a in ARMS]
    handles += [Line2D([],[],color='#444',marker='o',linestyle='',label='Geometry valid'),Line2D([],[],color='#444',marker='x',linestyle='',label='Invalid')]
    if any(missing.values()):handles.append(Line2D([],[],color='#444',marker='v',linestyle='',fillstyle='none',label='RMS unavailable'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Same frozen model architecture and paired fresh noise | all 32 outputs per arm',fontsize=14,y=.995)
    fig.text(.5,.03,'Both models: 9000 cumulative updates, 64 Euler steps, official fixed +/-1 corner domain. Loss weights are not applied during sampling.\nInvalid outputs remain in the plot and denominator; a missing RMS is marked separately, never replaced with zero.',ha='center',fontsize=9)
    save(fig,'fresh_paired_surface_rms')
    def mesh(path):
        path=Path(path)
        if not path.exists():return None
        sources[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as z:return z['vertices'].copy(),z['faces'].copy()
    target=mesh(root/'target/canonical_mesh.npz')
    if target is None:raise ValueError('Missing target mesh')
    meshes={(arm,i):mesh(root/'evaluation'/arm/f'{i:03d}_PPP_fixed/mesh.npz') for arm in ARMS for i in DISPLAY_IDS}
    vv=np.concatenate([target[0]]+[m[0] for m in meshes.values() if m is not None and len(m[0])]);lo=vv.min(0);hi=vv.max(0)
    center=(lo+hi)/2;span=float(np.max(hi-lo))*1.10;limits=np.c_[center-span/2,center+span/2]
    if not np.isfinite(limits).all() or span<=0:raise ValueError('Invalid plotting bounds')
    def panel(ax,data,title,row=None):
        ax.set_proj_type('ortho');ax.view_init(elev=ELEVATION,azim=AZIMUTH);ax.set_box_aspect((1,1,1))
        ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1]);ax.set_zlim(*limits[2]);ax.set_axis_off();ax.set_title(title,fontsize=10,pad=-2)
        if data is not None and len(data[1]):
            v,f=data;ax.add_collection3d(Poly3DCollection(v[f],facecolors='#b9c5cc' if row is None else '#70aabf',edgecolors='#244554',linewidths=.20))
            if row is None:caption=f'{len(v)} vertices | {len(f)} faces';color='#49656a'
            else:
                state='QUALITY PASS' if row['quality_pass'] else 'GEOMETRY VALID' if row['valid'] else 'INVALID'
                rms=row.get('rms_pct_bboxdiag');rms_text='NA' if rms is None or not np.isfinite(rms) else f'{rms:.3f}%'
                caption=f'{state} | {len(f)} faces\nRMS {rms_text} bbox diagonal';color='#00877f' if row['quality_pass'] else '#9a680c' if row['valid'] else '#a32222'
        else:
            ax.text2D(.5,.5,'NO SAVED MESH\nFailure retained',transform=ax.transAxes,ha='center',fontsize=9,color='#a32222');caption=str(row.get('status','missing'));color='#a32222'
        ax.text2D(.5,.035,caption,transform=ax.transAxes,ha='center',fontsize=8,color=color)
    fig=plt.figure(figsize=(16,7.8));fig.subplots_adjust(left=.01,right=.99,bottom=.10,top=.82,wspace=-.025,hspace=.14)
    for j,arm in enumerate(ARMS):
        panel(fig.add_subplot(2,5,j*5+1,projection='3d'),target,arm+(' old metric' if arm=='B0' else ' calibrated metric')+'\nTarget reference')
        for k,i in enumerate(DISPLAY_IDS):panel(fig.add_subplot(2,5,j*5+k+2,projection='3d'),meshes[arm,i],f'Fresh ID {i:02d} | free PPP',records[arm][i])
    fig.suptitle('Preselected fresh IDs 0–3 | same noise, camera and world scale for both arms',fontsize=14,y=.975)
    fig.text(.5,.895,'All displayed outputs are fully generated; the target is a visual reference only. No best-sample selection.\nNo repositioning, clipping, normal normalization, geometry repair or domain switching.',ha='center',fontsize=10)
    fig.text(.5,.027,'Actual face counts and audit labels are retained. Images alone do not certify manifoldness or the joint quality contract.\nAll panels share orthographic projection and the same physical axis limits.',ha='center',fontsize=9)
    save(fig,'fresh_samples_000_to_003')
    labels=['parent3000','M0_5000','M0_7000','M0_9000'];steps=[3000,5000,7000,9000]
    old=np.array([parent['summary']['old_field_gradient_rms'][k] for k in FIELDS]);new=np.array([parent['summary']['new_field_gradient_rms'][k] for k in FIELDS])
    fig,(left,right)=plt.subplots(1,2,figsize=(13.5,5.1),gridspec_kw={'width_ratios':[.85,1.45]});fig.subplots_adjust(left=.065,right=.99,bottom=.22,top=.77,wspace=.24)
    positions=np.arange(3);width=.33
    bars_old=left.bar(positions-width/2,old,width,color='#8ba7b8',label='Old metric at parent')
    bars_new=left.bar(positions+width/2,new,width,color='#087d73',label='Frozen M0 metric at parent')
    for bars in [bars_old,bars_new]:
        for bar in bars:left.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.025,f'{bar.get_height():.3f}',ha='center',fontsize=8)
    left.set_xticks(positions,['Position','Raw normal','Factor']);left.set_ylabel('Weighted field gradient RMS');left.grid(axis='y',alpha=.2);left.set_axisbelow(True);left.set_ylim(0,max(old.max(),new.max())*1.18)
    left.set_title('Calibration: same 3000-step model',fontsize=11);left.legend(fontsize=8,loc='upper left',frameon=False)
    trajectory={}
    for field in FIELDS:
        values=[probes[label]['summary']['new_field_gradient_rms'][field] for label in labels]
        endpoint=probes['B0_9000']['summary']['old_field_gradient_rms'][field]
        trajectory[field]=dict(M0=values,B0_endpoint=endpoint)
        right.plot(steps,values,color=FIELD_COLORS[field],marker='o',linewidth=1.6,label=FIELD_LABELS[field])
        right.scatter([9000],[endpoint],color=FIELD_COLORS[field],marker='*',s=125,edgecolors='white',linewidths=.6,zorder=4)
    right.set_xticks(steps);right.set_xlim(2750,9350);right.set_xlabel('Cumulative optimizer updates of saved checkpoint');right.set_ylabel('Weighted field gradient RMS');right.grid(alpha=.2)
    right.set_title('M0 frozen-weight trajectory; stars = B0 old-metric endpoint',fontsize=11)
    handles,legend_labels=right.get_legend_handles_labels();handles.append(Line2D([],[],color='#444',marker='*',markersize=10,linestyle='',label='B0 endpoint (old metric)'));legend_labels.append('B0 endpoint (old metric)')
    right.legend(handles,legend_labels,fontsize=8,frameon=False,ncol=2)
    fig.suptitle('Intervention check: shared-parameter gradients on the same 24 fixed conditional probes',fontsize=14,y=.975)
    fig.text(.5,.89,'RMS of ||a_b grad L_b|| across probes; 85,376 shared parameters, entire final output layer excluded.',ha='center',fontsize=10)
    fig.text(.5,.045,'All probe evaluations perform 0 optimizer updates and 0 mesh extractions. Weights are calibrated once and remain frozen.\nInitial total gradient RMS is matched using Gram cross terms; individual-field RMS need not stay equal after training.\nThese are parameter gradients, not AdamW update magnitudes or evidence of geometric success.',ha='center',fontsize=9)
    save(fig,'weighted_parameter_gradient_rms')
    report=dict(status='COMPLETE',plots=plots,counts=counts,display_fresh_ids=DISPLAY_IDS,paired_RMS_missing_ids=missing,
        camera=dict(elevation=ELEVATION,azimuth=AZIMUTH,projection='orthographic',axis_limits=limits.tolist(),shared_all_panels=True),
        gradient_trajectory=trajectory,gradient_checkpoint_labels=labels,gradient_probe_updates=0,shared_parameter_count=85376,
        source_files=[dict(path=p,sha256=h) for p,h in sorted(sources.items())],original_manifest_unmodified=sha(root/'MANIFEST.json')==manifest_hash,
        note='Post-analysis only. No different-objective training-loss comparison, no privileged field mixtures, no mesh repair or model execution.')
    assert report['original_manifest_unmodified'];dump(out/'VISUALIZATION.json',report)
    dump(out/'SUPPLEMENTAL_MANIFEST.json',dict(parent_manifest_sha256=manifest_hash,files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name!='SUPPLEMENTAL_MANIFEST.json']))
    print(json.dumps(dict(output=str(out),plots=plots,counts=counts,total_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())),indent=2))
if __name__=='__main__':main()