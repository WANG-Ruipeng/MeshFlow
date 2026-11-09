"""Read-only training-log visualization. No model import or recomputation."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path
import numpy as np
from . import common as C

VERSION='seam_c200_saved_training_curves_v1'
COLORS={'FULL4_C200':'#2468a0','SEAM_SLOT_C200':'#c45c27'}


def rows(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def rolling(values,width=20):
    values=np.asarray(values,float)
    result=np.full(len(values),np.nan)
    result[width-1:]=np.convolve(values,np.ones(width)/width,mode='valid')
    return result


def run(root):
    root=C.resolve(root);dest=root/'statistics';tick=time.perf_counter()
    names=('training_per_update.csv','training_per_micro.csv','learning_curve_time_buckets.csv','occupancy_training_per_micro.csv')
    paths=[dest/name for name in names];hashes={str(p):C.sha(p) for p in paths}
    identity=dict(version=VERSION,source_sha256=C.sha(__file__),inputs=hashes)
    receipt=dest/'training_curves_receipt.json'
    if receipt.exists():
        old=C.read(receipt)
        if old['identity']!=identity:raise ValueError('Frozen training plot identity changed')
        for p,h in old['files'].items():
            if C.sha(p)!=h:raise ValueError('Training plot artifact changed')
        return old
    if (dest/'training_curves.png').exists():raise RuntimeError('Unreceipted figure: no implicit overwrite')
    resources=C.resources();update,micro,buckets,head=map(rows,paths)
    if len(update)!=400 or len(micro)!=3200 or len(buckets)!=40 or len(head)!=1600:
        raise ValueError('Expected two complete 200-step arms and 1600 normal-forward occupancy records')
    curves=[];updates={};denominators={}
    for arm in C.ARMS:
        rr=sorted([r for r in update if r['arm']==arm],key=lambda r:int(r['local_step']))
        if [int(r['local_step']) for r in rr]!=list(range(1,201)):raise ValueError('Missing/repeated updates')
        if any(int(r['global_step'])!=12000+int(r['local_step']) for r in rr):raise ValueError('Global update identity differs')
        updates[arm]=rr;denominators[arm]=sorted({int(r['free_coordinate_denominator']) for r in rr})
        fm=rolling([float(r['FM']) for r in rr])
        for step,value in enumerate(fm,1):
            curves.append(dict(panel='FM_20_update_mean',arm=arm,local_step=step,value=float(value) if np.isfinite(value) else None,
                status='PASS' if np.isfinite(value) else 'NA_FIRST_19_UPDATES',denominator=denominators[arm]))
        mm=[r for r in micro if r['arm']==arm]
        if len(mm)!=1600 or any(sum(int(r['local_step'])==step for r in mm)!=8 for step in range(1,201)):
            raise ValueError('Normal training-forward microbatch count differs')
        # Check published buckets against their saved raw SSE/counts; no geometry/model work.
        for b in [r for r in buckets if r['arm']==arm]:
            subset=[r for r in mm if int(b['step_first'])<=int(r['local_step'])<=int(b['step_last']) and r['time_bin']==b['time_bin']]
            if len(subset)!=int(b['samples']):raise ValueError('Time-bin sample count mismatch')
            for prefix in ('source_interface_third','bulk'):
                denominator=sum(int(r[prefix+'_scalar_count']) for r in subset)
                numerator=sum(float(r[prefix+'_velocity_sse']) for r in subset)
                value=numerator/denominator
                if denominator!=int(b[prefix+'_scalar_count']) or not np.isclose(value,float(b[prefix+'_velocity_mse']),rtol=1e-12,atol=1e-14):
                    raise ValueError('Saved SSE/count does not reproduce bucket vMSE')
                curves.append(dict(panel=prefix+'_vMSE_by_t_and_step',arm=arm,step_first=int(b['step_first']),step_last=int(b['step_last']),
                    t_lower=int(b['time_bin'])*.2,t_upper=(int(b['time_bin'])+1)*.2,value=value,denominator=denominator,samples=len(subset),status='PASS'))
    head_step=[]
    for step in range(1,201):
        rr=[r for r in head if int(r['local_step'])==step]
        if len(rr)!=8 or {r['arm'] for r in rr}!={'SEAM_SLOT_C200'}:raise ValueError('Occupancy record identity differs')
        counts={key:sum(int(r[key]) for r in rr) for key in ('edge_count','positive_count','null_count','true_positive','true_negative','false_positive','false_negative')}
        if (counts['edge_count'],counts['positive_count'],counts['null_count'])!=(132,124,8):raise ValueError('Head exposure denominator differs')
        if counts['true_positive']+counts['false_negative']!=124 or counts['true_negative']+counts['false_positive']!=8:raise ValueError('Head classification count mismatch')
        loss=sum(float(r['weighted_BCE_contribution']) for r in rr)
        if not np.isclose(loss,float(updates['SEAM_SLOT_C200'][step-1]['occupancy_BCE']),rtol=1e-7,atol=1e-8):raise ValueError('Head BCE log mismatch')
        for r in rr:
            target=np.asarray(json.loads(r['labels']),bool);logit=np.asarray(json.loads(r['logits']),float);pred=logit>=0
            if int((pred&target).sum())!=int(r['true_positive']) or int((~pred&~target).sum())!=int(r['true_negative']):raise ValueError('Logged zero-threshold classification mismatch')
        head_step.append(dict(step=step,BCE=loss,positive_recall=counts['true_positive']/124,Null_recall=counts['true_negative']/8,
            exact_layout_fraction=sum(r['whole_layout_match'].lower()=='true' for r in rr)/8))
    for field in ('BCE','positive_recall','Null_recall','exact_layout_fraction'):
        for step,value in enumerate(rolling([r[field] for r in head_step]),1):
            curves.append(dict(panel='occupancy_training_'+field+'_20_update_mean',arm='SEAM_SLOT_C200',local_step=step,
                value=float(value) if np.isfinite(value) else None,status='PASS' if np.isfinite(value) else 'NA_FIRST_19_UPDATES',
                denominator='132 edges / 124 positive / 8 Null / 8 layouts per update; repeated clean+augmented TRAIN observations'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(17,9.9),dpi=160)
    x=np.arange(1,201)
    for arm in C.ARMS:
        values=np.array([float(r['FM']) for r in updates[arm]])
        axes[0,0].plot(x,values,color=COLORS[arm],alpha=.18,lw=.7)
        axes[0,0].plot(x,rolling(values),color=COLORS[arm],lw=2,label=arm+'; D='+','.join(map(str,denominators[arm])))
    axes[0,0].set_title('A. FM: 20-update mean\nDifferent free-coordinate denominators; no loss ranking')
    axes[0,0].set_ylabel('Logged FM (different objectives)');axes[0,0].legend(fontsize=8)
    for ax,prefix,title in ((axes[0,1],'source_interface_third','B. Source third-point vMSE'),(axes[0,2],'bulk','C. Bulk vMSE')):
        for arm in C.ARMS:
            selected=sorted([r for r in buckets if r['arm']==arm and r['time_bin']=='2'],key=lambda r:int(r['step_first']))
            centers=[(int(r['step_first'])+int(r['step_last']))/2 for r in selected]
            ax.plot(centers,[float(r[prefix+'_velocity_mse']) for r in selected],'-o',color=COLORS[arm],label=arm)
        ax.set_title(title+'\nt in [0.4, 0.6); four 50-update windows')
        ax.set_ylabel('Sum velocity SSE / sum active scalar count');ax.legend(fontsize=8)
        ax.set_xticks([25.5,75.5,125.5,175.5],['1-50','51-100','101-150','151-200'])
    for arm in C.ARMS:
        for start,style in ((1,'--'),(151,'-')):
            rr=sorted([r for r in buckets if r['arm']==arm and int(r['step_first'])==start],key=lambda r:int(r['time_bin']))
            axes[1,0].plot([.1+.2*int(r['time_bin']) for r in rr],[float(r['source_interface_third_velocity_mse']) for r in rr],
                style+'o',color=COLORS[arm],label=arm+(' 1-50' if start==1 else ' 151-200'))
    axes[1,0].set_title('D. Third-point vMSE across flow time\nFirst vs last 50 updates; same t buckets')
    axes[1,0].set_xlabel('Flow-time bucket midpoint');axes[1,0].set_ylabel('Source third-point velocity MSE');axes[1,0].legend(fontsize=7)
    bce=[r['BCE'] for r in head_step]
    axes[1,1].plot(x,bce,color=COLORS['SEAM_SLOT_C200'],alpha=.2,lw=.7)
    axes[1,1].plot(x,rolling(bce),color=COLORS['SEAM_SLOT_C200'],lw=2)
    axes[1,1].set_title('E. Occupancy head weighted BCE\nNormal training forwards; 20-update mean')
    axes[1,1].set_ylabel('Weighted BCE; 132 edges per update')
    for field,color,title in (('positive_recall','#348342','Positive recall (124/update)'),('Null_recall','#a14692','Null recall (8/update)'),
                              ('exact_layout_fraction','#67747a','Exact layouts (8/update)')):
        axes[1,2].plot(x,rolling([r[field] for r in head_step]),color=color,lw=2,label=title)
    axes[1,2].set_title('F. Training occupancy at logit >= 0\n20-update mean; repeated clean + augmented C')
    axes[1,2].set_ylabel('Training classification fraction');axes[1,2].set_ylim(-.03,1.03);axes[1,2].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.tick_params(labelsize=8)
        if ax is not axes[1,0]:ax.set_xlabel('Local optimizer update')
    fig.suptitle('SEAM_SLOT_C200 paired pilot: saved training logs only',fontsize=16,y=.99)
    fig.text(.5,.035,'Training interpolation contains source targets: these curves are not rollout adoption, seam closure, or unseen-C performance.',ha='center',fontsize=10)
    fig.text(.5,.015,'FM excludes fixed coordinates: FULL D=12348, SEAM D=11604. Third-point and bulk diagnostics use their own scalar denominators; no new forwards.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.965));buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=160,facecolor='white');plt.close(fig)
    image=dest/'training_curves.png'
    with image.open('xb') as handle:handle.write(buffer.getvalue())
    C.csvwrite(dest/'training_curves_plotted_values.csv',curves)
    if 'torch' in sys.modules:raise RuntimeError('Plot imported model framework')
    for path,digest in hashes.items():
        if C.sha(path)!=digest:raise ValueError('Training log changed during plotting')
    result=dict(status='COMPLETE',identity=identity,files={str(p):C.sha(p) for p in (image,dest/'training_curves_plotted_values.csv')},
        seconds=time.perf_counter()-tick,resources_before=resources,CPU_workers=1,BLAS_threads=1,
        input_records=dict(updates=400,normal_generator_microbatches=3200,normal_head_microbatches=1600,time_buckets=40),
        verified=dict(bucket_SSE_scalar_denominators=True,head_weighted_BCE_sum=True,zero_logit_threshold_counts=True),
        rolling_window=20,rolling_first_19='NA',step_windows=[[1,50],[51,100],[101,150],[151,200]],
        model_calls=0,new_training=0,new_scoring=0,source_target_metrics='Training interpolation fit only, not rollout or unique-GT ranking.',
        FM_comparison='Different unknown-coordinate sets and denominators, so FM scalar size is not a cross-arm success ranking.',
        occupancy_scope='Repeated four TRAIN C, clean plus augmentation. Log points are not independent test samples.',
        no_changes_to_frozen_evaluator_renderer_or_common=True)
    C.write(receipt,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(C.OUT));a=p.parse_args()
    result=run(a.root);print(json.dumps(dict(status=result['status'],files=result['files']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
