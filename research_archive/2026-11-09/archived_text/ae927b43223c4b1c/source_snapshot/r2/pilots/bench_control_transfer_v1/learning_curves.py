"""Preserve full Bench learning logs and plot declared, unselected trajectories."""
from __future__ import annotations
import argparse, csv, json, math, statistics, time
from pathlib import Path
import numpy as np
from . import train as T

def rows(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]

def csvwrite(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for v in values for k in v))
    with path.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(values)

def smooth(values,window):
    a=np.asarray([np.nan if v is None else float(v) for v in values]);out=np.full(len(a),np.nan)
    for i in range(len(a)):
        block=a[max(0,i-window+1):i+1];valid=block[np.isfinite(block)]
        if len(valid):out[i]=valid.mean()
    return out

def run(root,partial=False,window=50):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=T.resolve(root);out=root/'learning_curves';out.mkdir(exist_ok=True);reg=T.read(root/'registration.json');tick=time.perf_counter()
    T.write(out/'resources_before.json',T.resources());curves={};flat=[];summaries=[]
    for arm in T.ARMS:
        status=T.read(root/'training'/arm/'status.json')
        if not partial and (status['status']!='COMPLETE' or status['step']!=reg['updates']):raise RuntimeError('Final curves require both fixed endpoints')
        data=rows(root/'training'/arm/'training_log.jsonl')
        if [r['step'] for r in data]!=list(range(1,len(data)+1)):raise ValueError('Learning log has replay/gaps')
        curves[arm]=data;flat.extend(dict(arm=arm,**r) for r in data)
        exposure=rows(root/'training'/arm/'exposure.jsonl')
        for group,a in (('first100',data[:100]),('last100',data[-100:])):
            if not a:continue
            s=dict(arm=arm,window=group,steps=len(a),first_step=a[0]['step'],last_step=a[-1]['step'])
            for key in ('FM','seam_tip_velocity_MSE','bulk_velocity_MSE','occupancy_BCE','generator_gradient_norm','update_seconds','input_seconds'):
                valid=[float(r[key]) for r in a if r.get(key) is not None]
                s[key+'_mean']=statistics.mean(valid) if valid else None;s[key+'_valid_steps']=len(valid)
            summaries.append(s)
    csvwrite(out/'per_update_full.csv',flat);csvwrite(out/'fixed_window_summary.csv',summaries)
    fields=[('FM','Free-corner FM (different DOF denominators)'),('seam_tip_velocity_MSE','SEAM third-corner velocity MSE'),('bulk_velocity_MSE','Bulk velocity MSE'),('occupancy_BCE','Independent occupancy weighted BCE'),('generator_gradient_norm','Generator gradient norm before clipping'),('update_seconds','Measured update seconds (concurrent execution)')]
    colors={'PLAIN':'#2171b5','SEAM':'#d95f02'};fig,axes=plt.subplots(2,3,figsize=(15,8.5),dpi=150)
    for ax,(field,title) in zip(axes.flat,fields):
        for arm,data in curves.items():
            x=np.asarray([r['step'] for r in data]);v=[r.get(field) for r in data];y=np.asarray([np.nan if z is None else z for z in v])
            if not np.isfinite(y).any():continue
            ax.plot(x,y,color=colors[arm],alpha=.11,lw=.5)
            ax.plot(x,smooth(v,window),color=colors[arm],lw=1.4,label=arm)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Optimizer update');ax.grid(alpha=.2);ax.legend(fontsize=8)
        if field in ('FM','seam_tip_velocity_MSE','bulk_velocity_MSE','generator_gradient_norm'):ax.set_yscale('log')
    title='Bench control transfer — frozen 3000-step trajectories'
    if partial:title+=' (PARTIAL)'
    fig.suptitle(title,fontsize=14);fig.text(.5,.015,f'Faint: every actual update. Solid: trailing {window}-update mean. No checkpoint selection. Training-state errors do not replace free-generation quality.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95]);path=out/('learning_curves_partial.png' if partial else 'learning_curves.png');fig.savefig(path);plt.close(fig)
    T.write(out/'curve_manifest.json',dict(status='PARTIAL' if partial else 'COMPLETE',window=window,rows=len(flat),steps={a:len(v) for a,v in curves.items()},
        source_hashes={str(root/'training'/a/'training_log.jsonl'):T.sha(root/'training'/a/'training_log.jsonl') for a in T.ARMS},figure=str(path),
        semantics='FM objectives use different valid free-coordinate denominators; seam-tip metric is NA for PLAIN; all lines show actual training inputs, not rollouts.',
        wall_seconds=time.perf_counter()-tick,new_forwards=0,new_updates=0))
    T.write(out/'resources_after.json',T.resources());print(str(path))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(T.DEFAULT_ROOT));ap.add_argument('--partial',action='store_true');ap.add_argument('--window',type=int,default=50);a=ap.parse_args();run(a.root,a.partial,a.window)
if __name__=='__main__':main()
