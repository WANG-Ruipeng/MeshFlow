"""Prepared scientific plots from COMPLETE final tables only.

No training/model/mesh/score imports. No raw training log audit, new metrics,
resampling, checkpoint choice, or semantic verdict. Default action is status.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
EXPERIMENT=ROOT/'experiments/r2_seam_pool47_couple_v1'
OUT=EXPERIMENT/'summary_figures'
VERSION='pool47_couple_fixed_final_table_figures_v1'
LABELS=('B_LONG_LAYOUT','PLAIN3000','COUPLE3000')
DISPLAY={'B_LONG_LAYOUT':'B_STEP0','PLAIN3000':'PLAIN3000','COUPLE3000':'COUPLE3000'}
STAGES=('POST','BOUND015')
METRICS=(
    ('free_only_reached_bulk_area_fraction','Reachable bulk area (%)','Higher, descriptive'),
    ('FF_illegal_affected_free_fraction','FF affected free faces (%)','Lower'),
    ('ConflictCov_C','I: conflict coverage of C (%)','Lower'),
)
COLORS={'PLAIN3000':'#0072B2','COUPLE3000':'#D55E00'}
STAGE_COLORS={'POST':'#0072B2','BOUND015':'#D55E00'}
NOISE_COLORS=('#0072B2','#D55E00','#009E73','#CC79A7')


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def write_csv(path,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def number(value):
    if value is None or value=='':return None
    value=float(value)
    if not np.isfinite(value):raise ValueError('Nonfinite table entry; explicit NA required')
    return value

def consumed_inputs():
    return [EXPERIMENT/'training_analysis'/name for name in ('complete.json','rolling100.csv','time_binned_geometry.csv')]+[
        EXPERIMENT/'results/completion.json',EXPERIMENT/'results/per_output.csv']


def load_completed_tables():
    paths=consumed_inputs()
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise RuntimeError('Waiting for completed final tables: '+', '.join(missing))
    train=read_json(paths[0]);result=read_json(paths[3])
    if train.get('status')!='COMPLETE' or train.get('updates_per_arm')!=3000:
        raise ValueError('Require complete fixed 3000-update training analysis')
    if result.get('status') not in ('NUMERICAL_COMPLETE_VISUAL_PENDING','COMPLETE') or result.get('rows')!=252 or result.get('numeric_fields')!=124:
        raise ValueError('Require full 252-row, 124-field final numerical result')
    # Verify only consumed tables against their completed producer receipts;
    # do not rerun the training ledger audit or any geometric measurement.
    for receipt,selected in ((train,paths[1:3]),(result,paths[4:])):
        for p in selected:
            candidates=[h for q,h in receipt['files'].items() if Path(str(q).replace('\\','/')).name==p.name]
            if len(candidates)!=1 or candidates[0]!=sha(p):raise ValueError('Completed table hash differs: '+str(p))
    raw=read_csv(paths[4]);rows=[]
    for r in raw:
        rows.append({**r,'noise_index':int(r['noise_index']),**{m:number(r[m]) for m,_,_ in METRICS}})
    keys=[(r['label'],r['stage'],r['request_id']) for r in rows]
    if len(rows)!=252 or len(set(keys))!=252:raise ValueError('Duplicate or missing output row')
    if set(r['label'] for r in rows)!=set(LABELS) or set(r['stage'] for r in rows)!={'RAW','POST','BOUND015'}:
        raise ValueError('Unexpected comparison identities')
    for label in LABELS:
        for stage in ('RAW','POST','BOUND015'):
            rr=[r for r in rows if r['label']==label and r['stage']==stage]
            parents={r['parent_id'] for r in rr}
            if len(rr)!=28 or len(parents)!=7 or sum(r['split']=='TRAIN' for r in rr)!=16 or sum(r['split']=='DEV' for r in rr)!=12:
                raise ValueError('Registered TRAIN16/DEV12 panel differs')
            for parent in parents:
                if {r['noise_index'] for r in rr if r['parent_id']==parent}!={0,1,2,3}:
                    raise ValueError('Each parent must retain four fixed noise identities')
    rolling=read_csv(paths[1]);geometry=read_csv(paths[2])
    if {r['arm'] for r in rolling}!={'PLAIN3000','COUPLE3000'}:raise ValueError('Training arms differ')
    for arm in ('PLAIN3000','COUPLE3000'):
        ends=sorted(int(r['step_end']) for r in rolling if r['arm']==arm)
        if ends!=list(range(100,3001,25)):raise ValueError('Rolling-100 plot must use complete fixed producer windows')
    return rows,rolling,geometry,{str(p):sha(p) for p in paths}


def setup_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'legend.fontsize':8,'figure.titlesize':12,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    return plt

def save_figure(fig,name,plt,files):
    for ext in ('png','pdf'):
        p=OUT/(name+'.'+ext)
        if p.exists():raise FileExistsError('Refuse figure overwrite: '+str(p))
        fig.savefig(p,dpi=180,bbox_inches='tight')
        files[str(p)]=sha(p)
    plt.close(fig)


def training_figure(rolling,geometry,plt,files):
    fig,axes=plt.subplots(2,3,figsize=(14.5,8.3));plot_rows=[]
    settings=[(axes[0,0],'FM','FM objective (rolling 100)',False),
        (axes[0,1],'coupling','Unweighted coupling objective (rolling 100)',True),
        (axes[0,2],'couple_to_FM_velocity_norm_sum_ratio','lambda * coupling / FM velocity-norm sums',True)]
    for ax,metric,title,couple_only in settings:
        for arm in ('PLAIN3000','COUPLE3000'):
            if couple_only and arm!='COUPLE3000':continue
            rr=sorted([r for r in rolling if r['arm']==arm],key=lambda r:int(r['step_end']))
            vals=[number(r.get(metric)) for r in rr];x=[int(r['step_end']) for r in rr]
            ax.plot(x,[v if v is not None else np.nan for v in vals],color=COLORS[arm],label=arm,lw=1.8)
            plot_rows.extend(dict(plot='training',panel=metric,arm=arm,step=s,value=v,source='producer rolling100.csv') for s,v in zip(x,vals))
        ax.set_title(title);ax.set_xlabel('Additional optimizer updates');ax.grid(alpha=.22);ax.legend()
        if couple_only:ax.text(.02,.98,'PLAIN: term not applied',ha='left',va='top',transform=ax.transAxes,fontsize=8,color='#555555')
    for ax,region,metric,title in ((axes[1,0],'source_interface_third','endpoint_mse','Source seam-tip endpoint MSE'),
        (axes[1,1],'bulk','endpoint_mse','Bulk endpoint MSE'),
        (axes[1,2],'source_interface_third','velocity_mse','Source seam-tip velocity MSE')):
        for arm in ('PLAIN3000','COUPLE3000'):
            rr=sorted([r for r in geometry if r['arm']==arm and r['region']==region and r['time_bin']=='-1' and r['window'].startswith('BLOCK')],key=lambda r:int(r['window'][5:]))
            if len(rr)!=30:raise ValueError('Expected complete 30 x 100-update geometry blocks')
            x=[100*int(r['window'][5:]) for r in rr];vals=[number(r[metric]) for r in rr]
            ax.plot(x,[v if v is not None else np.nan for v in vals],color=COLORS[arm],label=arm,lw=1.8)
            plot_rows.extend(dict(plot='training',panel=region+'_'+metric,arm=arm,step=s,value=v,source='producer scalar-weighted 100-update blocks') for s,v in zip(x,vals))
        ax.set_title(title);ax.set_xlabel('Additional optimizer updates');ax.grid(alpha=.22);ax.legend()
    fig.suptitle('Fixed paired training: same B start, data stream and 3,000 updates',y=.995)
    fig.text(.5,.005,'Supervised interpolation diagnostics, not rollout quality. Norm sums are output-velocity diagnostics, not parameter-gradient ratios.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.965));save_figure(fig,'training_learning_curves',plt,files)
    return plot_rows


def metric_upper(rows,metric):
    vals=[100*r[metric] for r in rows if r['stage'] in STAGES and r.get(metric) is not None]
    return 100. if metric==METRICS[0][0] else max(1.,max(vals,default=0)*1.12)

def endpoint_figure(rows,plt,files):
    fig,axes=plt.subplots(2,3,figsize=(14.5,8.0));plot_rows=[]
    for rindex,split in enumerate(('TRAIN','DEV')):
        for cindex,(metric,title,direction) in enumerate(METRICS):
            ax=axes[rindex,cindex]
            for stage in STAGES:
                means=[];offset=-.11 if stage=='POST' else .11
                for j,label in enumerate(LABELS):
                    rr=sorted([r for r in rows if r['split']==split and r['stage']==stage and r['label']==label],key=lambda r:(r['parent_id'],r['noise_index']))
                    vals=[100*r[metric] for r in rr if r[metric] is not None]
                    jitter=np.linspace(-.065,.065,len(rr))
                    for dx,r in zip(jitter,rr):
                        v=100*r[metric] if r[metric] is not None else None
                        if v is not None:ax.scatter(j+offset+dx,v,s=13,color=STAGE_COLORS[stage],alpha=.38,edgecolors='none')
                        plot_rows.append(dict(plot='endpoints',split=split,stage=stage,label=label,request_id=r['request_id'],parent_id=r['parent_id'],noise_index=r['noise_index'],metric=metric,value_percent=v))
                    means.append(float(np.mean(vals)) if vals else np.nan)
                ax.plot(np.arange(3)+offset,means,marker='o',markersize=6,lw=1.5,color=STAGE_COLORS[stage],label=stage)
            ax.set_xticks(range(3),[DISPLAY[a] for a in LABELS]);ax.set_ylim(0,metric_upper(rows,metric));ax.grid(axis='y',alpha=.22)
            ax.set_title(split+' | '+title);ax.set_ylabel(direction)
            if cindex==0:ax.legend(loc='best')
    fig.suptitle('Final same-noise endpoints: original POST and the same 1.5%L binding',y=.995)
    fig.text(.5,.005,'Small dots: all registered outputs; large markers: means. TRAIN = 4 parents x 4 noises, DEV = 3 x 4. No confidence interval or semantic-success claim.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.035,1,.965));save_figure(fig,'endpoint_POST_BOUND015_comparison',plt,files)
    return plot_rows


def effect_figure(rows,plt,files):
    lookup={(r['label'],r['stage'],r['request_id']):r for r in rows};effects=[]
    fig,axes=plt.subplots(3,2,figsize=(13.5,11.2))
    categories=[('TRAIN','POST'),('TRAIN','BOUND015'),('DEV','POST'),('DEV','BOUND015')]
    comparisons=[('B_LONG_LAYOUT','PLAIN3000','Continuation: PLAIN3000 - B_STEP0'),
                 ('PLAIN3000','COUPLE3000','Coupling increment: COUPLE3000 - PLAIN3000')]
    for col,(base,candidate,heading) in enumerate(comparisons):
        for mi,(metric,title,direction) in enumerate(METRICS):
            ax=axes[mi,col]
            for j,(split,stage) in enumerate(categories):
                rr=sorted([r for r in rows if r['label']==candidate and r['stage']==stage and r['split']==split],key=lambda r:(r['parent_id'],r['noise_index']))
                vals=[]
                for dx,r in zip(np.linspace(-.13,.13,len(rr)),rr):
                    b=lookup[base,stage,r['request_id']];valid=r[metric] is not None and b[metric] is not None
                    delta=100*(r[metric]-b[metric]) if valid else None
                    if valid:vals.append(delta);ax.scatter(j+dx,delta,s=20,color=NOISE_COLORS[r['noise_index']],alpha=.7,edgecolors='none')
                    effects.append(dict(comparison=heading,baseline=base,candidate=candidate,split=split,stage=stage,request_id=r['request_id'],parent_id=r['parent_id'],noise_index=r['noise_index'],metric=metric,delta_pp=delta,status='PASS' if valid else 'NA'))
                if vals:ax.scatter(j,float(np.mean(vals)),marker='D',s=48,facecolor='white',edgecolor='black',linewidth=1.3,zorder=5)
            ax.axhline(0,color='#444444',lw=.8);ax.grid(axis='y',alpha=.2)
            ax.set_xticks(range(4),['TRAIN\nPOST','TRAIN\nBOUND015','DEV\nPOST','DEV\nBOUND015'])
            ax.set_ylabel('Delta, percentage points');ax.set_title(heading+'\n'+title+'; '+('positive is higher reach' if mi==0 else 'negative is lower conflict'))
    fig.suptitle('Do not combine continued training with the incremental coupling effect',y=.995)
    fig.text(.5,.005,'Colored points: paired fixed-noise differences; white diamonds: means. Four noises within a parent are repeated conditions, not new-parent evidence.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.965));save_figure(fig,'paired_continuation_vs_coupling_increment',plt,files)
    return effects


def parent_figures(rows,plt,files):
    lookup={(r['label'],r['stage'],r['parent_id'],r['noise_index']):r for r in rows}
    for split in ('TRAIN','DEV'):
        parents=sorted({r['parent_id'] for r in rows if r['split']==split})
        fig,axes=plt.subplots(len(parents),3,figsize=(15.4,2.9*len(parents)+1.4),squeeze=False)
        for i,parent in enumerate(parents):
            for j,(metric,title,direction) in enumerate(METRICS):
                ax=axes[i,j]
                for noise in range(4):
                    for stage,xs,style in (('POST',[0,1,2],'-'),('BOUND015',[4,5,6],'--')):
                        vals=[lookup[label,stage,parent,noise][metric] for label in LABELS]
                        ax.plot(xs,[100*v if v is not None else np.nan for v in vals],style,marker='o',markersize=3.5,lw=1.2,
                            color=NOISE_COLORS[noise],label='noise '+str(noise) if stage=='POST' else None)
                ax.axvline(3,color='#bbbbbb',lw=.6);ax.grid(axis='y',alpha=.18)
                ax.set_ylim(0,metric_upper(rows,metric));ax.set_xticks([0,1,2,4,5,6],['B0','PLAIN','COUPLE','B0','PLAIN','COUPLE'],fontsize=8)
                ax.set_title(parent[:8]+' | '+title);ax.set_xlabel('Original POST                         BOUND015')
                if i==0 and j==0:ax.legend(ncol=2,loc='best')
        fig.suptitle(split+': every parent and all four fixed noises',y=.995)
        fig.text(.5,.005,'Each colored line follows one noise across the three frozen models. Solid: POST; dashed: same 1.5%L binding. Shared metric scale across parents and splits.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.97));save_figure(fig,'per_parent_four_noise_'+split.lower(),plt,files)


def run():
    rows,rolling,geometry,hashes=load_completed_tables()
    done=OUT/'completion.json'
    if done.exists():
        value=read_json(done)
        if value['input_hashes']!=hashes or value['source_sha256']!=sha(__file__):raise ValueError('Completed visualization identity differs')
        for p,h in value['files'].items():
            if sha(p)!=h:raise ValueError('Completed figure changed')
        return value
    if (OUT/'registration.json').exists():raise RuntimeError('Incomplete figure attempt: preserve artifacts and require explicit recovery')
    OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    write_json(OUT/'registration.json',dict(status='FROZEN',version=VERSION,input_hashes=hashes,source_sha256=sha(__file__),
        selection='All fixed final 3000-step endpoints and all registered outputs; no metric-based subset or checkpoint choice',
        figures=['training_learning_curves','endpoint_POST_BOUND015_comparison','paired_continuation_vs_coupling_increment','per_parent_four_noise_train','per_parent_four_noise_dev'],
        formats=['png','pdf'],model_calls=0,quality_scans=0))
    plt=setup_plot();files={}
    training=training_figure(rolling,geometry,plt,files)
    endpoint=endpoint_figure(rows,plt,files)
    effects=effect_figure(rows,plt,files)
    parent_figures(rows,plt,files)
    for name,data in (('training_plot_values',training),('endpoint_plot_values',endpoint),('paired_effect_plot_values',effects)):
        p=OUT/(name+'.csv');write_csv(p,data);files[str(p)]=sha(p)
    for p,h in hashes.items():
        if sha(p)!=h:raise ValueError('Input table changed while rendering')
    value=dict(status='RENDERED_PENDING_VISUAL_QA',version=VERSION,source_sha256=sha(__file__),input_hashes=hashes,files=files,
        figures=5,formats=['png','pdf'],seconds=time.perf_counter()-started,model_calls=0,quality_scans=0,
        actual_image_review='PENDING; numerical rendering is not visual verification')
    write_json(done,value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('status','run'),default='status',nargs='?');args=p.parse_args()
    if args.stage=='run':result=run();result={k:v for k,v in result.items() if k not in ('files','input_hashes')}
    else:result=dict(status='PREPARED_NO_AUTO_RUN',completed_tables_present=all(p.exists() for p in consumed_inputs()),
        figures=5,model_calls=0,quality_scans=0)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
