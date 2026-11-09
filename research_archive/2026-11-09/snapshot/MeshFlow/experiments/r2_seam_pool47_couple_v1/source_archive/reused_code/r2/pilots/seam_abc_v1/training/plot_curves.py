"""CPU-only standalone learning curves with explicit scalar denominators.

Uses training.summarize/updates.csv plus preserved ordinary-forward diagnostics.
No torch/model/scorer imports. Never treats training-state endpoint fit as free
generation quality, and never equates shared-tip independent FM with the full
corner-domain FM. Existing outputs are immutable; choose a fresh output folder.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
COLORS={'SEAM200':'#777777','LONG':'#222222','A_COUPLE':'#2563eb','B_POOL47':'#d97706','C_SHARED_TIP':'#a21caf'}
ORDER=('SEAM200','LONG','A_COUPLE','B_POOL47','C_SHARED_TIP')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def number(value):return None if value in ('',None) else float(value)


def aggregate_geometry(path,arm,maximum_step,bin_index):
    data={}
    if not path.exists():return data
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line);step=int(row['local_step'])
            if step>maximum_step or int(row['time_bin'])!=bin_index:continue
            key=step+(0 if arm=='SEAM200' else 200)
            target=data.setdefault(key,defaultdict(float));target['microbatches']+=1
            for part in ('source_interface_third','source_interface_other_free','bulk'):
                for kind in ('scalar_count','velocity_sse','endpoint_sse'):
                    target[part+'_'+kind]+=row[part+'_'+kind]
    return data


def plot(root,summary,out,window=50,partial=False):
    if window!=50:raise ValueError('Frozen display uses 50 updates; do not tune smoothing after results')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root).resolve();summary=Path(summary).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    completion=json.loads((summary/'completion.json').read_text(encoding='utf-8-sig'))
    if completion['status'] not in (('PASS','PARTIAL') if partial else ('PASS',)):
        raise ValueError('A passed summary is required; --partial only permits a partial run')
    updates=read_csv(summary/'updates.csv')
    folders={arm:root/'training'/arm for arm in ORDER if arm not in ('LONG','SEAM200')}
    folders['LONG']=ROOT/'experiments/r2_seam_quality_four_v1/training/LONG'
    folders['SEAM200']=ROOT/'experiments/r2_seam_slot_c200_v1/training/SEAM_SLOT_C200'
    source_hashes={str(summary/'updates.csv'):digest(summary/'updates.csv'),
                   str(summary/'completion.json'):digest(summary/'completion.json')}
    byarm={arm:sorted((r for r in updates if r['arm']==arm),key=lambda r:int(r['local_step'])) for arm in ORDER}
    curves=[];geometry=[]
    for arm in ORDER:
        table=byarm[arm]
        if not table:continue
        for index,row in enumerate(table):
            recent=table[max(0,index-window+1):index+1]
            # A partial window is retained and explicitly counted, never
            # silently padded with old source steps from another recipe.
            den=sum(float(r['free_coordinate_denominator']) for r in recent)
            sse=sum(float(r['FM'])*float(r['free_coordinate_denominator']) for r in recent)
            item=dict(arm=arm,seam_step=int(row['seam_step']),local_step=int(row['local_step']),
                      updates_in_window=len(recent),window_first_step=int(recent[0]['seam_step']),
                      FM_weighted_mse=sse/den,FM_reconstructed_sse=sse,FM_scalar_denominator=den,
                      FM_domain=row['FM_domain'])
            for field in ('coupling','occupancy_BCE','pointer_CE','coupling_velocity_gradient_norm_sum','FM_velocity_gradient_norm_sum'):
                values=[number(r.get(field)) for r in recent];values=[v for v in values if v is not None]
                item[field+'_update_mean']=sum(values)/len(values) if values else None
                item[field+'_update_count']=len(values)
            coupling_den=sum(float(r.get('coupling_eligible_tips') or 0)*3 for r in recent)
            item['coupling_scalar_denominator_sum']=coupling_den
            item['occupancy_edge_denominator_sum']=sum(float(r['occupancy_edge_denominator']) for r in recent)
            item['pointer_query_denominator_sum']=sum(float(r.get('pointer_query_denominator') or 0) for r in recent)
            curves.append(item)
        p=folders[arm]/'geometry_diagnostics.jsonl'
        # Cap to summary cursor so later log appends cannot introduce unpaired
        # updates into this already-created summary's partial figures.
        g=aggregate_geometry(p,arm,max(int(r['local_step']) for r in table),2)
        if p.exists():source_hashes[str(p)]=digest(p)
        for row in table:
            current=int(row['seam_step']);start=max(1 if arm=='SEAM200' else 201,current-window+1)
            values=[g[s] for s in range(start,current+1) if s in g]
            for part in ('source_interface_third','source_interface_other_free','bulk'):
                count=sum(v[part+'_scalar_count'] for v in values)
                vs=sum(v[part+'_velocity_sse'] for v in values);es=sum(v[part+'_endpoint_sse'] for v in values)
                geometry.append(dict(arm=arm,seam_step=current,window_first_step=start,
                    partition=part,time_bin=2,time_lower=.4,time_upper=.6,
                    sampled_updates=len(values),microbatches=int(sum(v['microbatches'] for v in values)),
                    scalar_denominator=int(count),velocity_sse=vs,velocity_mse=vs/count if count else None,
                    endpoint_sse=es,endpoint_mse=es/count if count else None,
                    domain='original all-free diagnostic coordinates (aliases included for C)',
                    interpretation='training interpolation fit; not free rollout'))
    write_csv(out/'update_curves.csv',curves);write_csv(out/'partition_curves_t04_06.csv',geometry)
    plt.rcParams.update({'figure.dpi':140,'savefig.dpi':180,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def line(ax,rows,field,arm,label=None):
        valid=[r for r in rows if r.get(field) is not None]
        if not valid:return
        ax.plot([r['seam_step'] for r in valid],[r[field] for r in valid],color=COLORS[arm],
                label=label or arm,lw=1.6,ls='--' if arm=='SEAM200' else '-')
    def decorate(ax,ylabel):
        ax.set_xlabel('Total SEAM updates (source checkpoint = 200)');ax.set_ylabel(ylabel)
        ax.axvline(200,color='#aaaaaa',lw=.8,ls=':');ax.grid(alpha=.2)
        handles,labels=ax.get_legend_handles_labels()
        if handles:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'No completed updates in this snapshot',ha='center',va='center',transform=ax.transAxes,fontsize=9)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for arm in ORDER:
        rows=[r for r in curves if r['arm']==arm]
        target=axes[1] if arm=='C_SHARED_TIP' else axes[0]
        line(target,rows,'FM_weighted_mse',arm)
    axes[0].set_title('FM: original non-anchor coordinate domain')
    axes[1].set_title('C FM: independent variables only (different denominator)')
    for ax in axes:decorate(ax,'Velocity MSE (model units squared)')
    fig.suptitle('50-update scalar-weighted FM; stochastic training states')
    fig.savefig(out/'fm_curves.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for column,(part,title) in enumerate((('source_interface_third','Source seam third corners'),('bulk','Remaining bulk corners'))):
        for arm in ORDER:
            rows=[r for r in geometry if r['arm']==arm and r['partition']==part]
            line(axes[0,column],rows,'velocity_mse',arm);line(axes[1,column],rows,'endpoint_mse',arm)
        axes[0,column].set_title(title+' — velocity');axes[1,column].set_title(title+' — endpoint estimate')
        decorate(axes[0,column],'Velocity MSE (model units squared)')
        decorate(axes[1,column],'Endpoint MSE (model units squared)')
    fig.suptitle('Same t bin [0.4, 0.6); 50-update scalar-weighted source partitions\nC diagnostic includes aliased copies; this is training interpolation, not rollout')
    fig.savefig(out/'interface_bulk_curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for arm in ORDER:
        rows=[r for r in curves if r['arm']==arm]
        line(axes[2],rows,'occupancy_BCE_update_mean',arm)
        if arm=='A_COUPLE':line(axes[0],rows,'coupling_update_mean',arm)
        if arm=='C_SHARED_TIP':line(axes[1],rows,'pointer_CE_update_mean',arm)
    axes[0].set_title('A endpoint coupling (unweighted; lambda = 0.1)')
    axes[1].set_title('C pointer CE (independent small head)')
    axes[2].set_title('C-only occupancy weighted BCE')
    for ax,label in zip(axes,('Endpoint MSE (model units squared)','Cross entropy (all bulk queries)','Weighted BCE (all batch edges)')):decorate(ax,label)
    fig.suptitle('50-update objective means; objectives and denominators are distinct')
    fig.savefig(out/'auxiliary_curves.png');plt.close(fig)
    outputs={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()}
    result=dict(status='PARTIAL' if partial else 'PASS',summary_status=completion['status'],window_updates=window,
                model_calls=0,GPU_calls=0,source_hashes=source_hashes,outputs=outputs,code_sha256=digest(Path(__file__)),
                denominator_policy='Every CSV point stores scalar/query/edge denominator and support count; no source reconstruction ranking',
                interpretation='Training interpolation learning curves, not held-out generation quality')
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],files=list(outputs),update_points=len(curves),partition_points=len(geometry)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='experiments/r2_seam_abc_v1');p.add_argument('--summary',required=True);p.add_argument('--output',required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();plot(a.root,a.summary,a.output,partial=a.partial)
