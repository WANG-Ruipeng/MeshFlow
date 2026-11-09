"""Final paired result figures from complete, immutable evidence snapshots only.

No training, model import/forward, scoring, geometry analysis or review labeling.
Incomplete endpoints produce BLOCKED metadata and no figures. This is a display
adapter; original scientific definitions and source artifacts stay unchanged.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name,'2')
import csv,hashlib,io,json,math,sys
from pathlib import Path
from . import common as C

VERSION='full_contour_readout_final_paired_figures_v1'
STEPS=(0,1000,3000,6000)
WINDOWS=((1,1000),(1001,3000),(3001,6000))
ARMS=('ORIGINAL_FULL','CONTOUR_READOUT')
LABELS={'ORIGINAL_FULL':('FULL_S0000','FULL_S1000','FULL_S3000','FULL_S6000'),
        'CONTOUR_READOUT':('CONTOUR_S0','CONTOUR_S1000','CONTOUR_S3000','CONTOUR_S6000')}
NAMES={'ORIGINAL_FULL':'Original FULL','CONTOUR_READOUT':'Contour readout'}
COLORS={'ORIGINAL_FULL':'#4C566A','CONTOUR_READOUT':'#D55E00'}
QUALITY=(('CF','CF_illegal_affected_free_fraction','Illegal CF affected free faces (%)'),
         ('FF','FF_illegal_affected_free_fraction','Illegal FF affected free faces (%)'),
         ('I','ConflictCov_C','Conflict coverage of C-query weight (%)'))
DEST=C.OUT/'evaluation/result_plots'


def num(value):
    x=float(value)
    if not math.isfinite(x):raise ValueError('Nonfinite plotting value')
    return x


def csvread(path):
    with C.resolve(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def expected_hash(mapping,path):
    matches=[h for p,h in mapping.items() if C.resolve(p).resolve()==C.resolve(path).resolve()]
    if len(matches)!=1:raise ValueError('Output hash entry missing/ambiguous: '+str(path))
    return matches[0]


def snapshot(root,folder,names,sources,missing):
    pointer=root/folder/'latest.json'
    if not pointer.exists():
        missing.append(dict(reason='MISSING_LATEST',path=str(pointer)));return None,{}
    sources[str(pointer)]=C.sha(pointer);ref=C.read(pointer);path=C.resolve(ref['path'])
    if not path.exists():
        missing.append(dict(reason='MISSING_SNAPSHOT',path=str(path)));return None,{}
    if C.sha(path)!=ref['sha256']:raise ValueError('Snapshot pointer hash mismatch: '+str(pointer))
    sources[str(path)]=ref['sha256'];manifest=C.read(path)
    if manifest.get('status')!='COMPLETE':
        missing.append(dict(reason='SNAPSHOT_NOT_COMPLETE',path=str(path),actual_status=manifest.get('status')))
    tables={}
    for name in names:
        table=path.parent/name
        if not table.exists():
            missing.append(dict(reason='MISSING_TABLE',path=str(table)));continue
        h=C.sha(table)
        if h!=expected_hash(manifest['output_hashes'],table):raise ValueError('Table hash mismatch: '+str(table))
        sources[str(table)]=h;tables[name]=dict(path=str(table),sha256=h,rows=csvread(table))
    return manifest,tables


def unique(rows,key):
    out={}
    for row in rows:
        k=tuple(row[x] for x in key)
        if k in out:raise ValueError('Duplicate plotting identity: '+str(k))
        out[k]=row
    return out


def collect():
    """Return complete observed values, or explicit missing-input reasons."""
    if 'torch' in sys.modules:raise RuntimeError('Final plotting requires a fresh CPU-only process')
    regpath=C.OUT/'registration.json';reg=C.read(regpath)
    if reg.get('status')!='FROZEN' or tuple(reg['evaluation_labels'])!=LABELS['CONTOUR_READOUT']:
        raise ValueError('Wrong experiment registration')
    qs=reg['requests']
    if len(qs)!=24 or len({q['parent_id'] for q in qs})!=6:raise ValueError('Expected six-parent24request panel')
    for split in ('TRAIN','DEV'):
        selected=[q for q in qs if q['split']==split]
        if len(selected)!=12 or len({q['parent_id'] for q in selected})!=3:raise ValueError('TRAIN/DEV denominators differ')
    sources={str(regpath):C.sha(regpath),str(Path(__file__)):C.sha(__file__)};missing=[]
    comp,tables=snapshot(C.OUT,'evaluation/comparison',('joint_strata.csv','main_strata.csv','reference_strata.csv'),sources,missing)
    training={}
    for arm,root in (('ORIGINAL_FULL',C.BASELINE),('CONTOUR_READOUT',C.OUT)):
        manifest,data=snapshot(root,'training_analysis',('fixed_window_diagnostics.csv',),sources,missing)
        training[arm]=(manifest,data)
        if manifest and (manifest.get('logged_updates')!=6000 or manifest.get('last_complete_diagnostic_step')!=6000
                         or manifest.get('usable_complete_diagnostic_rows')!=48000):
            missing.append(dict(reason='INCOMPLETE6000_UPDATE_DIAGNOSTICS',arm=arm))
    if comp and not (comp.get('numerical_complete') and comp.get('joint_and_review_complete')):
        missing.append(dict(reason='NUMERICAL_OR_ACTUAL_REVIEW_INCOMPLETE'))
    if missing:return [],sources,missing
    # The comparison must still describe the actual reviewed CSV/image bytes.
    for path,h in comp['source_hashes'].items():
        if C.sha(path)!=h:raise ValueError('Comparison source changed; regenerate its summary first: '+str(path))
    joint_table=tables['joint_strata.csv'];j=unique(joint_table['rows'],('label','stratum'))
    quality_rows=tables['main_strata.csv']['rows']+tables['reference_strata.csv']['rows']
    qi=unique(quality_rows,('label','stage','stratum','metric'))
    result=[]
    for arm in ARMS:
        for step,label in zip(STEPS,LABELS[arm]):
            row=j.get((label,'ALL'))
            if row is None:
                missing.append(dict(reason='MISSING_JOINT_POINT',label=label));continue
            if row['status']!='COMPLETE' or int(row['expected'])!=24 or int(row['evaluated'])!=24:
                missing.append(dict(reason='INCOMPLETE_JOINT_POINT',label=label));continue
            counts={name:int(row[name]) for name in ('adoption_YES','adoption_NO','adoption_UNCERTAIN','legal_interface_outputs','explicit_joint_success')}
            if any(not 0<=v<=24 for v in counts.values()) or sum(counts[x] for x in ('adoption_YES','adoption_NO','adoption_UNCERTAIN'))!=24:
                raise ValueError('Invalid preserved review/count denominator')
            if counts['explicit_joint_success']>min(counts['adoption_YES'],counts['legal_interface_outputs']):
                raise ValueError('Joint count exceeds either component')
            for metric,field,stage,plotted in (
                ('RAW_ADOPTION_YES','adoption_YES','RAW',True),('POST_LEGAL_INTERFACE_OUTPUTS','legal_interface_outputs','POST',True),
                ('RAW_ADOPTION_AND_POST_LEGAL_INTERFACE','explicit_joint_success','RAW_AND_POST',True),
                ('RAW_ADOPTION_NO','adoption_NO','RAW',False),('RAW_ADOPTION_UNCERTAIN','adoption_UNCERTAIN','RAW',False)):
                result.append(dict(figure='adoption_interfaces_joint',arm=arm,label=label,local_step=step,stage=stage,stratum='ALL',metric=metric,
                    value=counts[field],display_value=counts[field],display_scale=1,display_unit='outputs / 24',denominator=24,
                    source_table=joint_table['path'],source_sha256=joint_table['sha256'],source_column=field,
                    source_row_identity=label+'|ALL',aggregation='count of fixed requests',
                    evidence='Actual saved nonblind AI_PROXY adoption; topology from original score. Candidate step0 explicitly inherited.',plotted=plotted,status='COMPLETE'))
            for stage in ('RAW','POST'):
                for split in ('TRAIN','DEV'):
                    for short,metric,_ in QUALITY:
                        r=qi.get((label,stage,split,metric))
                        if r is None:
                            missing.append(dict(reason='MISSING_QUALITY_POINT',label=label,stage=stage,stratum=split,metric=metric));continue
                        if (r['status']!='COMPLETE' or r['quantity']!='absolute' or int(r['valid_parent_means'])!=3
                            or int(r['complete_four_noise_parents'])!=3 or int(r['valid_noise_values'])!=12):
                            missing.append(dict(reason='INCOMPLETE_QUALITY_POINT',label=label,stage=stage,stratum=split,metric=metric));continue
                        source=tables['main_strata.csv' if arm=='CONTOUR_READOUT' else 'reference_strata.csv']
                        value=num(r['parent_macro_mean'])
                        if not 0<=value<=1:raise ValueError('Quality fraction outside[0,1]')
                        result.append(dict(figure=stage.lower()+'_quality',arm=arm,label=label,local_step=step,stage=stage,stratum=split,metric=short,
                            value=value,display_value=100*value,display_scale=100,display_unit='percent',expected_outputs=12,expected_parents=3,
                            denominator='Per-output free-face denominator' if short!='I' else 'Per-output C-query weight denominator',
                            source_table=source['path'],source_sha256=source['sha256'],source_column=metric,
                            source_row_identity='|'.join((label,stage,split,metric)),aggregation='equal parent mean of four fixed-noise per-output fractions; not pooled face counts',
                            evidence='Existing unchanged geometry metric',plotted=True,status='COMPLETE'))
        manifest,data=training[arm];table=data['fixed_window_diagnostics.csv'];rows=unique(table['rows'],('window_start','window_end','time_bin'))
        for lo,hi in WINDOWS:
            row=rows.get((str(lo),str(hi),'ALL'))
            if row is None or row['status']!='COMPLETE' or int(row['complete_diagnostic_updates_in_window'])!=hi-lo+1 or int(row['samples'])!=8*(hi-lo+1):
                missing.append(dict(reason='INCOMPLETE_FIXED_WINDOW',arm=arm,window_start=lo,window_end=hi));continue
            for scope in ('interface','other'):
                field=scope+'_velocity_mse';value=num(row[field]);sse=num(row[scope+'_velocity_sse']);count=int(row[scope+'_scalar_count'])
                if count<=0 or row[scope+'_status']!='PASS' or not math.isclose(value,sse/count,rel_tol=1e-12,abs_tol=1e-14):
                    raise ValueError('Training MSE numerator/denominator differs')
                result.append(dict(figure='ordinary_training_mse',arm=arm,stage='NORMAL_TRAINING_FORWARD',stratum='ALL_TIMES',metric=scope.upper()+'_VELOCITY_MSE',
                    window_start=lo,window_end=hi,value=value,display_value=value,display_scale=1,display_unit='model velocity squared',
                    numerator=sse,denominator=count,samples=int(row['samples']),updates=hi-lo+1,
                    source_table=table['path'],source_sha256=table['sha256'],source_column=field,source_row_identity=f'{lo}|{hi}|ALL',
                    aggregation='sum velocity SSE / sum scalar-coordinate count in the prespecified window',
                    evidence='Existing ordinary forwards with changing parent/time/noise/width inputs; not a fixed probe or rollout',plotted=True,status='COMPLETE'))
    if not missing:
        for lo,hi in WINDOWS:
            for scope in ('INTERFACE','OTHER'):
                rows=[r for r in result if r['figure']=='ordinary_training_mse' and r['window_start']==lo and r['metric']==scope+'_VELOCITY_MSE']
                if len(rows)!=2 or rows[0]['denominator']!=rows[1]['denominator']:
                    raise ValueError('Paired training coordinate counts differ')
    return result,sources,missing


def run():
    rows,sources,missing=collect()
    config=dict(version=VERSION,steps=STEPS,windows=WINDOWS,arms=ARMS,labels=LABELS,
        denominator='24 requests; TRAIN and DEV each3parents x4fixed noises',
        current_checkpoint_policy='Fixed0/1000/3000/6000, no best checkpoint',
        joint='RAW C_adoption YES and POST C_F_valid_edge_count>0 on the SAME request',
        visual_judgments='AI_PROXY nonblind, not human review; uncertainty saved in data.csv and not counted as YES',
        topology='A legal edge is not a claim of complete boundary attachment or watertightness',
        training='Prespecified windows; changing training inputs; pooled scalar-coordinate MSE; not a fixed probe or autonomous rollout',
        no_composite_score=True,no_errorbars_or_independent_noise_claim=True,
        missing_policy='Any missing formal endpoint/review/window blocks ALL final figures; no interpolation',
        display_units='Raw fractions remain in value; displayed percentages multiply by100')
    signature=hashlib.sha256(json.dumps(dict(sources=sources,config=config,missing=missing),sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
    dest=DEST/('snapshot_'+signature);final=dest/'manifest.json'
    if final.exists():
        saved=C.read(final)
        for p,h in saved['files'].items():
            if C.sha(p)!=h:raise ValueError('Saved figure/data changed')
        C.write(DEST/'latest.json',dict(path=str(final),sha256=C.sha(final),status=saved['status']),replace=True);return saved
    if dest.exists():raise RuntimeError('Unfinished plot attempt exists; record engineering recovery rather than overwrite')
    if missing:
        value=dict(status='BLOCKED_INCOMPLETE_FINAL_DATA',version=VERSION,reasons=missing,source_hashes=sources,
            config=config,files={},figures_created=0,model_calls=0,new_scores=0,new_reviews=0)
        C.write(final,value);C.write(DEST/'latest.json',dict(path=str(final),sha256=C.sha(final),status=value['status']),replace=True)
        return value
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C.write(dest/'render_started.json',dict(status='STARTED',source_hashes=sources,config=config))
    C.csvwrite(dest/'data.csv',rows);C.write(dest/'parameters.json',config)
    def finish(fig,name):
        for extension in ('png','svg'):
            buffer=io.BytesIO();fig.savefig(buffer,format=extension,dpi=190,metadata={'Date':None} if extension=='svg' else None)
            C._write_bytes(dest/(name+'.'+extension),buffer.getvalue())
        plt.close(fig)
    def curve(ax,arm,x,y):
        ax.plot(x,y,color=COLORS[arm],marker='o' if arm=='ORIGINAL_FULL' else 's',
            markerfacecolor='white' if arm=='ORIGINAL_FULL' else COLORS[arm],markersize=5,linewidth=1.7,label=NAMES[arm])
    def style(ax):
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2,linewidth=.6);ax.set_axisbelow(True)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(1,3,figsize=(11.4,3.6),sharey=True)
        for ax,metric,title in zip(axes,('RAW_ADOPTION_YES','POST_LEGAL_INTERFACE_OUTPUTS','RAW_ADOPTION_AND_POST_LEGAL_INTERFACE'),
            ('RAW adoption: AI_PROXY YES','POST outputs with a legal interface','Same-output adoption + legal interface')):
            for arm in ARMS:
                index={r['local_step']:r for r in rows if r['figure']=='adoption_interfaces_joint' and r['metric']==metric and r['arm']==arm}
                curve(ax,arm,STEPS,[index[s]['display_value'] for s in STEPS])
            style(ax);ax.set_title(title);ax.set_xticks(STEPS);ax.set_xlabel('Additional optimizer updates');ax.set_ylim(0,24.8);ax.set_yticks((0,6,12,18,24))
        axes[0].set_ylabel('Output count / 24 fixed requests');axes[0].legend(frameon=False,fontsize=8)
        fig.suptitle('Matched training budget | identical FULL C and fixed noise',fontsize=12,y=.98)
        fig.text(.5,.02,'AI_PROXY is nonblind, not human review. UNCERTAIN is not YES. Step0 is inherited exact geometry; no new sampling.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,.91));finish(fig,'adoption_interfaces_joint')
        for stage in ('RAW','POST'):
            fig,axes=plt.subplots(2,3,figsize=(11.4,6.2),sharex=True,sharey='col')
            for i,split in enumerate(('TRAIN','DEV')):
                for k,(short,_,ylabel) in enumerate(QUALITY):
                    ax=axes[i,k]
                    for arm in ARMS:
                        index={r['local_step']:r for r in rows if r['figure']==stage.lower()+'_quality' and r['stratum']==split and r['metric']==short and r['arm']==arm}
                        curve(ax,arm,STEPS,[index[s]['display_value'] for s in STEPS])
                    style(ax);ax.set_title(split+' | '+short);ax.set_ylabel(ylabel);ax.set_xticks(STEPS);ax.set_ylim(bottom=0)
                    if i==1:ax.set_xlabel('Additional optimizer updates')
            for k,(short,_,_) in enumerate(QUALITY):
                maximum=max(r['display_value'] for r in rows if r['figure']==stage.lower()+'_quality' and r['metric']==short)
                axes[0,k].set_ylim(0,maximum*1.12 if maximum>0 else 1)
            axes[0,0].legend(frameon=False,fontsize=8)
            fig.suptitle(stage+' quality costs | TRAIN and DEV kept separate',fontsize=12,y=.98)
            fig.text(.5,.018,'Each stratum:3parents x4fixed noises. Equal-parent means of per-output fractions; CF, FF and I are separate metrics.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.045,1,.94));finish(fig,stage.lower()+'_quality')
        fig,axes=plt.subplots(1,2,figsize=(8.8,3.7),sharey=True)
        x=np.arange(len(WINDOWS))
        for ax,scope in zip(axes,('INTERFACE','OTHER')):
            for arm in ARMS:
                index={r['window_start']:r for r in rows if r['figure']=='ordinary_training_mse' and r['arm']==arm and r['metric']==scope+'_VELOCITY_MSE'}
                curve(ax,arm,x,[index[lo]['display_value'] for lo,hi in WINDOWS])
            style(ax);ax.set_title('Source-interface coordinates' if scope=='INTERFACE' else 'Other free coordinates')
            ax.set_xticks(x,labels=[f'{lo}-{hi}' for lo,hi in WINDOWS]);ax.set_xlabel('Prespecified update window');ax.set_ylim(bottom=0)
        maximum=max(r['display_value'] for r in rows if r['figure']=='ordinary_training_mse')
        axes[0].set_ylim(0,maximum*1.12 if maximum>0 else 1)
        axes[0].set_ylabel('Pooled velocity MSE (model units)');axes[0].legend(frameon=False,fontsize=8)
        fig.suptitle('Ordinary training forwards | interface and other coordinates',fontsize=12,y=.98)
        fig.text(.5,.025,'Changing parent/time/noise/width inputs; sum(SSE)/sum(coordinate count). Not a fixed probe or rollout.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.07,1,.91));finish(fig,'ordinary_training_mse')
    for p,h in sources.items():
        if C.sha(p)!=h:raise ValueError('Input changed during final plotting')
    value=dict(status='COMPLETE',version=VERSION,path=str(dest),source_hashes=sources,config=config,
        rows=len(rows),plotted_values=sum(r['plotted'] for r in rows),figures_created=4,formats=['PNG','SVG'],
        visual_inspection='PENDING_CALLER_VIEW',model_calls=0,new_scores=0,new_reviews=0,
        files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
    C.write(final,value);C.write(DEST/'latest.json',dict(path=str(final),sha256=C.sha(final),status=value['status']),replace=True)
    return value


if __name__=='__main__':
    result=run();print(json.dumps({k:result[k] for k in ('status','path','rows','plotted_values','figures_created','reasons') if k in result},sort_keys=True))
