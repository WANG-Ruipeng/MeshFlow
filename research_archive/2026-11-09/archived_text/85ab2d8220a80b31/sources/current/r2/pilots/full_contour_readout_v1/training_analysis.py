"""CPU-only weighted summaries of already recorded CONTOUR training forwards.

No model, training, source scorer, sampling, or additional fixed probe is run.
Snapshots are immutable and keyed by the captured JSONL bytes; reading a growing
log never fabricates missing updates/diagnostics or treats announced exposure as
proof that its optimizer update completed.
"""
from collections import Counter, defaultdict
import csv, hashlib, io, json, math, sys
import numpy as np
from . import common as C

VERSION='full_contour_recorded_training_analysis_v1'
FIXED_WINDOWS=((1,1000),(1001,3000),(3001,6000))


def read_jsonl_snapshot(path):
    path=C.resolve(path)
    if not path.exists():return [],dict(path=str(path),exists=False,sha256=None,bytes=0,rows=0,incomplete_tail_bytes=0)
    content=path.read_bytes();lines=content.splitlines(keepends=True);rows=[];tail=0
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:row=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):
            if i==len(lines)-1 and not line.endswith((b'\n',b'\r')):tail=len(line);break
            raise ValueError('Malformed complete JSONL row: '+str(path)+' line '+str(i+1))
        rows.append(row)
    return rows,dict(path=str(path),exists=True,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),rows=len(rows),incomplete_tail_bytes=tail)


def csv_output(path,rows,empty_fields=('status',)):
    rows=list(rows)
    if rows:return C.csvwrite(path,rows)
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(empty_fields));writer.writeheader()
    return C._write_bytes(path,stream.getvalue().encode('utf-8-sig'))


def _finite(value,label):
    value=float(value)
    if not math.isfinite(value):raise ValueError('Nonfinite recorded '+label)
    return value


def _validate(logs,diagnostics,exposures,registration):
    batch=int(registration['batch']);limit=int(registration['updates_per_arm']);bins=registration['diagnostic_time_bin_edges']
    log_by={};diag_by=defaultdict(list);ex_by={};checks=0
    for row in logs:
        step=int(row['local_step'])
        if step in log_by or not 1<=step<=limit:raise ValueError('Duplicate/out-of-budget completed update')
        if int(row['global_step'])!=6000+step:raise ValueError('Completed update global clock differs')
        for key in ('FM','free_coordinate_denominator','update_seconds'):_finite(row[key],key)
        log_by[step]=row
    diag_ids=set()
    for row in diagnostics:
        step=int(row['local_step']);sid=int(row['sample_index']);t=float(row['t'])
        if sid in diag_ids or not 1<=step<=limit or not (step-1)*batch<=sid<step*batch:raise ValueError('Duplicate/out-of-budget diagnostic identity')
        diag_ids.add(sid)
        if int(row['global_step'])!=6000+step:raise ValueError('Diagnostic global clock differs')
        if not 0<=t<=1:raise ValueError('Recorded time outside unit interval')
        bi=min(len(bins)-2,int(np.searchsorted(bins,t,side='right')-1))
        if int(row['time_bin'])!=bi or float(row['time_bin_lower'])!=bins[bi] or float(row['time_bin_upper'])!=bins[bi+1]:raise ValueError('Recorded time bin differs from frozen bins')
        factor=float(np.float32(1)-np.float32(t))**2
        if not np.isclose(row['endpoint_error_factor'],factor,rtol=1e-12,atol=1e-15):raise ValueError('Endpoint diagnostic factor differs')
        for scope in ('interface','other'):
            count=int(row[scope+'_scalar_count']);sse=_finite(row[scope+'_velocity_sse'],scope+' velocity SSE')
            endpoint=_finite(row[scope+'_endpoint_scaled_sse'],scope+' endpoint SSE')
            if count<0 or sse<0 or endpoint<0:raise ValueError('Negative diagnostic count/SSE')
            if not np.isclose(endpoint,factor*sse,rtol=1e-12,atol=1e-12):raise ValueError('Endpoint scaling differs')
            if count:
                if not np.isclose(row[scope+'_velocity_mse'],sse/count,rtol=1e-12,atol=1e-12):raise ValueError('Recorded velocity MSE differs')
                if not np.isclose(row[scope+'_endpoint_scaled_mse'],endpoint/count,rtol=1e-12,atol=1e-12):raise ValueError('Recorded endpoint MSE differs')
            elif sse!=0 or endpoint!=0 or row[scope+'_velocity_mse'] is not None or row[scope+'_endpoint_scaled_mse'] is not None:raise ValueError('Missing coordinates must stay NA')
            checks+=4
        diag_by[step].append(row)
    for row in exposures:
        sid=int(row['sample_index']);step=int(row['local_step'])
        if sid in ex_by or not 1<=step<=limit or not (step-1)*batch<=sid<step*batch:raise ValueError('Duplicate/out-of-budget exposure identity')
        if row['condition_type']!='SEAT_FULL' or int(row['K'])<2:raise ValueError('Unexpected input family in recorded exposure')
        if int(row['global_step'])!=6000+step:raise ValueError('Exposure global clock differs')
        ex_by[sid]=row
    complete=[];incomplete=[]
    for step in sorted(set(log_by)|set(diag_by)):
        rows=diag_by.get(step,[]);why=[]
        if step not in log_by:why.append('NO_COMPLETED_UPDATE_IN_CAPTURED_TRAINING_LOG')
        if len(rows)!=batch:why.append('INCOMPLETE_DIAGNOSTIC_MICROBATCH_ROWS')
        if why:
            incomplete.append(dict(local_step=step,global_step=6000+step,diagnostic_rows=len(rows),expected_rows=batch,reasons=why));continue
        if {int(r['sample_index']) for r in rows}!=set(range((step-1)*batch,step*batch)):raise ValueError('Diagnostic microbatch indices differ')
        logged=log_by[step];den=sum(int(r['interface_scalar_count'])+int(r['other_scalar_count']) for r in rows)
        total=sum(float(r['interface_velocity_sse'])+float(r['other_velocity_sse']) for r in rows)
        if den!=int(logged['free_coordinate_denominator']):raise ValueError('Diagnostic denominator does not partition completed update')
        if not np.isclose(total/den,logged['FM'],rtol=2e-6,atol=1e-8):raise ValueError('Detached diagnostic differs from original FM')
        for row in rows:
            e=ex_by.get(int(row['sample_index']))
            if e is not None:
                if row['uid']!=e['uid'] or row['task_id']!=e['task_id']:raise ValueError('Exposure/diagnostic identity differs')
                if int(row['interface_scalar_count'])!=3*int(e['interface_corner_count']):raise ValueError('Source interface count differs')
                if int(row['interface_scalar_count'])+int(row['other_scalar_count'])!=int(e['free_coordinate_count']):raise ValueError('Free scalar count differs')
        complete.extend(sorted(rows,key=lambda r:r['sample_index']));checks+=3+len(rows)*3
    return log_by,complete,incomplete,ex_by,checks


def summarize(rows,log_by,lower,upper,time_bin='ALL',expected_end=None):
    selected=[r for r in rows if lower<=int(r['local_step'])<=upper and (time_bin=='ALL' or int(r['time_bin'])==time_bin)]
    steps=sorted({int(r['local_step']) for r in selected});end=upper if expected_end is None else expected_end
    complete_steps={int(r['local_step']) for r in rows if lower<=int(r['local_step'])<=upper}
    status='NA_NO_OBSERVATIONS' if not selected else ('COMPLETE' if len(complete_steps)==end-lower+1 else 'PARTIAL')
    result=dict(window_start=lower,window_end=end,observed_step_min=min(steps) if steps else None,observed_step_max=max(steps) if steps else None,
        observed_updates=len(steps),complete_diagnostic_updates_in_window=len(complete_steps),samples=len(selected),time_bin=time_bin,status=status)
    for scope in ('interface','other','all_free'):
        components=('interface','other') if scope=='all_free' else (scope,)
        count=sum(int(r[p+'_scalar_count']) for r in selected for p in components)
        velocity=math.fsum(float(r[p+'_velocity_sse']) for r in selected for p in components)
        endpoint=math.fsum(float(r[p+'_endpoint_scaled_sse']) for r in selected for p in components)
        result.update({scope+'_scalar_count':count,scope+'_velocity_sse':velocity,scope+'_endpoint_scaled_sse':endpoint,
            scope+'_velocity_mse':velocity/count if count else None,scope+'_endpoint_scaled_mse':endpoint/count if count else None,
            scope+'_status':'PASS' if count else 'NA_NO_OBSERVED_COORDINATES'})
    for kind in ('velocity','endpoint_scaled'):
        a=result['interface_'+kind+'_mse'];b=result['other_'+kind+'_mse']
        result['interface_over_other_'+kind+'_mse_ratio']=a/b if a is not None and b is not None and b>0 else None
    if time_bin=='ALL' and steps:
        den=math.fsum(float(log_by[s]['free_coordinate_denominator']) for s in steps)
        result.update(logged_FM_mean=float(np.mean([float(log_by[s]['FM']) for s in steps])),
            logged_FM_scalar_weighted_mean=math.fsum(float(log_by[s]['FM'])*float(log_by[s]['free_coordinate_denominator']) for s in steps)/den,
            mean_gradient_norm=float(np.mean([float(log_by[s]['gradient_norm']) for s in steps if log_by[s].get('gradient_norm') is not None])) if any(log_by[s].get('gradient_norm') is not None for s in steps) else None,
            update_seconds_sum=math.fsum(float(log_by[s]['update_seconds']) for s in steps))
    else:result.update(logged_FM_mean=None,logged_FM_scalar_weighted_mean=None,mean_gradient_norm=None,update_seconds_sum=None)
    result['interpretation']='Changing parent/time/noise/width training inputs; descriptive pooled SSE/count from existing ordinary forwards, not a fixed probe or rollout metric.'
    return result


def exposure_summary(exposures,log_by,complete_diagnostics,data,key):
    diag_steps={int(r['local_step']) for r in complete_diagnostics};groups=defaultdict(list)
    for r in exposures:groups[r[key]].append(r)
    registered=data['parent_ids'] if key=='uid' else data['train_task_ids'];rows=[]
    for identity in sorted(set(registered)|set(groups)):
        records=groups.get(identity,[]);completed=[r for r in records if int(r['local_step']) in log_by]
        diagnosed=[r for r in completed if int(r['local_step']) in diag_steps]
        steps=[int(r['local_step']) for r in completed]
        row=dict(identity=identity,identity_type='parent' if key=='uid' else 'task',registered=identity in registered,
            pre_update_exposure_records=len(records),completed_update_exposures=len(completed),complete_diagnostic_exposures=len(diagnosed),
            pending_or_unconfirmed_exposure_records=len(records)-len(completed),first_completed_step=min(steps) if steps else None,last_completed_step=max(steps) if steps else None,
            completed_clean_exposures=sum(float(r['alpha'])==1. for r in completed),completed_width_augmented_exposures=sum(float(r['alpha'])!=1. for r in completed),
            completed_free_scalar_exposure=sum(int(r['free_coordinate_count']) for r in completed),completed_interface_scalar_exposure=sum(3*int(r['interface_corner_count']) for r in completed),
            observed_N=sorted({int(r['N']) for r in completed}),observed_K=sorted({int(r['K']) for r in completed}),status='OBSERVED_COMPLETED_UPDATES' if completed else 'NA_NOT_YET_OBSERVED_COMPLETED_UPDATE',
            note='Exposure rows are appended before the update; only rows whose local_step appears in captured training_log are counted as completed-update exposures.')
        for lo,hi in FIXED_WINDOWS:row[f'completed_exposures_steps_{lo}_{hi}']=sum(lo<=int(r['local_step'])<=hi for r in completed)
        rows.append(row)
    return rows


def plot_curves(path,curves,last_logged,usable_updates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in curves if r['time_bin']=='ALL' and r['samples']]
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),dpi=145)
    for ax,kind,title in zip(axes,('velocity','endpoint_scaled'),('Velocity SSE / scalar count','(1-t)^2-scaled velocity SSE / scalar count')):
        for scope,color,label in [('interface','#c45524','Source interface corners'),('other','#326aab','Other free corners')]:
            x=[r['observed_step_max'] for r in rows if r[scope+'_'+kind+'_mse'] is not None]
            y=[r[scope+'_'+kind+'_mse'] for r in rows if r[scope+'_'+kind+'_mse'] is not None]
            if x:ax.plot(x,y,marker='o',markersize=3,color=color,label=label)
        ax.set_title(title);ax.set_xlabel('Observed local optimizer update');ax.set_ylabel('Pooled MSE, per 100-update window');ax.grid(alpha=.22)
        for boundary in (1000,3000):
            if last_logged>=boundary:ax.axvline(boundary,color='#888',linewidth=.8,linestyle='--')
        if rows:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'No complete recorded diagnostic update yet',ha='center',va='center',transform=ax.transAxes)
    fig.suptitle(f'CONTOUR recorded training diagnostics | logged update {last_logged}; usable updates {usable_updates}',fontsize=12)
    fig.text(.5,.025,'Changing training inputs: these are weighted ordinary-forward errors, not fixed probes or generated-mesh quality.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(C.checked(path));plt.close(fig)


def run():
    registration_path=C.OUT/'registration.json'
    if not registration_path.exists():return dict(status='NA_REGISTRATION_NOT_FROZEN',model_calls=0,training_updates=0)
    registration=C.read(registration_path)
    if registration.get('status')!='FROZEN':raise ValueError('Frozen training definition required for analysis')
    folder=C.OUT/'training'/C.TRAIN_LABEL
    logs,lmeta=read_jsonl_snapshot(folder/'training_log.jsonl')
    diagnostics,dmeta=read_jsonl_snapshot(folder/'interface_diagnostics.jsonl')
    exposures,emeta=read_jsonl_snapshot(folder/'exposure.jsonl')
    data=C.read(registration['interface_manifest_path']);bins=registration['diagnostic_time_bin_edges']
    log_by,complete,incomplete,ex_by,checks=_validate(logs,diagnostics,exposures,registration)
    last_logged=max(log_by,default=0);last_complete=max((int(r['local_step']) for r in complete),default=0)
    source=dict(training_log=lmeta,interface_diagnostics=dmeta,exposure=emeta,registration_sha256=C.sha(registration_path),analysis_code_sha256=C.sha(__file__))
    tag=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:12]
    out=C.OUT/'training_analysis'/f'snapshot_{last_logged:04d}_{last_complete:04d}_{tag}'
    final=out/'completion.json'
    if final.exists():
        receipt=C.read(final)
        for path,h in receipt['output_hashes'].items():assert C.sha(path)==h,path
        C.write(C.OUT/'training_analysis/latest.json',dict(path=str(final),sha256=C.sha(final),status=receipt['status'],last_logged_step=last_logged,last_complete_diagnostic_step=last_complete),replace=True)
        return receipt
    out.mkdir(parents=True,exist_ok=False)
    windows=[];bin_windows=[];curves=[]
    for lo,hi in FIXED_WINDOWS:
        windows.append(summarize(complete,log_by,lo,hi))
        for bi in range(len(bins)-1):
            row=summarize(complete,log_by,lo,hi,bi);row.update(time_bin_lower=bins[bi],time_bin_upper=bins[bi+1]);bin_windows.append(row)
    for lo in range(1,last_logged+1,100):
        hi=min(6000,lo+99)
        for bi in ['ALL',*range(len(bins)-1)]:
            row=summarize(complete,log_by,lo,hi,bi)
            row.update(time_bin_lower=bins[bi] if bi!='ALL' else None,time_bin_upper=bins[bi+1] if bi!='ALL' else None)
            curves.append(row)
    updates=[];complete_by_step=defaultdict(list)
    for item in complete:complete_by_step[int(item['local_step'])].append(item)
    for step,logged in sorted(log_by.items()):
        row=summarize(complete_by_step.get(step,[]),log_by,step,step)
        row.update(local_step=step,global_step=6000+step,recorded_FM=float(logged['FM']),recorded_free_coordinate_denominator=int(logged['free_coordinate_denominator']),
            recorded_gradient_norm=logged.get('gradient_norm'),recorded_update_seconds=logged.get('update_seconds'),recorded_input_seconds=logged.get('input_seconds'),
            recorded_cuda_peak_allocated_bytes=logged.get('cuda_peak_allocated_bytes'),recorded_cuda_peak_reserved_bytes=logged.get('cuda_peak_reserved_bytes'))
        updates.append(row)
    parent_rows=exposure_summary(exposures,log_by,complete,data,'uid');task_rows=exposure_summary(exposures,log_by,complete,data,'task_id')
    csv_output(out/'fixed_window_diagnostics.csv',windows)
    csv_output(out/'fixed_window_time_bins.csv',bin_windows)
    csv_output(out/'learning_curve_100_steps.csv',curves,('window_start','window_end','time_bin','status','samples','interface_velocity_mse','other_velocity_mse'))
    csv_output(out/'per_update.csv',updates,('local_step','global_step','status','recorded_FM'))
    csv_output(out/'parent_exposure.csv',parent_rows);csv_output(out/'task_exposure.csv',task_rows)
    csv_output(out/'incomplete_update_diagnostics.csv',incomplete,('local_step','global_step','diagnostic_rows','expected_rows','reasons'))
    plot_curves(out/'learning_curve_100_steps.png',curves,last_logged,len({int(r['local_step']) for r in complete}))
    C.write(out/'captured_sources.json',source)
    total=summarize(complete,log_by,1,6000)
    metadata=dict(status='COMPLETE' if len(log_by)==6000 and len(complete)==48000 else 'PARTIAL' if logs else 'NO_TRAINING_RECORDS_YET',version=VERSION,
        logged_updates=len(log_by),last_logged_step=last_logged,diagnostic_rows_observed=len(diagnostics),usable_complete_diagnostic_rows=len(complete),
        usable_complete_diagnostic_updates=len({int(r['local_step']) for r in complete}),last_complete_diagnostic_step=last_complete,
        incomplete_update_diagnostic_groups=len(incomplete),pre_update_exposure_records=len(exposures),
        completed_update_exposure_records=sum(int(r['local_step']) in log_by for r in exposures),
        expected_final_updates=6000,expected_final_samples=48000,frozen_time_bins=bins,record_consistency_checks=checks,
        weighted_total=total,source_snapshots=source,model_calls=0,additional_training_updates=0,new_quality_scans=0,
        limitations=['Inputs change across updates: parent, noise, time and width augmentation; this is not a fixed probe.',
            'MSE is pooled sum(SSE)/sum(scalar_count), not the arithmetic mean of per-example MSEs.',
            'Endpoint-scaled SSE is recorded (1-t)^2 times velocity SSE; not an additional rollout or independent endpoint forward.',
            'Only completed updates with all eight diagnostic rows enter SSE summaries; missing rows and incomplete JSONL tails remain explicit.',
            'Pre-update exposure rows alone do not establish a completed model/optimizer call. Completed exposure is cross-referenced to training_log.'],
        output_hashes={str(p):C.sha(p) for p in out.iterdir() if p.is_file()})
    C.write(final,metadata)
    C.write(C.OUT/'training_analysis/latest.json',dict(path=str(final),sha256=C.sha(final),status=metadata['status'],last_logged_step=last_logged,last_complete_diagnostic_step=last_complete),replace=True)
    if 'torch' in sys.modules:raise RuntimeError('Log-only analysis imported model framework')
    return metadata


if __name__=='__main__':
    result=run();print(json.dumps({k:result.get(k) for k in ('status','last_logged_step','usable_complete_diagnostic_updates','pre_update_exposure_records','record_consistency_checks','model_calls')},sort_keys=True))