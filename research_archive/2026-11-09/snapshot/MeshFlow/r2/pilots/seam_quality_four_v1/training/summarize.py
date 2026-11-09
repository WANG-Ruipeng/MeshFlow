"""CPU-only continuation curves and final ledger/state audit.

This reads saved training evidence. It never calls a model, optimizer, sampler
or scorer and never deletes rolling/final checkpoints. Historical SEAM200 is
shown once as shared history, not three independent replications.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import math
import time
import numpy as np
from . import support as S

PRIOR = S.ROOT/'experiments/r2_seam_slot_c200_v1'
HISTORY = 'SEAM200_SHARED_HISTORY'
KINDS = {'train_forward':8,'train_backward':8,'head_forward':8,'head_backward':8,
         'optimizer_update':1,'head_optimizer_update':1}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read_lines(path, *, live=False):
    if not Path(path).exists(): return []
    content=Path(path).read_text(encoding='utf-8')
    rows=[]; lines=content.splitlines()
    for i,line in enumerate(lines):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:
            if live and i==len(lines)-1 and not content.endswith('\n'): break
            raise
    return rows


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2)


def csvwrite(path, rows):
    rows=list(rows)
    if not rows: return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def near(actual, expected, name, *, atol=1.e-7, rtol=5.e-6):
    if not np.isclose(actual,expected,rtol=rtol,atol=atol):
        raise ValueError(f'{name} differs: {actual} versus {expected}')


def audit_calls(rows, count, base_step, *, history=False):
    begin={};done={};errors=[]
    for row in rows:
        step=int(row.get('context',{}).get('local_step',0))
        if not 1<=step<=count:
            if history and row.get('kind')=='preflight_forward': continue
            # During a live prefix audit a later update is expected to run.
            if step>count: continue
            if row.get('event')=='CALL_BEGIN': raise ValueError('Unbudgeted call outside update sequence')
            continue
        call=int(row['call_id'])
        if row['event']=='CALL_BEGIN':
            if call in begin: raise ValueError('Repeated call ID')
            begin[call]=row
        elif row['event']=='CALL_DONE':
            if call in done: raise ValueError('Repeated completion ID')
            done[call]=row
        else: errors.append(row)
    if errors or set(begin)!=set(done): raise ValueError('Failed/pending calls inside completed update prefix')
    observed=Counter(row['kind'] for row in begin.values())
    expected={kind:multiplier*count for kind,multiplier in KINDS.items()}
    if dict(observed)!=expected: raise ValueError(f'Call count differs: {observed} versus {expected}')
    updates=[row['context']['global_step'] for row in begin.values() if row['kind']=='optimizer_update']
    if updates!=list(range(base_step+1,base_step+count+1)):
        raise ValueError('Repeated/gapped optimizer sequence')
    for call,row in begin.items():
        final=done[call]
        if (row['kind'],row['context'])!=(final['kind'],final['context']):
            raise ValueError('Call begin/end identity differs')
    return dict(status='PASS',completed_updates=count,base_global_step=base_step,
        calls=dict(observed),pending=0,failed=0,extra_preflight=2 if history else 0,
        seconds_by_kind={kind:sum(float(r['seconds']) for r in done.values() if r['kind']==kind) for kind in KINDS})


def first_step(root):
    out=root/'training_analysis/first_step_check.json'
    if out.exists(): return S.read(out)
    arms={}
    for arm in S.ARMS:
        folder=root/'training'/arm
        logs=read_lines(folder/'training_log.jsonl',live=True)
        if not logs: arms[arm]={'status':'NOT_YET_COMPLETED_FIRST_UPDATE'};continue
        first=logs[0]
        if first['local_step']!=1 or first['global_step']!=12201:
            raise ValueError('Wrong first continuation update')
        calls=audit_calls(read_lines(folder/'calls.jsonl',live=True),1,12200)
        backward=S.read(folder/'first_formal_backward.json')
        if backward['status']!='PASS' or backward['no_extra_forward'] is not True:
            raise ValueError('First real backward check missing')
        diag=[r for r in read_lines(folder/'geometry_diagnostics.jsonl',live=True) if r['local_step']==1]
        if len(diag)!=8: raise ValueError('First-step diagnostics not complete')
        near(sum(r['FM_contribution'] for r in diag),first['FM'],'first-step FM')
        arms[arm]=dict(status='PASS',calls=calls,FM=first['FM'],BCE=first['occupancy_BCE'],
            generator_gradient_norm=first['generator_gradient_norm'],
            peak_allocated=first['cuda_peak_allocated_bytes'],peak_reserved=first['cuda_peak_reserved_bytes'])
    result=dict(status='PASS_COMPLETED_PREFIX_ONLY',arms=arms,model_calls_by_this_audit=0,
                source='live JSONL snapshot; later updates excluded, no claim of completed800')
    write(out,result);return result


def load_arm(root, arm, count, base_step, *, history=False):
    folder=root/'training'/('SEAM_SLOT_C200' if history else arm)
    paths={name:folder/(name+'.jsonl') for name in
        ('training_log','geometry_diagnostics','occupancy_training','exposure','calls')}
    logs={name:read_lines(path) for name,path in paths.items()}
    bystep={}
    for row in logs['training_log']:
        step=int(row['local_step'])
        if step in bystep: raise ValueError('Repeated logged update')
        bystep[step]=row
    if list(bystep)!=list(range(1,count+1)): raise ValueError('Incomplete update curve: '+arm)
    facts=audit_calls(logs['calls'],count,base_step,history=history)
    # A final audit must also reject any calls beyond its fixed endpoint.
    extra=[r for r in logs['calls'] if r['event']=='CALL_BEGIN' and int(r.get('context',{}).get('local_step',0))>count]
    if extra: raise ValueError('Budget exceeded after final step')
    grouped={name:defaultdict(list) for name in ('geometry_diagnostics','occupancy_training','exposure')}
    for name in grouped:
        for row in logs[name]: grouped[name][int(row['local_step'])].append(row)
        if set(grouped[name])!=set(bystep) or any(len(x)!=8 for x in grouped[name].values()):
            raise ValueError('Incomplete/duplicate microbatch evidence: '+name)
        for rows in grouped[name].values():
            if sorted(r['microbatch'] for r in rows)!=list(range(8)): raise ValueError('Repeated microbatch identity')
    curve=[]; parent_bins=defaultdict(list); head_edges=[]; exposure_identity=[]
    for step,log in bystep.items():
        diagnostics=grouped['geometry_diagnostics'][step]
        heads=grouped['occupancy_training'][step]
        exposure=sorted(grouped['exposure'][step],key=lambda r:r['microbatch'])
        if any(r['global_step']!=base_step+step for r in [log,*diagnostics,*heads,*exposure]):
            raise ValueError('Global/local update identity mismatch')
        denominator=sum(r['sample_free_coordinate_denominator'] for r in diagnostics)
        if denominator!=log['free_coordinate_denominator']: raise ValueError('Wrong batch FM scalar count')
        near(sum(r['FM_contribution'] for r in diagnostics),log['FM'],'per-update FM sum')
        near(sum(r['all_free_velocity_sse'] for r in diagnostics)/denominator,log['FM'],'FM reconstructed from coordinates')
        row=dict(arm=arm,local_step=step,seam_total_step=step if history else step+200,
            global_step=base_step+step,history_reused=history,FM=log['FM'],
            head_BCE=log['occupancy_BCE'],free_scalar_denominator=denominator,
            generator_gradient_norm=log['generator_gradient_norm'],head_gradient_norm=log['head_gradient_norm'],
            update_seconds=log['update_seconds'],peak_allocated_bytes=log['cuda_peak_allocated_bytes'],
            peak_reserved_bytes=log['cuda_peak_reserved_bytes'])
        for label in ('source_interface_third','source_interface_other_free','bulk'):
            scalar_count=sum(r[label+'_scalar_count'] for r in diagnostics)
            sse=sum(r[label+'_velocity_sse'] for r in diagnostics)
            endpoint_sse=sum(r[label+'_endpoint_sse'] for r in diagnostics)
            row.update({label+'_scalar_count':scalar_count,label+'_velocity_sse':sse,
                label+'_velocity_mse':sse/scalar_count if scalar_count else None,
                label+'_endpoint_mse':endpoint_sse/scalar_count if scalar_count else None})
        near(sum(row[label+'_velocity_sse'] for label in ('source_interface_third','source_interface_other_free','bulk'))/denominator,
             log['FM'],'FM partition')
        for key in ('true_positive','true_negative','false_positive','false_negative'):
            row['head_'+key]=sum(r[key] for r in heads)
        near(sum(r['weighted_BCE_contribution'] for r in heads),log['occupancy_BCE'],'BCE sum')
        for h in heads:
            logits=np.asarray(h['logits'],float);labels=np.asarray(h['labels'],bool);pred=logits>=0
            computed=dict(true_positive=int((pred&labels).sum()),true_negative=int((~pred&~labels).sum()),
                          false_positive=int((pred&~labels).sum()),false_negative=int((~pred&labels).sum()))
            if any(computed[k]!=h[k] for k in computed): raise ValueError('Head confusion differs')
            if h['edge_batch_denominator']!=132: raise ValueError('Occupancy denominator changed')
            bce=(np.logaddexp(0,logits)-labels*logits)*np.where(labels,66/124,8.25)
            near(float(bce.sum()/132),h['weighted_BCE_contribution'],'head BCE reconstruction')
            head_edges.append(dict(arm=arm,seam_total_step=row['seam_total_step'],global_step=row['global_step'],
                uid=h['uid'],alpha=h['alpha'],microbatch=h['microbatch'],edge_count=len(labels),**computed,
                logit_min=float(logits.min()),logit_max=float(logits.max()),layout_match=bool(np.array_equal(pred,labels))))
        if history:
            row.update(anti_cover=None,anti_cover_active_points=None,anti_cover_velocity_gradient_norm_sum=None)
        else:
            row.update({key:log[key] for key in ('anti_cover','anti_cover_active_points','anti_cover_velocity_gradient_norm_sum')})
            if arm=='ANTI_COVER':
                anti=[r['anti_cover'] for r in diagnostics]
                if any(a['denominator']!=248 for a in anti): raise ValueError('Anti-cover denominator changed')
                near(sum(a['raw_hinge_sum']/248 if a['time_active'] else 0. for a in anti),log['anti_cover'],'anti-cover loss sum')
                active=sum(a['active_points'] for a in anti if a['time_active'])
                if active!=log['anti_cover_active_points']: raise ValueError('Anti-cover active count mismatch')
                near(sum(a['velocity_gradient_l2_analytic'] for a in anti),log['anti_cover_velocity_gradient_norm_sum'],'analytic aux-gradient sum')
                row['anti_cover_over_FM']=log['anti_cover']/log['FM'] if log['FM'] else None
                row['anti_cover_time_active_microbatches']=sum(a['time_active'] for a in anti)
                row['anti_cover_nonzero_microbatches']=sum(a['time_active'] and a['raw_hinge_sum']>0 for a in anti)
                row['anti_cover_nonzero_gradient_microbatches']=sum(a['velocity_gradient_l2_analytic']>0 for a in anti)
                row['anti_cover_max_microbatch_loss']=max(a['raw_hinge_sum']/248 if a['time_active'] else 0. for a in anti)
                row['anti_cover_max_microbatch_velocity_gradient']=max(a['velocity_gradient_l2_analytic'] for a in anti)
                row['anti_cover_geometry_gated_points']=sum(a['gated_points'] for a in anti)
                row['anti_cover_degenerate_seams']=sum(a['degenerate_seams'] for a in anti)
            elif log['anti_cover'] or log['anti_cover_active_points'] or log['anti_cover_velocity_gradient_norm_sum']:
                raise ValueError('Control arm received auxiliary loss')
        for d in diagnostics:
            parent_bins[(arm,d['uid'],d['time_bin'])].append(d)
        for r in exposure:
            exposure_identity.append({key:r[key] for key in ('local_step','global_step','microbatch','uid','sample_index','alpha','t','N','K','free_coordinate_count','path')})
        curve.append(row)
    facts['input_files']={str(path):sha(path) for path in paths.values()}
    return curve,facts,parent_bins,head_edges,exposure_identity


def bins_rows(groups):
    rows=[]
    for (arm,uid,time_bin),group in sorted(groups.items()):
        out=dict(arm=arm,uid=uid,time_bin=time_bin,samples=len(group),
            t_min=group[0]['time_bin_lower'],t_max=group[0]['time_bin_upper'])
        for label in ('source_interface_third','bulk'):
            count=sum(r[label+'_scalar_count'] for r in group)
            out[label+'_scalar_count']=count
            for stage in ('velocity','endpoint'):
                sse=sum(r[label+'_'+stage+'_sse'] for r in group)
                out[label+'_'+stage+'_mse']=sse/count if count else None
        rows.append(out)
    return rows


def final_states(root):
    # CPU mmap only. No model object is constructed, no CUDA context opened.
    from meshflow_control import runtime as _runtime
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.training.schedule40 import tree_hash
    rows=[];hashes=[]
    for arm in S.ARMS:
        folder=root/'training'/arm;status=S.read(folder/'status.json')
        if status['status']!='COMPLETE' or (status['local_step'],status['global_step'])!=(800,13000):
            raise ValueError('Final continuation not complete: '+arm)
        receipts=[status['checkpoint'],status['inference_export'],status['occupancy_export']]
        for receipt in receipts:
            if sha(S.resolve(receipt['path']))!=receipt['sha256']: raise ValueError('Final weight bytes changed')
        checkpoint=torch.load(S.resolve(status['checkpoint']['path']),map_location='cpu',weights_only=True,mmap=True)
        if (checkpoint['schema'],checkpoint['arm'],checkpoint['local_step'],checkpoint['seam_total_step'],checkpoint['global_step'])!=(
            'r2_seam_quality_four_training_v1',arm,800,1000,13000): raise ValueError('Checkpoint identity differs')
        for key,func,expected in [('generator_state',state_hash,'generator_state_sha256'),
            ('optimizer',tree_hash,'optimizer_state_sha256'),('rng',tree_hash,'rng_sha256'),
            ('head_state',state_hash,'head_state_sha256'),('head_optimizer',tree_hash,'head_optimizer_state_sha256')]:
            if func(checkpoint[key])!=checkpoint[expected]: raise ValueError('Internal checkpoint state hash differs: '+key)
        generator_steps={float(v['step']) for v in checkpoint['optimizer']['state'].values()}
        head_steps={float(v['step']) for v in checkpoint['head_optimizer']['state'].values()}
        if generator_steps!={13000.} or head_steps!={1000.}: raise ValueError('Optimizer cursor differs')
        head_export=torch.load(S.resolve(status['occupancy_export']['path']),map_location='cpu',weights_only=True)
        if state_hash(head_export['head_state'])!=checkpoint['head_state_sha256']:
            raise ValueError('Export head differs from final live state')
        hashes.append(checkpoint['head_state_sha256'])
        rows.append(dict(arm=arm,status='PASS',generator_steps=sorted(generator_steps),head_steps=sorted(head_steps),
            generator_state_sha256=checkpoint['generator_state_sha256'],head_state_sha256=hashes[-1],
            head_optimizer_state_sha256=checkpoint['head_optimizer_state_sha256'],rng_sha256=checkpoint['rng_sha256'],
            optimizer_parameter_count=len(checkpoint['optimizer']['state']),head_parameter_tensors=len(checkpoint['head_optimizer']['state']),
            checkpoint_sha256=status['checkpoint']['sha256']))
        del checkpoint,head_export
    if len(set(hashes))!=1: raise ValueError('Independent head final states unexpectedly diverged across identical head streams')
    if torch.cuda.is_initialized(): raise RuntimeError('CPU audit initialized CUDA')
    return rows


def plot(out, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    keys=[('FM','Full free-coordinate FM'),('source_interface_third_velocity_mse','Seam third-point velocity MSE'),
          ('bulk_velocity_mse','Bulk velocity MSE'),('head_BCE','Occupancy weighted BCE'),
          ('anti_cover','Normalized anti-cover loss'),('anti_cover_velocity_gradient_norm_sum','Auxiliary output-gradient norm sum')]
    colors={HISTORY:'0.35','LONG':'#3f72af','ANTI_COVER':'#dd6633','LOCAL_NOISE':'#3e9960'}
    for ax,(key,title) in zip(axes.flat,keys):
        for arm in (HISTORY,*S.ARMS):
            group=[r for r in curves if r['arm']==arm and r.get(key) is not None]
            if not group:continue
            # Non-overlapping50-update means; raw every-update values in CSV.
            windows=[group[i:i+50] for i in range(0,len(group),50)]
            ax.plot([g[-1]['seam_total_step'] for g in windows],
                    [np.mean([r[key] for r in g]) for g in windows],label=arm,color=colors[arm],linewidth=1.8)
        ax.axvline(200,color='0.6',linestyle='--',linewidth=.8);ax.set_title(title)
        ax.set_xlabel('Total SEAM updates (shared history1..200)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Training interpolation diagnostics; not rollout quality. 50-update means.')
    fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)


def summarize(root):
    out=root/'training_analysis/final'
    complete=out/'completion.json'
    if complete.exists():
        value=S.read(complete)
        for path,expected in value['output_hashes'].items():
            if sha(out/path)!=expected: raise ValueError('Completed analysis output changed')
        return value
    if out.exists(): raise FileExistsError('Incomplete analysis directory exists; inspect before retry')
    for arm in S.ARMS:
        if S.read(root/'training'/arm/'status.json')['status']!='COMPLETE':
            raise ValueError('All three frozen final endpoints required before full audit')
    out.mkdir(parents=True);started=time.perf_counter()
    try:
        curves=[];audits=[];heads=[];groups={};exposures={}
        for directory,arm,count,base_step,history in [(PRIOR,HISTORY,200,12000,True)]+[(root,a,800,12200,False) for a in S.ARMS]:
            c,a,g,h,e=load_arm(directory,arm,count,base_step,history=history)
            curves.extend(c);audits.append(dict(arm=arm,**a));groups.update(g);heads.extend(h)
            if not history:exposures[arm]=e
        if not exposures['LONG']==exposures['ANTI_COVER']==exposures['LOCAL_NOISE']:
            raise ValueError('Arm exposures/randomstream identity differ')
        states=final_states(root)
        anti_rows=[r for r in curves if r['arm']=='ANTI_COVER']
        anti_activity=dict(updates=len(anti_rows),microbatches=len(anti_rows)*8,
            nonzero_loss_updates=sum(r['anti_cover']>0 for r in anti_rows),
            nonzero_gradient_updates=sum(r['anti_cover_velocity_gradient_norm_sum']>0 for r in anti_rows),
            nonzero_loss_microbatches=sum(r['anti_cover_nonzero_microbatches'] for r in anti_rows),
            nonzero_gradient_microbatches=sum(r['anti_cover_nonzero_gradient_microbatches'] for r in anti_rows),
            time_active_microbatches=sum(r['anti_cover_time_active_microbatches'] for r in anti_rows),
            active_points=sum(r['anti_cover_active_points'] for r in anti_rows),
            max_update_loss=max(r['anti_cover'] for r in anti_rows),
            max_microbatch_loss=max(r['anti_cover_max_microbatch_loss'] for r in anti_rows),
            max_microbatch_velocity_gradient=max(r['anti_cover_max_microbatch_velocity_gradient'] for r in anti_rows),
            summed_update_loss=sum(r['anti_cover'] for r in anti_rows),
            frozen_definition_unchanged=True,
            note='Zeros stay zeros; no threshold/weight/step adjustment. Output-gradient norm is not parameter-gradient norm.')
        inventory=[]
        for path in sorted((root/'checkpoints').glob('*.pt')):
            receipt=path.with_suffix('.json')
            header=S.read(receipt) if receipt.exists() else {}
            inventory.append(dict(path=str(path),bytes=path.stat().st_size,kind='rolling' if 'rolling' in path.name else 'final',
                local_step=header.get('local_step'),global_step=header.get('global_step'),sha256=sha(path),deleted=False))
        csvwrite(out/'curve_per_update.csv',curves);csvwrite(out/'per_parent_time_bin.csv',bins_rows(groups))
        csvwrite(out/'head_per_microbatch.csv',heads);csvwrite(out/'checkpoint_inventory.csv',inventory)
        write(out/'call_audit.json',audits);write(out/'final_state_audit.json',states)
        write(out/'anti_cover_activity.json',anti_activity)
        write(out/'exposure_audit.json',dict(status='PASS',all_three_equal=True,count_per_arm=6400,
            parent_counts=dict(Counter(r['uid'] for r in exposures['LONG'])),
            identity_fields=list(exposures['LONG'][0]),
            identity_sha256=hashlib.sha256(json.dumps(exposures['LONG'],sort_keys=True).encode()).hexdigest()))
        summaries=[]
        for arm in (HISTORY,*S.ARMS):
            rows=[r for r in curves if r['arm']==arm]
            for name,selection in [('all',rows),('first50',rows[:50]),('last50',rows[-50:])]:
                summary=dict(arm=arm,window=name,updates=len(selection))
                for key in ('FM','head_BCE','source_interface_third_velocity_mse','bulk_velocity_mse','anti_cover',
                            'anti_cover_active_points','anti_cover_velocity_gradient_norm_sum','generator_gradient_norm'):
                    values=[r[key] for r in selection if r.get(key) is not None]
                    summary[key]=float(np.mean(values)) if values else None
                summaries.append(summary)
        csvwrite(out/'training_window_summary.csv',summaries);plot(out,curves)
        result=dict(status='PASS',new_updates=2400,new_generator_forwards=19200,new_generator_backwards=19200,
            new_head_forwards=19200,new_head_backwards=19200,new_head_updates=2400,
            shared_historical_updates=200,history_included_once=True,extra_new_preflight=0,
            all_three_final_head_hashes_equal=True,all_three_exposure_identities_equal=True,
            ANTI_and_LONG_final_generator_hash_equal=(states[0]['generator_state_sha256']==states[1]['generator_state_sha256']),
            model_calls_by_this_audit=0,optimizer_updates_by_this_audit=0,deleted_weights=0,
            checkpoint_inventory_bytes=sum(r['bytes'] for r in inventory),seconds=time.perf_counter()-started,
            limits='Training interpolation curves do not establish rollout quality or convergence; auxiliary gradient is wrt predicted velocity, not parameter-gradient norm',
            summarizer_sha256=sha(__file__),output_hashes={str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()})
        write(complete,result);return result
    except Exception as exc:
        write(out/'failure.json',dict(status='FAILED_CPU_ANALYSIS',error=repr(exc),model_calls=0,optimizer_updates=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/r2_seam_quality_four_v1')
    parser.add_argument('--partial',action='store_true');args=parser.parse_args();root=S.resolve(args.root).resolve()
    result=first_step(root) if args.partial else summarize(root)
    print(json.dumps(result,ensure_ascii=False,indent=2)[:6000])
