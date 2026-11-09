"""Final read-only training audit and fixed-window curves; zero model calls.

Run after both fixed1000 arms complete. Reuses only recorded normal-forward
diagnostics and loads checkpoints on CPU without constructing a neural model.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,collections,csv,json,math,statistics,sys,time
import numpy as np
from . import training_data as D
from r2.pilots.seam_quality_wave1_wave2_v1 import training_summary as OLD_SUM

BASELINE='PLAIN_S2000_WINDOW'
ARMS=(BASELINE,*D.ARMS)
CATEGORIES=OLD_SUM.CATEGORIES

def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:return [json.loads(x) for x in f if x.strip()]

def same_path(left,right):
    # Exact strings identify the same path without redundant WSL filesystem
    # metadata requests. Different spellings retain the original canonical gate.
    return str(left)==str(right) or OLD_SUM.canonical(left)==OLD_SUM.canonical(right)

def normalized_rows(path,baseline=False):
    result=[]
    for row in jsonl(path):
        original=int(row['local_step'])
        if baseline and not 1001<=original<=2000:continue
        if not baseline and not 1<=original<=1000:raise ValueError('Unbudgeted training log row')
        row=dict(row);row['source_continuation_step']=original if baseline else 1000+original
        row['local_step']=original-1000 if baseline else original
        if row['global_step']!=14000+row['local_step']:raise ValueError('Wrong global training cursor')
        # Actual training diagnostics already provide a disjoint free mask
        # partition. Derive only this missing convenience field, no new score.
        if 'all_free_velocity_sse' in row and 'sample_free_coordinate_denominator' not in row:
            row['sample_free_coordinate_denominator']=sum(int(row[k+'_scalar_count']) for k in
                ('source_interface_third','source_interface_other_free','bulk'))
        result.append(row)
    return result

def load_data(root):
    folders={BASELINE:D.ROOT/'experiments/r2_seam_pool47_couple_v1/training/PLAIN3000',
        **{a:root/'training'/a for a in D.ARMS}}
    data={};inputs=[]
    for arm,folder in folders.items():
        paths=[folder/n for n in ('training_log.jsonl','geometry_diagnostics.jsonl','exposure.jsonl')]
        logs,diags,exposure=[normalized_rows(p,arm==BASELINE) for p in paths]
        if [r['local_step'] for r in logs]!=list(range(1,1001)):raise ValueError('Exactly1000 updates required '+arm)
        expected=[(s,m) for s in range(1,1001) for m in range(8)]
        for records in (diags,exposure):
            if [(r['local_step'],r['microbatch']) for r in records]!=expected:raise ValueError('Exactly8000 ordered samples required '+arm)
        inputs+=paths;data[arm]=dict(folder=folder,logs=logs,diags=diags,exposure=exposure)
    return data,inputs

def pair_inputs(data,manifest):
    rows=[];counter=collections.Counter()
    for i,base in enumerate(data[BASELINE]['exposure']):
        step,micro=base['local_step'],base['microbatch'];cache=manifest['steps'][step-1]['samples'][micro]
        if not same_path(base['path'],cache['path']) or base['uid']!=cache['uid']:raise ValueError('Baseline stream identity differs')
        for arm in D.ARMS:
            row=data[arm]['exposure'][i]
            for key in ('uid','N','K','alpha','t','sample_index'):
                if row[key]!=base[key]:raise ValueError('Paired exposure differs '+arm+' '+key)
            if not same_path(row['path'],base['path']) or row['input_sha256']!=cache['sha256']:raise ValueError('Cache hash identity differs')
            diag=data[arm]['diags'][i]
            if diag['sample_free_coordinate_denominator']!=base['free_coordinate_count']:raise ValueError('Free scalar counts differ')
        counter[base['uid']]+=1
        rows.append(dict(local_step=step,source_continuation_step=1000+step,global_step=14000+step,microbatch=micro,
            parent_id=base['uid'],N=base['N'],K=base['K'],alpha=base['alpha'],t=base['t'],source_sample_index=base['sample_index'],
            cache_path=base['path'],cache_sha256=cache['sha256'],all_three_base_random_identities_equal=True,
            CLEAN_path_same_as_PLAIN=True,SMOOTH_path='same underlying Gaussian IDs, bulkT.25 on Z/epsilon; cachedOT retained'))
    if dict(sorted(counter.items()))!=manifest['parent_exposures']:raise ValueError('Frozen perparent exposures differ')
    return rows,counter

def checkpoint_audit(root,reg,data):
    sys.path.insert(0,str(D.ROOT/'r2/src'))
    import torch
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.training.schedule40 import tree_hash
    from meshflow_control.config import MODEL_CONFIG
    if torch.cuda.is_initialized():raise RuntimeError('Checkpoint audit must stay CPU only')
    result={};inputs=[]
    expected_calls=dict(train_forward=8000,train_backward=8000,optimizer_update=1000)
    for arm in D.ARMS:
        folder=data[arm]['folder'];status=D.read(folder/'status.json');inputs.append(folder/'status.json')
        if (status['status'],status['local_step'],status['global_step'])!=('COMPLETE',1000,15000):raise ValueError('Incomplete fixed arm '+arm)
        calls=status['calls']
        if calls['attempted']!=expected_calls or calls['successful']!=expected_calls or calls['failed'] or calls['pending_call_ids']:raise ValueError('Formal calls differ '+arm)
        rawcalls=jsonl(folder/'calls.jsonl');inputs.append(folder/'calls.jsonl')
        begins=[r for r in rawcalls if r['event']=='CALL_BEGIN'];success=[r for r in rawcalls if r['event']=='CALL_DONE']
        # Ledger snapshot and physical CALL_BEGIN sequence independently agree.
        counts=collections.Counter(r['kind'] for r in begins)
        if dict(counts)!=expected_calls:raise ValueError('Physical call ledger differs '+arm)
        if dict(collections.Counter(r['kind'] for r in success))!=expected_calls:raise ValueError('Physical completed calls differ '+arm)
        call_identity=lambda r:(r['attempt_id'],r['call_id'],r['kind'])
        if len(set(map(call_identity,begins)))!=17000 or set(map(call_identity,begins))!=set(map(call_identity,success)):
            raise ValueError('Duplicate/unresolved physical call identity')
        if any(r['event'] not in ('CALL_BEGIN','CALL_DONE') for r in rawcalls):raise ValueError('Unexpected failed/other call event')
        seq=[r['context']['global_step'] for r in begins if r['kind']=='optimizer_update']
        if seq!=list(range(14001,15001)):raise ValueError('Replayed/gapped update')
        initpaths=sorted((folder/'attempts').glob('*/initialization.json'));inputs+=initpaths
        initial=[D.read(p) for p in initpaths if D.read(p)['global_step']==14000]
        if len(initial)!=1:raise ValueError('Exactlyone original restore required')
        initial=initial[0]
        if initial['generator_state_sha256']!=reg['source_generator_sha256'] or initial['rng_sha256']!=reg['source_rng_sha256']:raise ValueError('Initial state differs')
        restored=initial['optimizer_restore']
        if restored['actual_optimizer_sha256']!=reg['source_optimizer_sha256'] or not restored['exact_retained_step_and_moments'] or restored['removed_frozen_parameter_states']!=0:raise ValueError('Matched AdamW restore failed')
        ck=status['checkpoint'];exp=status['inference_export']
        for value in (ck,exp):
            if D.digest(value['path'])!=value['sha256']:raise ValueError('Output file hash changed')
        payload=torch.load(D.resolve(ck['path']),map_location='cpu',weights_only=True,mmap=True)
        if (payload['arm'],payload['global_step'],payload['local_step'],payload['update_in_progress'],payload['EMA'])!=(arm,15000,1000,False,False):raise ValueError('Final live cursor differs')
        if payload['model_config']!=MODEL_CONFIG or payload['registration_sha256']!=D.digest(root/'training_registration.json'):raise ValueError('Final model contract differs')
        for key,fn,hkey in (('generator_state',state_hash,'generator_state_sha256'),('optimizer',tree_hash,'optimizer_state_sha256'),('rng',tree_hash,'rng_sha256')):
            if fn(payload[key])!=payload[hkey] or payload[hkey]!=ck[hkey]:raise ValueError('Final internal state hash failed '+key)
        generator=payload['generator_state'];opt=payload['optimizer'];names=payload['optimizer_parameter_names']
        if len(generator)!=194 or len(opt['state'])!=194 or len(opt['param_groups'])!=1 or len(names)!=1:raise ValueError('Final parameter count differs')
        expected_names=['generator.'+n for n in generator]
        if names[0]!=expected_names:raise ValueError('Named optimizer identity differs')
        steps=sorted({float(s['step']) for s in opt['state'].values()})
        if steps!=[15000.]:raise ValueError('Final AdamW steps differ')
        for name,index in zip(names[0],opt['param_groups'][0]['params']):
            value=generator[name[len('generator.'):]];entry=opt['state'][index]
            if value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite/F32 model')
            for key in ('exp_avg','exp_avg_sq'):
                if entry[key].dtype!=torch.float32 or entry[key].shape!=value.shape or not bool(torch.isfinite(entry[key]).all()):raise ValueError('Invalid AdamW moment')
        exported=torch.load(D.resolve(exp['path']),map_location='cpu',weights_only=True,mmap=True)
        if exported['training_checkpoint_sha256']!=ck['sha256'] or exported['generator_state_sha256']!=payload['generator_state_sha256'] or state_hash(exported['generator_state'])!=payload['generator_state_sha256']:raise ValueError('Inference tensor export differs')
        if exported['global_step']!=15000 or exported['prior_rho']!=(.25 if arm=='SMOOTH_PRIOR' else 1.):raise ValueError('Export prior/cursor differs')
        result[arm]=dict(status='PASS',calls=calls,initial_generator_sha256=initial['generator_state_sha256'],
            initial_optimizer_sha256=restored['actual_optimizer_sha256'],initial_rng_sha256=initial['rng_sha256'],
            checkpoint=ck,inference_export=exp,final_internal_hashes={k:payload[k] for k in ('generator_state_sha256','optimizer_state_sha256','rng_sha256')},
            optimizer_states=194,optimizer_steps=steps,inference_equals_live_generator=True,model_constructions=0,neural_forwards=0,CUDA_initialized=False)
        del payload,exported,generator,opt
    if torch.cuda.is_initialized():raise RuntimeError('CPU audit unexpectedly initializedCUDA')
    return result,inputs

def objective(arm):
    return 'CLEAN weightedFM + .1 endpoint/vector edge loss' if arm=='CLEAN_EDGE_COH' else 'UnweightedFM, bulkT.25 altered prior and velocity target' if arm=='SMOOTH_PRIOR' else 'Original unweightedFM'

def plot(out,windows,lossrows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={BASELINE:'#666666','CLEAN_EDGE_COH':'#0072B2','SMOOTH_PRIOR':'#D55E00'}
    fig,axes=plt.subplots(2,3,figsize=(14,7.7),constrained_layout=True)
    for col,arm in enumerate(ARMS):
        ax=axes[0,col]
        for name,style in [('all_free','-'),('seam_third','--'),('bulk_total',':')]:
            sel=[r for r in windows if r['arm']==arm and r['category']==name and r['status']=='PASS']
            ax.plot([r['end_step'] for r in sel],[r['unweighted_velocity_mse'] for r in sel],style,label=name)
        ax.set_title(arm);ax.set_xlabel('New update, 100-update windows');ax.set_ylabel('Training-state velocity MSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax=axes[1,col];sel=[r for r in lossrows if r['arm']==arm]
        ax.plot([r['end_step'] for r in sel],[r['mean_FM'] for r in sel],label='Optimized FM',color=colors[arm])
        if arm=='CLEAN_EDGE_COH':ax.plot([r['end_step'] for r in sel],[r['mean_edge_loss'] for r in sel],label='Edge loss before lambda',color='#009E73')
        ax.set_ylabel('Own objective terms');ax.set_xlabel('New update');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same 8000 registered exposures; different objectives / SMOOTH prior. Training fit is not rollout quality.',fontsize=11)
    fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.svg');plt.close(fig)

def run(root):
    started=time.perf_counter();root=D.resolve(root).resolve();out=root/'training_summary'
    if out.exists():raise FileExistsError('Immutable summary exists; no overwrite')
    for arm in D.ARMS:
        if D.read(root/'training'/arm/'status.json')['status']!='COMPLETE':raise RuntimeError('Wait for both fixed arms')
    reg=D.read(root/'training_registration.json');manifest=D.read(D.resolve(reg['shared_data_manifest']))
    for path,sha in reg['protected_hashes'].items():
        if D.digest(path)!=sha:raise ValueError('Frozen source/input/training code changed '+path)
    data,inputs=load_data(root);pairing,exposure=pair_inputs(data,manifest)
    evidence,more=checkpoint_audit(root,reg,data);inputs+=more
    updates=[];windows=[];timebins=[];windowtime=[];overall=[];lossrows=[];phase={}
    for arm in ARMS:
        diags=data[arm]['diags'];logs=data[arm]['logs'];byupdate=collections.defaultdict(list);bytime=collections.defaultdict(list);both=collections.defaultdict(list)
        for row in diags:
            byupdate[row['local_step']].append(row);bytime[row['time_bin']].append(row);both[((row['local_step']-1)//100,row['time_bin'])].append(row)
        for step,group in sorted(byupdate.items()):updates+=OLD_SUM.pool(arm,group,'UPDATE',local_step=step,global_step=14000+step)
        for w in range(10):
            group=[r for r in diags if w*100<r['local_step']<=(w+1)*100]
            windows+=OLD_SUM.pool(arm,group,'WINDOW100',start_step=w*100+1,end_step=(w+1)*100)
            lo=logs[w*100:(w+1)*100]
            lossrows.append(dict(arm=arm,start_step=w*100+1,end_step=(w+1)*100,objective=objective(arm),
                mean_FM=statistics.fmean(r['FM'] for r in lo),mean_edge_loss=statistics.fmean(r.get('edge_coherence',0.) for r in lo),
                mean_total_loss=statistics.fmean(r.get('total_loss',r['FM']) for r in lo),
                directly_comparable_as_same_objective=False,
                reason='CLEAN weighted normalization plus edge differs; SMOOTH input path and velocity target differ'))
        for b,group in sorted(bytime.items()):timebins+=OLD_SUM.pool(arm,group,'TIME_BIN',time_bin=b,t_lower=group[0]['time_bin_lower'],t_upper=group[0]['time_bin_upper'])
        for (w,b),group in sorted(both.items()):windowtime+=OLD_SUM.pool(arm,group,'WINDOW100_TIME_BIN',start_step=w*100+1,end_step=(w+1)*100,
            time_bin=b,t_lower=group[0]['time_bin_lower'],t_upper=group[0]['time_bin_upper'])
        overall+=OLD_SUM.pool(arm,diags,'ALL_FIXED_WINDOW1000')
        phase[arm],more=OLD_SUM.resource_summary(data[arm]['folder'],logs,data[arm]['exposure']);inputs+=more
        phase[arm]['scope']='This exact14001..15000 global update window; baseline logs mapped from original1001..2000'
    lookup={(r['arm'],r['end_step'],r['category']):r for r in windows};comparisons=[]
    for arm in D.ARMS:
        for end in range(100,1001,100):
            for cat in CATEGORIES:
                b=lookup[(BASELINE,end,cat)];r=lookup[(arm,end,cat)]
                comparable=arm=='CLEAN_EDGE_COH' and b['status']=='PASS' and r['status']=='PASS'
                if b['status']=='PASS' and r['status']=='PASS' and b['scalar_count']!=r['scalar_count']:raise ValueError('Unweighted samefree counts differ')
                comparisons.append(dict(arm=arm,baseline=BASELINE,end_step=end,category=cat,
                    baseline_value=b['unweighted_velocity_mse'],candidate_value=r['unweighted_velocity_mse'],
                    paired_delta=r['unweighted_velocity_mse']-b['unweighted_velocity_mse'] if comparable else None,
                    status='PAIRED_TRAINING_FIT_ONLY' if comparable else 'NA_CHANGED_PRIOR_AND_TARGET' if arm=='SMOOTH_PRIOR' else 'NA_BASELINE_SPLIT_NOT_LOGGED',
                    interpretation='Never an output quality ranking; SMOOTH training states/targets differ'))
    # Retain measured overlap timings with no claim of isolated GPU throughput.
    concurrency=dict(status='MEASURED_OWNER_INTERVALS',arms={})
    for arm in D.ARMS:
        other=next(a for a in D.ARMS if a!=arm);start=phase[other]['owner_lifetime_start_unix'];end=phase[other]['owner_lifetime_end_unix'];groups=collections.defaultdict(list)
        for row in data[arm]['logs']:
            lo=row['time_unix']-row['update_seconds'];hi=row['time_unix'];overlap=max(0.,min(hi,end)-max(lo,start)) if start is not None and end is not None else None
            key='NA_OWNER_TIMES' if overlap is None else 'NO_OTHER_ARM_OWNER' if overlap==0 else 'OTHER_ARM_OWNER_OVERLAP'
            groups[key].append(row['update_seconds'])
        concurrency['arms'][arm]={key:OLD_SUM.numeric_stats(values) for key,values in groups.items()}
    concurrency['warning']='Owner overlap includes model loading/saving; no claim of simultaneous kernels or isolated speedup. User GPU work unassigned.'
    out.mkdir()
    for name,values in (('per_update_unweighted.csv',updates),('window100_unweighted.csv',windows),('time_bin_unweighted.csv',timebins),
        ('window100_time_bin_unweighted.csv',windowtime),('overall_unweighted.csv',overall),('objective_terms_separate.csv',lossrows),
        ('paired_window_deltas.csv',comparisons),('exposure_pairing.csv',pairing),
        ('per_parent_exposure.csv',[dict(parent_id=p,PLAIN_window=c,CLEAN_EDGE_COH=c,SMOOTH_PRIOR=c) for p,c in sorted(exposure.items())])):
        OLD_SUM.csv_write(out/name,values)
    D.write_new(out/'checkpoint_and_call_audit.json',evidence);D.write_new(out/'phase_resources.json',phase);D.write_new(out/'concurrency_phases.json',concurrency)
    plot(out,windows,lossrows)
    inputs += [root/'training_registration.json',D.resolve(reg['shared_data_manifest']),Path(__file__)]
    for arm in D.ARMS:
        for key in ('checkpoint','inference_export'):inputs.append(D.resolve(evidence[arm][key]['path']))
    note=dict(status='COMPLETE',model_constructions=0,neural_forwards=0,neural_backwards=0,optimizer_updates=0,geometry_scoring_calls=0,
        fixed_baseline_window='Original PLAIN3000 steps1001..2000, globals14001..15000',paired_examples=8000,parents=47,
        final_global_step=15000,source_assets_untouched=True,protected_source_files_rehashed=len(reg['protected_hashes']),
        final_rng_equal_between_arms=evidence[D.ARMS[0]]['final_internal_hashes']['rng_sha256']==evidence[D.ARMS[1]]['final_internal_hashes']['rng_sha256'],
        source_hashes={str(p):D.digest(p) for p in dict.fromkeys(inputs)},
        warnings=['SMOOTH MSE has a different noise path and velocity target, so no paired cross-prior quality conclusion.',
            'CLEAN weightedFM and edge loss form a combined arm; cannot isolate edge benefit without a matched CLEAN-only arm.',
            'Old PLAIN clean/dirty categories were not recorded; remain NA.',
            'Source face identities are training-only supervision, not inference labels.',
            'Training interpolation fit does not certify rollout attachment or quality.'],seconds=time.perf_counter()-started)
    D.write_new(out/'completion.json',note)
    return note

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='experiments/r2_seam_hinge_quality_v1');a=p.parse_args();r=run(a.root)
    print(json.dumps({k:v for k,v in r.items() if k!='source_hashes'},indent=2))
