"""Offline training evidence only: no torch, model, CUDA, OT or evaluator imports."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import argparse,csv,hashlib,json,math,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(v) for v in Path(p).read_text(encoding='utf-8').splitlines() if v]
def csvrows(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 if not rows:return
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(values):
 a=np.asarray(values,dtype=np.float64)
 if not len(a):return dict(n=0,mean=None,median=None,min=None,max=None,p95=None)
 if not np.isfinite(a).all():raise ValueError('Nonfinite source value')
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),p95=float(np.quantile(a,.95)))
def main(root):
 out=root/'analysis';out.mkdir(exist_ok=True)
 repo=root.parent/'MeshFlow-Control'
 source_paths=[repo/'src/meshflow_control/training/trainer.py',repo/'src/meshflow_control/models/readout.py',root/'audit/training_pairing.json']
 for arm in 'ABC':
  source_paths += [root/'training'/arm/n for n in ('training_log.csv','diagnostics.jsonl','status.json','start_import_audit.json','exposure.csv','events.jsonl')]
  source_paths.append(root/'audit/budget'/f'train_{arm}_resources.json')
 before={str(p):digest(p) for p in source_paths}
 logs={};diags={};status={};starts={};exposures={}
 hashes=('shared_batch_sha256','input_batch_sha256','label_batch_sha256')
 stringcols={'arm',*hashes}
 checks={};rows_observe=[];rows_parameter=[]
 expected_names={f'readouts.{l}.W_{k}.weight' for l in (3,6,9,12) for k in ('Q','K','V','O')}
 expected_observed=[1,2]+list(range(50,1001,50))
 for arm in 'ABC':
  f=root/'training'/arm
  logs[arm]=[{k:(v if k in stringcols else None if v=='' else float(v)) for k,v in r.items()} for r in csvrows(f/'training_log.csv')]
  diags[arm]=lines(f/'diagnostics.jsonl');status[arm]=read(f/'status.json');starts[arm]=read(f/'start_import_audit.json')
  exposures[arm]={int(r['sample_index']):r for r in csvrows(f/'exposure.csv')}
  assert [int(r['step']) for r in logs[arm]]==list(range(1,1001))
  assert [r['step'] for r in diags[arm]]==list(range(1,1001))
  assert status[arm]['status']=='COMPLETE' and status[arm]['step']==1000
  assert starts[arm]['old_tensors_equal'] and starts[arm]['optimizer_restored']
  assert starts[arm]['old_tensor_count']==194 and starts[arm]['original_optimizer_step']==5000
  assert len(exposures[arm])==8000
  for r,d in zip(logs[arm],diags[arm]):
   step=int(r['step']);assert r['conditional_total_step']==5000+step and r['old_adamw_step']==5000+step
   assert r['unused_parameters']==0
   assert r['new_adamw_step']==(None if arm=='A' else step)
   assert sum(x['FM_contribution'] for x in d['sample_losses'])==r['FM']
   assert [x['sample_index'] for x in d['sample_losses']]==list(range(8*(step-1),8*step))
   assert sum(x['free_coordinates'] for x in d['sample_losses'])==r['free_coordinate_denominator']
   assert all(x['batch_denominator']==r['free_coordinate_denominator'] for x in d['sample_losses'])
   assert set(d['readout_parameter_gradients'])==(set() if arm=='A' else expected_names)
   assert set(d['readout_parameter_updates'])==(set() if arm=='A' else expected_names)
   observed=d['readout_observations']
   assert len(observed)==(32 if arm!='A' and step in expected_observed else 0)
   for o in observed:
    assert o['branch_dtype']=='torch.float32' and o['source_mode']==('mixed' if arm=='B' else 'clean')
    e=exposures[arm][o['sample_index']]
    assert o['source_corners']==3*int(e['K']) and o['query_corners']==3*(int(e['N'])-int(e['K']))
    assert abs(o['residual_rms']/max(o['hidden_rms'],1e-30)-o['residual_to_hidden_rms'])<1e-15
    assert 0<=o['attention_entropy']<=math.log(o['source_corners'])+1e-5
    rows_observe.append(dict(arm=arm,step=step,layer=o['layer'],microbatch=o['microbatch'],sample_index=o['sample_index'],uid=e['uid'],t=float(e['t']),alpha=float(e['alpha']),source_corners=o['source_corners'],query_corners=o['query_corners'],hidden_rms=o['hidden_rms'],residual_rms=o['residual_rms'],residual_to_hidden_rms=o['residual_to_hidden_rms'],attention_entropy_nats=o['attention_entropy'],entropy_over_log_source_count=o['attention_entropy']/math.log(o['source_corners']),source_mode=o['source_mode'],branch_dtype=o['branch_dtype'],hidden_dtype=o['hidden_dtype']))
  checks[arm+'_complete_1000_and_diagnostic_accounting']=True
 for step in range(1000):
  assert all(logs[a][step][k]==logs['A'][step][k] for a in 'BC' for k in (*hashes,'free_coordinate_denominator'))
 assert all(logs[a][0]['FM']==logs['A'][0]['FM'] for a in 'BC')
 assert starts['B']['new_readout_state_sha256']==starts['C']['new_readout_state_sha256']
 pairing=read(root/'audit/training_pairing.json');assert pairing['status']=='PASS'
 checks['paired_all_1000_batches_all_three_hashes_and_denominator']=True
 checks['initial_FM_bitwise_python_float_equal']=True
 checks['BC_identical_initial_readout_state_hash']=True
 checks['existing_8000_exposure_pairing_audit_PASS']=True
 # Recount only immutable training records from one bounded snapshot of the shared,
 # currently append-only sampling ledger. Record the filtered byte hash, not a
 # misleading hash of the still-changing complete ledger.
 ledger=root/'audit/budget/calls.jsonl';snapshot_size=ledger.stat().st_size
 with ledger.open('rb') as f:snapshot=f.read(snapshot_size)
 counts={a:Counter() for a in 'ABC'};seen=set();training_hash=hashlib.sha256();selected_count=0
 for raw in snapshot.splitlines(keepends=True):
  if not raw.endswith(b'\n'):continue
  ev=json.loads(raw);kind=ev.get('kind');ctx=ev.get('context',{});arm=ctx.get('arm')
  if kind not in ('train_forward','train_backward','optimizer_update'):continue
  assert arm in 'ABC';event=ev['event'];assert event in ('RESERVE','RETURN')
  key=(arm,kind,event,ctx['step'],ctx.get('microbatch'))
  assert key not in seen;seen.add(key);counts[arm][kind+'_'+event]+=1
  training_hash.update(raw);selected_count+=1
 for a in 'ABC':
  for k,n in [('train_forward',8000),('train_backward',8000),('optimizer_update',1000)]:
   assert counts[a][k+'_RESERVE']==n and counts[a][k+'_RETURN']==n
 assert {(a,k,st,m) for a,k,e,st,m in seen if e=='RESERVE'}=={(a,k,st,m) for a,k,e,st,m in seen if e=='RETURN'}
 for a,k,e,st,m in seen:
  assert 1<=st<=1000 and (m is None if k=='optimizer_update' else m in range(8))
 checks['ledger_24000F_24000B_3000updates_unique_reserve_return_no_training_fail']=True
 occupancy=[r for r in lines(root/'audit/budget/gpu_occupancy.jsonl') if r.get('phase') in ('train_A','train_B','train_C')]
 stagewall={r['phase'][-1]:r['seconds'] for r in occupancy if r['event']=='END'}
 assert set(stagewall)==set('ABC')
 windows=[('all_1000',1,1000),('first_100',1,100),('steps_1_500',1,500),('steps_501_1000',501,1000),('last_100',901,1000)]
 numeric=('FM','old_gradient_norm','new_gradient_norm','new_actual_update_norm','grad_norm_before_clip','input_seconds','forward_seconds','backward_seconds','optimizer_seconds','update_seconds')
 arms={};window_rows=[]
 for a in 'ABC':
  aw={}
  for name,lo,hi in windows:
   rs=logs[a][lo-1:hi]
   v={k:stats([r[k] for r in rs]) for k in numeric}
   v['forward_backward_seconds']=stats([r['forward_seconds']+r['backward_seconds'] for r in rs])
   v['clip_count']=sum(r['grad_norm_before_clip']>1 for r in rs);v['clip_fraction']=v['clip_count']/len(rs)
   # torch clip uses 1/(norm+1e-6); diagnostic does not rerun clipping.
   v['torch_clip_factor']=stats([min(1.,1./(r['grad_norm_before_clip']+1e-6)) for r in rs])
   aw[name]=dict(first_step=lo,last_step=hi,**v)
   window_rows.append(dict(arm=a,window=name,first_step=lo,last_step=hi,steps=len(rs),**{k+'_mean':v[k]['mean'] for k in numeric},forward_backward_seconds_mean=v['forward_backward_seconds']['mean'],clip_count=v['clip_count'],clip_fraction=v['clip_fraction']))
  allocations=max(r['cuda_peak_allocated_bytes'] for r in logs[a]);reserved=max(r['cuda_peak_reserved_bytes'] for r in logs[a])
  resources=read(root/'audit/budget'/f'train_{a}_resources.json')
  arms[a]=dict(label={'A':'BASELINE_CONT','B':'MIXED_READOUT','C':'CLEAN_READOUT'}[a],steps=1000,examples=8000,counts=dict(counts[a]),windows=aw,old_adamw=dict(import_step=5000,first_logged_step=5001,final_step=6000,parameter_tensors=194,parameter_count=starts[a]['old_parameter_count'],restored=starts[a]['optimizer_restored']),new_adamw=None if a=='A' else dict(import_step=0,first_logged_step=1,final_step=1000,parameter_tensors=16,parameter_count=starts[a]['new_parameter_count']),training_loop_wall_seconds=status[a]['wall_seconds'],gpu_worker_occupancy_seconds=stagewall[a],peak_allocated_bytes=int(allocations),peak_reserved_bytes=int(reserved),peak_allocated_GiB=allocations/2**30,peak_reserved_GiB=reserved/2**30,observed_own_process_peak_MiB=None,pre_dispatch_gpu_snapshot=resources['gpu'],checkpoint_metadata=status[a]['checkpoint'],readout_observed_steps=[] if a=='A' else expected_observed,readout_observations=0 if a=='A' else 704)
  if a!='A':
   first=diags[a][0]
   assert all(v==0 for n,v in first['readout_parameter_gradients'].items() if '.W_O.' not in n)
   assert all(v==0 for n,v in first['readout_parameter_updates'].items() if '.W_O.' not in n)
   assert all(v>0 for n,v in first['readout_parameter_gradients'].items() if '.W_O.' in n)
   assert all(v>0 for n,v in first['readout_parameter_updates'].items() if '.W_O.' in n)
   arms[a]['initial_zero_WO_gate']=dict(QKV_gradient_and_update_zero=True,all_four_WO_gradients_and_updates_positive=True)
   for pname in sorted(expected_names):
    for wn,lo,hi in windows:
     gs=stats([d['readout_parameter_gradients'][pname] for d in diags[a][lo-1:hi]])
     us=stats([d['readout_parameter_updates'][pname] for d in diags[a][lo-1:hi]])
     rows_parameter.append(dict(arm=a,parameter=pname,layer=int(pname.split('.')[1]),projection=pname.split('.')[2],window=wn,first_step=lo,last_step=hi,gradient_mean=gs['mean'],gradient_min=gs['min'],gradient_max=gs['max'],actual_update_mean=us['mean'],actual_update_min=us['min'],actual_update_max=us['max'],positive_gradient_steps=sum(d['readout_parameter_gradients'][pname]>0 for d in diags[a][lo-1:hi]),positive_update_steps=sum(d['readout_parameter_updates'][pname]>0 for d in diags[a][lo-1:hi])))
 paired=[];paired_summary={}
 for i in range(1000):
  r=dict(step=i+1,free_coordinate_denominator=int(logs['A'][i]['free_coordinate_denominator']),shared_batch_sha256=logs['A'][i]['shared_batch_sha256'])
  r.update({a+'_FM':logs[a][i]['FM'] for a in 'ABC'})
  for c,b in [('B','A'),('C','A'),('C','B')]:r[f'{c}_minus_{b}_FM']=r[c+'_FM']-r[b+'_FM']
  paired.append(r)
 for c,b in [('B','A'),('C','A'),('C','B')]:
  contrast=f'{c}_minus_{b}';paired_summary[contrast]={}
  for wn,lo,hi in windows:
   ds=[r[f'{contrast}_FM'] for r in paired[lo-1:hi]];base=np.mean([r[b+'_FM'] for r in paired[lo-1:hi]])
   paired_summary[contrast][wn]=dict(delta_FM=stats(ds),relative_mean_delta_percent=100*float(np.mean(ds))/base,negative_steps=sum(d<0 for d in ds),positive_steps=sum(d>0 for d in ds),zero_steps=sum(d==0 for d in ds))
 layerrows=[];layer_windows=[]
 for a in 'BC':
  for layer in (3,6,9,12):
   for st in expected_observed:
    rs=[r for r in rows_observe if r['arm']==a and r['layer']==layer and r['step']==st];assert len(rs)==8
    z=dict(arm=a,step=st,layer=layer,samples=len(rs))
    for k in ('residual_to_hidden_rms','residual_rms','hidden_rms','attention_entropy_nats','entropy_over_log_source_count','source_corners','query_corners'):
     ss=stats([r[k] for r in rs]);z.update({k+'_'+q:ss[q] for q in ('mean','min','max')})
    layerrows.append(z)
   for wn,lo,hi in windows:
    rs=[r for r in rows_observe if r['arm']==a and r['layer']==layer and lo<=r['step']<=hi]
    layer_windows.append(dict(arm=a,layer=layer,window=wn,observed_steps=len({r['step'] for r in rs}),observations=len(rs),residual_to_hidden_rms=stats([r['residual_to_hidden_rms'] for r in rs]),attention_entropy_nats=stats([r['attention_entropy_nats'] for r in rs]),entropy_over_log_source_count=stats([r['entropy_over_log_source_count'] for r in rs])))
 csvwrite(out/'training_paired_steps.csv',paired)
 csvwrite(out/'training_windows.csv',window_rows)
 csvwrite(out/'training_readout_observations.csv',rows_observe)
 csvwrite(out/'training_readout_layers.csv',layerrows)
 csvwrite(out/'training_readout_parameters.csv',rows_parameter)
 # The plots use fixed contiguous 50-step descriptive blocks (no model fits).
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,3,figsize=(12,7.3),layout='constrained')
 colors={'A':'#444444','B':'#246eac','C':'#cb7025'};x=np.arange(25.5,1000,50)
 block=lambda values:np.asarray(values,dtype=float).reshape(20,50).mean(axis=1)
 for a in 'ABC':
  axes[0,0].plot(x,block([r['FM'] for r in logs[a]]),label=a+' '+arms[a]['label'],color=colors[a])
 axes[0,0].set(title='FM on paired training inputs',ylabel='Batch FM (50-step mean)');axes[0,0].legend(fontsize=7.3)
 for c,b,col in [('B','A',colors['B']),('C','A',colors['C']),('C','B','#6c488f')]:
  axes[0,1].plot(x,block([r[f'{c}_minus_{b}_FM'] for r in paired]),label=c+' - '+b,color=col)
 axes[0,1].axhline(0,color='.6',lw=.7);axes[0,1].set(title='Paired FM differences',ylabel='Candidate - reference (50-step mean)');axes[0,1].legend()
 for a in 'BC':
  axes[0,2].plot(x,block([r['new_gradient_norm'] for r in logs[a]]),color=colors[a],label=a+' new-gradient')
  axes[1,0].plot(x,block([r['new_actual_update_norm'] for r in logs[a]]),color=colors[a],label=a+' actual new-parameter update')
 axes[0,2].set(title='Readout gradient before global clip',ylabel='L2 norm (50-step mean)');axes[0,2].legend(fontsize=8)
 axes[1,0].set(title='Measured AdamW readout parameter change',ylabel='Actual L2 update (50-step mean)');axes[1,0].legend(fontsize=7.8)
 lc={3:'#2574a9',6:'#d07928',9:'#339759',12:'#925ba2'}
 for a,ls in [('B','-'),('C','--')]:
  for l in (3,6,9,12):
   rs=[r for r in layerrows if r['arm']==a and r['layer']==l]
   for ax,k in [(axes[1,1],'residual_to_hidden_rms_mean'),(axes[1,2],'entropy_over_log_source_count_mean')]:ax.plot([r['step'] for r in rs],[r[k] for r in rs],color=lc[l],ls=ls,label=f'{a} layer {l}',lw=1.25)
 axes[1,1].set(title='Readout residual / incoming hidden RMS',ylabel='Mean of 8 sample ratios');axes[1,1].ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
 axes[1,2].set(title='Attention entropy / log(source corners)',ylabel='Sample mean; descriptive normalization');axes[1,2].legend(ncol=2,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.22))
 for ax in axes.flat:ax.set_xlabel('Training update');ax.grid(alpha=.16)
 fig.suptitle('Completed A / B / C training: mechanism evidence, not generated quality',fontsize=12)
 fig.savefig(out/'training_evidence.png',dpi=160,facecolor='white');plt.close(fig)
 source_unchanged=all(digest(Path(k))==v for k,v in before.items());assert source_unchanged
 evidence=dict(schema='native_direct_context_training_evidence_v1',status='PASS',created_unix=time.time(),model_calls=0,CUDA_calls=0,OT_calls=0,checkpoint_tensor_loads=0,checks=checks,source_files=[dict(path=k,sha256=v,bytes=Path(k).stat().st_size) for k,v in before.items()],source_files_unchanged=source_unchanged,analysis_source=dict(path=str(Path(__file__).resolve()),sha256=digest(__file__)),ledger_training_only=dict(path=str(ledger),snapshot_bytes=snapshot_size,training_event_count=selected_count,filtered_raw_line_sha256=training_hash.hexdigest(),scope='Only train_forward/train_backward/optimizer_update; active sampling excluded'),arms=arms,paired_FM=paired_summary,readout_layer_windows=layer_windows,definitions=dict(FM='Python-ordered sum of 8 microbatch free-coordinate squared-error contributions, each using the same full-batch 9*sum(N-K) denominator.',clip='Count grad_norm_before_clip > 1; actual torch coefficient is min(1, 1/(norm+1e-6)).',gradients='Pre-global-clip Euclidean norms; old/new are parameter partitions, not separate loss gradients or directional cosines.',actual_update='Euclidean norm of directly measured post-AdamW minus pre-AdamW NEW readout parameters, measured every step; old/full parameter update was not captured.',residual_to_hidden='For each observed sample and layer, FP64 RMS(raw FP32 residual) / RMS(incoming query hidden); before cast/add to BF16 hidden. It is not the realized BF16 hidden delta.',entropy='Natural-log attention entropy, averaged across queries and heads. New offline descriptive normalization divides each observation by log(source corner count), then averages samples equally.',observations='Readout telemetry sampled at steps 1,2 and every50:22 steps/arm,8samples*4layers per observed step. It is not full1000-step telemetry.',FM_comparison='Same-step paired descriptive deltas only. Across windows inputs/t vary; no fixed held-out loss or generated-control-quality conclusion.',timing='F/B/optimizer timings include CUDA synchronization and ledger overhead. update_seconds excludes stream fetch but includes collation, diagnostics, clipping, optimizer and finite checks; wall includes checkpoint/log IO. A overlapped CPU audit, so actual timings are not independent matched performance benchmarks.',VRAM='Maximum recorded torch allocated/reserved is cumulative within each training process from its stage reset, includes resident parameters/optimizer; no process nvidia-smi peak was measured. Pre-dispatch free memory is not a peak measurement.'),limits=['No generated-output metric is used here; attention magnitude/entropy and training FM do not establish useful control or causality.','No gradient directions/cosines or original/full-model actual parameter deltas were logged.','The branch observer records raw FP32 residual before BF16 cast/add; possible quantization loss cannot be measured from these scalar summaries.','No new inference, training, scoring, OT, checkpoint tensor loading, or source modifications.'])
 products=('training_paired_steps.csv','training_windows.csv','training_readout_observations.csv','training_readout_layers.csv','training_readout_parameters.csv','training_evidence.png')
 evidence['products']={n:dict(path=str(out/n),sha256=digest(out/n),bytes=(out/n).stat().st_size) for n in products}
 (out/'training_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 # Brief Chinese memo; Unicode escapes keep shell transit strictly ASCII.
 txt=['# \u5b9e\u9645\u8bad\u7ec3\u673a\u5236\u4e0e\u6210\u672c\u8bc1\u636e','', '\u4e09\u81c2\u5747\u5b8c\u6210 1000 \u6b21\u66f4\u65b0\uff0c\u5404 8000 \u6b21\u4e3b FM forward/backward\u3002\u91cd\u8bfb\u8d26\u672c\u786e\u8ba4 24000 F / 24000 B / 3000 updates\uff0c\u8bad\u7ec3\u8bb0\u5f55\u65e0\u5931\u8d25\u3001\u91cd\u590d\u6216\u672a\u5b8c\u6210\u914d\u5bf9\u3002\u5168\u90e8 1000 \u6279\u7684\u8f93\u5165\u3001\u6807\u7b7e\u3001\u5f52\u4e00\u5316\u5206\u6bcd\u4e00\u81f4\uff0c\u539f 8000 \u4f8b\u540c\u6d41\u5ba1\u8ba1 PASS\u3002','', '| Arm | FM 1-500 | FM 501-1000 | clip /1000 | old grad mean | new grad mean | new actual update mean | wall s | F+B mean s | allocated / reserved GiB |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for a in 'ABC':
  w=arms[a]['windows'];v=w['all_1000']
  txt.append(f"| {a} {arms[a]['label']} | {w['steps_1_500']['FM']['mean']:.8f} | {w['steps_501_1000']['FM']['mean']:.8f} | {v['clip_count']} | {v['old_gradient_norm']['mean']:.6g} | {v['new_gradient_norm']['mean']:.6g} | {v['new_actual_update_norm']['mean']:.6g} | {arms[a]['training_loop_wall_seconds']:.2f} | {v['forward_backward_seconds']['mean']:.4f} | {arms[a]['peak_allocated_GiB']:.3f} / {arms[a]['peak_reserved_GiB']:.3f} |")
 txt+=['','\u540c\u6b65\u914d\u5bf9 FM\uff08\u4ee5\u5404\u7a97\u53e3\u53c2\u7167\u5747\u503c\u4e3a\u5206\u6bcd\uff09\uff1a','', '| contrast | all1000 delta | relative % | 1-500 relative % | 501-1000 relative % | last100 relative % |','|---|---:|---:|---:|---:|---:|']
 for k,v in paired_summary.items():txt.append(f"| {k} | {v['all_1000']['delta_FM']['mean']:+.9f} | {v['all_1000']['relative_mean_delta_percent']:+.4f} | {v['steps_1_500']['relative_mean_delta_percent']:+.4f} | {v['steps_501_1000']['relative_mean_delta_percent']:+.4f} | {v['last_100']['relative_mean_delta_percent']:+.4f} |")
 txt+=['','\u7b2c 1000 \u6b65\u56db\u5c42\u8bfb\u51fa\uff08\u6bcf\u5c42 8 \u4f8b\u7b49\u6743\uff09\uff1a','', '| Arm / layer | residual/hidden RMS | entropy nats | entropy/log(Kcorners) |','|---|---:|---:|---:|']
 for r in layerrows:
  if r['step']==1000:txt.append(f"| {r['arm']} / {r['layer']} | {r['residual_to_hidden_rms_mean']:.8g} | {r['attention_entropy_nats_mean']:.5f} | {r['entropy_over_log_source_count_mean']:.5f} |")
 txt+=['','\u65e7 194 \u5f20\u91cf\u7684 AdamW \u5b8c\u6574\u6062\u590d\u81ea step5000\uff0c\u672b\u6b65\u5747\u4e3a 6000\uff1bB/C \u65b0 16 \u5f20\u91cf\uff082,359,296 \u53c2\u6570\uff09\u4ece 0 \u5230 1000\u3002B/C \u521d\u59cb\u8bfb\u51fa\u6743\u91cd hash \u4e00\u81f4\u3002\u7b2c\u4e00\u6b65 Q/K/V \u68af\u5ea6\u4e0e\u771f\u5b9e update \u5747\u4e3a 0\uff0c\u56db\u5c42 WO \u7684\u68af\u5ea6\u4e0e update \u5747\u4e3a\u6b63\uff0c\u7b26\u5408 WO \u96f6\u521d\u59cb\u5316\u3002\u9010\u53c2\u6570\u8868\u4fdd\u7559\u540e\u7eed\u68af\u5ea6\u53ca\u771f\u5b9e update \u3002','', '\u5c42\u7edf\u8ba1\u53ea\u8986\u76d6 1\u30012 \u548c\u6bcf 50 \u6b65\uff0c\u6bcf\u81c2 22 \u6b65\u3001704 \u6761\u89c2\u6d4b\u3002residual/hidden \u662f\u539f FP32 \u6b8b\u5dee\u5728\u8f6c\u4e3a BF16 \u5e76\u76f8\u52a0\u4e4b\u524d\u7684\u6bd4\u7387\uff0c\u4e0d\u662f\u76f8\u52a0\u540e\u5b9e\u9645 hidden \u6539\u53d8\uff1b\u4e0d\u80fd\u7531\u8be5\u6bd4\u503c\u76f4\u63a5\u5224\u65ad\u63a7\u5236\u6709\u6548\u6216\u91cf\u5316\u635f\u5931\u3002\u71b5\u7528\u81ea\u7136\u5bf9\u6570\uff0c\u539f\u503c\u53d7\u6765\u6e90\u89d2\u70b9\u6570\u5f71\u54cd\uff1b\u65b0\u8868\u53e6\u5217 entropy/log(Kcorners) \u4ec5\u4f5c\u63cf\u8ff0\u3002','', '\u65e7\u53c2\u6570\u548c\u5168\u7f51\u7684\u5b9e\u9645 update \u3001\u68af\u5ea6\u65b9\u5411\u6216\u4f59\u5f26\u6ca1\u6709\u8bb0\u5f55\uff0c\u4e0d\u4f2a\u9020\u6216\u7528\u65b0\u53c2\u6570 update \u4ee3\u66ff\u3002\u68af\u5ea6\u8303\u6570\u4e3a\u5168\u5c40 clip \u524d\u7684\u503c\uff1b\u5b9e\u9645\u65b0\u53c2\u6570 update \u5305\u542b AdamW \u52a8\u91cf\u4f5c\u7528\u3002','', '\u6210\u672c\u662f\u672c\u6b21\u771f\u5b9e\u8fd0\u884c\u8bb0\u5f55\u3002A \u671f\u95f4\u5b58\u5728\u5e76\u884c CPU \u5ba1\u8ba1\uff0c\u4e0d\u4f5c\u72ec\u7acb\u6027\u80fd\u57fa\u51c6\u3002F+B \u5305\u542b\u540c\u6b65\u4e0e\u8d26\u672c\u5f00\u9500\uff1bwall \u5305\u542b\u4fdd\u5b58\u4e0e\u65e5\u5fd7\u3002VRAM \u4e3a\u5355\u8bad\u7ec3\u8fdb\u7a0b\u4ece\u9636\u6bb5\u91cd\u7f6e\u8d77\u7d2f\u79ef\u7684 PyTorch allocated/reserved \u5cf0\u503c\uff0c\u975e\u6574\u4e2a\u8fdb\u7a0b\u72ec\u5360\u663e\u5b58\u5cf0\u503c\uff1b\u540e\u8005\u4e3a NA\u3002','', '\u8fd9\u4e9b\u6570\u636e\u786e\u8ba4\u914d\u5bf9\u8bad\u7ec3\u4e0e\u5206\u652f\u66f4\u65b0\u53d1\u751f\uff0c\u4e0d\u662f\u751f\u6210\u8d28\u91cf\u6216\u63a7\u5236\u63d0\u5347\u8bc1\u636e\u3002\u672a\u6267\u884c\u4efb\u4f55\u6a21\u578b\u3001GPU\u3001OT\u6216\u65b0\u8bc4\u5206\uff0c\u672a\u4fee\u6539\u79d1\u5b66\u6e90\u7801\u4e0e\u539f\u59cb\u65e5\u5fd7\u3002','', '\u6765\u6e90\u6587\u4ef6\u53ca SHA256\u3001\u53e3\u5f84\u3001\u914d\u5bf9\u5ba1\u8ba1\u548c\u5bfc\u51fa\u8868\u5747\u89c1 [training_evidence.json](training_evidence.json)\uff1b\u56fe\u89c1 [training_evidence.png](training_evidence.png)\u3002']
 (out/'training_evidence.md').write_text('\n'.join(txt)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='PASS',arms={a:dict(FM=arms[a]['windows']['all_1000']['FM']['mean'],clip=arms[a]['windows']['all_1000']['clip_count'],old_gradient=arms[a]['windows']['all_1000']['old_gradient_norm']['mean'],new_gradient=arms[a]['windows']['all_1000']['new_gradient_norm']['mean'],new_update=arms[a]['windows']['all_1000']['new_actual_update_norm']['mean'],wall=arms[a]['training_loop_wall_seconds'],FB=arms[a]['windows']['all_1000']['forward_backward_seconds']['mean'],reservedGiB=arms[a]['peak_reserved_GiB']) for a in 'ABC'},paired={k:v['all_1000'] for k,v in paired_summary.items()},rows=dict(observations=len(rows_observe),layerrows=len(layerrows),parameters=len(rows_parameter))),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,default=Path('/mnt/e/MeshFlow-Control-artifacts'));main(p.parse_args().artifacts)
