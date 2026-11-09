"""Independent CPU verification of completed Bench training (no replay/forward).

This does not import or modify a checkpoint until both workers report COMPLETE.
It verifies physical call accounting, paired example identities, final model and
AdamW tensors, and the serialized RNG states without initializing CUDA.
"""
from __future__ import annotations
import argparse, collections, copy, csv, gc, json, math, random, statistics, time
from pathlib import Path
import numpy as np
from . import train as T
import torch
from meshflow_control.training.schedule40 import tree_hash
from meshflow_control.checkpoints import state_hash

def lines(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]

def table(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def require(ok,message):
    if not ok:raise ValueError(message)

def optimizer_check(saved,parameters,expected_step,lr,label):
    groups=saved['param_groups'];states=saved['state'];names=list(parameters)
    require(len(groups)==1,label+': one group required');g=groups[0]
    require((g['lr'],tuple(g['betas']),g['weight_decay'],g['eps'])==(lr,(.9,.95),0.,1e-8),label+': AdamW hyperparameters differ')
    ids=g['params'];require(len(ids)==len(set(ids))==len(names),label+': parameter ownership differs')
    require(set(states)==set(ids),label+': missing or extra AdamW states')
    rows=[]
    for name,sid in zip(names,ids):
        target=parameters[name];s=states[sid]
        require(set(s)=={'step','exp_avg','exp_avg_sq'},label+': AdamW state fields differ')
        require(float(s['step'])==expected_step,label+': cumulative AdamW clock differs')
        for field in ('exp_avg','exp_avg_sq'):
            x=s[field];require(x.dtype==torch.float32 and x.shape==target.shape,label+': moment dtype/shape differs '+name)
            require(bool(torch.isfinite(x).all()),label+': nonfinite moment '+name)
        require(bool((s['exp_avg_sq']>=0).all()),label+': negative second moment '+name)
        rows.append(dict(owner=label,parameter=name,optimizer_id=sid,step=float(s['step']),shape=str(tuple(target.shape)),dtype=str(target.dtype),moments_finite=True,second_moment_nonnegative=True))
    return rows

def rng_check(rng):
    require(set(rng)=={'python','numpy','torch_cpu','torch_cuda'},'RNG schema differs')
    before_python=random.getstate();before_np=np.random.get_state();before_torch=torch.get_rng_state().clone()
    n=rng['numpy'];expected_cpu={k:rng[k] for k in ('python','numpy','torch_cpu')}
    try:
        random.setstate(rng['python']);np.random.set_state((n['kind'],np.asarray(n['keys'],np.uint32),n['pos'],n['has_gauss'],n['cached']))
        torch.set_rng_state(rng['torch_cpu']);current=np.random.get_state()
        actual_cpu=dict(python=random.getstate(),numpy=dict(kind=current[0],keys=current[1].tolist(),pos=current[2],has_gauss=current[3],cached=current[4]),torch_cpu=torch.get_rng_state())
        require(tree_hash(actual_cpu)==tree_hash(expected_cpu),'CPU RNG roundtrip differs')
    finally:random.setstate(before_python);np.random.set_state(before_np);torch.set_rng_state(before_torch)
    require(len(rng['torch_cuda'])==1,'One owned CUDA RNG state expected')
    for x in rng['torch_cuda']:require(x.device.type=='cpu' and x.dtype==torch.uint8 and x.ndim==1 and x.numel()>0,'Malformed saved CUDA RNG')
    require(not torch.cuda.is_initialized(),'Audit unexpectedly initialized CUDA')
    return dict(rng_sha256=tree_hash(rng),CPU_rng_roundtrip=True,CUDA_rng_tensor_sizes=[int(x.numel()) for x in rng['torch_cuda']],CUDA_rng_restored=False,CUDA_rng_reason='CPU audit preserves serialized bytes; no new CUDA context or model call')

def audit(root):
    root=T.resolve(root);reg=T.checked(root);total=int(reg['updates']);out=root/'training_audit';out.mkdir(exist_ok=True)
    statuses={arm:T.read(root/'training'/arm/'status.json') for arm in T.ARMS}
    require(all(s['status']=='COMPLETE' and s['step']==total for s in statuses.values()),'Both workers must be COMPLETE before checkpoint audit')
    require(not torch.cuda.is_initialized(),'CPU-only audit precondition failed')
    started=time.perf_counter();T.write(out/'resources_before.json',T.resources())
    schedule=T.read(root/'input_schedule.json')['steps'];cases=[T.arrays(r['case_path']) for r in reg['records']]
    records=reg['records'];all_summary={};exposure_maps={};step_rows=[];parent_rows=[];tensor_rows=[];checkpoint_rows=[]
    initializations=[]
    for arm in T.ARMS:
        folder=root/'training'/arm
        require(not (folder/'failure.json').exists(),arm+': recorded failure requires explicit review')
        ledger=lines(folder/'ledger.jsonl');exposure=lines(folder/'exposure.jsonl');logs=lines(folder/'training_log.jsonl')
        groups=collections.defaultdict(list)
        for row in ledger:groups[int(row['step'])].append(row)
        require(sorted(groups)==list(range(1,total+1)),arm+': physical step range differs')
        expected_one={'UPDATE_STARTED':1,'UPDATE_COMPLETE':1,'FORWARD_STARTED':8,'FORWARD_COMPLETE':8,'BACKWARD_STARTED':8,'BACKWARD_COMPLETE':8,'OPTIMIZER_STARTED':1,'OPTIMIZER_COMPLETE':1}
        if arm=='SEAM':expected_one.update(HEAD_FORWARD_STARTED=8,HEAD_FORWARD_COMPLETE=8,HEAD_BACKWARD_STARTED=8,HEAD_BACKWARD_COMPLETE=8,HEAD_OPTIMIZER_STARTED=1,HEAD_OPTIMIZER_COMPLETE=1)
        for step,rows in sorted(groups.items()):
            counts=dict(collections.Counter(r['event'] for r in rows));require(counts==expected_one,f'{arm} step{step}: call budget differs')
            require(rows[0]['event']=='UPDATE_STARTED' and rows[-1]['event']=='UPDATE_COMPLETE',arm+': incomplete update boundary')
            for kind in ('FORWARD_STARTED','FORWARD_COMPLETE','BACKWARD_STARTED','BACKWARD_COMPLETE'):
                require([r['micro'] for r in rows if r['event']==kind]==list(range(8)),arm+': replayed/skipped microbatch')
            step_rows.append(dict(arm=arm,step=step,**counts))
        require(len(logs)==total and [r['step'] for r in logs]==list(range(1,total+1)),arm+': learning curve cursor differs')
        require(len(exposure)==total*8,arm+': exposure count differs');mapping={}
        for row in exposure:
            key=(row['step'],row['micro']);require(key not in mapping,arm+': duplicate exposure');mapping[key]=row
            planned=schedule[row['step']-1]['samples'][row['micro']]
            require(all(row[k]==v for k,v in planned.items()),arm+': random stream identity differs')
            record=records[planned['index']];require(row['parent_id']==record['parent_id'],arm+': parent differs')
            case=cases[planned['index']];n=len(case['full_target']);k=len(case['known_ids'])
            require((row['N'],row['K'])==(n,k),arm+': N/K differs')
            z=float(np.random.RandomState(planned['time_seed']).standard_normal());expected_t=float(np.float32(1/(1+np.exp(-z))))
            require(row['t']==expected_t,arm+': logged time differs from registered draw')
        for metric in logs:
            rows=schedule[metric['step']-1]['samples'];den=0
            for planned in rows:
                c=cases[planned['index']];den+=9*(len(c['full_target'])-len(c['known_ids']))-(6*int(c['occupied_labels'].sum()) if arm=='SEAM' else 0)
            require(metric['free_coordinate_denominator']==den,arm+': loss denominator differs')
            require(math.isfinite(metric['FM']) and math.isfinite(metric['generator_gradient_norm']),arm+': nonfinite learning value')
        exposure_maps[arm]=mapping
        parents=collections.Counter(r['parent_id'] for r in exposure)
        for p,count in sorted(parents.items()):parent_rows.append(dict(arm=arm,parent_id=p,exposures=count,clean=sum(r['parent_id']==p and r['alpha']==1. for r in exposure),augmented=sum(r['parent_id']==p and r['alpha']!=1. for r in exposure)))
        init=T.read(folder/'initialization_audit.json');initializations.append(init)
        require(init['fresh_AdamW'] and init['role_nonzero']==0 and init['Geo_exit_nonzero']==0,arm+': initial adaptation contract differs')
        require(init['official_sha256']==reg['official_sha256'],arm+': official category checkpoint differs')
        cp=folder/f'step{total}.pt';receipt=T.read(cp.with_suffix('.json'));cp_hash=T.sha(cp)
        require(cp_hash==receipt['sha256'],arm+': final checkpoint file changed')
        p=torch.load(cp,map_location='cpu',weights_only=True,mmap=True)
        require(p['schema']=='bench_control_training_v1' and p['arm']==arm and p['step']==total and p['updates']==total and p['EMA'] is False,arm+': checkpoint identity differs')
        require(p['registration_sha256']==T.sha(root/'registration.json') and p['model_config']==reg['model_config'],arm+': registration/config differs')
        require(p['initialization']==init,arm+': embedded initialization audit differs')
        gs=p['generator_state'];require(state_hash(gs)==p['generator_state_sha256']==receipt['generator_state_sha256'],arm+': generator state hash differs')
        require(all(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) for x in gs.values()),arm+': generator is not finite FP32')
        # Build a CPU architecture solely to establish exact parameter names and shapes.
        model=T.build_model(copy.deepcopy(reg['model_config']),readout_mode='none',trainable=True).eval()
        expected=dict(model.named_parameters());require(set(expected)==set(gs),arm+': generator tensor identity differs')
        for name,x in gs.items():require(x.shape==expected[name].shape,arm+': generator tensor shape differs '+name)
        tensor_rows.extend(dict(arm=arm,**r) for r in optimizer_check(p['optimizer'],expected,total,reg['lr'],'generator'))
        checkpoint_fact=dict(arm=arm,step=total,path=str(cp),file_sha256=cp_hash,generator_state_sha256=p['generator_state_sha256'],generator_parameters=sum(x.numel() for x in gs.values()),optimizer_sha256=tree_hash(p['optimizer']),optimizer_steps=[total])
        if arm=='SEAM':
            head=T.OccupancyHead(24).eval();hp=dict(head.named_parameters());hs=p['head_state']
            require(set(hp)==set(hs),'Occupancy state identity differs')
            require(all(hs[n].shape==x.shape and hs[n].dtype==torch.float32 and bool(torch.isfinite(hs[n]).all()) for n,x in hp.items()),'Occupancy tensor invalid')
            tensor_rows.extend(dict(arm=arm,**r) for r in optimizer_check(p['head_optimizer'],hp,total,reg['head_lr'],'occupancy'))
            checkpoint_fact.update(head_state_sha256=state_hash(hs),head_optimizer_sha256=tree_hash(p['head_optimizer']),head_parameters=sum(x.numel() for x in hs.values()))
            del head,hp,hs
        else:require(p['head_state'] is None and p['head_optimizer'] is None,'PLAIN unexpectedly owns occupancy head')
        checkpoint_fact.update(rng_check(p['rng']));checkpoint_rows.append(checkpoint_fact)
        del p,model,expected,gs;gc.collect()
        all_summary[arm]=dict(status='PASS',physical_updates=total,train_forward=8*total,train_backward=8*total,head_forward=8*total if arm=='SEAM' else 0,
            head_backward=8*total if arm=='SEAM' else 0,head_optimizer_updates=total if arm=='SEAM' else 0,
            independent_parents=len(parents),exposures=len(exposure),per_parent_min=min(parents.values()),per_parent_max=max(parents.values()),
            recorded_elapsed_before_final_checkpoint_seconds=statuses[arm]['elapsed_seconds'],physical_update_span_seconds=ledger[-1]['unix']-ledger[0]['unix'],mean_logged_update_seconds=statistics.mean(r['update_seconds'] for r in logs),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in logs),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in logs))
    require(initializations[0]==initializations[1],'Arms did not use identical official and conditional initialization')
    require(set(exposure_maps['PLAIN'])==set(exposure_maps['SEAM']),'Paired exposure keys differ');paired=[]
    fields=('step','micro','parent_id','index','alpha','N','K','t','noise_seed','epsilon_seed','time_seed','permutation_seed')
    for key,a in sorted(exposure_maps['PLAIN'].items()):
        b=exposure_maps['SEAM'][key];require(all(a[k]==b[k] for k in fields),'Between-arm paired input stream differs')
        row={k:a[k] for k in fields};c=cases[a['index']];row.update(PLAIN_free_coordinates=9*(a['N']-a['K']),SEAM_free_coordinates=9*(a['N']-a['K'])-6*int(c['occupied_labels'].sum()),pair_equal=True);paired.append(row)
    table(out/'physical_calls_per_update.csv',step_rows);table(out/'paired_exposure_full.csv',paired);table(out/'exposure_per_parent.csv',parent_rows);table(out/'optimizer_tensor_audit.csv',tensor_rows);T.write(out/'checkpoint_audit.json',checkpoint_rows)
    result=dict(status='PASS',experiment=reg['schema'],arms=all_summary,same_initialization=True,paired_rows=len(paired),paired_stream_exact=True,
        initialization=initializations[0],CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),new_generator_forwards=0,new_optimizer_updates=0,
        no_checkpoint_replay_or_generation=True,checkpoint_audit=str(out/'checkpoint_audit.json'),wall_seconds=time.perf_counter()-started,
        limits='This verifies recorded execution and tensor state. It does not claim independent retraining repeatability or cross-category zero-shot generalization. Status elapsed excludes the last checkpoint serialization; concurrent arm times are not isolated speed benchmarks.')
    require(not result['CUDA_initialized'],'Unexpected CUDA context');T.write(out/'resources_after.json',T.resources());T.write(out/'FINAL_TRAINING_AUDIT.json',result)
    print(json.dumps(result,ensure_ascii=False));return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(T.DEFAULT_ROOT));a=ap.parse_args();audit(a.root)
if __name__=='__main__':main()
