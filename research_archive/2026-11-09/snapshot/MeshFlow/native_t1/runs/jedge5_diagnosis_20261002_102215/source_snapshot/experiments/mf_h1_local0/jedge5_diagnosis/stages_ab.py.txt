"""Bounded A/B workers; each process owns one frozen model and result file.

No scoring, OT, optimizer, geometry guidance or stage-D training occurs here.
The orchestrator decides concurrency from measured resources before launching.
"""
from native_t1.runtime import configure_stable_runtime
from pathlib import Path
import argparse
import json
import time
import traceback
import numpy as np
import torch
from native_t1.artifacts import file_sha256, state_sha256, array_hash
from native_t1.sampling import clamped_sample, new_counts as production_counts
from .sampling import sample, new_counts

MODELS = ('B','J')
SETTINGS = (('bf16',50),('bf16',100),('fp32',50),('fp32',100))
CUTS = (20,50,80)


def plan_A(label, seeds):
    if label not in MODELS or len(seeds)!=2 or len(set(seeds))!=2:
        raise ValueError('A requires B/J and exactly two registered seeds')
    main=[dict(model=label,parent=p,patch=k,seed=int(seed),precision=precision,nfe=nfe)
          for precision,nfe in SETTINGS for p in (0,1) for k in range(5) for seed in seeds]
    reference=[dict(model=label,parent=p,patch=k,seed=int(seeds[0]),precision='fp32',nfe=200)
               for p in (0,1) for k in (0,4)]
    return main+reference


def plan_B(label, seeds):
    if label not in MODELS or len(seeds)!=2 or len(set(seeds))!=2:
        raise ValueError('B requires B/J and exactly two registered seeds')
    other='J' if label=='B' else 'B'
    replays=[dict(source_model=label,continuation_model=label,parent=0,patch=0,seed=int(seeds[0]),start_step=cut,same_model_replay=True) for cut in CUTS]
    cross=[dict(source_model=other,continuation_model=label,parent=p,patch=k,seed=int(seed),start_step=cut,same_model_replay=False)
           for p in (0,1) for k in (0,2,4) for seed in seeds for cut in CUTS]
    return replays+cross


def _noise_mapping(out):
    from .common import noise_bank
    bank=noise_bank(out)
    if not isinstance(bank,dict):
        bank={key:bank[key] for key in bank.files}
    mapped={int(str(k).replace('seed_','')):np.asarray(v,dtype=np.float32) for k,v in bank.items()}
    if len(mapped)!=2 or any(v.shape!=(1,112,9) for v in mapped.values()):
        raise ValueError('Registered noise bank must contain exactly two [1,112,9] arrays')
    return dict(sorted(mapped.items()))


def _key(item, phase):
    if phase=='A':
        return f"{item['model']}_P{item['parent']}_patch{item['patch']}_seed{item['seed']}_{item['precision']}_Euler{item['nfe']}"
    return f"{item['source_model']}_to_{item['continuation_model']}_P{item['parent']}_patch{item['patch']}_seed{item['seed']}_k{item['start_step']}"+('_same_replay' if item['same_model_replay'] else '')


def _write_npz(path, **arrays):
    path=Path(path)
    if path.exists():
        raise FileExistsError('Never overwrite a diagnostic trajectory: '+str(path))
    begin=time.perf_counter()
    with path.open('xb') as stream:
        np.savez_compressed(stream,**arrays)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha256(path),archive_write_seconds=time.perf_counter()-begin)


def _save_trajectory(out, folder, key, raw, states, audit, C, *, keep_all):
    nfe=audit['nfe'];start=audit['start_step']
    indices=list(range(start,nfe+1)) if keep_all else sorted({start,nfe,*[int(nfe*t) for t in (.2,.5,.8) if int(nfe*t)>=start]})
    selected=states[np.asarray(indices)-start]
    record=_write_npz(folder/(key+'.npz'),raw=raw,states=selected,state_indices=np.asarray(indices,dtype=np.int64),
                      C=np.asarray(C,dtype=np.float32).reshape(12,3,3))
    record['path']=str(Path(record['path']).relative_to(out))
    record['state_indices']=indices
    record['all_states_saved']=keep_all
    return record


def _input_tensors(case, noise, device):
    return torch.from_numpy(noise.copy()).to(device),torch.from_numpy(np.asarray(case['C'],dtype=np.float32).copy()).to(device)


def _verify_model_unchanged(model, before):
    after=state_sha256(model)
    if before!=after:
        raise RuntimeError('Frozen model state changed during diagnostic integration')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError('Diagnostic model left read-only/no-gradient state')
    return after


def _failure_record(result, exc, folder):
    result['status']='STOPPED'
    result['failure']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    if hasattr(exc,'partial_path'):
        result['failure']['partial_states']=_write_npz(folder/'failed_partial_states.npz',states=exc.partial_path,
            state_indices=np.asarray(exc.partial_state_indices,dtype=np.int64))


def _production_replay(model, noise, C, formal_states, counts, folder, out):
    """One already-registered extra full B/firstcase BF16/50 replay."""
    local=production_counts();tick=time.perf_counter()
    try:
        raw,path,audit=clamped_sample(model,noise,C,local)
        equal=path.dtype==formal_states.dtype and path.shape==formal_states.shape and np.array_equal(path.view(np.uint32),formal_states.view(np.uint32))
        saved=_write_npz(folder/'production_BF16_Euler50_replay.npz',raw=raw,states=path,C=C.cpu().numpy())
        saved['path']=str(Path(saved['path']).relative_to(out))
        if not equal:
            raise RuntimeError('Independent BF16/50 loop differs bitwise from unchanged production sampler')
        return dict(status='PASS',all_51_states_bitwise=True,scope='Full original production clamped_sample versus formal A B/firstcase',
                    sampling=audit,counts=local,count_note='These counts are included in worker total counts, not additional to it',archive=saved,seconds=time.perf_counter()-tick)
    finally:
        for key in ('model_forward_attempts','model_forward_returns','euler_updates','complete_rollouts'):
            counts[key]+=local[key]
        counts['full_rollout_attempts']+=local['rollout_attempts']
        if local['complete_rollouts']!=1:
            counts['sampling_failures']+=1


def run_A(out, label):
    from .common import load_model,load_cases,write_json
    if label not in MODELS:
        raise ValueError('A worker model must be B/J')
    out=Path(out);folder=out/'A'/label;folder.mkdir(parents=True,exist_ok=False)
    path=folder/'results.json';tick=time.perf_counter();counts=new_counts()
    result=dict(stage='A',model=label,status='RUNNING',rows=[],counts=counts,formal_expected=84,
                formal_completed=0,extra_complete_rollouts=0,extra_full_budget=1 if label=='B' else 0,
                no_OT=True,no_geometry_callbacks=True)
    write_json(path,result)
    model=None
    try:
        noises=_noise_mapping(out);seeds=list(noises);cases=load_cases();plan=plan_A(label,seeds)
        result['seeds']=seeds;result['planned_formal_forward_calls']=sum(x['nfe'] for x in plan)
        load_start=time.perf_counter();model,identity=load_model(label);result['load_seconds']=time.perf_counter()-load_start
        result['model_identity']=identity;before=state_sha256(model);result['state_before']=before
        device=next(model.parameters()).device
        for index,item in enumerate(plan):
            result['active_request']=dict(item,index=index);write_json(path,result)
            if counts['complete_rollouts']>=84+(1 if label=='B' else 0):
                raise RuntimeError('A worker full-rollout budget exhausted')
            noise,C=_input_tensors(cases[item['parent'],item['patch']],noises[item['seed']],device)
            raw,states,audit=sample(model,noise,C,nfe=item['nfe'],precision=item['precision'],counts=counts)
            is_first=index==0 and label=='B'
            record=_save_trajectory(out,folder,_key(item,'A'),raw,states,audit,cases[item['parent'],item['patch']]['C'],
                                    keep_all=(item['precision']=='fp32' and item['nfe']==100) or is_first)
            row=dict(item,phase='A',audit=audit,noise_sha256=array_hash(noise),**record)
            result['rows'].append(row);result['formal_completed']+=1;write_json(path,result)
            if is_first:
                result['active_request']=dict(kind='production_replay',**item);write_json(path,result)
                result['production_replay']=_production_replay(model,noise,C,states,counts,folder,out)
                result['extra_complete_rollouts']=1;write_json(path,result)
        expected_F=6800+(50 if label=='B' else 0)
        if len(result['rows'])!=84 or counts['model_forward_attempts']!=expected_F or counts['model_forward_returns']!=expected_F:
            raise RuntimeError('A fixed matrix or forward budget mismatch')
        result['state_after']=_verify_model_unchanged(model,before)
        result['status']='PASS';result['active_request']=None
    except BaseException as exc:
        _failure_record(result,exc,folder)
        raise
    finally:
        result['wall_seconds']=time.perf_counter()-tick;write_json(path,result)
        if model is not None:
            del model
            import gc;gc.collect();torch.cuda.empty_cache()
    return result


def _A_index(out, label):
    path=Path(out)/'A'/label/'results.json';r=json.loads(path.read_text())
    if r.get('status')!='PASS' or r.get('formal_completed')!=84:
        raise RuntimeError('B requires both complete A model matrices')
    return {(x['parent'],x['patch'],x['seed']):x for x in r['rows'] if x['precision']=='fp32' and x['nfe']==100}


def _validate_B_gate_payload(result, label, seeds):
    """Pure metadata gate: all three registered same-model suffixes, no extras."""
    from .common import ASSETS
    if (result.get('status')!='PASS' or result.get('stage')!='B' or result.get('mode')!='gate'
            or result.get('model')!=label or result.get('seeds')!=list(seeds)):
        raise RuntimeError('Missing/failed or mismatched B same-model gate: '+label)
    if result.get('state_before')!=ASSETS[label]['state'] or result.get('state_after')!=ASSETS[label]['state']:
        raise RuntimeError('B same-model gate identity changed: '+label)
    rows=result.get('same_model_replays',[])
    if len(rows)!=3 or result.get('rows')!=[]:
        raise RuntimeError('B same-model gate must contain exactly three replays and zero cross suffixes')
    actual=set()
    for row in rows:
        identity=(row.get('source_model'),row.get('continuation_model'),row.get('parent'),row.get('patch'),row.get('seed'),row.get('start_step'))
        actual.add(identity)
        audit=row.get('audit',{})
        if (row.get('same_model_full_suffix_bitwise') is not True or not row.get('same_model_replay')
                or row.get('precision')!='fp32' or row.get('nfe')!=100
                or audit.get('known_all_states_bitwise') is not True
                or audit.get('integration_steps')!=100-row.get('start_step',-1)):
            raise RuntimeError('B same-model suffix proof failed: '+label)
    expected={(label,label,0,0,int(seeds[0]),cut) for cut in CUTS}
    counts=result.get('counts',{})
    if actual!=expected or any(counts.get(key)!=value for key,value in (
            ('model_forward_attempts',150),('model_forward_returns',150),('euler_updates',150),
            ('suffix_attempts',3),('suffix_returns',3),('complete_rollouts',0),
            ('sampling_failures',0),('model_backward_calls',0),('optimizer_updates',0))):
        raise RuntimeError('B same-model gate matrix/counts differ: '+label)
    return True


def require_all_B_gates(out, seeds):
    """Fail before any cross worker model/array load unless both models passed."""
    out=Path(out);evidence=[]
    for label in MODELS:
        path=out/'B'/label/'gate_results.json'
        if not path.is_file():
            raise RuntimeError('Both B same-model gates must finish before any cross suffix: missing '+label)
        result=json.loads(path.read_text())
        _validate_B_gate_payload(result,label,seeds)
        for row in result['same_model_replays']:
            archived=out/row['path']
            if not archived.is_file() or file_sha256(archived)!=row['sha256']:
                raise RuntimeError('Saved same-model suffix proof changed: '+label)
        evidence.append(dict(model=label,path=str(path.relative_to(out)),sha256=file_sha256(path),
                             same_model_suffixes=3,forward_calls=150))
    return dict(status='PASS',all_six_same_model_suffixes_bitwise=True,models=evidence,
                counts_note='Gate calls are recorded only in gate_results.json, not in cross worker counts')


def run_B(out, continuation_label, *, mode):
    from .common import load_model,load_cases,write_json,registered_seeds
    if continuation_label not in MODELS or mode not in ('gate','cross'):
        raise ValueError('B requires B/J and explicit mode gate/cross')
    out=Path(out);seeds=list(registered_seeds(out))
    # This is deliberately before loading Gaussian arrays, cases, or a model.
    gate_evidence=require_all_B_gates(out,seeds) if mode=='cross' else None
    indices={label:_A_index(out,label) for label in MODELS}
    noises=_noise_mapping(out)
    if list(noises)!=seeds:
        raise RuntimeError('Registered seed order differs from noise bank')
    cases=load_cases();base=out/'B'/continuation_label
    folder=base/('gates' if mode=='gate' else 'cross');folder.mkdir(parents=True,exist_ok=False)
    path=base/('gate_results.json' if mode=='gate' else 'results.json')
    if path.exists():
        raise FileExistsError('Never overwrite a B worker ledger')
    tick=time.perf_counter();counts=new_counts()
    selected=[x for x in plan_B(continuation_label,seeds) if x['same_model_replay']==(mode=='gate')]
    expected_count=3 if mode=='gate' else 36;expected_forward=150 if mode=='gate' else 1800
    result=dict(stage='B',mode=mode,model=continuation_label,status='RUNNING',rows=[],same_model_replays=[],counts=counts,
                cross_suffix_expected=0 if mode=='gate' else 36,same_model_suffix_expected=3 if mode=='gate' else 0,
                global_same_model_gate_evidence=gate_evidence,seeds=seeds,no_OT=True,no_noise_draw=True,
                own_model_endpoints_reused_from_A=True)
    write_json(path,result);model=None
    try:
        model,identity=load_model(continuation_label);result['model_identity']=identity
        before=state_sha256(model);result['state_before']=before;device=next(model.parameters()).device
        for index,item in enumerate(selected):
            result['active_request']=dict(item,index=index);write_json(path,result)
            if counts['suffix_attempts']>=expected_count:
                raise RuntimeError('B suffix attempt budget exhausted')
            source=indices[item['source_model']][item['parent'],item['patch'],item['seed']]
            source_path=out/source['path']
            if file_sha256(source_path)!=source['sha256']:
                raise RuntimeError('A source trajectory archive changed')
            with np.load(source_path,allow_pickle=False) as z:
                if not np.array_equal(z['state_indices'],np.arange(101)) or z['states'].shape!=(101,112,9) or z['states'].dtype!=np.float32:
                    raise RuntimeError('Suffix source is not all101 original FP32 states')
                source_states=z['states'].copy();source_C=z['C'].copy()
            case=cases[item['parent'],item['patch']]
            if source_C.tobytes()!=np.asarray(case['C'],dtype=np.float32).tobytes():
                raise RuntimeError('A source C differs from current registered case')
            noise,C=_input_tensors(case,noises[item['seed']],device)
            initial=torch.from_numpy(source_states[item['start_step']][None].copy()).to(device)
            raw,states,audit=sample(model,noise,C,nfe=100,precision='fp32',counts=counts,initial_state=initial,start_step=item['start_step'])
            record=_save_trajectory(out,folder,_key(item,'B'),raw,states,audit,case['C'],keep_all=True)
            row=dict(item,phase='B',model=continuation_label,precision='fp32',nfe=100,audit=audit,
                     source_archive=dict(path=source['path'],sha256=source['sha256']),**record)
            if item['same_model_replay']:
                equal=np.array_equal(states.view(np.uint32),source_states[item['start_step']:].view(np.uint32))
                row['same_model_full_suffix_bitwise']=equal
                result['same_model_replays'].append(row);write_json(path,result)
                if not equal:
                    raise RuntimeError('Same-model absolute-time suffix does not reproduce original trajectory')
            else:
                result['rows'].append(row);write_json(path,result)
        if (len(result['rows'])+len(result['same_model_replays'])!=expected_count
                or counts['model_forward_attempts']!=expected_forward or counts['model_forward_returns']!=expected_forward):
            raise RuntimeError('B suffix fixed matrix or forward budget mismatch')
        result['state_after']=_verify_model_unchanged(model,before)
        result['status']='PASS';result['active_request']=None
    except BaseException as exc:
        _failure_record(result,exc,folder)
        raise
    finally:
        result['wall_seconds']=time.perf_counter()-tick;write_json(path,result)
        if model is not None:
            del model
            import gc;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--stage',choices=('A','B'),required=True)
    parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--mode',choices=('gate','cross'))
    args=parser.parse_args()
    if args.stage=='B' and args.mode is None:
        parser.error('Stage B requires explicit --mode gate or --mode cross')
    if args.stage=='A' and args.mode is not None:
        parser.error('--mode is only valid for Stage B')
    result=run_A(args.out,args.model) if args.stage=='A' else run_B(args.out,args.model,mode=args.mode)
    print(json.dumps(dict(stage=args.stage,mode=args.mode,model=args.model,status=result['status'],counts=result['counts'])),flush=True)


if __name__=='__main__':
    main()
