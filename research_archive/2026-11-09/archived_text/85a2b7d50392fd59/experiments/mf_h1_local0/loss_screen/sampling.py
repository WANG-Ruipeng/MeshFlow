"""Budgeted orchestration around the unchanged Native T1+Geo Euler sampler.

No geometry loss, GT endpoint, target/noise OT, auxiliary head, guidance, or
repair participates in generation. Training-only relation modules are not
imported. Every sampler's local counts are merged exactly once; no additional
model forward hook is attached. Failure records retain partial actual counts.
"""
from pathlib import Path
import time
import traceback
import json
import numpy as np
import torch

from native_t1.artifacts import array_hash, file_sha256, state_sha256
from native_t1.sampling import make_noise, clamped_sample, new_counts
from .common import ARMS, write, event, release, record, seal

FORMAL_CAP = 240
REPLAY_CAP = 3


def _ledger(r):
    counts = r.setdefault('counts', {}).setdefault('sampling', {})
    for key in new_counts():
        counts.setdefault(key, 0)
    for key in ('formal_output_attempts', 'formal_outputs_saved', 'replay_output_attempts',
                'replay_outputs_verified', 'generation_attempts', 'generation_returns',
                'dispatch_attempts', 'dispatch_failures', 'noise_generation_calls',
                'noise_generation_returns'):
        counts.setdefault(key, 0)
    counts.setdefault('generation_wall_seconds', 0.)
    counts.setdefault('native_sampling_seconds', 0.)
    return counts


def _phase_models(r, phase):
    if phase == 'development':
        if set(r.get('arms', {})) != set(ARMS):
            raise ValueError('All seven arm statuses must be recorded before development')
        if any(v.get('status') in ('STARTING','RUNNING','PENDING','TRAINING') for v in r['arms'].values()):
            raise ValueError('Finish the fixed training screen before generation')
        models=tuple(m for m in ARMS if r['arms'][m].get('status')=='PASS')
    elif phase == 'confirmation':
        selection=r.get('selection', {})
        if selection.get('status')!='FROZEN':
            raise ValueError('Confirmation requires a frozen development selection')
        body={k:v for k,v in selection.items() if k not in ('selection_sha256','artifact')}
        if seal(body)!=selection.get('selection_sha256'):
            raise ValueError('Frozen selection seal changed')
        selected=selection.get('selected_models')
        if not isinstance(selected,list) or len(selected)>2 or len(set(selected))!=len(selected) or any(m not in ARMS[1:] for m in selected):
            raise ValueError('At most two distinct registered nonbaseline candidates')
        models=tuple(['L0']+selected)
        if selection.get('confirmation_models')!=list(models):
            raise ValueError('Frozen confirmation model list is inconsistent')
        for model in models:
            if selection.get('checkpoint_identities',{}).get(model)!=r['arms'][model]['checkpoints']['500']:
                raise ValueError('Selected checkpoint identity changed: '+model)
    else:
        raise ValueError('Expected development or confirmation')
    if 'L0' not in models:
        raise ValueError('The fresh L0 baseline must complete training')
    for model in models:
        arm=r['arms'][model]
        if arm.get('status')!='PASS' or arm.get('completed_updates')!=500 or '500' not in arm.get('checkpoints',{}):
            raise ValueError('Require a PASS additional500/cumulative3000 endpoint: '+model)
        if arm['checkpoints']['500'].get('cumulative_updates')!=3000:
            raise ValueError('Wrong endpoint update identity')
    return models


def _bank(r):
    from .evaluation import load_bank
    info=r['assets']['evaluation_inputs']
    return load_bank(info['path'],info['sha256'])


def _historical_noise_checks(r, noises, seeds):
    """Read only old original Gaussians where present; never use old GT or OT."""
    info=r.get('assets',{}).get('historical_G0_results')
    if info is None:
        return dict(status='NOT_AVAILABLE',checked=0,reason='No registered historical result asset')
    path=Path(info['path'])
    if file_sha256(path)!=info['sha256']:
        raise ValueError('Historical G0 results changed')
    saved=json.loads(path.read_text(encoding='utf-8-sig'))
    checked=[];missing=[]
    for seed in seeds:
        rows=[v for v in saved.get('generated',[]) if v.get('model')=='G0' and int(v['seed'])==seed]
        if not rows:
            missing.append(dict(seed=seed,reason='No historical output with this seed'));continue
        row=sorted(rows,key=lambda v:(int(v['patch']),int(v['parent'])))[0]
        raw=Path(row['raw_path'])
        if file_sha256(raw)!=row['raw_sha256']:
            raise ValueError('Historical raw identity changed')
        with np.load(raw,allow_pickle=False) as archive:
            if 'initial_gaussian' not in archive:
                missing.append(dict(seed=seed,reason='Historical raw has no initial_gaussian'));continue
            old=archive['initial_gaussian']
            if old.dtype!=np.float32 or old.shape!=(1,112,9) or old.tobytes()!=noises[seed].tobytes():
                raise ValueError('Original same-seed Gaussian differs from historical G0')
        checked.append(dict(seed=seed,raw_path=str(raw),raw_sha256=row['raw_sha256'],bitwise=True))
    return dict(status='PASS',checked=len(checked),records=checked,not_available=missing,
                GT_or_OT_used=False)


def _noises(r, out, phase, seeds, *, create):
    key=phase+'_noise';info=r['assets'].get(key)
    if info is None:
        if not create:
            raise ValueError('Replay requires the registered fixed phase noise file')
        path=Path(out)/(phase+'_noise.npz')
        if path.exists():
            raise FileExistsError('Unregistered noise file already exists: '+str(path))
        noises={};counts=_ledger(r)
        for seed in seeds:
            counts['noise_generation_calls']+=1
            noises[seed]=make_noise(seed).numpy().copy()
            counts['noise_generation_returns']+=1
        historical=_historical_noise_checks(r,noises,seeds)
        with path.open('xb') as stream:
            np.savez_compressed(stream,**{f'noise_seed{s}':noises[s] for s in seeds})
        r['assets'][key]=record(path)
        r.setdefault('noise_audits',{})[phase]=dict(status='PASS',seeds=list(seeds),
            generator='native_t1.sampling.make_noise: private CPU torch.Generator(seed)',
            source='original Gaussian, never OT-matched',historical=historical,
            reused_across_models_and_contexts=True)
        write(r,out)
        info=r['assets'][key]
    path=Path(info['path'])
    if file_sha256(path)!=info['sha256']:
        raise ValueError('Fixed phase noise archive changed')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={f'noise_seed{s}' for s in seeds}:
            raise ValueError('Fixed Gaussian archive keys differ from registered seeds')
        result={s:archive[f'noise_seed{s}'].copy() for s in seeds}
    for seed,value in result.items():
        if value.dtype!=np.float32 or value.shape!=(1,112,9) or not np.isfinite(value).all():
            raise ValueError('Fixed Gaussian must be finite FP32[1,112,9]')
        value.setflags(write=False)
    return result


def _load_endpoint(r, model_name):
    from .checkpoint import load
    info=r['arms'][model_name]['checkpoints']['500']
    if file_sha256(info['path'])!=info['sha256']:
        raise ValueError('Endpoint file hash changed: '+model_name)
    model,audit=load(info['path'],protocol_sha=r['protocol_sha256'])
    if audit['arm']!=model_name or audit['state_sha256']!=info['state_sha256'] or audit['cumulative_updates']!=3000:
        raise ValueError('Strict endpoint identity differs: '+model_name)
    if audit['sha256']!=info['sha256'] or audit['auxiliary_network_parameters']!=0 or audit['geometry_loss_modules_constructed']!=0:
        raise ValueError('Unexpected inference network or checkpoint')
    return model,audit


def _validate_arrays(raw, states, noise, C):
    if raw.dtype!=np.float32 or raw.shape!=(112,9) or states.dtype!=np.float32 or states.shape!=(51,112,9):
        raise ValueError('Expected raw FP32[112,9] and all51 FP32 states')
    if C.dtype!=np.float32 or C.shape not in ((12,3,3),(12,9)):
        raise ValueError('Registered original K12 C required')
    if noise.dtype!=np.float32 or noise.shape!=(1,112,9):
        raise ValueError('Registered FP32 original Gaussian required')
    if not np.isfinite(raw).all() or not np.isfinite(states).all() or not np.isfinite(noise).all() or not np.isfinite(C).all():
        raise ValueError('Nonfinite generated or input coordinates')
    expected=np.broadcast_to(C.reshape(1,12,9),(51,12,9))
    if states[:,:12].tobytes()!=expected.tobytes():
        raise ValueError('C changed in one of the 51 stored states')
    if states[0,12:].tobytes()!=noise[0,12:].tobytes():
        raise ValueError('Free initial noise differs from original Gaussian slots')
    if states[-1].tobytes()!=raw.tobytes():
        raise ValueError('Raw endpoint differs from final trajectory state')


def _merge_counts(total, local):
    for key in new_counts():
        total[key]+=int(local[key])


def _sample_one(r,out,model,model_name,parent,patch,seed,phase,noise,C,*,reference=None):
    totals=_ledger(r);replay=reference is not None
    capkey='replay_output_attempts' if replay else 'formal_output_attempts'
    if totals[capkey]>=(REPLAY_CAP if replay else FORMAL_CAP):
        raise RuntimeError('Hard generation budget exceeded before dispatch')
    identity=dict(phase=phase,model=model_name,parent=parent,patch=patch,seed=seed,K=12)
    path=Path(out)/'raw'/phase/f'{model_name}_P{parent}_patch{patch}_seed{seed}.npz'
    if not replay and path.exists():
        raise FileExistsError(path)
    local=new_counts();attempt=dict(identity,replay=replay,status='DISPATCHED',counts=local)
    r.setdefault('sampling_attempts',[]).append(attempt)
    totals[capkey]+=1;totals['dispatch_attempts']+=1
    started=time.perf_counter();native_seconds=0.
    try:
        write(r,out)
        z=torch.from_numpy(noise.copy()).cuda()
        fixed=torch.from_numpy(C.copy()).cuda()
        totals['generation_attempts']+=1
        raw,states,audit=clamped_sample(model,z,fixed,local)
        totals['generation_returns']+=1;native_seconds=float(audit['seconds'])
        if (local['model_forward_attempts'],local['model_forward_returns'],local['euler_updates'],
                local['complete_rollouts'],local['rollout_attempts'])!=(50,50,50,1,1):
            raise RuntimeError('Original 50-step sampler did not complete exactly once')
        if any(local[k]!=0 for k in ('model_backward_calls','optimizer_updates','geometry_gradient_calls','postprocessing_repairs')):
            raise RuntimeError('Generation executed forbidden backward/update/geometry/repair')
        _validate_arrays(raw,states,noise,C)
        row=dict(identity,reused=False,ranked=not replay,sampling=audit,
                 C_all51_bitwise=True,known_all51_bitwise=True,total_faces=112,free_faces=100,
                 output_sha256=array_hash(raw),path_sha256=array_hash(states),
                 noise_sha256=audit['noise_sha256'],initial_gaussian_sha256=array_hash(noise),
                 fixed_noise_asset=r['assets'][phase+'_noise'],
                 geometry_loss_calls=0,geometry_gradient_calls=0,OT_calls=0,
                 auxiliary_network_parameters=0,postprocessing_repairs=0)
        if replay:
            if file_sha256(reference['raw_path'])!=reference['raw_sha256']:
                raise ValueError('Replay reference raw changed')
            with np.load(reference['raw_path'],allow_pickle=False) as old:
                if old['initial_gaussian'].tobytes()!=noise.tobytes() or old['path'].tobytes()!=states.tobytes() or old['output'].tobytes()!=raw.tobytes():
                    raise RuntimeError('Fresh strict reload trajectory is not bitwise identical')
            row.update(reference_path=reference['raw_path'],reference_sha256=reference['raw_sha256'],
                       all51_bitwise_equal=True,fresh_model_load=True)
            r.setdefault('replays',[]).append(row);totals['replay_outputs_verified']+=1
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:
                np.savez_compressed(stream,output=raw,path=states,initial_gaussian=noise)
            # Verify the persisted original Gaussian, not only an in-memory hash.
            with np.load(path,allow_pickle=False) as saved:
                if saved['initial_gaussian'].tobytes()!=noise.tobytes():
                    raise RuntimeError('Saved initial Gaussian differs from fixed phase archive')
            row.update(raw_path=str(path.resolve()),raw_sha256=file_sha256(path))
            r.setdefault('generated',[]).append(row);totals['formal_outputs_saved']+=1
        attempt['status']='PASS'
        event('loss_screen_sample',**identity,replay=replay,formal_saved=totals['formal_outputs_saved'])
        return row
    except BaseException as error:
        attempt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)
        totals['dispatch_failures']+=1
        raise
    finally:
        _merge_counts(totals,local)
        elapsed=time.perf_counter()-started
        totals['generation_wall_seconds']+=elapsed;totals['native_sampling_seconds']+=native_seconds
        attempt.update(wall_seconds=elapsed,native_sampling_seconds=native_seconds)
        write(r,out)


def sample_phase(r,out,phase):
    """Generate each eligible fixed500-update endpoint once; never resume/retry."""
    from .evaluation import _seeds,_expected,_identity
    out=Path(out);models=_phase_models(r,phase);seeds=_seeds(r,phase)
    phases=r.setdefault('sampling_phases',{})
    if phase in phases or any(x.get('phase')==phase for x in r.get('generated',[])):
        raise RuntimeError('Phase already dispatched; no automatic repeated generation')
    totals=_ledger(r)
    if totals['formal_output_attempts']+24*len(models)>FORMAL_CAP:
        raise RuntimeError('Registered phase would exceed total formal budget')
    block=dict(status='STARTING',phase=phase,models=list(models),seeds=list(seeds),
               expected_outputs=24*len(models),completed_outputs=0,models_audit={},
               automatic_retry=False)
    phases[phase]=block;started=time.perf_counter();write(r,out)
    try:
        banks=_bank(r);noises=_noises(r,out,phase,seeds,create=True)
        for name in models:
            model=None;entry=dict(status='STARTING',completed_outputs=0)
            block['models_audit'][name]=entry
            try:
                model,loaded=_load_endpoint(r,name);entry['load']=loaded
                for patch in (0,1,2):
                    for parent in (0,1):
                        C=banks[patch]['cases_by_key'][parent,12]['constraints']
                        for seed in (seeds if patch==0 else seeds[:2]):
                            _sample_one(r,out,model,name,parent,patch,seed,phase,noises[seed],C)
                            entry['completed_outputs']+=1;block['completed_outputs']+=1
                entry['state_after']=state_sha256(model)
                if entry['state_after']!=loaded['state_sha256']:
                    raise RuntimeError('Frozen generator changed during sampling')
                rows=[v for v in r['generated'] if v['phase']==phase and v['model']==name]
                if len(rows)!=24 or {_identity(v) for v in rows}!=_expected((name,),seeds):
                    raise RuntimeError('Registered model output matrix incomplete or duplicated')
                entry['status']='PASS'
            except BaseException as error:
                entry.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)
                if name=='L0' or not isinstance(error,Exception):
                    raise
                # A failed candidate is retained but not retried or ranked by a
                # partial matrix. Other independently trained candidates proceed.
            finally:
                if model is not None and 'state_after' not in entry:
                    try:
                        entry['state_after']=state_sha256(model)
                        entry['frozen_state_unchanged']=entry['state_after']==entry.get('load',{}).get('state_sha256')
                    except Exception as error:
                        entry['state_hash_after_failure_error']=repr(error)
                model=None;release();write(r,out)
        block['status']='PASS' if all(v['status']=='PASS' for v in block['models_audit'].values()) else 'PASS_WITH_FAILED_CANDIDATES'
        block['failed_models']=[m for m,v in block['models_audit'].items() if v['status']!='PASS']
        return block
    except BaseException as error:
        block.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        block['seconds']=time.perf_counter()-started;write(r,out)


def replay_once(r,out):
    """One prespecified fresh-load L0/P0/patch0/development-first-seed replay."""
    from .evaluation import _seeds
    if 'sampling_replay' in r or r.get('replays'):
        raise RuntimeError('The one registered replay was already attempted')
    seeds=_seeds(r,'development');seed=seeds[0]
    matches=[v for v in r.get('generated',[]) if (v.get('phase'),v['model'],int(v['parent']),int(v['patch']),int(v['seed']))==('development','L0',0,0,seed)]
    if len(matches)!=1:
        raise ValueError('Fixed baseline replay reference is absent or duplicated')
    _phase_models(r,'development')
    block=dict(status='STARTING',model='L0',parent=0,patch=0,seed=seed,selection_by_result=False)
    r['sampling_replay']=block;model=None;started=time.perf_counter();write(r,out)
    try:
        banks=_bank(r);noises=_noises(r,Path(out),'development',seeds,create=False)
        model,loaded=_load_endpoint(r,'L0');block['load']=loaded
        row=_sample_one(r,out,model,'L0',0,0,seed,'development',noises[seed],
                        banks[0]['cases_by_key'][0,12]['constraints'],reference=matches[0])
        after=state_sha256(model)
        if after!=loaded['state_sha256']:
            raise RuntimeError('Fresh replay changed frozen generator')
        block.update(status='PASS',state_after=after,all51_bitwise_equal=True,row=row)
        return block
    except BaseException as error:
        block.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)
        raise
    finally:
        block['seconds']=time.perf_counter()-started;model=None;release();write(r,out)


def run_cpu_fixtures():
    """Only scalar/count/array guards, no model construction, sampling, or RNG."""
    raw=np.zeros((112,9),np.float32);noise=np.ones((1,112,9),np.float32)
    C=np.arange(108,dtype=np.float32).reshape(12,3,3)/100
    states=np.zeros((51,112,9),np.float32);states[:,:12]=C.reshape(12,9)
    states[0,12:]=noise[0,12:];raw[:12]=C.reshape(12,9)
    _validate_arrays(raw,states,noise,C)
    for kind in ('C','noise','endpoint','nonfinite','shape'):
        a=raw.copy();b=states.copy()
        if kind=='C':b[17,0,0]+=1
        elif kind=='noise':b[0,12,0]+=1
        elif kind=='endpoint':a[-1,0]+=1
        elif kind=='nonfinite':b[12,15,0]=np.nan
        else:b=b[:50]
        try:_validate_arrays(a,b,noise,C)
        except ValueError:pass
        else:raise AssertionError('Guard accepted '+kind)
    r={};totals=_ledger(r);local=new_counts();local.update(model_forward_attempts=7,model_forward_returns=6,euler_updates=5,rollout_attempts=1)
    _merge_counts(totals,local)
    assert totals['model_forward_attempts']==7 and totals['model_forward_returns']==6 and totals['euler_updates']==5 and totals['complete_rollouts']==0
    arms={m:dict(status='PASS',completed_updates=500,checkpoints={'500':dict(path=m,sha256=m,state_sha256=m,cumulative_updates=3000)}) for m in ARMS}
    value={'arms':arms};assert _phase_models(value,'development')==ARMS
    selected=dict(status='FROZEN',selected_models=['L2','L6'],confirmation_models=['L0','L2','L6'],checkpoint_identities={m:arms[m]['checkpoints']['500'] for m in ('L0','L2','L6')})
    selected['selection_sha256']=seal(selected);value['selection']=selected
    assert _phase_models(value,'confirmation')==('L0','L2','L6')
    selected['selected_models']=['L1']
    try:_phase_models(value,'confirmation')
    except ValueError:pass
    else:raise AssertionError('Changed selection accepted')
    return dict(status='PASS',checks=10,model_forwards=0,model_constructions=0,sampling_calls=0,
                OT_calls=0,autograd_calls=0,optimizer_updates=0,random_draws=0)
