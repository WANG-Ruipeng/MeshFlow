"""Resume only unstarted sampling after one audited Windows/WSL JSON lock."""
from native_t1.runtime import configure_stable_runtime
from pathlib import Path
import json, time, traceback
import numpy as np
from experiments.mf_h1_local0 import tgeo_m10_run as run
from experiments.mf_h1_local0.tgeo_m10_evaluation import load_bank

OUT=Path('experiments/mf_h1_local0/outputs/tgeo_m10_20260930_173200').resolve()

def identity(row):
    return row['model'],row['parent'],row['patch'],row['seed']

def main():
    r=json.loads((OUT/'results.json').read_text())
    if r['status']!='STOPPED' or r.get('io_recovery'):
        raise RuntimeError('Only the recorded first IO interruption is eligible')
    if r.get('error')!="PermissionError(13, 'Permission denied')":
        raise RuntimeError('Not the audited file replace error')
    if "counts['rollout_attempts']+=1;write(r,out)" not in r.get('traceback',''):
        raise RuntimeError('Wrong interruption location')
    counts=r['counts']
    expected=dict(optimizer_updates=1000,model_forward_attempts=9552,model_forward_returns=9552,
        sampling_forward_attempts=1550,sampling_forward_returns=1550,rollout_attempts=32,
        complete_rollouts=31,formal_rollouts=31,replay_rollouts=0,euler_updates=1550)
    if any(counts[k]!=v for k,v in expected.items()) or len(r['sampling_attempts'])!=31 or len(r['generated'])!=31:
        raise RuntimeError('The recorded no-forward interruption state differs')
    if run.seal(r['protocol'])!=r['protocol_sha256']: raise RuntimeError('Protocol changed')
    run.verify_sources(r); run.verify_assets(r)
    if any(r['arms'][a]['status']!='PASS' for a in run.ARMS): raise RuntimeError('Training incomplete')
    for row in r['generated']:
        if run.file_sha256(row['raw_path'])!=row['raw_sha256']: raise RuntimeError('Existing raw changed')
    if len(list((OUT/'raw').glob('*.npz')))!=31: raise RuntimeError('Unrecorded raw output exists')
    snapshot=OUT/'sampling_io_interruption.json'
    with snapshot.open('x') as f: json.dump(r,f,indent=2)
    r['io_recovery']=dict(status='RUNNING',error=r['error'],original_traceback=r['traceback'],
        interruption_snapshot=run.record(snapshot),source=run.record(__file__),
        stage='Before model call: sampling dispatcher JSON atomic rename',
        cause='Windows read handle blocked WSL atomic replace; subsequent observation uses WSL reads only',
        preserved_formal_outputs=31,unstarted_dispatch_attempts=1,
        interrupted_model_forwards=0,interrupted_complete_rollouts=0,
        resumed_missing_formal=41,planned_replays=1,training_restarted=False,
        original_execution_sources_unchanged=True,no_sampler_or_loss_changes=True)
    for key in ('error','traceback','automatic_retry'): r.pop(key,None)
    r['status']='SAMPLING_IO_RECOVERY'
    run.write(r,OUT)
    banks=load_bank(r['assets']['evaluation_inputs']['path'],r['assets']['evaluation_inputs']['sha256'])
    with np.load(r['assets']['evaluation_noise']['path'],allow_pickle=False) as saved:
        noises={s:saved[f'noise_seed{s}'].copy() for s in run.SEEDS}
    tick=time.perf_counter()
    for name in run.MODELS:
        expected_ids=[(name,p,j,s) for j in (0,1,2) for p in (0,1)
                      for s in (run.SEEDS if j==0 else run.SEEDS[:2])]
        done={identity(row) for row in r['generated']}
        missing=[key for key in expected_ids if key not in done]
        if not missing and name!='Geo_M10': continue
        r['active_stage']='sampling_'+name
        model,_=run.load_model(r,name); before=run.state_sha256(model)
        for _,parent,patch,seed in missing:
            run._sample_one(r,OUT,model,name,parent,patch,seed,noises[seed],
                            banks[patch]['cases_by_key'][parent,12]['constraints'])
            run.event('sampling_recovered',formal=counts['formal_rollouts'],model=name)
        if name=='Geo_M10':
            reference=next(row for row in r['generated'] if identity(row)==(name,0,0,run.SEEDS[0]))
            run._sample_one(r,OUT,model,name,0,0,run.SEEDS[0],noises[run.SEEDS[0]],
                           banks[0]['cases_by_key'][0,12]['constraints'],reference=reference)
        if run.state_sha256(model)!=before: raise RuntimeError('Sampling changed weights')
        del model;run.release()
    wanted=dict(model_forward_attempts=11652,model_forward_returns=11652,optimizer_updates=1000,
        formal_rollouts=72,replay_rollouts=1,complete_rollouts=73,rollout_attempts=74,
        sampling_forward_attempts=3650,sampling_forward_returns=3650,euler_updates=3650)
    if any(counts[k]!=v for k,v in wanted.items()): raise RuntimeError('Unexpected recovery final counts')
    r['io_recovery'].update(status='PASS',resumption_wall_seconds=time.perf_counter()-tick,
        initial_sampling_wall_seconds='NOT_RECORDED_BEFORE_IO_EXIT',
        original_raw_files_preserved=True,final_model_forward_count=11652)
    r['timing']['sampling_resumption_wall_seconds']=r['io_recovery']['resumption_wall_seconds']
    r['timing']['model_sampling_seconds']=sum(x['sampling']['seconds'] for x in r['generated']+r['replays'])
    r['timing']['sampling_total_seconds']=None
    r['timing']['sampling_total_status']='NOT_RECORDED: initial wall timer interrupted; per-rollout model timing complete'
    r['status']='SAMPLING_PASS';run.write(r,OUT)
    run.event('sampling_pass_after_io',counts=counts)

if __name__=='__main__':
    try: main()
    except BaseException as error:
        if (OUT/'results.json').exists():
            r=json.loads((OUT/'results.json').read_text())
            r.update(status='STOPPED',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)
            run.write(r,OUT)
        raise
