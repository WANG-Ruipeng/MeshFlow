"""Read only the completed first100 steps; no torch import or model calls."""
import sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from r2.pilots.seam_quality_wave1_wave2_v1 import training_data as D
OUT=Path(__file__).resolve().parent
EXP=OUT.parent

def read_prefix(path,limit=100,context=False):
    rows=[]
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            r=json.loads(line);step=r.get('context',{}).get('local_step',0) if context else r.get('local_step',0)
            if step>limit:break
            rows.append(r)
    return rows

go=D.read(EXP/'training_dispatch_go.json')
for p,h in go['code_hashes'].items():assert D.digest(p)==h,p
reg=D.read(EXP/'training_registration.json')
manifest=D.read(D.resolve(reg['shared_data_manifest']))
folders={a:EXP/'training'/a for a in D.ARMS}
folders['PLAIN_S1000']=ROOT/'experiments/r2_seam_pool47_couple_v1/training/PLAIN3000'
data={a:{k:read_prefix(f/(v+'.jsonl')) for k,v in [('exposure','exposure'),('logs','training_log'),('diags','geometry_diagnostics')]} for a,f in folders.items()}
wanted=[(s,m) for s in range(1,101) for m in range(8)]
for a,d in data.items():
    assert [r['local_step'] for r in d['logs']]==list(range(1,101)),a
    for key in ('exposure','diags'):assert [(r['local_step'],r['microbatch']) for r in d[key]]==wanted,a
for index,base in enumerate(data['PLAIN_S1000']['exposure']):
    rec=manifest['steps'][base['local_step']-1]['samples'][base['microbatch']]
    for arm in D.ARMS:
        row=data[arm]['exposure'][index]
        for key in ('uid','N','K','alpha','t','sample_index','free_coordinate_count'):assert row[key]==base[key],(arm,index,key)
        assert D.resolve(row['path'])==D.resolve(base['path'])==D.resolve(rec['path'])
        assert row['input_sha256']==rec['sha256']
result={}
for arm in D.ARMS:
    folder=folders[arm]
    inits=[D.read(p) for p in (folder/'attempts').glob('*/initialization.json')]
    init=next(r for r in inits if r['global_step']==13000)
    assert init['generator_state_sha256']==reg['source_generator_sha256']
    assert init['rng_sha256']==reg['source_rng_sha256']
    assert init['source_optimizer_sha256']==reg['source_optimizer_sha256']
    assert init['optimizer_restore']['exact_retained_step_and_moments']
    assert not init['head_loaded'] and init['head_forward_calls']==0
    frozen=next(r for r in read_prefix(folder/'frozen_state_checks.jsonl') if r['local_step']==100)
    assert frozen['status']=='PASS' and frozen['initial']==frozen['actual']==init['initial_frozen_state_sha256']
    calls=read_prefix(folder/'calls.jsonl',context=True)
    begins=Counter(r['kind'] for r in calls if r['event']=='CALL_BEGIN')
    done=Counter(r['kind'] for r in calls if r['event']=='CALL_DONE')
    assert begins==done==Counter(train_forward=800,train_backward=800,optimizer_update=100)
    for step,log in enumerate(data[arm]['logs'],1):
        diags=data[arm]['diags'][(step-1)*8:step*8]
        plain=sum(r['sample_free_coordinate_denominator'] for r in diags)
        weighted=sum(r['sample_weighted_free_coordinate_denominator'] for r in diags)
        assert plain==log['free_coordinate_denominator'] and np.isclose(weighted,log['weighted_free_coordinate_denominator'])
        for r in diags:
            assert sum(r[k+'_scalar_count'] for k in ('quality_clean_bulk','quality_dirty_bulk','quality_seam_third'))==r['sample_free_coordinate_denominator']
            assert np.isclose(sum(r[k+'_velocity_sse'] for k in ('quality_clean_bulk','quality_dirty_bulk','quality_seam_third')),r['all_free_velocity_sse'])
        loss_sse=sum(r['quality_weighted_velocity_sse' if arm=='CLEAN_BULK_FM' else 'all_free_velocity_sse'] for r in diags)
        divisor=weighted if arm=='CLEAN_BULK_FM' else plain
        assert np.isclose(loss_sse/divisor,log['FM'],rtol=3e-6,atol=1e-7),(arm,step)
        assert np.isclose(plain/divisor,log['effective_unit_seam_coefficient_ratio_vs_plain'])
        assert all(np.isfinite(v) and v>0 for v in log['preprocessing_gradient_norms'].values())
    result[arm]=dict(status='PASS',updates_checked=100,examples_checked=800,calls=dict(done),
        source_generator_sha256=init['generator_state_sha256'],source_rng_sha256=init['rng_sha256'],
        actual_restored_optimizer_sha256=init['optimizer_restore']['actual_optimizer_sha256'],
        retained_parameter_states=init['optimizer_restore']['retained_parameter_states'],
        removed_frozen_parameter_states=init['optimizer_restore']['removed_frozen_parameter_states'],
        exact_retained_moment_and_step13000_receipt=True,frozen_step100_receipt=frozen,
        source_identity_matches_existing_PLAIN=True,loss_recomputed_from_detached_SSE=True,
        all_unweighted_categories_partition_exact=True,preprocessing_gradients_all_finite_nonzero=True,
        prefix_records_sha256=hashlib.sha256(json.dumps(data[arm],sort_keys=True).encode()).hexdigest())
receipt=dict(status='PASS',scope='Existing first100 normal updates only; no checkpoint/model load or additional forward',
    neural_network_constructions=0,neural_forwards=0,neural_backwards=0,extra_optimizer_updates=0,
    registered_code_hashes_unchanged=True,arms=result,script_sha256=D.digest(__file__))
D.write_new(OUT/'early_log_check.json',receipt)
print(json.dumps(receipt,indent=2))
