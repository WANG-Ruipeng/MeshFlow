"""Freeze existing parents, new paired training randomness and a 12-request panel."""
import copy, json
from . import common as C

def prepare():
    source=C.read(C.PRIOR/'data/manifest.json')
    train=[r for r in source['train'] if r['variant']=='NATURAL']
    assert len(train)==30 and len({r['parent_id'] for r in train})==30
    manifest=dict(schema=C.EXPERIMENT+'_data',train=train,independent_new_parents=0,new_C=0,low_variants=0)
    C.verify({r['case_path']:r['case_sha256'] for r in train})
    old=C.read(C.PRIOR/'data/schedule_PLAIN1500.json')
    coefficients=C.read(C.PRIOR/'data/coefficients_PLAIN1500.json')
    if isinstance(coefficients,dict): coefficients=coefficients['steps']
    steps=copy.deepcopy(old['steps'][:267])
    for e in steps:
        for mi,r in enumerate(e['samples']):
            for field in ('noise_seed','epsilon_seed','time_seed','permutation_seed'):
                r[field]=C.seed('SUPERVISED',e['stage_step'],mi,field)
    C.write(C.OUT/'data/manifest.json',manifest)
    C.write(C.OUT/'data/schedule.json',dict(schema=C.EXPERIMENT+'_schedule',steps=steps))
    C.write(C.OUT/'data/coefficients.json',coefficients[:267])
    ep=C.read(C.PRIOR/'data/eval_manifest.json')
    requests=[q for q in ep['requests'] if any(q['parent_id'].startswith(p) for p in C.PANEL_PREFIXES)]
    assert len(requests)==12 and len({q['parent_id'] for q in requests})==6
    panel=copy.deepcopy(ep); panel.update(requests=requests,experiment=C.EXPERIMENT,
        subset_policy='Predeclared six prefixes; both original n0/n1; no outcome-based replacement',
        baseline_manifest_sha256=C.sha(C.PRIOR/'data/eval_manifest.json'))
    C.write(C.OUT/'data/eval_manifest.json',panel)
    heldout={q['parent_id'] for q in requests if q['eval_group'] in ('DEV','NEW_PARENT_HOLDOUT','HIGH_HOLDOUT')}
    assert not heldout & {r['parent_id'] for r in train}
    rl_cases=[next(r for r in train if r['parent_id'].startswith(p)) for p in C.RL_PREFIXES]
    opportunities=[]
    for i in range(16):
        row=rl_cases[i%4]
        opportunities.append(dict(opportunity=i,case=row,trajectories=[dict(index=j,noise_seed=C.seed('RL',i,j,'initial'),
            policy_noise_seed=C.seed('RL',i,j,'transition')) for j in range(4)]))
    C.write(C.OUT/'data/rl_schedule.json',dict(schema=C.EXPERIMENT+'_RL',opportunities=opportunities))
    C.write(C.OUT/'data/panel_scope.json',dict(requests=12,parents=6,train_prefixes=['a8e987','27eb79','bcb4f5'],
        dev_prefixes=['459f90','f495bd'],heldout_prefixes=['df6fd3'],fixed_original_noises=12,
        new_labels=list(C.NEW_LABELS),cached_label='EXPOSURE',new_raw=72,new_D0=72,
        all_new_raw_before_D0=True,stages=['RAW','BASE_POST','CLEAN_FINAL_POST'],new_score_rows=216,reused_score_rows=36,
        no_checkpoint_selection=True,AI_PROXY_required=True))
    resources=C.register_resource('PREPARATION_CPU_ONLY')
    C.write(C.OUT/'data/preparation.json',dict(status='COMPLETE',resources=resources,model_calls=0,
        parents=30,supervised_steps=267,RL_opportunities=16,eval_requests=12,
        files={str(p):C.sha(p) for p in (C.OUT/'data').glob('*.json') if p.name!='preparation.json'}))
    print(json.dumps(dict(status='COMPLETE',parents=30,eval_requests=12,model_calls=0)))

if __name__=='__main__': prepare()
