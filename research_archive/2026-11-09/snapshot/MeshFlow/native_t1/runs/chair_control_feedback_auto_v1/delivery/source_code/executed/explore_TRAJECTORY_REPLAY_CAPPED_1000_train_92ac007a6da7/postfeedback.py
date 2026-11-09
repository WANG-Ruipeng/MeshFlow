"""Registered fixed-input post-training rollout diagnostic, not a DEV substitute."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LP_NUM_THREADS'):os.environ[name]='2'
from .common import *
from .run import call_module,ready
from .evaluation import score_set
from .aggregation import paired
from experiments.mf_h1_local0.chair_control_usability_quick_v1.scoring import write_csv

def main():
    out=OUT;plan=read_json(out/'decisions/posttraining_feedback_plan.json')
    manifest=out/'manifests/TRAIN_FEEDBACK.json';assert digest(manifest)==plan['manifest_sha256']
    assert plan['status']=='REGISTERED' and plan['extra_rollouts_max']==48
    for label in plan['labels']:
        identity=label.rsplit('_',1)[0]
        done=read_json(out/'training'/f'{identity}_complete.json')
        assert done['status']=='PASS' and done['steps']==1000
        checkpoint=out/'checkpoints'/f'{label}.pt';assert checkpoint.exists()
        if not ready(out,label,'TRAIN_FEEDBACK'):
            args=['sample','--out',str(out),'--label',label,'--checkpoint',str(checkpoint),'--manifest',str(manifest)]
            process,handle=call_module('worker',args,out/'logs'/f'{label}_postfeedback.log')
            result=process.wait();handle.close()
            if result:raise RuntimeError('Postfeedback rollout failed; inspect accounting before any recovery')
        if not (out/'evaluation'/label/'TRAIN_FEEDBACK/summary.json').exists():
            result=score_set(manifest,label,out/'runtime'/label,out,workers=2)
            assert result['status']=='COMPLETE'
    requests=read_json(manifest)['rows'];reference=read_json(out/'evaluation/START/TRAIN_FEEDBACK/per_output.json')
    folder=out/'derived/posttraining_feedback';folder.mkdir(parents=True,exist_ok=True)
    for label in plan['labels']:
        rows=read_json(out/'evaluation'/label/'TRAIN_FEEDBACK/per_output.json')
        comparisons={'vs_START':paired(rows,reference,requests)}
        if label=='explore_TRAJECTORY_REPLAY_1000':
            control=read_json(out/'evaluation/explore_REPLAY_FM_CONTROL_1000/TRAIN_FEEDBACK/per_output.json')
            comparisons['vs_matched']=paired(rows,control,requests)
        for name,result in comparisons.items():
            freeze_json(folder/f'{label}_{name}.json',dict(role='TRAINING_COHORT_DIAGNOSTIC_ONLY',**result))
            for key in ('outputs','parents','noise'):write_csv(folder/f'{label}_{name}_{key}.csv',result[key])
    freeze_json(folder/'complete.json',dict(status='PASS',registered_plan_sha256=digest(out/'decisions/posttraining_feedback_plan.json'),
        labels=plan['labels'],outputs=48,training_or_feedback_targets_modified=False,
        note='No promotion from this panel; it measures actual post-training outputs on the original feedback parents/noises.'))
    print('POSTTRAINING_FEEDBACK COMPLETE 48 real outputs',flush=True)

if __name__=='__main__':main()
