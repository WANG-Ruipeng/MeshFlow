"""Register calibration and held-out tests before any new model calls."""
from .bootstrap import bind, ROOT, OUT, OLD
D, E = bind()
import argparse
import json
from pathlib import Path
import numpy as np

VERSION = 'R2_WINDOW_POLICY_RL_V1'
WINDOW = list(range(10, 30))
LABELS = {'FULL_DET': 0., 'S005': .005, 'S010': .01, 'S020': .02}
FIELDS = ('condition_id','parent_id','task_id','N','K','condition_source','condition_source_sha256',
          'split','condition_type','region_type','tier','variant','alpha','selection_order')

def freeze(q0, rid, namespace, index, extra=None):
    q = {k:q0[k] for k in FIELDS if k in q0}
    q.update(request_id=rid,base_request_id=rid,candidate=index,
             condition_type='SEAT_FULL',region_type='SEAT',tier='FULL',
             historical_metadata_policy='Actual frozen FULL array/source; historical HALF descriptors not inherited',
             panel=namespace,evaluation_noise_partition='FRESH')
    if extra: q.update(extra)
    c = D.arrays(q['condition_source'])['C'].reshape(q['K'],3,3)
    q = D._freeze_request(q,c,VERSION+'/'+namespace,OUT)
    seed = D.noise_seed(VERSION+'/'+namespace+'/TRANSITION',rid)
    epsilon = np.zeros((20,1,q['N'],9),np.float32)
    epsilon[:,:,q['K']:] = np.random.RandomState(seed).randn(20,1,q['N']-q['K'],9).astype(np.float32)
    p = D.save_npz(OUT/'inputs/transitions'/(rid+'.npz'),epsilon=epsilon,k=np.asarray(WINDOW,np.int64))
    q.update(transition_noise_path=str(p),transition_noise_sha256=D.sha(p),
             transition_Gaussian_sha256=D.array_sha(epsilon),transition_noise_seed=seed,
             transition_noise_namespace=VERSION+'/'+namespace+'/TRANSITION')
    return q

def prepare():
    target=OUT/'inputs/manifest.json'
    if target.exists():
        m=D.read(target)
        for q in m['calibration_requests']+m['evaluation_requests']: D.load_request(q)
        return m
    old=D.read(OLD/'inputs/manifest.json')
    previous_frozen=D.read(ROOT/'experiments/r2_nft_training_c_frozen_v1/inputs/manifest.json')
    conditions=old['train_conditions']
    assert len(conditions)==8
    train_first={c['condition_id']:next(q for q in old['training_requests'] if q['condition_id']==c['condition_id']) for c in conditions}
    dev={q['condition_id']:q for q in old['evaluation_requests'] if q['split']=='DEV'}
    assert len(dev)==3
    queries=D.read(OLD/'inputs/training_queries.json')['queries']
    queries=queries+[q for q in old['evaluation_queries'] if q['condition_id'] in dev]
    assert len(queries)==11 and len({q['condition_id'] for q in queries})==11
    calib=[];evaluation=[]
    for c in conditions:
        q0=train_first[c['condition_id']]
        for i in range(8):
            calib.append(freeze(q0,'WPCAL__'+c['task_id']+'__n'+str(i),'CALIBRATION',i))
    all_conditions=list(train_first.values())+list(dev.values())
    for q0 in all_conditions:
        for i in range(4):
            evaluation.append(freeze(q0,'WPEVAL__'+q0['parent_id']+'__n'+str(i),'FINAL_EVALUATION',i))
    old_requests=old['training_requests']+old['evaluation_requests']+previous_frozen['evaluation_requests']
    prior_seeds={q['noise_seed'] for q in old_requests}; prior_hashes={q['Gaussian_sha256'] for q in old_requests}
    qs=calib+evaluation
    assert not ({q['noise_seed'] for q in qs}&prior_seeds)
    assert not ({q['Gaussian_sha256'] for q in qs}&prior_hashes)
    assert len({q['noise_seed'] for q in qs})==108
    assert len({q['transition_noise_seed'] for q in qs})==108
    assert not ({q['noise_seed'] for q in qs}&{q['transition_noise_seed'] for q in qs})
    protected={}
    for p in [OLD/'inputs/manifest.json',OLD/'inputs/training_queries.json',
              ROOT/'experiments/r2_nft_training_c_frozen_v1/inputs/manifest.json',Path(old['source_binding']['path'])]:
        protected[str(p)]=D.sha(p)
    assert protected[old['source_binding']['path']]==old['source_binding']['sha256']
    for q in qs:
        p=q['condition_source'];h=D.sha(p)
        assert h==q['condition_source_sha256'];protected[p]=h
    for q in queries:
        assert D.sha(q['path'])==q['sha256'];protected[q['path']]=q['sha256']
    m=dict(version=VERSION,status='FROZEN',source_binding=old['source_binding'],train_conditions=conditions,
           calibration_requests=calib,evaluation_requests=evaluation,evaluation_queries=queries,queries=queries,
           protected_hashes=protected,window=WINDOW,calibration_labels=LABELS,dt=.02,
           training_requests_extension='inputs/training_manifest.json, only after separately frozen finite training budget',
           noise_novelty=dict(previous_requests=len(old_requests),new_initial_arrays=108,new_transition_arrays=108,
                              initial_seed_collisions=0,initial_array_collisions=0,selection='fixed SHA namespaces, no resampling'),
           scope='Same eight TRAIN C; three historical DEV C; fresh noises do not establish unseen-parent generalization')
    D.write(target,m)
    registration=dict(experiment=VERSION,status='CALIBRATION_FROZEN',manifest_sha256=D.sha(target),
        nearest_prior='R29 NFT off-policy endpoint adaptation; now true windowed stochastic-action score-function learning, no endpoint regression',
        attachment_has_measured_results=False,
        calibration=dict(window=WINDOW,sigma_interpretation='constant per-transition FP32 state standard deviation, NOT sigma0*sqrt(dt)',
                         labels=LABELS,conditions=8,trajectories_per_C=8,paired_initial_and_transition_arrays=True,
                         RAW=256,POST=256,generator_forwards=12800,D0_forwards=256,original_scores=512),
        gate=dict(origin='Preregistered bounded engineering viability thresholds chosen locally; not literature guarantees',
                  all_C_and_finite_and_original_scorer_PASS=True,mixed_conditions_minimum=4,
                  stochastic_connection_rate_minimum_fraction_of_DET=.8,
                  paired_DET_positive_retention_minimum=.75,
                  max_mean_CF_delta_percentage_points=2.,max_mean_FF_delta_percentage_points=3.,max_mean_I_delta_percentage_points=1.,
                  max_new_POST_body_NO_relative_to_DET_YES=2,max_drop_POST_body_YES_count=2,
                  body_review='Actual model-blind AI_PROXY all 256 POST; UNKNOWN not imputed YES; do not certify human quality',
                  selection='Smallest sigma satisfying every gate; no ranking by best connection rate',
                  if_no_sigma_passes='STOP without RL updates; retain complete negative calibration evidence'),
        proposed_training=dict(batches=64,B=1,G=8,maximum_optimizer_updates=64,rollouts=512,
                  inactive_entire_batch='SKIP optimizer.step incl AdamW/momentum; count batch, no redraw or makeup',
                  optimizer='fresh AdamW on rank4 alpha4 LoRA only; exact hyperparameters freeze before formal updates',
                  reward='binary C_F_valid_edge_count>0 after same single official D0/GEO_C/source winding',
                  objective='RLOO REINFORCE one step per fresh batch; sum full d=9(N-K) joint logpdf and 20 steps; no sigma or d averaging',
                  deterministic_steps='physical LoRA bypass via reference branch; no trainable phi dependence',
                  forward_precision='BF16 forward, FP32 parameters/integration; joint logpdf and KL arithmetic FP64',
                  after_step_KL='old-buffer states, full joint Gaussian KL; diagnostic not extra loss',
                  budget_status='User preference pending; freeze accepted or stated bounded default before training'),
        final_evaluation=dict(requests=44,TRAIN=32,DEV=12,models=['FULL_DET','FULL_STOCH','RL_STOCH'],RAW=132,POST=132,
                  generator_forwards=6600,D0_forwards=132,RL_DET=False,
                  outcome='RL vs same stochastic sampler first; adoption/body/legal length/CF/FF/I and deterministic baseline separately',
                  noise='Frozen before calibration; distinct from calibration, prior NFT, later training; no checkpoint selection'),
        scientific_retry='No automatic replay after a GPU failure; preserve attempts and decide explicit recorded engineering recovery within budget',
        finish='Stop after gate failure or finite training+full evaluation; no default/commit/push')
    D.write(OUT/'registration.json',registration)
    from r2.pilots.terminal_reward_nft_v1 import common as W
    D.write(OUT/'resources_before_preparation.json',W.resource_snapshot())
    print(json.dumps({'status':'FROZEN','calibration_requests':len(calib),'evaluation_requests':len(evaluation),'model_calls':0}))
    return m

def training(batches):
    m=prepare();target=OUT/'inputs/training_manifest.json'
    if target.exists():
        t=D.read(target);assert t['batches']==batches;return t
    assert batches in (64,128)
    old=D.read(OLD/'inputs/manifest.json');requests=[];groups=[]
    ordered=sorted(m['train_conditions'],key=lambda c:c['selection_rank'])
    for b in range(batches):
        c=ordered[b%8];q0=next(q for q in old['training_requests'] if q['condition_id']==c['condition_id'])
        group=[]
        for i in range(8):
            q=freeze(q0,f'WPTRAIN__b{b:03d}__'+c['task_id']+f'__n{i}','TRAINING',i,dict(batch_index=b,cycle=b//8))
            requests.append(q);group.append(q['request_id'])
        groups.append(dict(batch_index=b,condition_id=c['condition_id'],requests=group))
    previous=D.read(ROOT/'experiments/r2_nft_training_c_frozen_v1/inputs/manifest.json')
    used=m['calibration_requests']+m['evaluation_requests']+old['training_requests']+old['evaluation_requests']+previous['evaluation_requests']
    assert not ({q['noise_seed'] for q in used}&{q['noise_seed'] for q in requests})
    assert not ({q['Gaussian_sha256'] for q in used}&{q['Gaussian_sha256'] for q in requests})
    assert len({q['noise_seed'] for q in requests})==len(requests)
    allnew=m['calibration_requests']+m['evaluation_requests']+requests
    assert len({q['transition_noise_seed'] for q in allnew})==len(allnew)
    assert not ({q['noise_seed'] for q in allnew}&{q['transition_noise_seed'] for q in allnew})
    result=dict(status='FROZEN',batches=batches,max_optimizer_updates=batches,B=1,G=8,
                training_requests=requests,groups=groups,parent_exposure={c['parent_id']:batches for c in ordered},
                zero_advantage='Skip whole inactive batch without replacement; not counted as optimizer update',
                evaluation_manifest_sha256=D.sha(OUT/'inputs/manifest.json'))
    D.write(target,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training-batches',type=int);args=p.parse_args()
    training(args.training_batches) if args.training_batches else prepare()
