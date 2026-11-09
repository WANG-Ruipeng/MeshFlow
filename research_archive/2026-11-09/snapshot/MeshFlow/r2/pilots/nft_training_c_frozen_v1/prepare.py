"""Register all32 unseen Gaussian arrays and identical source queries before GPU."""
from .bootstrap import bind, OLD, OUT, ROOT
D,E=bind()
from collections import Counter
import json

VERSION='R2_NFT_TRAINING_C_FROZEN_V1'

def run():
    target=OUT/'inputs/manifest.json'
    if target.exists():
        m=D.read(target)
        for q in m['evaluation_requests']:D.load_request(q)
        for p,h in m['protected_hashes'].items():
            if D.sha(p)!=h:raise ValueError('Protected source changed: '+p)
        return m
    old_path=OLD/'inputs/manifest.json'
    m0=D.read(old_path)
    queries_path=OLD/'inputs/training_queries.json'
    queries=D.read(queries_path)['queries']
    conditions=m0['train_conditions']
    if len(conditions)!=8 or len(queries)!=8:raise ValueError('Must retain all original8 reward-training conditions')
    protected={str(old_path):D.sha(old_path),str(queries_path):D.sha(queries_path)}
    def protect(p,expected=None):
        h=D.sha(p)
        if expected is not None and h!=expected:raise ValueError('Source mismatch: '+str(p))
        protected[str(D.resolve(p))]=h
    protect(m0['source_binding']['path'],m0['source_binding']['sha256'])
    for arm in ['NFT','SELF_FM']:
        cp=OLD/f'checkpoints/{arm}/round15.pt';receipt=D.read(cp.with_suffix('.json'))
        if (receipt['step'],receipt['round_index'])!=(256,15):raise ValueError('Final endpoint differs')
        protect(cp,receipt['sha256']);protect(cp.with_suffix('.json'))
    oldqs=m0['training_requests']+m0['evaluation_requests']
    oldseeds={q['noise_seed'] for q in oldqs}
    oldhash={q['Gaussian_sha256'] for q in oldqs}
    oldfree={q.get('free_Gaussian_sha256') for q in oldqs}
    qs=[]
    for c in conditions:
        protect(c['condition_source'],c['condition_source_sha256'])
        ca=D.arrays(c['condition_source'])['C'].reshape(c['K'],3,3)
        q0=next(q for q in m0['training_requests'] if q['condition_id']==c['condition_id'])
        if D.array_sha(ca)!=q0['C_sha256']:raise ValueError('Condition changed')
        for ni in range(4):
            rid='NFTTRAINCFRESH__'+c['task_id']+'__n'+str(ni)
            q={k:q0[k] for k in ['condition_id','parent_id','task_id','N','K','condition_source','condition_source_sha256','split','condition_type','region_type','tier','variant','alpha','selection_order']}
            q.update(request_id=rid,base_request_id=rid,candidate=ni,evaluation_noise_partition='FRESH',panel='REWARD_TRAIN_C_FRESH_NOISE_FROZEN',reward_training_condition=True)
            q=D._freeze_request(q,ca,VERSION+'/EVAL_ONLY',OUT)
            if q['noise_seed'] in oldseeds or q['Gaussian_sha256'] in oldhash or q['free_Gaussian_sha256'] in oldfree:
                raise ValueError('Old-noise collision; no silent redraw')
            qs.append(q)
    if len({q['noise_seed'] for q in qs})!=32 or set(Counter(q['condition_id'] for q in qs).values())!={4}:
        raise ValueError('Fresh-noise count or uniqueness mismatch')
    if {c['condition_id'] for c in conditions}!={q['condition_id'] for q in queries}:raise ValueError('Query identity mismatch')
    for q in queries:protect(q['path'],q['sha256'])
    manifest=dict(status='FROZEN',version=VERSION,train_conditions=conditions,evaluation_requests=qs,evaluation_queries=queries,queries=queries,
        source_binding=m0['source_binding'],protected_hashes=protected,source_manifest=str(old_path),source_manifest_sha256=D.sha(old_path),
        noise_novelty=dict(compared_old_training_requests=512,compared_old_final_requests=36,seed_collisions=0,full_array_collisions=0,free_array_collisions=0,new_arrays=32,method='Same MT19937 FP32 free-only Gaussian; new fixed SHA256 namespace, no outcome selection'),
        budgets=dict(new_RAW=96,new_POST=96,generator_forwards=4800,D0_forwards=96,training_updates=0,extra_validation_forwards=0),
        scope='Same eight reward-supervised C on unseen noise, not new-parent generalization; all8 preserved; no checkpoint selection')
    D.write(target,manifest)
    registration=dict(experiment=VERSION,status='FROZEN',manifest_sha256=D.sha(target),sampling=D.read(OLD/'registration.json')['sampling'],budgets=manifest['budgets'],
        reward=dict(definition='UNCHANGED geometry_review_reward raw R; no within-group normalization for model comparison',source=str(ROOT/'r2/pilots/terminal_reward_nft_v1/evaluation.py'),sha256=D.sha(ROOT/'r2/pilots/terminal_reward_nft_v1/evaluation.py'),unknown='ABSTAIN, matched-known pairs and explicit unknown bounds; never imputed as0'),
        comparison=['FULL_S6000 frozen','NFT final256','SELF_REFLOW final256'],queries='Reuse original eight2048 frozen C query banks exactly; zero new queries',
        visual_review='All32 paired pages, all96 outputs; actual AI_PROXY model-blind with original RAW/POST adoption/body rubric',
        interpretation='Four user-specified exits; mixed or insufficient evidence remains inconclusive, no forced positive decision',
        stop='After96 RAW/POST, fullpaired scoring/review and report; no further training/default/commit/push')
    D.write(OUT/'registration.json',registration)
    from r2.pilots.terminal_reward_nft_v1 import common as W
    D.write(OUT/'resources_before_execution.json',W.resource_snapshot())
    print(json.dumps({'status':'FROZEN','requests':len(qs),'conditions':len(conditions),'new_noise_arrays':32,'model_calls':0}),flush=True)
    return manifest

if __name__=='__main__':run()
