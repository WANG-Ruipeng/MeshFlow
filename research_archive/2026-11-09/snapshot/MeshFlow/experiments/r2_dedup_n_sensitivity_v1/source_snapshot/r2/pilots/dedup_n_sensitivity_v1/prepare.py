"""Freeze count sensitivity, exact existing Gaussian prefixes and B decision gate."""
import json
import shutil
import numpy as np
from . import common as C

PRIOR = C.ROOT/'experiments/r2_target_cleanup_c200_v1'

def prepare():
    dest=C.OUT/'registration.json'
    if dest.exists():
        r=C.read(dest)
        for path,digest in r['protected'].items():
            assert C.sha(path)==digest,path
        for q in r['requests']:C.old().read_input(q)
        return dict(status='VERIFIED_FROZEN',sha256=C.sha(dest))
    old=C.read(PRIOR/'registration.json')
    assert C.read(PRIOR/'completion.json')['status']=='COMPLETE'
    protected={str(PRIOR/'registration.json'):C.sha(PRIOR/'registration.json')}
    ds=C.ROOT/'experiments/r2_target_cleanup_v1'
    for p in (ds/'data_manifest.json',ds/'paired_parent_summary.csv',C.CODE/'common.py',C.CODE/'prepare.py'):
        protected[str(p)]=C.sha(p)
    ratio_num,ratio_den=2717,3047
    original=old['requests'];qs=[];counts=[]
    for src in original:
        q=dict(src);a=C.old().read_input(src);n,k=int(src['N']),int(src['K'])
        free=(2*(n-k)*ratio_num+ratio_den)//(2*ratio_den)
        low=max(128,min(256,k+free));assert k<low<n
        b={'C':a['C'].copy(),'z':a['z'][:,:low].copy(),'N':np.array(low,dtype=a['N'].dtype),'K':a['K'].copy()}
        p=C.OUT/'inputs/requests'/f"{q['request_id']}.npz";C.frozen_npz(p,b)
        q.update(N=low,ratio=k/low,input_path=str(p),input_sha256=C.sha(p),Gaussian_sha256=C.array_sha(b['z']),
            free_Gaussian_sha256=C.array_sha(b['z'][:,k:]),action='NEW',original_N=n,original_ratio=src['ratio'],
            original_input_path=src['input_path'],original_input_sha256=src['input_sha256'],
            original_Gaussian_sha256=src['Gaussian_sha256'],N_policy='TRAIN_POOL_FREE_COUNT_RATIO_2717_OVER_3047',
            Gaussian_prefix_contract='Copy original z[:, :N_low, :]; retain C zeros and exact original absolute free slot identities; no RNG draw',
            face_bin=20,original_N_embedding_bin=n//20,N_embedding_bin=low//20,
            full_noise_reused=False,noise_provenance='BYTE_EXACT_PREFIX_OF_REGISTERED_ORIGINAL_FULL_N_NOISE; no new noise',
            source_full_N_is_not_requested_N=True)
        assert C.same(b['C'],a['C']) and C.same(b['z'][:,k:],a['z'][:,k:low])
        C.old().read_input(q);qs.append(q)
        for path,digest in ((src['input_path'],src['input_sha256']),(src['condition_source'],src['condition_source_sha256'])):
            protected[path]=digest
        if not any(x['parent_id']==q['parent_id'] for x in counts):
            counts.append(dict(parent_id=q['parent_id'],K=k,N_original=n,N_low=low,free_original=n-k,free_low=low-k,
                original_N_embedding_bin=n//20,N_embedding_bin=low//20,noise_prefix_length=low-k,
                inferred_source_dedup_N=False))
    for query in old['queries']:
        protected[query['path']]=query['sha256']
    cache={};models={}
    for stem,source in (('ORIGINAL','ORIGINAL_C200'),('DEDUP','DEDUP_C200')):
        b=C.read(PRIOR/'evaluation/bindings'/f'{source}.json')
        assert C.sha(b['path'])==b['sha256'];protected[b['path']]=b['sha256']
        for kind in ('LOW_N','ORIG_N'):
            label=stem+'_'+kind;models[label]={**b,'label':label,'source_arm':source}
        ref=stem+'_ORIG_N';cache[ref]={}
        for q in original:
            item={'historical_label':source}
            for stage in ('raw','post'):
                p=PRIOR/'evaluation'/stage/source/q['request_id']/'receipt.json'
                item[stage+'_path']=str(p);item[stage+'_sha256']=C.sha(p)
                protected[str(p)]=C.sha(p)
            cache[ref][q['request_id']]=item
    gate=dict(version='pilot_B_gate_v1',purpose='Conservative pilot continuation screen, not statistical significance or universal metric thresholds',
        mandatory_execution='48 new RAW+48 single official D0 and all fixed scores; C exact all; no hidden model calls',
        DEDUP_low_vs_original_low='POST FF mean strictly lower; all six parent differences reported',
        DEDUP_low_vs_DEDUP_orig=dict(
            deleted_fraction='Mean GEO deleted_free/(N-K) strictly lower; raw count decrease alone never passes',
            FF='POST macro FF no higher',
            CF_I='Each POST macro increase <=0.5 percentage points; worst output increase <=5 percentage points',
            connectivity='Every formerly legal-CF request retains at least one legal CF edge; number reaching >=30% oriented free area does not decrease',
            actual_visual='At least 3 requests across >=2 parents improve visible cracks/fragments/body surface; improved count exceeds worsened; no new clear major body loss; C role retained',
            review='Nonblind AI_PROXY, same scales/views, UNKNOWN permitted; compare geometry, not unique GT style'),
        interpretation='Lower-N response does not prove prior N was incorrect or target duplication uniquely caused cracks. If both models improve similarly, call generic count effect.',
        if_pass='Proceed only to B paired SEAM original/DEDUP short continuation, 200 updates per arm, after CPU owner/data/occupancy feasibility freeze; do not add new loss/longer budget/RL.',
        if_fail='Stop after A report; preserve B readiness, no B training.')
    C.write(C.OUT/'B_GATE.json',gate)
    resources={**old['resources'],'generator_incremental_estimate_MiB':2600,'experiment_incremental_cap_MiB':6144,
        'CPU_score_workers':2,'CPU_threads_per_process':2,'dispatch_policy':'One worker first; measure representative sampling peak/compute. Incremental second worker only if <=6GiB experimental incremental budget, >=2GiB headroom and throughput permit. D0 sized separately.'}
    r=dict(version=C.VERSION,status='FROZEN',utc=C.utc(),evaluation_labels=list(C.ARMS+C.REFERENCES),
        requests=qs,original_requests=original,queries=old['queries'],conditions=old['conditions'],models=models,cache=cache,
        d0_checkpoint=old['d0_checkpoint'],protected=protected,resources=resources,
        budget=dict(training_updates=0,backward=0,new_rollouts=48,new_sampling_forward=2400,new_D0_forward=48,
            new_quality_scans=96,cached_original_N_requests=48,cached_original_N_scores=96,total_score_rows=192),
        count_policy=dict(numerator=ratio_num,denominator=ratio_den,source='Already-frozen 16-parent training free counts; not evaluation parent geometry',
            rule='N_low=max(128,min(256,K+floor((N_orig-K)*2717/3047+0.5)))',one_lower_count_only=True,face_bin=20),
        hypothesis='Does one lower count reduce collapse and visible cracks while preserving dedup FF benefit and connections? Compare weight x N interaction.',
        prior_routes=['Target cleanup C200: positive POST FF but more deleted faces and visible cracks; no N intervention yet',
            'R31 SEAM: bounded connection gain, residual bulk FF; original four parents have no exact target duplicates, so B cannot be a relabelled same-target test'],
        gate_path=str(C.OUT/'B_GATE.json'),gate_sha256=C.sha(C.OUT/'B_GATE.json'),
        inference_contract=dict(Euler_steps=50,precision='BF16 forward FP32 parameters/integration',all_48_RAW_before_D0=True,
            post='one official D0, original GEO_C, source C winding and free orientation',same_C_queries=True,source_GT_rank=False),
        failures='No silent model retries. Preserve unresolved calls and logs; stop numerical/OOM failures without scientific setting changes.',
        no_default_switch=True,no_commit_or_push=True,no_old_asset_cleanup=True)
    C.write(dest,r);C.csvwrite(C.OUT/'inputs/count_policy.csv',counts)
    C.write(C.OUT/'resources/initial.json',dict(snapshot=C.resources(),disk_free_bytes=shutil.disk_usage(C.ROOT).free,
        windows_initial='RTX5080 total16303MiB used3076 free12904 utilization4%; per-processVRAM NA WDDM; Windows CIM unavailable, WSL /proc authoritative'))
    C.event('REGISTRATION_FROZEN',sha256=C.sha(dest),new_rollouts=48,new_D0=48,gate_sha256=r['gate_sha256'])
    return dict(status='FROZEN',sha256=C.sha(dest),counts=counts)

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
