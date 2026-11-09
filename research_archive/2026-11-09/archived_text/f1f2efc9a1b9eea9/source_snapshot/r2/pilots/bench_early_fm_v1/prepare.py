"""Freeze same USER stream and one weighting change, with CPU-only gates."""
from .common import *
import copy, json, difflib, math
import numpy as np

PROTOCOL = '''# R2_BENCH_EARLY_FM_V1 — frozen 1500-update test

User authorized continuation of the next test after the actual-trajectory crossover.
Question: can emphasizing early-time conditional FM improve real RAW C adoption,
while preserving familiar positives and postprocessed body quality?

## Single variable and closest prior evidence
Reuse USER v2's exact COMMON12000/A step0 generator, fresh AdamW, RNG, 25-parent /
81-condition data, 1500-update schedule, batch8/micro1, xyz augmentation, restricted
OT followed by HYBRID .25, original SEAM masks, BF16 forward and FP32 integration.
Only multiply each sample's original coordinate-normalized FM contribution by
w(t)=1.5 for t<.5 and .5 otherwise. Keep the original whole-batch free-coordinate
denominator; do NOT normalize away the weight per microbatch. Original t is
sigmoid(N(0,1)), NOT uniform. Its symmetry implies population E[w]=1; the actual
frozen stream's counts and scalar-weighted mean are reported, not assumed exact.
This is an experimental allocation of supervision, not a causal conclusion from
the crossover and not a new loss/architecture/OT/solver. Late supervision remains.

Closest evidence: USER v2 ordinary1500 (direct same-stream comparator), the
actual-trajectory crossover (USER prefix retained familiar gains, DEV zero),
R02 injection/time settings (no stable gain), R07/R19 auxiliary-loss schedules
(different objectives), and Bench asynchronous SEAM completion (changed trajectory
and corner clocks, distinct from weighting original FM). None establishes that
this weighting will work. No sweep or second candidate arm is authorized here.

## Fixed computation and evaluation
One new arm, 1500 optimizer updates, 12000 forward/backward microbatches. Endpoint
1500 only, no best-checkpoint selection. Safe full-cursor checkpoints every250;
no silent replay after a partial update, numerical error or OOM. No extra neural
probes. Log original SSE/counts and weighted objective separately, time-bin errors,
gradient clipping, exposures, allocator peaks, process resources and all attempts.

Same crossover16 requests: four CURRENT_TRAIN and four DEV parents, two old frozen
noises each. Same C/N/production layouts, source-free inference, fixed queries and
positions of front/side sections. New800 Euler50 calls and16 single official Bench
D0 calls; no occupancy/head calls, no new noise and no source/oracle layouts.
Save all RAW before D0. Same C/seam restore, GEO_C, source winding, HINGE_OPP015,
ALPHA/HOLE final CPU chain. Preserve RAW, BASE_POST, CLEAN_FINAL_POST and all original
scorer fields. Reuse COMMON and USER old results read-only; new scoring48 outputs.
This is a repeatedly observed small development panel, not a generalization test
on fresh parents/noises. No unique-source reconstruction ranking.

## Decision and limits
Review actual identical-view/section images for all16 RAW and finalPOST. Record
AI_PROXY YES/PARTIAL/NO/UNCERTAIN separately from exact topology. Primary evidence
is paired adoption, particularly fixed failures and DEV; keep prior good support /
assembly outputs, body completeness and local surface damage visible. Report CF,
FF affected-face fraction AND illegal-pair counts, I with support status, small
angles, actual free count/area, deletions and legal bulk reach. An improvement in
training MSE/I alone does not pass. Do not trade away body surfaces to lower FF.
If no repeatable paired adoption signal, do not extend this weighting recipe.

## Resource and preservation policy
Resource audit before each GPU wave. One authorized training arm means one training
process; use measured actual backward/optimizer peak, not inference allocation.
Leave at least max(2GiB,10%VRAM) extra free headroom. Independent CPU scoring/render
jobs may overlap and are thread-capped. Incremental sampling workers only if actual
resources and throughput warrant. Never stop user processes or change precision /
batch to add workers. Keep all old data/weights/results and scoring definitions.
After report and recoverable artifacts stop; no default change, cleanup, commit/push.
'''

def main():
    if (OUT/'registration.json').exists():
        from .training import checked
        checked(); print('FROZEN_ALREADY_VERIFIED'); return
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'PROTOCOL.md').write_text(PROTOCOL,encoding='utf-8')
    from r2.pilots.bench_user_v2_training_v1.training import checked as old_checked
    old, manifest, steps = old_checked()
    assert read(OLD_TRAIN/'training/USER_V2_SEAM/status.json')['status']=='COMPLETE'
    cross=read(CROSS/'registration.json'); qs=cross['requests']
    save(OUT/'eval_manifest.json',dict(requests=qs,source=str(CROSS/'registration.json')))
    orig_exposure=[json.loads(s) for s in (OLD_TRAIN/'training/USER_V2_SEAM/exposure.jsonl').read_text().splitlines()]
    assert len(orig_exposure)==12000
    weights=[]; bytime=[]
    for entry in steps:
        for mi,row in enumerate(entry['samples']):
            t=float(np.float32(1/(1+np.exp(-float(np.random.RandomState(row['time_seed']).standard_normal())))))
            record=orig_exposure[(entry['step']-1)*8+mi]
            assert record['condition_id']==row['condition_id'] and record['t']==t
            w=time_weight(t); weights.append(w)
            bytime.append(dict(local_step=entry['step'],microbatch=mi,condition_id=row['condition_id'],t=t,
                time_weight=w,free_scalar_count=record['all_free_scalar_count']))
    csvwrite(OUT/'frozen_time_weights.csv',bytime)
    import torch
    from meshflow_control.seam.model_adapter import corner_masked_fm_loss
    checks=[]
    pred=torch.tensor(np.arange(54).reshape(1,6,9)/100,dtype=torch.float32,requires_grad=True)
    target=torch.zeros_like(pred);valid=torch.tensor([[True]*5+[False]])
    fixed=torch.tensor([[[True]*3,[True,True,False],[False]*3,[False]*3,[False]*3,[False]*3]])
    denominator=int((valid[:,:,None]&~fixed).sum())*3
    base=corner_masked_fm_loss(pred,target,valid,fixed,denominator=denominator)
    grad=torch.autograd.grad(base,pred,retain_graph=True)[0]
    for t in (0.,.499999,.5,.999999,1.):
        w=time_weight(t); got=torch.autograd.grad(base*w,pred,retain_graph=True)[0]
        assert torch.allclose(got,grad*w,rtol=1e-6,atol=1e-8)
        assert torch.count_nonzero(got.reshape(1,6,3,3)[fixed|~valid[:,:,None]])==0
        checks.append(dict(t=t,weight=w,gradient_scaling='PASS',masked_gradient_zero='PASS'))
    assert not torch.cuda.is_initialized()
    counts=np.array([r['free_scalar_count'] for r in bytime],np.float64)
    gate=dict(status='PASS',training_updates=0,model_forwards=0,CUDA_initialized=False,
        checked_original_time_rows=12000,early_count=int(sum(w>1 for w in weights)),late_count=int(sum(w<1 for w in weights)),
        sample_mean_weight=float(np.mean(weights)),coordinate_weighted_mean=float(np.average(weights,weights=counts)),
        weight_gradient_tests=checks,reference_checked='Original USER training checked() unchanged')
    save(OUT/'cpu_gate.json',gate)
    paths={**old['training_code_hashes']}
    for p in Path(__file__).parent.glob('*.py'):paths[str(p)]=sha(p)
    reg=copy.deepcopy(old)
    reg.update(schema='BENCH_EARLY_FM_REGISTRATION_V1',experiment=EXPERIMENT,arms=list(ARMS),
        time_weight=dict(split=.5,early=1.5,late=.5,per_batch_renormalize=False),
        training_code_hashes=paths,expected_new_raw=16,expected_new_d0=16,expected_new_score_rows=48,
        generator_forward_sampling=800,protocol_sha256=sha(OUT/'PROTOCOL.md'),cpu_gate_sha256=sha(OUT/'cpu_gate.json'),
        eval_manifest=str(OUT/'eval_manifest.json'),eval_manifest_sha256=sha(OUT/'eval_manifest.json'),
        evaluation_manifest=str(OUT/'eval_manifest.json'),evaluation_manifest_sha256=sha(OUT/'eval_manifest.json'),
        evaluation_labels={'EARLY':dict(arm=ARM,subset='full',local_step=1500,global_step=13500,
            generator_path=str(OUT/'training'/ARM/'generator_step1500.pt'))},
        requests=qs,reused_baseline=dict(root=str(OLD),label='USER',training_registration_sha256=sha(OLD_TRAIN/'training_registration.json')))
    protected={str(resolve(reg[k])):sha(reg[k]) for k in ('source_checkpoint','manifest','input_schedule')}
    protected.update({str(resolve(r['case_path'])):r['case_sha256'] for r in manifest['rows']})
    protected.update(cross['protected_hashes'] if 'protected_hashes' in cross else {})
    for q in qs:
        for k in ('inference_npz','metadata_npz','noise_npz','layout_npz','C_export_metadata','frame_json'):
            protected[q[k]]=sha(q[k])
        protected[q['query_descriptor']['path']]=q['query_descriptor']['sha256']
        for label in ('COMMON','USER'):
            for p in (OLD/'evaluation'/label/q['request_id']).rglob('*.json'):protected[str(p)]=sha(p)
            for p in (OLD/'evaluation'/label/q['request_id']).rglob('*.npz'):protected[str(p)]=sha(p)
    reg['protected_files']=protected
    save(OUT/'registration.json',reg);save(OUT/'training_registration.json',reg)
    a=(ROOT/'r2/pilots/bench_user_v2_training_v1/training.py').read_text();b=(Path(__file__).parent/'training.py').read_text()
    (OUT/'training_change.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='original_USER_training.py',tofile='early_training.py')),encoding='utf-8')
    from .training import checked
    checked()
    snap=resources();save(OUT/'resource_initial.json',snap)
    gpu=snap['gpu']; assert gpu['returncode']==0
    fields=gpu['stdout'].splitlines()[0].split(',');free=float(fields[3]);total=float(fields[1])
    assert free >= max(2048,.1*total)+6000,'Wait for resources; do not dispatch'
    save(OUT/'training_dispatch_go.json',dict(authorized=True,arms=list(ARMS),
        training_registration_sha256=sha(OUT/'training_registration.json'),actual_workers=1,
        reason='One candidate arm; observe actual first backwards and optimizer allocation. No extra probes.',
        safety_headroom_mib=max(2048,.1*total),initial_free_mib=free,resources=snap))
    print(json.dumps(dict(status='FROZEN_CPU_PASS',gate=gate,updates=1500,new_RAW=16,new_D0=16)))

if __name__=='__main__':main()
