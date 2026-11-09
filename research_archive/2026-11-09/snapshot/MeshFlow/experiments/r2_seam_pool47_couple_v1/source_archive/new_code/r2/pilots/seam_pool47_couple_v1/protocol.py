"""Fixed scope for the approved three-part SEAM follow-up; no model calls."""
from pathlib import Path
import json
from r2.pilots.seam_abc_v1 import generate as G
ROOT=G.REPOSITORY
OUT=ROOT/'experiments/r2_seam_pool47_couple_v1'
ARMS=('PLAIN3000','COUPLE3000')

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        reg=G.read(path);G.verify_files(reg['protected_files']);return reg
    old=ROOT/'experiments/r2_seam_followup_v1/layout/registration.json'
    layout=G.read(old)
    protected={str(old):G.sha(old)}
    for q in layout['requests']:
        for key in ('inference_npz','noise_npz','metadata_npz'):
            protected[q[key]]=G.sha(q[key])
        protected[q['query_descriptor']['path']]=q['query_descriptor']['sha256']
    for item in layout['layouts'].values():
        for key in ('path','source_path','receipt'):protected[item[key]]=G.sha(item[key])
    source=ROOT/'experiments/r2_seam_abc_v1/checkpoints/B_POOL47_step1000.pt'
    protected[str(source)]=G.sha(source)
    reg=dict(schema='r2_seam_pool47_couple_v1',status='FROZEN',
        authorization='User approved two arms each3000 additional updates, isolating couple increment; CPU binding and C-only corner diagnostic also approved.',
        source_checkpoint=str(source),source_checkpoint_sha256=protected[str(source)],
        train=dict(arms=list(ARMS),updates_each=3000,global_steps=[13000,16000],seam_steps=[1000,4000],
            pool='Same original TRAIN47 and source owner layouts; no dedup, no DEV training',
            stream='Continue registered B parent/augmentation stream; one shared24000 sample OT cache; same Gaussian/time/hybrid draws',
            optimizer='Restore matching B live-generator AdamW; lr1e-5 betas[.9,.95] eps1e-8 wd0; no reset',
            batch=8,microbatch=1,precision='BF16 forward; FP32 parameters/integration/AdamW',
            loss=dict(PLAIN3000='Original free-corner FM OT then HYBRID0.25',COUPLE3000='Same FM + lambda=.1 existing source-correspondence one-step tip/bulk MSE; both sides differentiable'),
            occupancy_head='Frozen and unused for training layout; no head training or inference forward',
            recovery_checkpoints=[1000,2000,3000],main_endpoint=3000,early_stop_metric=None),
        evaluation=dict(requests=layout['requests'],layouts=layout['layouts'],layout_policy='Exact cached LONG C-only layout for both arms; no source alias decisions at inference',
            outputs_per_arm=28,split={'TRAIN':16,'DEV':12},RAW_total=56,generator_forwards=2800,D0_forwards=56,
            sampling='Same saved Gaussian arrays, requested N/C, Euler50; all56RAW before D0',
            stages=['RAW','single official D0 + original GEO_C + source winding POST','same POST + .015L actual-cluster binding'],
            binding_new_cells=56,metrics='Full original124 numerical fields; per-noise/per-parent costs and connectivity; AI_PROXY actual gallery review',
            reference='Existing B_LONG_LAYOUT step0 outputs and same-bind CPU results, frozen LONG optional context; no new step0 generation'),
        cpu_binding=dict(source='B_LONG_LAYOUT',requests=28,caps=[.015,.020],cells=56,
            diagnostics='DEV finite tip/side-edge to bulk open geometry; source alias fate diagnostic only'),
        corner_rule=dict(scope='Frozen C-only deterministic geometric feasibility proposals, TRAIN47+DEV3; labels only after proposals for diagnostic comparison',
            inference_use=False,automatic_source_alias_identification=False),
        stopping='No extra arms, checkpoint selection, tuning, automatic retries, default change, cleanup, commit or push. Preserve failures and exact completed update counts.',
        interpretation='Couple incremental effect at equal extended budget; does not prove previous pool undertraining, universal alias identifiability or a unique correct completion.',
        resources_before=G.resources(),protected_files=protected)
    G.write(path,reg)
    G.event(OUT/'ledger.jsonl','PROTOCOL_FROZEN',updates=6000,new_RAW=56,new_D0=56,cpu_binding_cells=112)
    return reg

if __name__=='__main__':
    r=freeze();print(json.dumps(dict(status=r['status'],path=str(OUT/'protocol.json'))))
