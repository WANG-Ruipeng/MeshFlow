"""Immutable protocol, input identities and scientific budget. No model calls."""
from pathlib import Path
import json
from r2.pilots.seam_abc_v1 import generate as G

ROOT=G.REPOSITORY
OUT=ROOT/'experiments/r2_seam_quality_wave1_wave2_v1'
WAVE1=('PLAIN_S1000','PLAIN_S2000','WISE_B_PLAIN_050')
WAVE2=('CLEAN_BULK_FM_S1000','SURGICAL_LATE4_S1000')
ARMS=WAVE1+WAVE2
BASELINES=('LONG','B_LONG_LAYOUT','PLAIN3000')
SOURCE=ROOT/'experiments/r2_seam_abc_v1/checkpoints/B_POOL47_step1000.pt'

def cleanup_gate():
    p=OUT/'cleanup/completion.json'
    r=G.read(p)
    if not str(r.get('status','')).startswith('COMPLETE'):
        raise ValueError('Stage0 must be complete before scientific computation')
    return r

def freeze():
    cleanup_gate();OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'protocol.json'
    if p.exists():
        r=G.read(p);G.verify_files(r['protected_files']);return r
    oldpath=ROOT/'experiments/r2_seam_pool47_couple_v1/protocol.json'
    old=G.read(oldpath)
    protected={str(oldpath):G.sha(oldpath)}
    requests=old['evaluation']['requests'];layouts=old['evaluation']['layouts']
    assert len(requests)==28 and len(layouts)==7
    for q in requests:
        for key in ('inference_npz','noise_npz','metadata_npz'):
            protected[q[key]]=G.sha(q[key])
        protected[q['query_descriptor']['path']]=q['query_descriptor']['sha256']
    for v in layouts.values():
        for key in ('path','source_path','receipt'):protected[v[key]]=G.sha(v[key])
    sources={'B_POOL47':SOURCE}
    for step in (1000,2000,3000):
        sources[f'PLAIN{step}']=ROOT/f'experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step{step}.pt'
    for v in sources.values():
        meta=G.read(v.with_suffix('.json'))
        if G.sha(v)!=meta['sha256']:raise ValueError('Source checkpoint hash differs')
        protected[str(v)]=meta['sha256'];protected[str(v.with_suffix('.json'))]=G.sha(v.with_suffix('.json'))
    r=dict(schema='r2_seam_quality_wave1_wave2_v1',status='FROZEN',
        authorization='Explicit user task attachment 3a4e99e9-7378-4c0f-88fa-0fb549027563; all stages authorized, no automatic extension',
        sources={k:str(v) for k,v in sources.items()},
        wave1=dict(arms=WAVE1,updates=0,weight_alpha=.5,RAW=84,generator_forwards=4200,D0_forwards=84,score_cells=252),
        wave2=dict(arms=WAVE2,updates_each=1000,instances_each=8000,forward_each=8000,backward_each=8000,
            global_steps=[13000,14000],source=str(SOURCE),source_kind='live nonEMA and matched AdamW/RNG',
            stream='Exact first1000 old pool47 updates; no new OT, random draws, source geometry or schedule',
            batch=8,microbatch=1,learning_rate=1e-5,gradient_clip=1.,hybrid=.25,
            CLEAN_BULK_FM='Source illegal/unknown bulk face weight .2, others1; owner thirdpoints always1; weighted effective batch scalar denominator',
            SURGICAL_LATE4='Freeze actual backbone.layers.0..7; train all other existing parameters; inherit exact AdamW states; retain autograd through frozen layers',
            RAW=56,generator_forwards=2800,D0_forwards=56,score_cells=168),
        evaluation=dict(requests=requests,layouts=layouts,arms=list(ARMS),baselines=list(BASELINES),
            split={'TRAIN':16,'DEV':12},stages=['RAW','POST','BOUND015'],
            layout_policy='Exact saved LONG C-only layouts, no head forward, no source Y at inference',
            sampling='Euler50 BF16 forward FP32 parameters/integration; exact saved Gaussian; all51 C/seam anchor states bitwise checked',
            barrier='All RAW in each wave before any D0 in that wave; Wave1 must finish scoring before Wave2 training',
            post='Single officialD0, C/seam anchors restored, originalGEO_C/source winding; then independent frozen1.5%L real-cluster binding',
            metrics='Same complete124 numeric fields plus statuses, denominator, per-output, per-parent paired values; TRAIN/DEV separately',
            visual='Actual nonblind AI_PROXY same-camera 4-view POST review all140new outputs, critical BOUND015 changes',
            equal_budget_control='PLAIN_S1000 fixed in advance; no posthoc checkpoint selection',
            panel_scope='Repeatedly observed exploratory panel; DEV untrained parents but not untouched test set'),
        budgets=dict(new_RAW=140,new_D0=140,generator_forwards=7000,new_score_cells=420,new_binding_cells=140,training_updates=2000,logical_rows=672),
        interpretation=[
            'Weighted denominator raises relative seam gradient coefficient despite unchanged owner weights; report coefficient explicitly.',
            'Old PLAIN clean/dirty category MSE was not recorded; NA rather than extra training forwards.',
            'A positive intervention supports a mechanism hypothesis; it does not uniquely prove source pollution or representation drift.',
            'Bulk30 is graph reachability, not whole-boundary closure; I is ConflictCov_C, not global collision area.',
            'No automatic 2000-step extension or combination arm; no default change, commit or push.'],
        prior_routes=dict(SEAM='R31 bounded positive; same representation preserved',
            CLEAN='Earlier DEDUP removed faces/changed N and caused holes; this trial keeps all47/fullN and only weights static bulk FM',
            SURGICAL='R01 frozen-backbone adapters did not establish upgrade; here existing learned SEAM and exact optimizer continuation, only late4 trainable',
            COUPLE='R07 and recent pool47 couple lacked stable DEV benefit; no couple term in either new arm'),
        protected_files=protected)
    G.write(p,r);G.event(OUT/'ledger.jsonl','PROTOCOL_FROZEN',**r['budgets']);return r

if __name__=='__main__':
    r=freeze();print(json.dumps(dict(status=r['status'],arms=r['evaluation']['arms'],budgets=r['budgets'])))
