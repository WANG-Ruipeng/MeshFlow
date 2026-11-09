"""Frozen task identities and budget; no neural calls on import or prepare."""
from pathlib import Path
from r2.pilots.seam_abc_v1 import generate as G

ROOT=G.REPOSITORY
OUT=ROOT/'experiments/r2_seam_hinge_quality_v1'
OLD=ROOT/'experiments/r2_seam_quality_wave1_wave2_v1'
ARMS=('CLEAN_EDGE_COH','SMOOTH_PRIOR')
BASELINE='PLAIN_S2000'
SOURCE=ROOT/'experiments/r2_seam_pool47_couple_v1/checkpoints/PLAIN3000_step1000.pt'

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        r=G.read(path);G.verify_files(r['protected_files']);return r
    task=Path('/mnt/c/Users/warpwang/.codex/attachments/b2b4dc89-4027-4165-904a-49c8f3d0c1b0/Pasted text.txt')
    (OUT/'TASK.txt').write_bytes(task.read_bytes())
    old=G.read(OLD/'protocol.json')
    meta=G.read(SOURCE.with_suffix('.json'))
    assert G.sha(SOURCE)==meta['sha256']
    protected={str(SOURCE):meta['sha256'],str(SOURCE.with_suffix('.json')):G.sha(SOURCE.with_suffix('.json')),
        str(OLD/'protocol.json'):G.sha(OLD/'protocol.json'),str(OUT/'TASK.txt'):G.sha(OUT/'TASK.txt')}
    qq=old['evaluation']['requests'];layouts=old['evaluation']['layouts']
    assert len(qq)==28 and sum(q['split']=='TRAIN' for q in qq)==16
    for q in qq:
        for key in ('inference_npz','noise_npz','metadata_npz'):
            protected[q[key]]=G.sha(q[key])
        desc=q['query_descriptor'];assert G.sha(desc['path'])==desc['sha256']
        protected[desc['path']]=desc['sha256']
        for stage,files in {'raw':['raw.npz','trajectory.npz','source_condition.npz'],
            'post':['proposal.npz','pre_geo.npz','post.npz']}.items():
            for name in files:
                p=OLD/'evaluation'/BASELINE/q['request_id']/stage/name;protected[str(p)]=G.sha(p)
    for layout in layouts.values():
        for key in ('path','source_path','receipt'):protected[layout[key]]=G.sha(layout[key])
    for rel in ('gallery/wave1_POST/definition.json','results/final/per_output.csv'):
        p=OLD/rel;protected[str(p)]=G.sha(p)
    r=dict(schema='r2_seam_hinge_quality_v1',status='FROZEN',arms=ARMS,source=str(SOURCE),baseline=BASELINE,
        authorization='User explicitly requests all attached three candidates; no further questions; fixed defaults and same-position sections required',
        training=dict(source=str(SOURCE),source_kind='PLAIN_S1000 live parameters and matching AdamW/RNG',
            arms=ARMS,updates_each=1000,instances_each=8000,global_steps=[14000,15000],
            old_plan_steps=[1001,2000],batch=8,microbatch=1,lr=1e-5,clip=1.,hybrid=.25,
            precision='BF16 forward / FP32 parameters; original deterministic runtime',
            CLEAN_EDGE_COH=dict(dirty_bulk_weight=.2,edge_weight=.1,
                edges='Source valid opposite-winding two-face seam-bulk and bulk-bulk edges, excluding either face implicated in illegal/UNKNOWN source Y/Y pairs',
                loss='mean_edge[.5*(||i0-j1||²+||i1-j0||²)+.5*||(i1-i0)-(j0-j1)||²] on Yhat; CLEAN weighted effective scalar FM denominator',
                empty_edges='Zero auxiliary term with explicit count, never NaN'),
            SMOOTH_PRIOR=dict(rho=.25,transform='mean_corner(Z)+rho*(Z-mean_corner(Z)), only bulk slots',
                order='Keep exact old restricted OT target and corner assignment; transform cached Z and epsilon before HYBRID. This is an allowed fixed-assignment path variant, not a new OT optimum.',
                inference='Same transform on frozen Gaussian bulk arrays, seam third points and fixed anchors unchanged'),
            checkpoint='Predetermined final1000 only for comparison; recovery states allowed, no best selection'),
        evaluation=dict(requests=qq,layouts=layouts,arms=ARMS,baseline=BASELINE,
            split={'TRAIN':16,'DEV':12},stages=['RAW','POST','BOUND015'],
            layout_policy='Frozen original LONG C-only layouts, no source occupancy at inference',
            barrier='All56 new RAW saved before any newD0',
            post='Same single officialD0, restoreC/seamanchors, originalGEO_C and sourcewinding; BOUND015 separate unchanged comparator',
            sections='Inherit original fixed central front/side planes, scale and crop, plus true halfspace cutaways; RAW/D0/RESTORED/POST retained for botharms; 608n3 and3fn2 fixed-position comparisons',
            metrics='Original124 numeric fields preserved; new legal-edge and vertex-only topology separately versioned; all28 outputs, TRAIN/DEV separate',
            visual='Actual AI_PROXY_NONBLIND review, no labels inferred from scores',
            repeated_dev_panel=True),
        hinge=dict(base='POST only, independent ofBOUND015',arms=[BASELINE,*ARMS],thresholds_L=[.015,.025],cells=168,
            candidates='Two incidence1 openedges already shareonevertex and have opposite directed incidence; deterministic distance/id order',
            movement='Free endpoint clusters midpoint; C/seam fixed anchors immovable; endpoint gap and cumulative PRE_GEO member displacement each <=delta',
            veto=['any new illegalCF','any new illegalFF','any lost legalCF/FF edge','new nonmanifoldedge','new/worsened angle below5deg','normal change>60deg','degeneracy'],
            delete_faces=False,reorient_faces=False,uncross='NOT_IMPLEMENTED_NO_DEFINED_OPERATOR',
            raw_reference='Diagnostic distances only; do not choose geometry by baseline case outcome'),
        budgets=dict(training_updates=2000,training_forward=16000,training_backward=16000,
            new_RAW=56,generator_forwards=2800,new_D0=56,new_BOUND=56,hinge_cells=168,head_forwards=0),
        interpretation=[
            'Two examples establish real edge-disconnection but do not invalidate small-angle penalties for other defects or uniquely identify FM dynamics.',
            'Vector term is algebraically coupled to endpoint errors, not an independent topology guarantee.',
            'CLEAN_EDGE_COH combines CLEAN weighting and edge loss; no isolated edge-effect claim without its missing equal-window CLEAN-only arm.',
            'Transformed prior is full-rank correlated Gaussian, not bounded noise and not a guarantee of fewer final intersections.',
            'Same generated slot across changed model weights is not certified same physical face; same-position sections are primary visual pairing.',
            'Legal-edge topology differs from previous all-index-edge component audit; keep both rather than overwrite definitions.',
            'No automatic extension, cleanup, default change, commit or push.'],
        prior_routes=dict(edge='R07/R09 shared endpoint supervision and recentCOUPLE; substantive change is clean full free shared-edge pairs includingbulk-bulk inSEAM, fixed1000window',
            prior='RecentLOCAL_NOISE and CREF priors; newtarget-independent full-rank bulk centroid contraction identically used for trainGaussian components and eval',
            hinge='Partialedge and seam-tip bindings; now existing vertex hinges acrossbulk-bulk withzero-newFF hardveto'),
        protected_files=protected)
    G.write(path,r);G.event(OUT/'ledger.jsonl','PROTOCOL_FROZEN',**r['budgets']);return r

if __name__=='__main__':
    import json
    r=freeze();print(json.dumps(dict(status=r['status'],budgets=r['budgets'])))
