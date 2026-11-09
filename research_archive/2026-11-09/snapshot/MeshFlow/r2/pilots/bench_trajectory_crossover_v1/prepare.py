"""Freeze actual t=.5 rollout states; read only C metadata, never source Y."""
from .common import *
import numpy as np
from r2.pilots.seam_slot_c200_v1.generate import raw_arrays

def main():
    if (OUT / 'registration.json').exists():
        verify(registration()['files']); print('Already frozen'); return
    oldreg = read(OLD / 'registration.json')
    scorereg = read(OLD / 'evaluation/score_registrations/COMMON.json')
    qs = scorereg['requests']; assert len(qs) == 16
    assert len({q['condition_id'] for q in qs}) == 8
    verify(scorereg['source_hashes'])
    files = {}; checks = []; manifest = []
    def track(p):
        files[str(resolve(p))] = sha(p)
    for path in (OLD/'registration.json', OLD/'evaluation/score_registrations/COMMON.json', OLD/'visual_review/main.csv'):
        track(path)
    checkpoints = {k: oldreg['checkpoints'][k] for k in ('COMMON','USER')}
    for p in checkpoints.values():
        assert sha(p) == oldreg['files'][p]; track(p)
    for q in qs:
        rid = q['request_id']; cid = q['condition_id']; k = q['K']; n = q['N']
        lp = OLD / 'layouts' / cid / 'layout.npz'; lay = arrays(lp); track(lp)
        assert int(lay['N']) == n and int(lay['K']) == k
        fixed = lay['anchor_ids'] >= 0
        # Only C provenance fields are read from source metadata.
        with np.load(resolve(q['source_npz']), allow_pickle=False) as bank:
            source = {key: bank[key].copy() for key in ('C_source_corner_permutations','C_original_source_vertex_ids')}
        cp = OUT / 'inputs' / cid / 'C_export_metadata.npz'; array_file(cp, source); track(cp)
        track(q['source_npz'])
        starts = {}; baseline = {}
        for letter, model in MODELS.items():
            tp = OLD / 'evaluation' / model / rid / 'raw/trajectory.npz'
            tr = arrays(tp); track(tp)
            assert tr['states'].shape == (51,n,3,3) and tr['states'].dtype == np.float32
            assert tr['velocities'].shape == (50,n,3,3)
            assert np.isfinite(tr['states']).all() and np.isfinite(tr['velocities']).all()
            anchor = lay['C_vertices'][lay['anchor_ids'][fixed]]
            assert all(np.array_equal(x[fixed], anchor) for x in tr['states'])
            # Check inherited Euler arithmetic without any model calls.
            for step in range(50):
                x = tr['states'][step] + np.float32(.02) * tr['velocities'][step]
                x[fixed] = anchor
                assert np.array_equal(x, tr['states'][step+1])
            rp = raw_path(letter*2, rid); raw = arrays(rp); track(rp)
            reproduced = raw_arrays(tr['states'][-1], lay, lay['occupancy_logits'], source)
            assert set(reproduced) == set(raw)
            assert all(np.array_equal(reproduced[key], raw[key]) for key in raw)
            startp = OUT / 'inputs' / rid / (model+'_t050.npz')
            array_file(startp, dict(state=tr['states'][25], absolute_step=np.asarray(25)))
            track(startp); starts[model] = str(startp)
            for fn in ('row.json','score.json'):
                p = score_path(letter*2, rid) / fn
                if p.exists(): track(p)
            baseline[model] = dict(trajectory=str(tp), raw=str(rp), score=str(score_path(letter*2,rid)))
        c0 = arrays(baseline['COMMON']['trajectory'])['states'][0]
        u0 = arrays(baseline['USER']['trajectory'])['states'][0]
        assert np.array_equal(c0,u0)
        for key in ('inference_npz','metadata_npz','noise_npz'): track(q[key])
        # Query files are the original frozen 2,048 C samples; no redraw.
        for path,digest in q['query_descriptor'].get('files',{}).items():
            assert sha(path)==digest; track(path)
        qp = q['query_descriptor'].get('path')
        if qp: track(qp)
        framep = OLD / 'gallery_full/main' / (rid+'.json'); track(framep)
        cleanq = {key:q[key] for key in ('request_id','condition_id','parent_id','noise_index','N','K','split','part_type','boundary_edges','inference_npz','metadata_npz','noise_npz','query_descriptor')}
        cleanq.update(layout_npz=str(lp),C_export_metadata=str(cp),starts=starts,baseline=baseline,frame_json=str(framep))
        manifest.append(cleanq)
        checks.append(dict(request_id=rid,K=k,N=n,finite=True,all_fixed_states_exact=True,
            cached_euler_recurrence_checks=100,raw_export_roundtrip=True,identical_initial_state=True,source_Y_read=False))
    own = {str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    inherited = dict(scorereg['source_hashes'])
    for name in ('gpu.py','common.py'):
        p=ROOT/'r2/pilots/bench_trajectory_diagnostic_v1'/name; inherited[str(p)]=sha(p)
    resources_before = resources(); save(OUT/'resources/preparation.json',resources_before)
    save(OUT/'cpu_contract_checks.json',dict(status='PASS',requests=checks,model_calls=0,OT_calls=0,new_noise=0))
    reg=dict(status='FROZEN',version='actual_rollout_crossover_t050_v1',checkpoints=checkpoints,requests=manifest,
        files=files,source_hashes={**inherited,**own},combinations=list(COMBOS),new_combinations=['CU','UC'],
        switch_t=.5,absolute_steps=list(range(25,50)),dt=.02,generator_budget=800,D0_budget=0,head_budget=0,
        optimizer_updates=0,new_noise=0,new_OT=0,source_Y_generation=False,
        runtime='BF16 forward / FP32 params and Euler; strict deterministic; math SDPA; TF32 off',
        closest_prior_route='Bench trajectory diagnostic: source-informed continuation; now actual generated states, same production layout, complete-model suffix exchange',
        endpoint_policy='All 16 requests; no time scan, no best selection; no automatic replay',
        primary='RAW adoption + body completeness from all fixed-camera pages; original full geometry metrics and LOW_SUPPORT retained',
        interpretation='Cross failure can reflect state compatibility; no proof of a harmful suffix; no new-parent success inferred from training-C gains')
    save(OUT/'registration.json',reg)
    print(json.dumps(dict(status='FROZEN',requests=16,new_outputs=32,generator_calls=800,resources=resources_before),ensure_ascii=False),flush=True)

if __name__ == '__main__': main()
