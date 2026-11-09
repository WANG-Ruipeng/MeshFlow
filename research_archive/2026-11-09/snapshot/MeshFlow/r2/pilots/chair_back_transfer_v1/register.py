"""Freeze the already-authorized 8-C/16-noise Chair back transfer experiment."""
from __future__ import annotations
from . import worker as W
G,ROOT,OUT=W.G,W.ROOT,W.OUT


def run():
    if (OUT/'registration.json').exists():
        r=G.read(OUT/'registration.json')
        assert r['eval_manifest_sha256']==G.sha(OUT/'eval_manifest.json')
        return r
    m=G.read(OUT/'eval_manifest.json')
    assert len(m['conditions'])==8 and len(m['requests'])==16
    for q in m['requests']:W.assets(q)
    code=[W.__file__,G.__file__,__file__,
        ROOT/'r2/pilots/seam_slot_c200_v1/postprocess.py',
        ROOT/'r2/prototypes/seam_slot_c200_v1/model_adapter.py',
        ROOT/'r2/prototypes/seam_slot_c200_v1/corner_contract.py',
        ROOT/'r2/prototypes/seam_slot_c200_v1/slot_data.py',
        ROOT/'r2/src/meshflow_control/seam/slot_data.py',
        ROOT/'r2/src/meshflow_control/seam/inference.py',
        ROOT/'r2/src/meshflow_control/seam/occupancy.py',
        ROOT/'r2/src/meshflow_control/runtime.py',
        ROOT/'r2/src/meshflow_control/source_winding.py',
        ROOT/'r2/src/meshflow_control/geo_c.py',
        ROOT/'r2/src/meshflow_control/denoise_model.py',
        ROOT/'r2/research/hinge_opposite_v1/replay_array.py',
        ROOT/'r2/research/hinge_v2_cpu_v1/replay_array.py']
    generator=G.read(W.GEN.with_suffix('.json'));head=G.read(W.HEAD.with_suffix('.json'))
    assert G.sha(W.GEN)==generator['sha256'] and G.sha(W.HEAD)==head['sha256']
    review=[]
    notes={
      'f4f1':'Tall planar two-triangle back surface; coherent and interpretable.',
      '6c36':'Sloped/folded internal back surface of an armchair; source occlusion noted; C itself coherent.',
      '8c9b':'Ladder back with open spaces; do not assume every open boundary must be sealed.',
      'b631':'Curved full back surface, fine elongated source triangulation retained.',
      'fb8c':'Tall planar two-triangle back surface; coherent and interpretable.',
      'abbe':'Arched full back panel; coherent.',
      'f199':'Arched full back panel with a lower projecting portion; retained as frozen source region.',
      '9d81':'Arched back frame with central opening; do not treat its hole as required fill.'}
    for c in m['conditions']:
        prefix=c['uid'][:4]
        review.append(dict(condition_id=c['condition_id'],reviewer='ROOT_AI_PROXY',blinded=False,
          stage='INPUT_ONLY_BEFORE_GENERATION',actually_opened=True,status='ELIGIBLE',notes=notes[prefix]))
    G.write(OUT/'input_review/actual_review.json',dict(rows=review,selection_changed_after_visual_review=False))
    r=dict(status='FROZEN',experiment='R2_CHAIR_BACK_TRANSFER_V1',
       question='Does the retained Chair seat-trained SEAM pipeline carry over to full-back inputs?',
       conditions=8,requests=16,per_group_conditions=dict(FAMILIAR=4,DEV=4),noises_per_C=2,
       eval_manifest_sha256=G.sha(OUT/'eval_manifest.json'),data_registration_sha256=G.sha(OUT/'data_registration.json'),
       training_updates=0,backwards=0,generator_forwards=800,head_forwards=8,D0_forwards=16,
       checkpoint_selection='Fixed PLAIN_S2000 live global15000; fixed LONG C-only head; no oracle layouts',
       inference='Euler50 standard Gaussian; BF16 forward; FP32 parameters/integration; strict deterministic math SDPA',
       model_hashes={str(W.GEN):generator['sha256'],str(W.HEAD):head['sha256'],str(W.D0):'6a31f4c7dcbea7a6ac58c26075b20c897fd19aad4c33b5295cc4c93ed7078c36'},
       scientific_code_hashes={str(p):G.sha(p) for p in code},
       stages=['RAW','single official Chair D0','anchor restoration and GEO_C/source winding','HINGE_OPP015','ALPHA_HOLE'],
       metric_policy='Preserve original scorers: CF/FF/I, exact legal interfaces, oriented reachability, seam-to-original-bulk reachability, small angles, fragmentation, deletion and displacement. Full per-output fields; fixed original-face denominator for added bridges.',
       descriptive_bulk_threshold=.30,visual_review='All16 outputs; same view/scale and source-C-based fixed section locations across stages; nonblind AI_PROXY',
       inference_source_Y_used=False,unique_source_reconstruction_success=False,
       transfer_scope='FAMILIAR parents seen in TRAIN47, BACK_FULL may be seen in early EXP67; DEV unseen in conditional adaptation but official pretrain exposure UNKNOWN.',
       replay_policy='No automatic repeats of a started neural call. Reuse completed receipts; before-forward engineering fixes may be recorded. No resampling or condition substitution.',
       dispatch='Incremental measured GPU scheduling with >=2GiB headroom. All16RAW before any D0. CPU post/scoring may use independent workers.',
       stop='Complete fixed16, report all outcomes, then stop. No training, default switch, cleanup, commit or push.',
       registered_unix=__import__('time').time())
    G.write(OUT/'registration.json',r)
    print('FROZEN16',r['eval_manifest_sha256'])
    return r

if __name__=='__main__':run()
