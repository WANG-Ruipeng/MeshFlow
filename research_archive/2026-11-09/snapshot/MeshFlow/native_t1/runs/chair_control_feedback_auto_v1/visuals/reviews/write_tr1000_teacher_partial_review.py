"""Persist actual assigned-four-parent visual review, without rendering/scoring."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from PIL import Image

OUT = Path('/mnt/e/MeshFlow/native_t1/runs/chair_control_feedback_auto_v1')
GALLERY = OUT/'visuals/DEV_START_explore_FM_CONTROL_1000_explore_REPLAY_FM_CONTROL_1000_explore_TRAJECTORY_REPLAY_1000_d3c014cf5b53'
CANDIDATE = 'explore_TRAJECTORY_REPLAY_1000'
MATCHED = 'explore_REPLAY_FM_CONTROL_1000'
STEM = CANDIDATE+'_vs_'+MATCHED+'_coupling_partial'

# These are manual observations made after actually viewing all two-noise,
# four-view-plus-wire parent pages and every saved S/I/E feedback page below.
# They are not produced from metrics, earlier reviews or an automated classifier.
NOTES = [
    ('86ec9c796a6c1789e792598ed58dd1cc', 0, 'MIXED', 'MIXED',
     'Both teacher and TR retain a complete slatted chair. TR removes the teacher conspicuous under-seat triangular flap visible from the front, but the seat edge is more skewed/layered and one front leg has a narrow split. The crown is mostly present with small holes in both. There is no main-body breakup.',
     'The known blue uprights/legs remain distinct. TR has a more visible narrow blue/orange gap along the upper left back and small seat-junction gaps; style is not used as a quality criterion.',
     'S contains green and red in both; TR has visible additional red along an upper vertical boundary. I is mainly gray with small colored local markers, E predominantly green. These local overlays do not establish watertightness.'),
    ('86ec9c796a6c1789e792598ed58dd1cc', 1, 'MIXED', 'MIXED',
     'TR improves some teacher crown perforations and removes several sharp lower-back/seat-junction protrusions. It also has a more raised/thick front seat apron and a conspicuous gap between the right crown edge and the blue known upright. Slender irregular back slits remain. The chair body is retained.',
     'Cleaner portions of the orange back do not resolve the blue/orange top-right separation. Seat/leg seams remain visible.',
     'S is mostly red in the back/leg boundary in both. The teacher has a purple I cluster near the upper right upright that is less conspicuous in TR, while TR retains small lower-junction colored markers. E is mostly green; no scalar improvement is inferred.'),
    ('bdd29e651e5f6fb2b079317292bdc5d4', 0, 'SIMILAR', 'MIXED',
     'Teacher and TR have very similar complete silhouettes and irregular thin gaps across the seat. TR changes a narrow vertical orange strip by the blue back and some front-seat triangular folds; there is no clear overall repair or newly missing chair component. Both have fewer small back pinholes than START, with other junction defects persisting.',
     'The blue known back and projecting seat triangle do not merge cleanly with the orange geometry. A thin orange/blue back split remains.',
     'S stays mostly red at the seat and back boundaries. I shows localized purple/red at the junction with changed distribution; E is mostly green with purple unknown markers. Unknown is not counted as a confirmed failure.'),
    ('bdd29e651e5f6fb2b079317292bdc5d4', 1, 'MIXED', 'MIXED',
     'TR closes many teacher seat slits and reduces the jagged folded front lip. At the same time, its orange back has several long white vertical gaps, and thin leg splits remain. The teacher also has back slits, but the TR seat improvement cannot be generalized to the back.',
     'TR removes a prominent orange intrusion over the blue back visible in the teacher rear view, while the long free-back openings and incomplete seat contact remain.',
     'S is predominantly red. Teacher I has a pronounced purple vertical band near the known back junction; TR shows fewer such upper markers but local seat markers remain. E is mainly green with unknown purple regions.'),
    ('c1b8c2f0c4f3cc43aff7c08b060f5ed6', 0, 'WORSE', 'MIXED',
     'Teacher and TR both retain a jagged fan crown, sparse back strips and triangular lower-back openings. TR has a particularly clear long white split and thin side in a front leg, while the teacher corresponding leg is broader. Seat/body remain coherent, but this local leg regression prevents an improvement judgment.',
     'Orange back fragments remain in front of the fixed curved blue back. The new leg split lies outside most C-local feedback markers.',
     'S is mostly red around C. I has red/purple back clusters in both, differently distributed; TR appears less densely purple than teacher, but E still includes red internal back patches. Marker color density is not an area-weighted metric.'),
    ('c1b8c2f0c4f3cc43aff7c08b060f5ed6', 1, 'WORSE', 'MIXED',
     'TR has clear substantial new upper-back fragmentation against the teacher: the teacher keeps a broad upper orange triangle and simple lower opening, whereas TR forms a jagged fan crown, sharp narrow pieces and long irregular openings. The teacher is imperfect and has one hanging lower-back triangular point. TR retains a seat and legs, so this is recorded as a severe local back regression rather than disappearance of the whole chair. Relative to START, TR removes the obvious seat spikes/under-seat sheet but worsens the back.',
     'Jagged orange pieces overlap visually with the fixed curved blue back. This is not judged worse merely because of a style change; the broken crown and needle-like panels are the evidence.',
     'Teacher I is mostly gray with edge-local colors; TR adds red and purple internal back patches. E in TR has conspicuous internal red patches where teacher is largely green. S remains mostly red.'),
    ('f59627a077dcc2566c63b8b48495261a', 0, 'BETTER', 'MIXED',
     'Against teacher, TR has more continuous legs and a broader continuous back instead of thin split legs and widely gapped horizontal strips. It also directly exposes the blue C seat. TR still has a ragged perforated rounded crown and a few back gaps, so it is not a clean-mesh or usability pass. START has sturdier legs than teacher but still hides C under an orange seat, making that comparison mixed.',
     'The blue seat is clearly visible and no longer mostly covered by a raised orange layer. Boundary attachment still looks incomplete; new rounded styling alone is neither rewarded nor penalized.',
     'Teacher E at the seat is mostly red while TR is predominantly green. Teacher I has a local purple rim and TR is mainly gray. S remains red around much of the seat even though exposure is visibly improved.'),
    ('f59627a077dcc2566c63b8b48495261a', 1, 'MIXED', 'MIXED',
     'TR has a somewhat more regular broad arm-chair frame and removes several teacher narrow irregular slats. It introduces a large downward triangular sheet spanning part of the under-seat side, very clear in the side view, while teacher has smaller thin sheets/strips. The main chair is recognizable; improved smooth frame regions trade against the new hanging sheet.',
     'More of the known blue seat is exposed in TR, but the new free side sheet is outside what the C exposure markers alone can certify.',
     'TR E at C is predominantly green, with red at one rim, compared with much more red in teacher/START. S remains mostly red and I is mainly gray with small local markers. Exposure improvement does not remove the visible side-sheet defect.'),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest_path = GALLERY/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'COMPLETE' and manifest['requests'] == 16
    assert manifest['labels'] == ['START', 'explore_FM_CONTROL_1000', MATCHED, CANDIDATE]
    assert manifest['all_geometry_unchanged'] and manifest['all_vertices_inside']
    assert manifest['software_GL'] and manifest['CPU_threads'] == 2
    pages = {Path(p['path']).name: p for p in manifest['parent_pages']}
    registered = json.loads((OUT/'manifests/DEV.json').read_text())['rows']
    bykey = {(r['uid'], r['noise_index']): r for r in registered}
    rows, images = [], {}
    for uid, noise, judgment, vsstart, geometry, condition, feedback in NOTES:
        registered_row = bykey[(uid, noise)]
        paths = [str(GALLERY/(uid+s+'.png')) for s in ['', '_feedback']]
        for p in paths:
            if p not in images:
                entry = pages[Path(p).name]
                assert sha(p) == entry['sha256']
                with Image.open(p) as im:
                    assert list(im.size) == entry['size']
                images[p] = dict(path=p, sha256=entry['sha256'], size=entry['size'], actually_viewed=True)
        rows.append(dict(request_id=registered_row['id'], uid=uid, noise_index=noise,
            candidate=registered_row['candidate'], candidate_label=CANDIDATE, matched_label=MATCHED,
            reference_label='START', review_type='AI_VISUAL_REVIEW', reviewer='AI agent /root/coupling',
            actually_viewed=True, scope_status='COMPLETE', judgment=judgment,
            versus_START_judgment=vsstart, geometry_observation=geometry,
            condition_compatibility_observation=condition, feedback_overlay_observation=feedback,
            new_obvious_main_body_collapse_vs_matched=False,
            new_obvious_main_body_collapse_vs_START=False,
            severe_local_regression=(uid.startswith('c1b8') and noise == 1),
            status='PASS', gate_interpretation='No clear new whole-main-body collapse in this row; local WORSE does not mean usable or clean.',
            image_paths=paths, image_sha256={p: images[p]['sha256'] for p in paths},
            mesh_views=['front_3q', 'back_3q', 'front', 'side', 'wire'], feedback_views=['S', 'I', 'E'],
            automatic_metric_values_used_for_judgment=False, human_usability='NOT_RUN'))
    assert len(rows) == 8 and len(images) == 8
    result = dict(schema='chair_matched_dev_partial_ai_visual_review_v1',
        status='INCOMPLETE', scope_status='PARTIAL_FULL_COHORT', assigned_scope_status='COMPLETE',
        created_utc=datetime.now(timezone.utc).isoformat(), review_type='AI_VISUAL_REVIEW',
        reviewer='AI agent /root/coupling', candidate_label=CANDIDATE, matched_label=MATCHED,
        reference_label='START', human_usability='NOT_RUN',
        scope=dict(split='DEV', assigned_parents=4, assigned_requests=8, full_cohort_requests=16,
                   assigned_parent_uids=sorted({r['uid'] for r in rows}),
                   four_shaded_views_plus_wire=True, all_S_I_E_overlays=True,
                   viewed_columns=['C', 'START', 'explore_FM_CONTROL_1000', MATCHED, CANDIDATE, 'GT'],
                   display_regions=[[0,0,2595,2162],[0,2162,2595,4251]],
                   display_method='Both adjacent lossless full-resolution regions covered each saved mesh page completely; each feedback page viewed whole. Original files were not changed.',
                   CONFIRM_read=False, model_calls=0, new_metric_calls=0),
        gate=dict(status='INCOMPLETE', assigned_rows_status='PASS',
                  full_cohort_decision='Must merge with root actual review of the first four parents. This receipt does not assert full-cohort PASS.',
                  narrow_definition='No new obvious main-body collapse, separately from local geometry/C-compatibility deterioration.',
                  warning='c1b noise1 has a substantial new back-fragmentation regression against the teacher; candidate cannot be described as uniformly improved.'),
        descriptive_judgment_counts_vs_matched=dict(sorted(Counter(r['judgment'] for r in rows).items())),
        limitations=['AI review, not blinded human usability.', 'No manifoldness or exact intersection certification from images.',
                     'Saved feedback markers are qualitative x-ray overlays; their visual density is not an area-weighted score.',
                     'Only the assigned latter half of DEV was inspected in this 1000 matched-control review; no other checkpoint review is substituted.'],
        source_manifest=dict(path=str(manifest_path), sha256=sha(manifest_path)),
        source_images=list(images.values()),
        rendering_source_files={p:h for p,h in manifest['source_files'].items() if p.endswith('.py')},
        source_script=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        rows=rows, records=rows)
    destination = OUT/'visuals/reviews'
    path = destination/(STEM+'.json')
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Matched DEV1000 AI visual review: assigned latter four parents', '',
        '**Assigned scope COMPLETE: 4 parents, 8 requests, all four shaded views, wire and S/I/E overlays. Full-cohort gate: INCOMPLETE pending root first-half review. Human usability NOT_RUN.**', '',
        f'Candidate `{CANDIDATE}`; matched teacher `{MATCHED}`. START retained in every comparison. All eight original pages were actually viewed and hash-bound below.', '',
        'The main chair remains recognizable in these assigned rows, but local defects matter: c1b noise1 newly fragments the upper back relative to the teacher; c1b noise0 has a conspicuous thin split leg. f596 noise0 improves leg/back continuity and C-seat exposure relative to the teacher while retaining crown defects. f596 noise1 trades better exposure for a large hanging under-seat triangle. The other rows mostly retain tradeoffs rather than uniform improvement.', '',
        '| Parent | Noise | TR vs teacher | TR vs START | Observation |', '|---|---:|---|---|---|']
    for r in rows:
        lines.append(f"| {r['uid'][:6]} | {r['noise_index']} | {r['judgment']} | {r['versus_START_judgment']} | {r['geometry_observation']} |")
    lines += ['', f'Complete image/hash/source binding: `{path}`.', '',
              'No model, scoring, mesh repair or CONFIRM access was performed. Do not interpret an individual narrow gate PASS as clean geometry or usability.', '']
    (destination/(STEM+'.md')).write_text('\n'.join(lines))
    print(json.dumps(dict(status=result['status'], assigned_scope_status='COMPLETE', requests=8,
                          source_images=8, path=str(path), sha256=sha(path),
                          counts=result['descriptive_judgment_counts_vs_matched']), indent=2))


if __name__ == '__main__':
    main()
