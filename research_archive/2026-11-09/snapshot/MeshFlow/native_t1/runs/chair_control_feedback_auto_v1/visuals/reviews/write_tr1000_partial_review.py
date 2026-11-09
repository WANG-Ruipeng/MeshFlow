"""Persist actual read-only DEV visual observations, without model/scoring calls."""
from pathlib import Path
from collections import Counter
import hashlib
import json

BASE = Path(__file__).resolve().parents[2]
GALLERY = BASE / 'visuals/DEV_START_explore_TRAJECTORY_REPLAY_1000_142454cfd0a3'
LABEL = 'explore_TRAJECTORY_REPLAY_1000'
OBS = [
 ('86ec9c796a6c1789e792598ed58dd1cc', 0, 'MIXED',
  'Recognizable tall chair remains. Some cap holes close, but diagonal seat gaps and an irregular front-leg sheet are more conspicuous.',
  'The blue back/leg frame remains visually present; an orange-to-blue upper-back boundary gap persists. No whole-body collapse is seen.',
  'S has long red stretches on known-leg and upper-back boundaries in both outputs, with some locations changing from green to red. I is mainly gray with sparse red/purple near the seat/leg region. E is mainly green. These are qualitative marker locations, not a recomputed score.'),
 ('86ec9c796a6c1789e792598ed58dd1cc', 1, 'WORSE',
  'The overall chair remains intact, but the top cap stays ragged and a pronounced gap separates part of it from the blue upper frame. The orange seat/apron is higher and thicker with irregular triangular sheets.',
  'The blue upper-frame notch remains exposed below the cap; the seat-to-known-bar relation is visibly offset. This is localized deterioration, not disappearance of the chair.',
  'S remains red across much of the upper frame and several leg boundaries. I is mostly gray with small colored clusters. E remains mostly green, with localized red around junctions; no aggregate improvement is inferred from marker density.'),
 ('bdd29e651e5f6fb2b079317292bdc5d4', 0, 'MIXED',
  'Both have similar complete chair silhouettes. Small orange-back holes are reduced, while thin seat strips and a pointed front-seat feature remain.',
  'A long narrow separation beside the blue back is more visible. The blue seat fragment still projects beyond the orange seat assembly.',
  'S is mostly red around the triangular known seat and the outer back boundary in both. I retains colored clusters near the vertical back junction and seat junction. E is largely green with purple unknown bands; this does not resolve exact contact.'),
 ('bdd29e651e5f6fb2b079317292bdc5d4', 1, 'WORSE',
  'Seat and legs still form a complete chair, but the orange back develops several conspicuous long vertical slits and ragged fragments at its lower edge.',
  'The orange back does not cleanly continue the curved blue back; the junction remains stepped or separated.',
  'S remains predominantly red along the known seat and outer-back lines. I shows local colored points near the seat junction. E is mostly green with unknown purple regions, despite the visible orange-back deterioration.'),
 ('c1b8c2f0c4f3cc43aff7c08b060f5ed6', 0, 'MIXED',
  'The seat becomes simpler and broad chair proportions remain, but the ragged arched orange back layer persists. One front leg has a long open/split-looking thin region.',
  'Orange back triangles still lie in front of the blue curved back, without a clean complement like the contextual GT. Clean attachment is not established.',
  'S is predominantly red on the blue back outline in both. I retains red/purple clusters over the upper back. E remains green with appreciable red patches. The visible defect has not disappeared.'),
 ('c1b8c2f0c4f3cc43aff7c08b060f5ed6', 1, 'MIXED',
  'Trajectory replay removes START seat spikes and its large under-seat hanging sheet, but introduces a conspicuously fragmented rounded arch and long open triangular back strips.',
  'The new rounded orange structure overlays the square/curved blue known back instead of continuing its triangular complement. This is a significant localized C-compatibility regression alongside cleaner lower-body geometry.',
  'S stays largely red. I changes from a narrow right-side purple band to red clusters over the left/top and lower back. E has a more spread-out red back patch. Marker locations support the local concern but are not used to estimate numerical differences.'),
 ('f59627a077dcc2566c63b8b48495261a', 0, 'BETTER',
  'The orange seat cover is removed from much of the known blue seat, and the body becomes a coherent broad-backed chair with contiguous legs.',
  'The blue seat is plainly visible in the intended seat region, unlike its broad orange cover in START. The surrounding frame still has gaps, so exposure is not proof of attachment.',
  'S remains red around the seat perimeter and I is mostly gray. E changes from a large red seat patch in START to predominantly green. The rounded cap remains heavily faceted and perforated; rounded versus rectangular style alone is not the basis for the improvement judgment.'),
 ('f59627a077dcc2566c63b8b48495261a', 1, 'MIXED',
  'The coarse armchair frame becomes more regular and the upright START seat spike disappears, but a conspicuous downward triangular sheet now hangs below the seat.',
  'More of the blue known seat is exposed, although the generated arm/seat framework remains layered above and around it. The armless GT style is not imposed as a requirement.',
  'S is mostly red along the seat boundary. I stays mostly gray. E changes from a largely red seat patch to mainly green with a remaining red edge. The exposure improvement coexists with an obvious new hanging sheet.'),
]

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    mp = GALLERY / 'manifest.json'
    manifest = json.loads(mp.read_text())
    assert manifest['status'] == 'COMPLETE'
    pages = {Path(p['path']).name: p for p in manifest['parent_pages']}
    records, sources = [], {}
    for uid, noise, judgment, mesh_note, c_note, feedback_note in OBS:
        image_paths = []
        for suffix in ('.png', '_feedback.png'):
            p = GALLERY / (uid + suffix)
            actual = digest(p)
            assert actual == pages[p.name]['sha256']
            sources[str(p)] = actual
            image_paths.append(str(p))
        records.append(dict(
            request_id=f'DEV__{uid}_C1__n{noise}', uid=uid, candidate=noise,
            candidate_label=LABEL, matched_label='START', comparison_reference='START',
            matched_training_control_review='NOT_AVAILABLE',
            review_type='AI_VISUAL_REVIEW', reviewer='AI agent /root/coupling',
            actually_viewed=True, judgment=judgment, overall_chair_and_artifacts=mesh_note,
            C_compatibility=c_note, feedback_overlay_observation=feedback_note,
            obvious_whole_body_collapse_observed=False,
            severe_local_back_regression=(uid.startswith('c1b8') and noise == 1),
            image_paths=image_paths, image_sha256={p:sources[p] for p in image_paths},
            observed_views=['front_3q','back_3q','front','side','wire'],
            observed_feedback=['S','I','E'], human_usability='NOT_RUN'))
    assert len(records) == len({r['request_id'] for r in records}) == 8
    result = dict(
        status='PARTIAL_FULL_COHORT_REVIEW', assigned_scope_status='COMPLETE',
        review_type='AI_VISUAL_REVIEW', candidate_label=LABEL, matched_label='START',
        reviewer='AI agent /root/coupling', human_usability='NOT_RUN',
        source_manifest=str(mp), source_manifest_sha256=digest(mp), source_images=sources,
        source_script=str(Path(__file__).resolve()), source_script_sha256=digest(__file__),
        scope=dict(parents=4, requests=8, full_DEV_requests=16,
                   assigned_parents=sorted({r['uid'] for r in records}),
                   remaining_four_parents_reviewer='AI agent /root; not represented by this file',
                   four_views_and_wire=True, S_I_E_overlays=True,
                   mesh_display='Two original-resolution, lossless in-memory PNG crops covering the full 1779x4251 page: (0,0,1779,2162) and (0,2162,1779,4251).',
                   overlay_display='Each entire original 963x2619 saved PNG displayed at original detail.',
                   saved_images_modified=False, new_render_calls=0, model_calls=0,
                   numeric_metric_values_read=False, CONFIRM_read=False),
        records=records, judgment_counts=dict(Counter(r['judgment'] for r in records)),
        conclusion='No whole-chair disappearance is observed in these eight requests. Changes are mixed: visible seat exposure improves at f596..., while local back/C incompatibility and fragmentation persist or worsen elsewhere. c1b8... noise 1 trades lower-body cleanup for substantial new back fragmentation. This partial review does not establish a full-cohort visual gate or superiority to the matched training control.',
        limitations=[
            'Only the assigned latter four DEV parents are covered; combine with the root reviewer receipt before asserting all-16 coverage.',
            'All comparisons here are against START, not an unavailable matched control.',
            'AI categories are descriptive, not human usability labels or acceptance rates.',
            'Overlay markers are existing classifications, displayed in an x-ray projection with possible overlap; no pixel counting, metric recomputation or bounds estimates were performed.',
            'Mostly green exposure markers do not establish seam attachment or coherent free surfaces.',
            'Raster views do not prove exact bytes, watertightness, connectivity or absence of intersections.',
            'GT is contextual; legitimate style variation alone is not counted as degradation.'])
    dst = Path(__file__).parent / 'explore_TRAJECTORY_REPLAY_1000_vs_START_coupling_partial.json'
    dst.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    lines = ['# TR1000 assigned DEV visual review', '',
             'Actual review completed by AI agent /root/coupling for the latter four DEV parents, all 8 requests, four views plus wireframe and corresponding complete S/I/E overlay pages. Human usability: NOT_RUN. Other four parents are owned by /root and are not claimed here.', '',
             'Reference is START; the matched training control has not been reviewed. No overall chair disappearance is visible in this subset, but local quality and C compatibility have mixed changes. The f596 seat becomes more exposed; c1b8 noise 1 removes lower-body artifacts while introducing a fragmented back arch across the known context. bdd noise 1 has markedly worse vertical back slits. No blanket visual-improvement or full-cohort gate claim follows.', '',
             '| Parent | Noise | AI comparison to START | Mesh observation |', '|---|---:|---|---|']
    for r in records:
        lines.append(f"| {r['uid']} | {r['candidate']} | {r['judgment']} | {r['overall_chair_and_artifacts']} {r['C_compatibility']} |")
    lines += ['', 'All eight source PNGs and the gallery manifest are bound by verified SHA256 in the JSON. Existing feedback-marker observations are qualitative, not numerical scores. Overlay overlap prevents reliable visual area counting. No model, rendering, scoring or CONFIRM access was performed.']
    dst.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'path':str(dst), 'assigned_scope_status':result['assigned_scope_status'], 'counts':result['judgment_counts']}))

if __name__ == '__main__':
    main()
