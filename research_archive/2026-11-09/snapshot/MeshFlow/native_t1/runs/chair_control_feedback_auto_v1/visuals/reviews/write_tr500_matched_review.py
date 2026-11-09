"""Bind manually observed, complete DEV images to the AI review receipt.

The observations below were made by agent /root/coupling after actually viewing
all eight saved parent pages, their two full-resolution noise regions, and all
eight saved S/I/E feedback pages. This script records that review; it does not
perform an automated visual assessment, render, score, or access CONFIRM.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image

OUT = Path('/mnt/e/MeshFlow/native_t1/runs/chair_control_feedback_auto_v1')
GALLERY = OUT / 'visuals/DEV_START_explore_FM_CONTROL_500_explore_REPLAY_FM_CONTROL_500_explore_TRAJECTORY_REPLAY_500_b3d953028828'
CANDIDATE = 'explore_TRAJECTORY_REPLAY_500'
MATCHED = 'explore_REPLAY_FM_CONTROL_500'
LABELS = ['START', 'explore_FM_CONTROL_500', MATCHED, CANDIDATE]
STEM = CANDIDATE + '_vs_' + MATCHED


def observation(uid, noise, judgment, versus_start, geometry, condition, feedback, collapse=False):
    return dict(uid=uid, noise_index=noise, judgment=judgment,
                versus_START_judgment=versus_start,
                geometry_observation=geometry, condition_compatibility_observation=condition,
                feedback_overlay_observation=feedback,
                new_obvious_main_body_collapse_vs_matched=collapse,
                new_obvious_main_body_collapse_vs_START=collapse)


OBSERVATIONS = [
    observation('33ec1e64a02ff718faaadcdc0221eb72', 0, 'BETTER', 'MIXED',
        'The teacher has broad diagonal side sheets descending through the seat and under-seat region. TR removes those large drooping sheets and has a more regular open arm frame and coherent broad back. The orange seat remains layered and rough, with an interior triangular protrusion.',
        'The blue box-like known base remains distinct from the raised orange seat layers. Removing the side sheet is an improvement but does not establish a sealed C/free interface.',
        'S is mostly red along the known base in both columns; I is mainly gray with localized red/purple seat markers; E is predominantly green. These are qualitative marker observations, not integrated score estimates.'),
    observation('33ec1e64a02ff718faaadcdc0221eb72', 1, 'WORSE', 'MIXED',
        'The teacher retains a broad continuous back. TR has a jagged slatted back and a conspicuous long downward needle/sliver under the seat, visible in the side view. The slatted style alone is not judged a defect; the new needle and jagged edges are. The main chair remains recognizable.',
        'Both have an orange raised seat over the blue base. The lower needle is outside most of the known-condition overlay region.',
        'Mostly red S, mostly gray I, and mainly green E at C do not reveal the hanging free-region needle.'),
    observation('341bcae1e7f366a233d93f57729688db', 0, 'WORSE', 'WORSE',
        'Clear new main-body deterioration against both the teacher and START: TR loses much of the side/support structure and its seat and under-seat region break into dense narrow strips, gaps, jagged triangles and long protrusions. The teacher is already changed from the START box chair, but retains broad contiguous panels and a more coherent seat/frame. This decision concerns fragmentation and support loss, not an armchair-to-armless style preference.',
        'The small fixed blue support/known region does not constrain the large missing and fragmented free body sufficiently to certify usability.',
        'The known foot/boundary remains red in S; I is mostly gray and E mostly green on the known bar. Those C-local overlays do not compensate for the visible body deterioration.', True),
    observation('341bcae1e7f366a233d93f57729688db', 1, 'WORSE', 'WORSE',
        'The teacher and START preserve a coherent box-like chair. TR retains that overall shell but has a more buckled/layered front seat/base, a narrow white crack inside the back, warped upper edges, and ragged lower parts. Local deterioration is visible without the severe body breakup of noise 0.',
        'The blue known region remains visible; the new interior cracks and base roughness are free-geometry defects.',
        'S remains red at the known foot patch, with mainly gray I and green E at the known bar. No quantitative superiority is inferred from these sparse condition-local markers.'),
    observation('534b0e6e2c53a8e1a7cdfd3874ed3723', 0, 'WORSE', 'WORSE',
        'TR has a more conspicuously raised/floating orange seat and sharp protruding seat triangle, with a larger apparent junction gap by the known leg. An orange leg sliver is pulled away from the blue leg. Some small back holes reduce, but the seat/junction defect is worse. The complete chair silhouette survives.',
        'The orange seat remains offset above the blue known triangle and the narrow orange leg is visibly separated from the blue leg in some views.',
        'S is predominantly red along the C leg/seat. Local I markers shift around the seat; E retains a red known-seat patch and green known leg. These observations do not replace the saved numerical classifications.'),
    observation('534b0e6e2c53a8e1a7cdfd3874ed3723', 1, 'MIXED', 'SIMILAR',
        'TR closes part of the teacher lower-back gap and reduces several long back slits. Sharp seat lips, raised seat mismatch, and a narrow split leg/sliver remain. Back continuity improves while local seat/leg defects persist.',
        'C/free seat and leg alignment is still visibly imperfect; back filling alone is not a global compatibility pass.',
        'S is mostly red in both. TR shows a purple seat band in I where the teacher has gray/red markers, and the known-seat red E patch remains.'),
    observation('866e7248434968051cc7750d0afca796', 0, 'SIMILAR', 'BETTER',
        'Teacher and TR both produce a coherent chair with a circular back opening aligned in style with the blue known back. Differences are small cap/rim adjustments; ragged circular edges and narrow slits persist. Both look more consistent with C than the START rectangular slatted/holey back.',
        'The round opening follows the known circular feature qualitatively, but small rim gaps remain. Image agreement does not prove exact surface attachment.',
        'S mixes red and green around the boundary; I has localized colored markers on the known back/leg and E is predominantly green.'),
    observation('866e7248434968051cc7750d0afca796', 1, 'BETTER', 'BETTER',
        'TR has a more open and regular circular hole than the teacher, whose opening contains inward jagged triangles. Seat/body are coherent and improve on the more fragmented START result. A thin hanging sliver and a small detached-looking triangle below the seat remain; the teacher also has under-seat sliver/triangle artifacts.',
        'The round known-back shape is followed better, but remaining free fragments prevent calling the result fully usable.',
        'S is mostly red; I mostly gray; E mainly green with localized red at the back. No numeric reduction is asserted from image colors alone.'),
    observation('86ec9c796a6c1789e792598ed58dd1cc', 0, 'WORSE', 'WORSE',
        'Against the teacher relatively coherent slatted back/cap, TR has a ragged open triangular crown and a long white diagonal back gap between large orange triangles. A thin split front leg is also visible. Seat, legs and the main back remain present, so this is substantial local deterioration rather than the clear whole-body breakup seen at 341b noise 0.',
        'Known blue frame remains, but the orange cap and triangular panels fit it irregularly. A solid-versus-slatted style change is not itself the reason for the worse judgment.',
        'Upper S markers that are partly green in the teacher are largely red in TR; I has local purple/red seat-junction markers. E stays mainly green with localized red.'),
    observation('86ec9c796a6c1789e792598ed58dd1cc', 1, 'MIXED', 'MIXED',
        'TR fills the teacher slatted back with larger triangular panels, improving continuity in places but leaving diagonal/edge gaps and a ragged cap. The front leg still has a thin split. The chair body remains intact.',
        'Back filling trades apparent additional coverage/overlap of the known uprights for continuity; it cannot be treated as uniformly better.',
        'I shows purple clusters along the known back uprights and seat horizontal in TR, while E has visible localized red along those parts. Marker extent is described qualitatively, not as a surface-area fraction.'),
    observation('bdd29e651e5f6fb2b079317292bdc5d4', 0, 'WORSE', 'WORSE',
        'The teacher orange back is mostly continuous with small holes. TR introduces a long conspicuous white vertical slit and additional narrow holes; the seat still has thin gaps and a rough lip. The complete chair body is retained.',
        'The blue back and seat triangle project beyond the orange surfaces. The new long back opening is a local compatibility/continuity concern.',
        'Known-seat S stays mostly red, with some green along the inner back edge. I markers move near the junction; E remains green with purple unknown markers. Unknown colors are not treated as confirmed failures.'),
    observation('bdd29e651e5f6fb2b079317292bdc5d4', 1, 'MIXED', 'WORSE',
        'TR removes a large diagonal under-seat sheet present in the teacher, but its long back gaps become more conspicuous. The main seat and legs remain. START has less open back fragmentation in this noise.',
        'The step between the blue and orange back remains; removal of the under-seat sheet is a real local improvement despite worse back continuity.',
        'S is mostly red. A TR red/purple I cluster is visible near the known seat triangle; E is predominantly green with unknown purple regions. No scalar metric value was consulted.'),
    observation('c1b8c2f0c4f3cc43aff7c08b060f5ed6', 0, 'SIMILAR', 'SIMILAR',
        'Teacher and TR both retain seat and legs but have a severely ragged fan-like crown and narrow back strips. TR changes the gaps and thickens parts of the seat/frame; there is no clear cleanup of the pre-existing fragmented back and no new disappearance of the overall chair.',
        'Many orange back strips lie in front of the large blue known curved back. The incompatible double structure remains in all generated columns.',
        'S is mostly red around the C back. I has red/purple clusters in both, with the teacher showing a dense upper patch and TR a differently distributed patch. E is green with red local areas in both; the pattern does not establish full attachment.'),
    observation('c1b8c2f0c4f3cc43aff7c08b060f5ed6', 1, 'WORSE', 'MIXED',
        'The teacher has a broad coherent orange back with only small edge gaps. TR replaces much of it with jagged fan-like crown pieces, long narrow strips and large openings: substantial new back fragmentation. Seat and legs remain as a recognizable chair. Relative to START, TR also removes the obvious seat spikes and hanging under-seat sheet, so that separate comparison is mixed.',
        'TR has many fragmented orange pieces in front of the fixed curved blue back instead of the teacher broad panels. The judgment is based on jagged breakup, not on a preference for a particular back style.',
        'Teacher I is mainly gray with edge-local colors; TR adds internal red/purple back patches. TR E has more visibly red internal back patches. S remains mostly red. These are qualitative visual observations only.'),
    observation('f59627a077dcc2566c63b8b48495261a', 0, 'MIXED', 'MIXED',
        'TR exposes the blue C seat and gives a more continuous broad back, while teacher/START put a raised orange layer over the seat and have many horizontal back slits. TR also introduces a ragged perforated rounded crown and heavier legs. The main chair is coherent, but improved C exposure trades against a malformed crown.',
        'The fixed blue seat is directly visible in TR rather than hidden under an orange duplicate layer. The rounded back style is not itself penalized; the cap holes and jagged facets are.',
        'E changes from a largely red teacher seat to a largely green TR seat, consistent with the visible exposure. I is mostly gray in TR versus a teacher local purple patch. S remains red around much of the boundary despite exposure improvement.'),
    observation('f59627a077dcc2566c63b8b48495261a', 1, 'MIXED', 'MIXED',
        'TR removes the teacher side-pointing back needle and retains a coherent arm-chair frame, but adds large triangular sheets across the under-seat/side region and heavier fragmented side supports. START has its own upright seat spike. This is a tradeoff, not an across-the-board improvement.',
        'More of the blue C seat is exposed in TR, but the large new free side triangles remain visually problematic outside the C-local exposure test.',
        'E at the known seat becomes predominantly green in TR versus much more red in teacher/START. S remains red and I shows a small localized red/purple cluster in TR. Better exposure does not validate the new side sheets.'),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest_path = GALLERY / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'COMPLETE' and manifest['split'] == 'DEV'
    assert manifest['labels'] == LABELS
    assert manifest['requests'] == 16 and manifest['parents'] == 8
    assert manifest['all_geometry_unchanged'] and manifest['all_vertices_inside']
    assert manifest['feedback_overlay_status'] == 'COMPLETE'
    assert manifest['software_GL'] and manifest['CPU_threads'] == 2
    assert manifest['model_calls'] == manifest['new_metric_calls'] == 0
    dev_path = OUT / 'manifests/DEV.json'
    registered = json.loads(dev_path.read_text())['rows']
    index = {(r['uid'], r['noise_index']): r for r in registered}
    assert len(index) == len(OBSERVATIONS) == 16
    page_index = {Path(p['path']).name: p for p in manifest['parent_pages']}
    source_images = []
    for uid in sorted({r['uid'] for r in OBSERVATIONS}):
        for suffix in ['', '_feedback']:
            page = page_index[uid + suffix + '.png']
            actual_hash = sha256(page['path'])
            assert actual_hash == page['sha256'], page['path']
            with Image.open(page['path']) as image:
                assert list(image.size) == page['size']
            source_images.append(dict(path=page['path'], sha256=actual_hash,
                                      size=page['size'], actually_viewed=True))
    rows = []
    for note in OBSERVATIONS:
        row = dict(note)
        registered_row = index[(row['uid'], row['noise_index'])]
        image_paths = [str(GALLERY / (row['uid'] + '.png')),
                       str(GALLERY / (row['uid'] + '_feedback.png'))]
        row.update(request_id=registered_row['id'], candidate=registered_row['candidate'],
                   candidate_label=CANDIDATE, matched_label=MATCHED, reference_label='START',
                   review_type='AI_VISUAL_REVIEW', reviewer='AI agent /root/coupling',
                   actually_viewed=True, scope_status='COMPLETE',
                   status='FAIL' if row['new_obvious_main_body_collapse_vs_matched'] else 'PASS',
                   image_paths=image_paths,
                   image_sha256={p: page_index[Path(p).name]['sha256'] for p in image_paths},
                   mesh_views=['front_3q', 'back_3q', 'front', 'side', 'wire'],
                   feedback_views=['S', 'I', 'E'], human_usability='NOT_RUN',
                   automatic_metric_values_used_for_judgment=False)
        rows.append(row)
    failure_ids = [r['request_id'] for r in rows if r['status'] == 'FAIL']
    counts = dict(sorted(Counter(r['judgment'] for r in rows).items()))
    result = dict(
        schema='chair_complete_matched_dev_ai_visual_review_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='FAIL' if failure_ids else 'PASS', scope_status='COMPLETE',
        review_type='AI_VISUAL_REVIEW', reviewer='AI agent /root/coupling',
        candidate_label=CANDIDATE, matched_label=MATCHED, reference_label='START',
        human_usability='NOT_RUN',
        scope=dict(split='DEV', parents=8, requests=16, all_registered_requests=True,
                   four_shaded_views_and_wire_per_request=True,
                   all_three_feedback_overlays_per_request=True,
                   viewed_mesh_columns=['Input C'] + LABELS + ['GT: C + free'],
                   viewed_feedback_columns=LABELS,
                   mesh_original_pages=8, feedback_original_pages=8,
                   mesh_display_regions=[[0, 0, 2595, 2162], [0, 2162, 2595, 4251]],
                   display_method='Original-resolution saved PNG; mesh page shown as two adjacent lossless in-memory crops covering every pixel. Feedback page viewed whole. No image was rewritten.',
                   geometry_changes=0, new_render_calls_during_review=0,
                   model_calls=0, new_metric_calls=0, CONFIRM_read=False,
                   scalar_metric_tables_read_for_judgment=False),
        gate=dict(status='FAIL' if failure_ids else 'PASS',
                  definition='No new obvious main-body collapse in TR relative to the matched teacher control; START is retained as a separate constraint.',
                  status_vs_matched='FAIL' if failure_ids else 'PASS',
                  status_vs_START='FAIL' if failure_ids else 'PASS',
                  failing_request_ids=failure_ids,
                  reason='341b noise 0 has new fragmentation of seat and support/body regions against both controls. Other local worsenings and improvements are retained in all16 rows.',
                  interpretation='PASS on an individual row means this narrow new-main-body-collapse gate was not triggered. It is not a usability, clean-mesh, or numerical-metric pass.'),
        descriptive_judgment_counts_vs_matched=counts,
        counts_interpretation='AI qualitative case counts; neither blinded human preference nor human usability rate.',
        limitations=['Views can show gaps, thin sheets and apparent intersections but do not certify manifoldness, exact attachment or physical support.',
                     'Feedback markers are the saved classifications projected as x-ray points. Color density is not an area-weighted score; unknown remains unknown.',
                     'Candidate and method names were visible. This was not a blinded human assessment.',
                     'Style changes alone were not treated as failures; fragmentation, protrusions, missing structure and C compatibility were assessed explicitly.'],
        source_manifest=dict(path=str(manifest_path), sha256=sha256(manifest_path)),
        registered_DEV_manifest=dict(path=str(dev_path), sha256=sha256(dev_path)),
        source_images=source_images,
        rendering_source_files={p: h for p, h in manifest['source_files'].items() if p.endswith('.py')},
        gallery_generation={k: manifest[k] for k in [
            'CPU_threads', 'GL_workers', 'software_GL', 'unique_cached_mesh_panels',
            'overlay_panels', 'mesh_render_seconds', 'wall_seconds',
            'all_geometry_unchanged', 'all_vertices_inside', 'per_parent_camera_shared']},
        timing_note='mesh_render_seconds is accumulated renderer time, excluding layout/overlay assembly. wall_seconds is gallery generation wall time, not model sampling or visual-review time.',
        source_script=dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__)),
        rows=rows, records=rows)
    destination = OUT / 'visuals/reviews'
    destination.mkdir(exist_ok=True)
    result_path = destination / (STEM + '.json')
    assert not result_path.exists(), 'Do not overwrite a completed independent review'
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    lines = [
        '# Complete DEV500 matched-control AI visual review', '',
        '**Visual gate: FAIL. Coverage: COMPLETE (8 parents, 16 noises). Human usability: NOT_RUN.**', '',
        f'Candidate `{CANDIDATE}` is compared primarily with `{MATCHED}`. START and ordinary FM500 are also visible in every page. All four shaded views, the wire view and all S/I/E feedback overlays were actually inspected for every request.', '',
        'The decisive failure is **341b noise 0**: compared with both the teacher control and START, TR newly fragments the seat and lower/side support body into narrow strips, gaps and protruding triangles. The teacher itself differs from START, but that does not explain away the additional TR deterioration.', '',
        'The review is not uniformly negative. For example, 33ec noise 0 loses large unwanted side sheets, and 866e noise 1 has a cleaner round back opening. Conversely, c1b noise 1 gains severe back fragmentation relative to the teacher; f596 exposes the known seat more clearly but retains or adds malformed crown/side sheets. C-local feedback improvements cannot by themselves certify the rest of the free chair body.', '',
        f'Descriptive judgments against the teacher: `{json.dumps(counts, sort_keys=True)}`. These counts are AI observations, not human success rates. Local WORSE rows need not trigger the narrower whole-body-collapse gate.', '',
        '| Parent | Noise | TR vs teacher | TR vs START | Main-body gate | Observation |',
        '|---|---:|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['uid'][:6]} | {r['noise_index']} | {r['judgment']} | {r['versus_START_judgment']} | {r['status']} | {r['geometry_observation']} |")
    lines += ['', 'Every original mesh page and feedback page is listed with an actual SHA256 in the JSON. The original gallery manifest binds the renderer, raw inputs, feedback vectors, fixed cameras and panel receipts. The review recorder source and its hash are retained.', '',
              f'Gallery: `{GALLERY}`', f'JSON: `{result_path}`', '',
              'No model, geometry repair, scoring, noise generation or CONFIRM inspection was performed for this review. The software-rendered gallery used two CPU threads. Viewing cropped each tall mesh page into two lossless adjacent display regions; all original pixels were covered, and original files were unchanged.', '']
    (destination / (STEM + '.md')).write_text('\n'.join(lines))
    print(json.dumps(dict(status=result['status'], scope_status=result['scope_status'],
                          requests=len(rows), source_images=len(source_images),
                          counts=counts, failing_request_ids=failure_ids,
                          path=str(result_path), sha256=sha256(result_path)), indent=2))


if __name__ == '__main__':
    main()
