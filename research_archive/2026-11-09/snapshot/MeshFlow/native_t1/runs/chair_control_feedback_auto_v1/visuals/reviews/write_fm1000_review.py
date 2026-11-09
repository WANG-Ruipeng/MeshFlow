"""Retain manually observed AI review; hash saved images, do not score/render meshes."""
from pathlib import Path
import collections
import hashlib
import json

BASE = Path(__file__).resolve().parents[2]
GALLERY = BASE / 'visuals/DEV_START_explore_FM_CONTROL_1000_d57752c4b6d8'
LABEL = 'explore_FM_CONTROL_1000'
# These observations were written only after actually viewing both complete page
# halves at original pixel resolution. They are not inferred from metric values.
OBS = [
 ('33ec1e64a02ff718faaadcdc0221eb72', 0, 'WORSE', False,
  'Chair remains recognizable, but the back becomes a jagged pile of horizontal strips and triangles.',
  'The orange seat and side assembly is layered above the blue lower frame; exact attachment is not established from pixels.',
  'Large diagonal orange sheets run through the side/under-seat region; ragged back and thin supports are more conspicuous than START.'),
 ('33ec1e64a02ff718faaadcdc0221eb72', 1, 'MIXED', False,
  'Both outputs retain a slender chair silhouette. FM fills much of the open back into a broad panel.',
  'Raised orange seat and blue lower frame remain visibly layered, with thin irregular seat edges.',
  'Greater back continuity is visible, but solid versus slatted style alone is not a quality improvement; no clear overall winner.'),
 ('341bcae1e7f366a233d93f57729688db', 0, 'WORSE', True,
  'START has a coherent box-chair shell. FM retains an L-like chair silhouette but loses substantial shell/support coherence.',
  'The blue lower bar protrudes beneath sparse, irregular orange support regions.',
  'Fragmented back cap, long under-seat strips and hanging plates are prominent across side and oblique views. This is structural deterioration beyond an armchair-to-armless style change.'),
 ('341bcae1e7f366a233d93f57729688db', 1, 'SIMILAR', False,
  'Both outputs retain coherent box-armchair shells with similar proportions.',
  'Blue lower-bar ends remain partly exposed beneath both shells; exact contact is unresolved.',
  'Rough lower boundaries and thin scraps remain, without a large new loss of chair structure.'),
 ('534b0e6e2c53a8e1a7cdfd3874ed3723', 0, 'SIMILAR', False,
  'Both outputs are complete recognizable plain chairs with very similar back, seat and leg silhouettes.',
  'The orange seat covers much of the blue seat fragment, with layered edges and a thin orange sliver beside the known leg.',
  'Minor back slits and uneven surface faceting persist. Any upper-back hole reduction is small, not a clear material improvement.'),
 ('534b0e6e2c53a8e1a7cdfd3874ed3723', 1, 'SIMILAR', False,
  'Both outputs retain the same recognizable chair structure.',
  'The seat-to-back and known-leg junction remains patchy; visual contact is not a topology test.',
  'Back slits and a triangular seat lip persist with local changes but no clear overall winner or wholesale collapse.'),
 ('866e7248434968051cc7750d0afca796', 0, 'BETTER', False,
  'FM makes a more coherent rounded back whose opening follows the circular feature already present in C.',
  'The completion follows the partial circular known back more plausibly than the START rectangular/slatted construction.',
  'A ragged opening rim, back slits, visible seat opening and uneven back-to-seat join remain; improvement does not establish usability.'),
 ('866e7248434968051cc7750d0afca796', 1, 'BETTER', False,
  'FM has a more coherent seat and rounded-back outline, with less global fragmentation than START.',
  'The roundish back shape better follows C, although its opening and surrounding surface remain irregular.',
  'Rim holes/slits and a detached-looking triangular sliver below the seat remain; actual connectivity cannot be inferred from the raster.'),
 ('86ec9c796a6c1789e792598ed58dd1cc', 0, 'SIMILAR', False,
  'Both outputs are recognizable tall-back chairs with comparable seat and leg bodies.',
  'The blue back frame/legs remain visually consistent; no obvious second chair body appears.',
  'Small cap holes and ragged slat boundaries persist. Filling a few tiny gaps is not a clear overall quality change.'),
 ('86ec9c796a6c1789e792598ed58dd1cc', 1, 'WORSE', False,
  'FM retains the chair body but the back cap becomes a more fragmented open triangle framework.',
  'The orange completion still uses the blue back frame; exact boundary compatibility is unresolved.',
  'A long irregular diagonal back sheet and narrow slits accompany the cap deterioration. This is localized, not whole-body collapse.'),
 ('bdd29e651e5f6fb2b079317292bdc5d4', 0, 'MIXED', False,
  'Both outputs have similar full chair silhouettes; FM smooths some small holes in the orange back.',
  'A narrow vertical separation/step remains between the curved blue back and the flatter orange completion, and appears stronger in FM.',
  'Streaky seat strips/gaps persist. Local smoothing and boundary separation trade off, with no clear overall winner.'),
 ('bdd29e651e5f6fb2b079317292bdc5d4', 1, 'WORSE', False,
  'Chair silhouette and legs remain intact, but the FM seat surface is more visibly broken into thin strips.',
  'The blue/orange back junction remains stepped or gapped.',
  'Conspicuous white seat gaps and long back slits are more pronounced than START; this is localized surface deterioration.'),
 ('c1b8c2f0c4f3cc43aff7c08b060f5ed6', 0, 'SIMILAR', False,
  'Both outputs retain a recognizable chair with similar broad seat and legs.',
  'Both put an irregular orange back layer in front of the blue curved back; clean continuation is not demonstrated.',
  'Ragged upper-back triangles, long strips, open patches and split-looking leg tips remain. The FM seat is locally simpler but the main back defect persists.'),
 ('c1b8c2f0c4f3cc43aff7c08b060f5ed6', 1, 'BETTER', False,
  'FM removes the large under-seat hanging sheet and several upright seat spikes seen in START, producing a much cleaner chair silhouette.',
  'The orange upper-back complement follows the blue back more cleanly, while thin gaps remain around its lower boundary.',
  'A downward triangular flap remains in the open lower-back region and leg tips stay rough. It is improved, not proven fully usable.'),
 ('f59627a077dcc2566c63b8b48495261a', 0, 'WORSE', False,
  'Both outputs are tall-back chairs, but FM legs become more kinked/split-looking and less coherent.',
  'In both outputs a broad orange seat sits over the blue seat fragment, leaving only a small exposed blue rim.',
  'Back gaps, ragged edges and irregular leg sheets are more conspicuous in FM. The missing GT stretchers alone are not used as failure evidence.'),
 ('f59627a077dcc2566c63b8b48495261a', 1, 'SIMILAR', False,
  'Both outputs form coarse armchairs rather than the armless GT style; style difference alone is not treated as degradation.',
  'The known blue seat is surrounded and partly covered by multiple orange seat/arm strips in both outputs.',
  'START has an upright seat spike; FM trades it for broad triangular back/seat sheets. Both retain irregular gaps and strip-like construction, without a clear overall winner.'),
]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    manifest_path = GALLERY / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'COMPLETE' and manifest['requests'] == 16
    pages = {Path(p['path']).name: p for p in manifest['parent_pages']}
    records = []
    source_images = {}
    for uid, noise, judgment, severe, chair, context, artifacts in OBS:
        path = GALLERY / (uid + '.png')
        actual = digest(path)
        assert actual == pages[path.name]['sha256']
        assert pages[path.name]['size'] == [1779, 4251]
        source_images[str(path)] = actual
        records.append(dict(
            request_id=f'DEV__{uid}_C1__n{noise}', uid=uid, candidate=noise,
            candidate_label=LABEL, matched_label='START', review_type='AI_VISUAL_REVIEW',
            actually_viewed=True, judgment=judgment, severe_visual_regression=severe,
            overall_chair=chair, C_compatibility=context, artifacts_and_missing_regions=artifacts,
            image_paths=[str(path)], image_sha256={str(path): actual},
            observed_views=['front_3q', 'back_3q', 'front', 'side', 'wire'],
            actual_display_region=[0, 0 if noise == 0 else 2162, 1779, 2162 if noise == 0 else 4251],
            human_usability='NOT_RUN', reviewer='AI agent /root/coupling'))
    assert len(records) == 16 and len({r['request_id'] for r in records}) == 16
    counts = dict(collections.Counter(r['judgment'] for r in records))
    result = dict(
        version=1, status='COMPLETE', review_type='AI_VISUAL_REVIEW',
        candidate_label=LABEL, matched_label='START', reviewer='AI agent /root/coupling',
        human_usability='NOT_RUN', source_manifest=str(manifest_path),
        source_manifest_sha256=digest(manifest_path), source_images=source_images,
        source_script=str(Path(__file__).resolve()), source_script_sha256=digest(__file__),
        scope=dict(parents=8, requests=16, four_mesh_views_and_wire_per_request=True,
                   columns=['Input C', 'START', LABEL, 'GT: C + free'],
                   page_pixels=[1779, 4251],
                   complete_page_regions=[[0, 0, 1779, 2162], [0, 2162, 1779, 4251]],
                   display='Lossless in-memory PNG crops at original resolution covering each entire saved page.',
                   saved_images_modified=False, new_render_calls=0, model_calls=0,
                   metric_values_read_for_judgments=False, feedback_overlay_review='NOT_VIEWED_IN_THIS_REVIEW',
                   CONFIRM_read=False),
        records=records, judgment_counts=counts,
        conclusion='Mixed visual changes, including a severe regression at 341b... noise 0. This full DEV review does not support a blanket claim that FM continuation preserves or improves visual quality.',
        interpretation_limits=[
            'Judgments are descriptive AI comparisons to START, not human usability labels or acceptance rates.',
            'No new numerical gate is defined here. COMPLETE means review coverage, not quality PASS.',
            'The saved blue context appears consistent; raster inspection does not establish exact bytes, watertightness, connectivity, intersections or editability.',
            'GT is a contextual reference; legitimate style differences and slatted versus solid backs are not automatically failures.',
            'No CONFIRM image, output or score was accessed.'])
    dest = Path(__file__).parent / 'explore_FM_CONTROL_1000_vs_START.json'
    dest.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# FM_CONTROL1000 full DEV AI visual review', '',
        'All 8 parent pages were actually viewed, covering all 16 requests, four shaded views and wireframe. Each original 1779 x 4251 page was displayed as two lossless halves covering every pixel. C is blue and free geometry orange in START, FM and GT.', '',
        f'Descriptive comparisons to START: {counts}. These are AI reviewer judgments, not human usability rates. Human usability: NOT_RUN. Review completion is not a quality gate PASS.', '',
        'FM has mixed effects. The clearest severe regression is 341b... noise 0: a coherent START shell becomes sparse, malformed support and hanging strips. Localized deterioration also occurs in 33ec... n0, 86ec... n1, bdd... n1 and f596... n0. Conversely, both 866e... noises better follow the round C back, and c1b8... n1 loses conspicuous spikes and a large hanging sheet. Residual holes, strips and uncertain attachment remain in the improved examples.', '',
        '| Parent | Noise | Judgment | Observation |', '|---|---:|---|---|']
    for r in records:
        lines.append(f"| {r['uid']} | {r['candidate']} | {r['judgment']} | {r['overall_chair']} {r['artifacts_and_missing_regions']} |")
    lines += ['', 'The JSON retains each viewed source path and verified SHA256, full scope and limitations. No feedback-score values were used for these judgments; feedback overlays were outside this particular review. No CONFIRM data were viewed. Exact topology, intersections, welds and human usability require separate evidence.']
    dest.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'path': str(dest), 'status': result['status'], 'requests': len(records), 'counts': counts}))

if __name__ == '__main__':
    main()
