"""Persist eight actual SEAM image reviews; no model or geometry calculations."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LABEL = 'SEAM_S13000'
GALLERY_NAME = 'NATIVE_S13000__SEAM_S13000_process'
GALLERY = ROOT / 'gallery' / GALLERY_NAME
PRIOR = ROOT / 'visual_review/NATIVE_S10000_process'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def state(body, use, role, layer, thin, note):
    return dict(body_completeness=body, C_visual_coherent_use=use,
        construction_intended_role_visible=role, visible_offset_or_redundant_layer=layer,
        thin_or_dangling_sheet_appearance=thin, notes=note)

C='COMPLETE_MACRO'; P='PARTIAL'; Y='YES'; U='UNCERTAIN'; PL='PLAUSIBLE'
notes = [
('2d1e5eeb255e79dcab53bf7fa22f3636__BACK_SURFACE_1__n0',
 state(P,U,Y,Y,Y,'The blue board is in the back region of a recognizable bench, while a broad gray back surface remains behind it. The side section shows blue and gray inclined traces at different depths. Short gray extensions appear near the blue ends, but their apparent approach does not establish a junction. The lower gray body has a broad curved sheet and irregular end flaps rather than a clean continuous-looking surface.'),
 state(P,U,Y,Y,Y,'The back-board arrangement and its offset gray layer remain. The gray lower curved sheet is slightly flatter, with a conspicuous triangular opening at an end in the surface view. The blue panel stays in a plausible back location, but neither the offset layer nor the irregular lower structure is visibly resolved.'),
 'Local smoothing leaves the offset gray back and irregular curved lower sheet/end opening visible.'),
('459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0',
 state(P,U,'PARTIAL',Y,Y,'The blue input includes most of the seat slab and one curved end support. A narrower gray support stands at the other end. Several long folded gray strips lie above the blue slab, with raised peaks and openings near their ends. The front and side sections retain separate upper gray and blue seating layers. The complementary gray structure appears irregular, despite the substantial blue assembly.'),
 state(P,U,'PARTIAL',Y,Y,'The gray upper strips become flatter but still form a raised layer above the blue seat. The opposite gray support becomes narrower and partly line-like. The blue assembly remains intact in the same position; the extra gray layer and weak complementary support persist.'),
 'POST flattens the upper gray layer and thins the free opposite support, without resolving the layered seating structure.'),
('459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1__n0',
 state(C,U,Y,Y,U,'A recognizable gray bench with a broad open back structure and substantial side supports surrounds the blue seat-height plane. A broad gray slab remains lower than the blue input in both sections. Small gray sloping or vertical extensions descend near the blue ends in the side section; these do not by themselves establish exact shared connectivity.'),
 state(C,U,Y,Y,U,'The broad back and supports remain recognizable and substantial. The gray lower seat layer is smoother, but its separation from the blue plane persists. Small gray end extensions are still visible in the fixed side section. The input has a clear seat-region role, while integration with the lower gray slab remains uncertain.'),
 'The macro bench is retained; smoothing does not remove the separate gray layer below the blue seating plane.'),
('9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0',
 state(C,PL,Y,U,U,'The blue U-shaped frame forms one end of a recognizable backless bench. The gray seat lies inside and below its top beam and extends to gray legs at the opposite end. Multiple nearby gray seat traces remain in the sections. The frame location is compatible with a support role, without image-based proof of a shared junction.'),
 state(P,PL,Y,U,Y,'The gray seat is flatter, while its opposite free legs become very thin strands in both surface views. The blue end frame retains its robust area and plausible support position. The simplification of gray section traces does not establish that the frame and seat are connected.'),
 'The input end-frame role remains plausible, but POST strongly thins the free legs at the opposite end.'),
('122a480cfcdd742650c626aa72455dae__LEG_SUPPORT_1__n0',
 state(C,PL,Y,Y,U,'A recognizable bench has broad gray seat and back surfaces with large curved gray end supports. The blue tapered leg is largely hidden within or behind one curved support, with its top near the seat underside. Gray curves and horizontal traces cross the blue projection in the finite sections. The position is compatible with a leg role, but projected overlap does not certify an intersection or a shared connection.'),
 state(C,PL,Y,Y,Y,'The broad bench and blue leg location remain. Some curved gray support edges become thinner or more open-looking, while the blue leg stays partly occluded. Gray traces still cross its projection. Macro placement is plausible, but local geometric integration is not resolved by these views.'),
 'The macro shape and blue leg placement are retained; curved free edges look thinner after POST and the junction stays occluded.'),
('122a480cfcdd742650c626aa72455dae__SEAT_SURFACE_1__n0',
 state(C,U,'NO',Y,U,'The generated gray bench has a slatted seat well above the narrow blue slat. The blue element lies around a lower brace or apron height and is partly hidden by a broad gray front panel. Both sections show the gray seating surface above it. A brace interpretation remains possible, so this is not labeled a categorical failure of all coherent use; the construction-intended seat-slat role is not visible.'),
 state(P,U,'NO',Y,Y,'The gray seating strips become notably thinner and more open-looking, while the front apron still partly hides the blue lower slat. The blue-to-gray seat height offset is unchanged. The input could still serve as a lower brace, but it is not at the generated seating surface and the gray seat looks less robust.'),
 'The source-intended seat-height relation remains absent; POST makes the upper gray slats thinner without clarifying the alternative lower-brace use.'),
('20222f467d257022c8bbbd98dee48cb__BACK_SURFACE_1__n0',
 state(C,U,Y,Y,U,'The blue open rectangular frame sits at back height within a recognizable gray bench. Folded or angled gray surfaces extend above and below it toward a broad gray rear back, leaving multiple back layers. The side section contains gray lines approaching the blue upper and lower segments, but no exact connection can be inferred. The back role is apparent while the folded surrounding geometry remains ambiguous.'),
 state(C,U,Y,Y,U,'The broad body and blue back-frame location are retained. The gray back still folds around the frame and includes a rear layer at a different depth. Some short gray section traces disappear or simplify; this alone is not evidence of a hole or topological change. The visual relationship remains layered.'),
 'POST retains the folded gray back around the blue frame; apparent edge approach does not establish a new legal junction.'),
('20222f467d257022c8bbbd98dee48cb__COMBO_SEAT_SUPPORT__n0',
 state(C,U,'PARTIAL',U,U,'A recognizable backed bench surrounds the blue rear-seat slat and wedge-shaped end-support panel. The slat lies near the rear seat edge and the wedge is near a gray end support, making both roles plausible in location. The side section still leaves small offsets and short gray traces near the blue element. Whether the input assembly forms one continuous structure with the gray body is uncertain.'),
 state(P,U,'PARTIAL',U,Y,'The gray back and seat are flatter, but free legs become very thin or nearly line-like. The blue wedge retains its area and the rear slat stays in the same position. Nearby gray traces do not visibly resolve the assembly junction. The loss of robust-looking free supports is a separate quality concern.'),
 'The assembly location is retained, while POST substantially weakens the visible free legs; the local joining remains uncertain.'),
]

manifest_path = ROOT / 'gallery_manifests' / (GALLERY_NAME + '.json')
manifest = read(manifest_path)
completion = read(GALLERY / 'completion.json')
pages = manifest['pages']
assert len(pages) == 8 and len(completion['images']) == 8
assert [p['request_id'] for p in pages] == [n[0] for n in notes]
schema = read(PRIOR / 'review_schema.json')
schema.update(checkpoint=LABEL, actual_view_count=8,
    scope='Eight fixed process requests; SEAM columns 3/4 of existing joint NATIVE/SEAM pages; 16 RAW/POST observations',
    prior_schema_sha256=sha(PRIOR / 'review_schema.json'))
schema['limitations'].append('Existing joint pages also display NATIVE columns. Review is nonblind; only SEAM columns 3/4 are labeled here. No direct arm-improvement claim is made.')
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
rows=[]; pairs=[]; image_records=[]
definition_sha=sha(GALLERY / 'definition.json')
for page, record in zip(pages, notes):
    rid, raw, post, change = record
    path = GALLERY / (page['page_id'] + '.png')
    image_receipt = path.with_suffix('.json')
    ir = read(image_receipt)
    image_sha=sha(path)
    assert image_sha == ir['image_sha256']
    assert ir['definition_sha256'] == definition_sha
    assert page['cells'][2]['label'] == LABEL + '_RAW'
    assert page['cells'][3]['label'] == LABEL + '_POST'
    image_records.append(dict(request_id=rid, path=str(path), sha256=image_sha,
        receipt_sha256=sha(image_receipt), reviewed_column_indices_zero_based=[2,3]))
    for stage, observation, column in (('RAW',raw,2), ('POST',post,3)):
        rows.append(dict(review_version=schema['review_version'], reviewer='AI_PROXY', actual_image_viewed=True,
            human_reviewed=False, nonblind=True, reviewed_at_utc=now, label=LABEL, global_step=13000,
            request_id=rid, condition_id=page['condition_id'], parent_id=page['parent_id'], split=page['split'],
            stage=stage, K=page['K'], N=page['N'], part_description=page['kind'], image_path=str(path),
            image_sha256=image_sha, image_receipt_sha256=sha(image_receipt), gallery_definition_sha256=definition_sha,
            reviewed_column_index_zero_based=column, model_stage_label=page['cells'][column]['label'],
            views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
            body_recognizable_as_bench=True, **observation, exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
            topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY', metric_values_used_for_labels=False,
            source_full_mesh_shown=False, scope='Fixed process checkpoint observation; no checkpoint/data/seed selection or training-policy changes'))
    pairs.append(dict(request_id=rid, split=page['split'], part_description=page['kind'], RAW_to_POST=change,
        image_sha256=image_sha, reviewer='AI_PROXY', actual_image_viewed=True))

def write(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        f.write(value)

write('review.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
write('raw_post_comparison.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in pairs))
write('review_schema.json', json.dumps(schema, ensure_ascii=False, indent=2)+'\n')
counts={stage:{field:dict(collections.Counter(r[field] for r in rows if r['stage']==stage))
    for field in ('body_completeness','C_visual_coherent_use','thin_or_dangling_sheet_appearance')}
    for stage in ('RAW','POST')}
write('observed_label_counts.json', json.dumps(dict(source='Actual image review labels only', counts=counts), indent=2)+'\n')
md=['# SEAM_S13000 process image review (AI_PROXY)', '',
    'Actually inspected all eight existing joint gallery pages, using SEAM RAW/POST columns 3/4, both surface views and the fixed front/side sections. Saved 16 stage observations and eight paired comparisons. The other NATIVE columns were visible, so this review is nonblind. No model calls, sampling, rescoring or new rendering were performed.', '',
    'The blue end frame of 9fd5 and blue leg of 122a have plausible support placement. Other cases retain separate gray seat/back layers or ambiguous use. In particular, the 122a seat slat is below the generated seating surface; it may serve as a lower brace, so the source-intended role and overall coherent use are deliberately recorded separately.', '',
    'POST frequently flattens gray surfaces while making free support or seat-strip surfaces visibly thinner. This is clear for 9fd5 opposite legs, 20222 combo free legs, and 122a upper seat strips. The blue surfaces retain their positions. A plausible location does not certify a legal shared edge, and an apparent short section trace does not certify a disconnected fragment.', '',
    'These are process observations from one fixed noise per condition, not a convergence or SEAM-versus-NATIVE quality verdict. Exact connectivity is NOT_ASSESSED_FROM_IMAGES and disconnectedness is UNKNOWN_IMAGE_ONLY in all rows. Source part names are construction diagnostics, not semantic inputs or a requirement to reproduce the unique source style.', '',
    '| Request | RAW to POST observation |', '|---|---|']
for pair in pairs:
    md.append('|'+pair['request_id']+'|'+pair['RAW_to_POST']+'|')
md.extend(['', 'See review.jsonl for stage-specific notes and observed_label_counts.json for counts of these visual labels only. No checkpoint or training-policy decision is made here.', ''])
write('REVIEW.md', '\n'.join(md))
write('receipt.json', json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE', reviewer='AI_PROXY', reviewed_at_utc=now,
    images_viewed=8, image_view_calls_per_page=1, stage_rows=16, paired_rows=8, model_calls=0, new_samples=0,
    rescoring=0, data_audits=0, new_renders=0, metric_values_used=False, joint_gallery_name=GALLERY_NAME,
    reviewed_column_indices_zero_based=[2,3], nonblind=True, protected_gallery_definition_sha256=definition_sha,
    protected_manifest_sha256=sha(manifest_path), protected_completion_sha256=sha(GALLERY/'completion.json'),
    image_records=image_records, files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()}), ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE', images=8, stage_rows=16, paired_rows=8, counts=counts)))
