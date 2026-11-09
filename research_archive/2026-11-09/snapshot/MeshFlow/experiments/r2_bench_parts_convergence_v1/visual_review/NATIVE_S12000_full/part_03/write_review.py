"""Persist the 16 actual image-only reviews of sorted requests 32..47.

Images were individually viewed before these notes were written. This script
validates immutable image identities; it never computes geometry metrics.
"""
import collections
import datetime
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
LABEL='NATIVE_S12000'
GALLERY=ROOT/'gallery'/f'{LABEL}_full'
PRIOR=ROOT/'visual_review/NATIVE_S10000_process'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(body,use,role,layer,thin,note):return dict(body_completeness=body,C_visual_coherent_use=use,
    construction_intended_role_visible=role,visible_offset_or_redundant_layer=layer,
    thin_or_dangling_sheet_appearance=thin,notes=note)
C='COMPLETE_MACRO';P='PARTIAL';Y='YES';U='UNCERTAIN';N='NO_OBVIOUS_IN_VIEWS';PL='PLAUSIBLE';NO='NO_VISIBLE_COHERENT_USE'
notes=[
('9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0',
 state(C,PL,Y,U,U,'The blue U-shaped end frame sits at one end of a recognizable backless bench. The gray seat extends from inside this frame to two gray far-end legs. The side section places the seat below the blue top beam; several closely spaced gray traces remain around the seat. The placement is compatible with a support role, but the views do not establish a shared junction.'),
 state(P,PL,Y,U,Y,'The seat is flatter, but both free far-end legs are reduced to very thin, partly intermittent-looking strands in the two surface views. The blue end frame remains robust and in the same plausible support position. Gray seat traces inside the frame become simpler; this is not evidence that a shared connection was created.'),
 'Seat smoothing is accompanied by strong thinning of the free opposite legs; the blue end-frame role remains visually plausible.'),
('9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n1',
 state(C,PL,Y,U,U,'A broad gray seat and two gray far-end legs form a bench with the blue end frame. The gray seat is visibly lower than the frame top beam and has an irregular layered edge. The side section shows the gray seat inside the U opening, supporting the end-frame interpretation without proving attachment.'),
 state(P,PL,Y,U,Y,'The gray seat remains but its free legs become almost line-like. The blue frame is unchanged. The side section retains the seat below the blue beam with fewer gray irregularities; the thin free support surfaces are a visible quality concern.'),
 'The frame placement is retained, while POST makes the opposite free supports substantially thinner.'),
('9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n0',
 state(C,NO,'NO',Y,N,'The long blue seat slat is visibly elevated above a separate gray bench seat. Both front and side sections show a substantial vertical gap, and both surface views show the blue strip above the gray body. The gray bench has its own seat and legs; no coherent incorporation of the input slat is visible.'),
 state(P,NO,'NO',Y,Y,'The elevated blue slat and the gap above the gray seat persist. The gray seat becomes smoother, but the free legs are now very thin strands. The image does not show a new bridge from the generated body to the input slat.'),
 'The seat-height mismatch is unchanged; the gray body is smoother but its legs are much thinner.'),
('9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n1',
 state(C,NO,'NO',Y,N,'The blue slat lies above a complete gray backless bench rather than at its seating surface. The two finite sections agree on the large vertical separation. Gray seat and leg geometry are recognizable without using the blue slat.'),
 state(P,NO,'NO',Y,Y,'The blue slat remains clearly above the gray seat. POST makes the gray legs narrow and in places barely visible; the surface is flatter, but the input-use discrepancy remains in both sections.'),
 'No visible adoption improvement; smoothing coexists with free-leg thinning.'),
('f495bd67af8d9f0cadd1cf65e8dc4de__COMBO_PART_ASSEMBLY__n0',
 state(P,U,'PARTIAL',Y,Y,'The blue upper strip and two short blue legs coexist with a lower gray shelf or seat and a tall gray rear strip. The front section shows the blue horizontal element well above the gray horizontal layer. Several small gray upright sheet-like protrusions appear around the lower layer. The assembly could serve as a top component, but coherent support of it is unclear.'),
 state(P,U,'PARTIAL',Y,Y,'The lower gray layer and its uprights become thinner and more open-looking. The blue upper component stays separated in height from the main gray horizontal layer. Small upright or narrow sheet appearances remain; the images do not resolve whether the assembly is mechanically integrated.'),
 'POST simplifies the gray lower layer, while the ambiguous upper/lower assembly relation and thin protrusions remain.'),
('f495bd67af8d9f0cadd1cf65e8dc4de__COMBO_PART_ASSEMBLY__n1',
 state(C,PL,Y,Y,Y,'The blue upper surface occupies a seat-height position within a broad gray end-frame bench. Gray longitudinal strips lie over or close to the blue surface, partly hiding it, and extra thin gray supports hang below the seat. The side section contains several closely spaced seat traces. The input assembly has a plausible role, but overlapping layers remain visible.'),
 state(C,PL,Y,Y,Y,'The blue seating surface becomes more exposed as the surrounding gray strips become narrow. The large end frames remain, along with thin additional gray supports beneath the seat. The central sections are simpler but retain nearby seat layers; exact connectivity is not established.'),
 'The input surface is more visible after POST, but nearby gray layers and thin extra supports remain.'),
('f495bd67af8d9f0cadd1cf65e8dc4de__LEG_SUPPORT_1__n0',
 state(C,U,'PARTIAL',Y,U,'The gray body is a long seat with two curved or loop-like end supports. The short blue rectangular leg is partly hidden beneath the seat and extends lower than nearby curved gray elements. In the sections, gray curves pass through the blue leg projection. Its position permits a support interpretation, but its coherent relation to the existing curved supports is ambiguous.'),
 state(P,U,'PARTIAL',Y,Y,'The broad seat survives while the curved gray supports are reduced to narrow strips. The blue leg remains beneath the slab and partly obscured. The side section retains a gray curve crossing the blue projection; this projection is not a certified triangle intersection or junction.'),
 'POST makes the curved free supports considerably thinner; the blue leg remains visually ambiguous within them.'),
('f495bd67af8d9f0cadd1cf65e8dc4de__LEG_SUPPORT_1__n1',
 state(C,U,'PARTIAL',Y,Y,'The gray body resembles a long rack-like bench with a lower slatted seat and a raised rear strip. The blue vertical leg extends both below and above the gray seating layer; the front section shows this mismatch clearly. It can be read as a support, but its upper protrusion and surrounding layers make its intended use uncertain.'),
 state(P,U,'PARTIAL',Y,Y,'The gray long rails and end supports become thinner and more open-looking, while the blue leg still projects through the seating-level region. POST does not remove the upper protrusion or clarify how the blue leg belongs to the body.'),
 'The blue-leg height relation is unchanged; gray rail and support surfaces become thinner.'),
('f5d370e9fe5b7452dbbc9440457e303e__BACK_SURFACE_1__n0',
 state(C,PL,Y,U,U,'The broad blue board forms the visible back of a bench with gray seat strips and robust gray end supports. The side section places the gray seat near the lower portion of the blue board. No second broad gray back panel is apparent in these views. This is plausible back-board use, with actual shared connectivity unknown.'),
 state(P,PL,Y,U,Y,'The blue back and gray end supports remain in the same positions. The gray seat becomes a set of much thinner parallel strips with conspicuous openings in the surface view; the side section loses many gray seat traces. The plausible back role remains, but seat surface robustness is reduced.'),
 'The back-board role is retained, while POST substantially thins the free seat strips.'),
('f5d370e9fe5b7452dbbc9440457e303e__BACK_SURFACE_1__n1',
 state(C,U,Y,Y,Y,'The blue board sits in the back region, but the generated bench also has gray slats forming another back layer in front of it. The side section shows the blue and gray inclined elements at different depths. Thin gray protrusions are visible at the ends. The intended back role is evident, while coherent integration of the two back layers is uncertain.'),
 state(C,U,Y,Y,Y,'The generated slatted back remains in front of the blue board, with the same depth separation in the side section. Some end details and support surfaces become thinner, but the broad shape is retained. POST does not visibly eliminate the extra back layer.'),
 'Gray and blue back layers remain separated; local smoothing does not resolve the ambiguous duplicated back structure.'),
('f5d370e9fe5b7452dbbc9440457e303e__SEAT_SURFACE_1__n0',
 state(C,U,Y,Y,N,'The blue rectangle occupies the seat region of a recognizable bench with a gray back and broad supports. The sections nevertheless show the blue plane above the gray segmented seat slab with a visible gap. The view supports a seat role but does not establish that the blue surface is integrated with the lower gray slab.'),
 state(C,U,Y,Y,N,'The broad back and supports remain. The gray slab below the blue seat becomes smoother, but the visible height separation remains in both sections. No major field of small floating-looking free sheets is apparent in these views.'),
 'The macro bench is retained; the blue seat remains above a separate gray seat-layer trace.'),
('f5d370e9fe5b7452dbbc9440457e303e__SEAT_SURFACE_1__n1',
 state(C,U,Y,Y,U,'A large gray back and box-like lower bench surround the blue seat-height rectangle. The input is at a plausible seating level, but the side and front sections show a substantial gap to the lower gray horizontal layer. Some short gray feet and panel edges are thin; the exact support relation of the blue plane is not visible.'),
 state(P,U,Y,Y,Y,'The large back and lower box-like panels remain, but the short free feet become narrow strands. The blue seat stays at the same height above the lower gray layer, without a newly visible connecting surface in the fixed sections.'),
 'The layered seat relation persists and the small free feet become thinner after POST.'),
('fd355d06238853c7f0d9805ad6c2059c__COMBO_SEAT_SUPPORT__n0',
 state(C,PL,Y,Y,Y,'The blue seat and lower two-leg frame form recognizable parts of a gray backed bench with side arms. A gray horizontal layer remains below the blue seat in the front and side sections. Pointed gray sheet-like extensions hang near the blue leg tops. The input assembly has a plausible overall role, while local layers and flaps remain.'),
 state(C,PL,Y,Y,Y,'The same bench arrangement and blue input roles remain. Gray panels are slightly flatter, but the lower gray seat layer and pointed gray extensions beside the blue legs persist. The images do not certify the panel/frame junctions.'),
 'The plausible assembly use is retained; small smoothing does not remove the lower layer or pointed gray extensions.'),
('fd355d06238853c7f0d9805ad6c2059c__COMBO_SEAT_SUPPORT__n1',
 state(C,PL,Y,Y,Y,'The blue seating rectangle is framed by gray side arms and a gray back, with the blue lower frame supporting the arrangement visually. Both sections retain a gray horizontal element below the blue seat. Triangular gray flaps extend down alongside the upper blue legs, so apparent macro completeness does not imply a clean surface.'),
 state(C,PL,Y,Y,Y,'The gray back and arms become slightly flatter while the blue seat/frame arrangement stays the same. The below-seat gray layer and pointed extensions remain. No clear visual gain in local joining can be asserted.'),
 'Macro arrangement is stable; POST mostly smooths existing panels and leaves the visible layers/flaps.'),
('fd355d06238853c7f0d9805ad6c2059c__LEG_SUPPORT_2__n0',
 state(C,NO,'NO',Y,U,'A complete gray backless bench with its own legs sits above and offset from the blue lower frame. Both surface views show a clear spatial separation between the two structures. The C-centered sections contain the blue frame but miss the gray body, consistent with a positional offset rather than evidence that the gray body is absent.'),
 state(P,NO,'NO',Y,Y,'The gray bench is still separate from the blue frame, and its own free legs become very thin strands. The input frame is not visibly adopted as the bench support. The lack of gray section traces is not treated as a topology test.'),
 'The separate gray-bench/blue-frame arrangement persists, while POST thins the gray legs.'),
('fd355d06238853c7f0d9805ad6c2059c__LEG_SUPPORT_2__n1',
 state(C,PL,Y,U,Y,'The blue lower frame is under a gray backed bench, unlike the spatially separated n0 result. Its legs are in a plausible support position. The front section still places the gray seat/rail above the blue top beam with a gap, and pointed gray panels extend near the blue leg tops. These views suggest a coherent overall role but do not show exact joining.'),
 state(C,PL,Y,U,Y,'The backed bench and blue support frame retain their relative positions. Gray panels are flatter, while the finite sections still show an offset from the blue top beam and pointed extensions remain near the legs. Shared topology remains unknown.'),
 'The plausible support arrangement is stable across POST; small gaps and pointed gray extensions remain.'),
]

manifest_path=ROOT/'gallery_manifests'/f'{LABEL}_full.json'
manifest=read(manifest_path);completion=read(GALLERY/'completion.json')
pages=sorted(manifest['pages'],key=lambda p:p['request_id'])
assert len(pages)==48 and len(completion['images'])==48
selected=pages[32:48]
assert [p['request_id'] for p in selected]==[n[0] for n in notes]
schema=read(PRIOR/'review_schema.json')
schema.update(checkpoint=LABEL,scope='Sorted request_id indices32..47 of fixed48 full requests; 16 pages,32 RAW/POST observations; part03',
    actual_view_count=16,prior_schema_sha256=sha(PRIOR/'review_schema.json'))
schema['limitations']=[s for s in schema['limitations'] if not s.startswith('No pretraining image baseline')]
schema['limitations'].append('This part covers16 DEV requests, two fixed noises per condition. It is not the full-panel aggregate or a model-improvement claim.')
now=datetime.datetime.now(datetime.timezone.utc).isoformat();rows=[];pairs=[];image_records=[]
for page,record in zip(selected,notes):
    rid,raw,post,change=record;path=GALLERY/(page['page_id']+'.png');ir=read(path.with_suffix('.json'))
    assert sha(path)==ir['image_sha256'] and ir['definition_sha256']==sha(GALLERY/'definition.json')
    image_records.append(dict(request_id=rid,path=str(path),sha256=sha(path),receipt_sha256=sha(path.with_suffix('.json'))))
    for stage,observation in (('RAW',raw),('POST',post)):
        rows.append(dict(review_version=schema['review_version'],reviewer='AI_PROXY',actual_image_viewed=True,
            human_reviewed=False,nonblind=True,reviewed_at_utc=now,label=LABEL,global_step=12000,
            request_id=rid,condition_id=page['condition_id'],parent_id=page['parent_id'],split=page['split'],stage=stage,
            K=page['K'],N=page['N'],part_description=page['kind'],image_path=str(path),image_sha256=sha(path),
            image_receipt_sha256=sha(path.with_suffix('.json')),gallery_definition_sha256=sha(GALLERY/'definition.json'),
            views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
            body_recognizable_as_bench=True,**observation,exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
            topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY',metric_values_used_for_labels=False,
            source_full_mesh_shown=False,scope='Fixed full-panel image observation; no checkpoint/data/seed selection or training-policy changes'))
    pairs.append(dict(request_id=rid,split=page['split'],part_description=page['kind'],RAW_to_POST=change,
        image_sha256=sha(path),reviewer='AI_PROXY',actual_image_viewed=True))

def write(name,text):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(text)
write('review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
write('raw_post_comparison.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
write('review_schema.json',json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
counts={stage:{field:dict(collections.Counter(r[field] for r in rows if r['stage']==stage))
    for field in ('body_completeness','C_visual_coherent_use','thin_or_dangling_sheet_appearance')} for stage in ('RAW','POST')}
write('observed_label_counts.json',json.dumps(dict(source='Actual image review labels only',counts=counts),indent=2)+'\n')
md=['# NATIVE_S12000 full image review: part03 (AI_PROXY)','',
    'Actually inspected16 existing image pages (sorted request_id indices32..47), each with RAW/POST, two surface views and fixed front/side sections. All16 requests are DEV. Saved32 stage observations and16 paired comparisons. No metric values were read to assign labels, and no generation, rescoring, rendering or model calls were run.','',
    'The images separate recognizable bench shape from visible use of C. Both 9fd5 seat-slat requests retain a blue slat well above the generated gray seat. fd355 lower-frame n0 has a separate gray bench above/offset from the blue frame; n1 has a much more plausible support arrangement. These are image observations, not topological connection judgments.','',
    'POST frequently flattens the gray seat or panels while making free supports very thin. This is particularly visible in the9fd5 requests, the curved supports of f495 leg n0, the seat strips of f5d3 back n0 and the feet of f5d3 seat n1. Plausible blue back/frame placement can coexist with this quality cost. Other images retain parallel seat/back layers or pointed gray extensions.','',
    'The source part descriptions are construction diagnostics, not semantic inputs or unique required completions. Exact shared connectivity is NOT_ASSESSED_FROM_IMAGES and topological disconnectedness is UNKNOWN_IMAGE_ONLY throughout. Section traces can be missed by the plane; a short or missing line alone does not prove a disconnected triangle or hole.','',
    '| Request | RAW to POST visual observation |','|---|---|']
for p in pairs:md.append('|'+p['request_id']+'|'+p['RAW_to_POST']+'|')
md.extend(['','See review.jsonl for stage-specific observations and observed_label_counts.json for counts of these visual labels only. This shard is not the complete48-request result and does not select a checkpoint or change training.',''])
write('REVIEW.md','\n'.join(md))
write('receipt.json',json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',reviewer='AI_PROXY',reviewed_at_utc=now,
    part='part_03',sorted_request_indices=[32,47],images_viewed=16,image_view_calls_per_page=1,stage_rows=32,paired_rows=16,
    model_calls=0,new_samples=0,rescoring=0,data_audits=0,new_renders=0,metric_values_used=False,
    protected_gallery_definition_sha256=sha(GALLERY/'definition.json'),protected_manifest_sha256=sha(manifest_path),
    protected_completion_sha256=sha(GALLERY/'completion.json'),image_records=image_records,
    files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()}),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',images=16,stage_rows=32,paired_rows=16,counts=counts)))
