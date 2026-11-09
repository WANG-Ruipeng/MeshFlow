"""Archive the actual last16 joint-page reviews; images were viewed individually.

This writer validates existing PNG identities only. It does not compute geometry
metrics, render, invoke models, choose a checkpoint, or alter training.
"""
import collections
import datetime
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
NAME='NATIVE_S15000__SEAM_S15000_full'
GALLERY=ROOT/'gallery'/NAME
PRIOR=ROOT/'visual_review/NATIVE_S10000_process/review_schema.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def st(body,use,role,layer,thin,note):return dict(body_completeness=body,C_visual_coherent_use=use,
    construction_intended_role_visible=role,visible_offset_or_redundant_layer=layer,
    thin_or_dangling_sheet_appearance=thin,notes=note)
C='COMPLETE_MACRO';P='PARTIAL';PL='PLAUSIBLE';U='UNCERTAIN';Y='YES';NO='NO_VISIBLE_COHERENT_USE';N='NO_OBVIOUS_IN_VIEWS'
notes=[]
def add(rid,native,seam):notes.append((rid,{'NATIVE':native,'SEAM':seam}))

add('9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0',
 (st(C,PL,Y,U,U,'The blue U-shaped end frame sits at one end of a recognizable gray bench. The gray seat extends to substantial opposite-end legs. The sections place the gray seat inside and slightly below the blue top beam, with several nearby gray traces. This is plausible support placement, without certification of a shared junction.'),
  st(P,PL,Y,U,Y,'The gray seat becomes flatter, while the opposite free legs become almost line-like in both surface views. The robust blue frame and its plausible support location remain. The simpler seat traces do not establish new connectivity.'),
  'Seat smoothing accompanies pronounced thinning of the opposite free legs; the blue end-frame role remains plausible.'),
 (st(C,PL,Y,U,U,'The gray seat and opposite legs form a recognizable bench with the blue end frame. Small gray edge irregularities remain near the frame interior in the side section. The seat is below the frame top beam, consistent with a support role, but precise joining is not visible.'),
  st(P,PL,Y,U,Y,'The free opposite legs become very thin strands, while the seat is flatter and the blue frame unchanged. A short gray turn remains near the seat edge inside the frame. No exact shared connection can be inferred from that trace.'),
  'The plausible end-frame placement is retained; POST substantially thins the gray opposite legs.'))

add('9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n1',
 (st(C,PL,Y,U,U,'The blue end frame and a gray seat with two opposite free legs form a backless bench. The side section shows the seat below the blue beam, with irregular nearby gray edge traces. The input frame has a plausible support position.'),
  st(P,PL,Y,U,Y,'The gray seat is smoother but its opposite legs are reduced to narrow, partly intermittent-looking strands. The blue frame remains robust and in the same position. Missing short traces are not treated as proof of deleted topology.'),
  'POST simplifies the seat and weakens the visible free legs, leaving input placement unchanged.'),
 (st(C,PL,Y,U,U,'The blue end frame is in a plausible support location for a broad gray seat. The opposite legs are visibly present, and the side section contains a small curl near the gray seat edge inside the frame. The view does not resolve exact attachment.'),
  st(P,PL,Y,U,Y,'The gray opposite legs become line-like, while the seat flattens. A small turned gray section segment remains inside the blue frame. The robust input frame stays in place, but free support thickness is visibly reduced.'),
  'The same macro arrangement persists, with substantially thinner free supports after POST.'))

for noise in (0,1):
 add(f'9fd5b67a2b20b5d1b19874039c8f3e05__SEAT_SURFACE_1__n{noise}',
  (st(C,NO,'NO',Y,U,'The blue slat is visibly above a separate gray bench seat. Both sections show a substantial height gap, with no visible bridge from the gray body to the blue strip. The gray body has its own seat and legs. No coherent incorporation of the slat is apparent in these fixed views.'),
   st(P,NO,'NO',Y,Y,'The elevated blue strip and its gap above the gray seat persist. POST flattens the gray seat but makes its legs very thin strands. No new bridging surface is visible.'),
   'The input remains above the generated seat; POST mainly smooths the gray body and thins its legs.'),
  (st(C,NO,'NO',Y,Y,'The gray bench remains well below the blue slat. Gray flanges descend from the slat, forming a small cap-like structure above the main body. The side section shows those flanges ending above the lower gray seat. Their presence does not show that the elevated input assembly joins the body.'),
   st(P,NO,'NO',Y,Y,'The short gray flanges remain around the elevated blue slat, with a visible gap to the main gray seating layer. The main bench legs become very thin. The cap-like assembly still does not visibly integrate with the body.'),
   'The input-adjacent gray flanges persist but do not visibly span the main gap; POST thins the main free legs.'))

add('f495bd67af8d9f0cadd1cf65e8dc4de__COMBO_PART_ASSEMBLY__n0',
 (st(P,U,'PARTIAL',Y,Y,'The blue upper surface and short blue legs coexist with a lower gray seat or shelf and a gray rear strip. The sections show a clear upper-blue/lower-gray height separation. Several small upright gray pieces appear around the lower layer. An upper-component role is possible, but coherent support of the assembly is unclear.'),
  st(P,U,'PARTIAL',Y,Y,'The gray lower layer and small uprights become thinner and more open-looking. The blue upper component remains above the gray horizontal layer. The surface view is still rack-like, without a clearly resolved assembly junction.'),
  'POST simplifies the lower gray layer, while the separated upper/lower relation and thin upright pieces remain.'),
 (st(P,U,'PARTIAL',Y,Y,'The blue upper assembly lies above a broad gray lower panel, with gray end panels and small hanging projections. Short gray surfaces descend from the blue ends, but the side section retains a gap to the lower layer. The arrangement is recognizable as bench-like yet locally layered and ambiguous.'),
  st(P,U,'PARTIAL',Y,Y,'The gray lower panel is flatter, and thin vertical remnants remain underneath. Gray extensions at the blue ends stay above or beside the lower layer without an unambiguous bridge in the fixed section. The upper/lower separation persists.'),
  'POST flattens the lower panel and leaves thin remnants; the blue upper assembly is still visually separate in height.'))

add('f495bd67af8d9f0cadd1cf65e8dc4de__COMBO_PART_ASSEMBLY__n1',
 (st(C,PL,Y,Y,Y,'The blue upper surface is at the seat level of a bench with broad gray end supports. Several gray longitudinal strips overlap or cover parts of it, and additional pointed gray supports hang underneath. The side section contains multiple nearby seat traces. The input assembly has a plausible role, with visible layers and extra thin parts.'),
  st(C,PL,Y,Y,Y,'The blue seating surface becomes more exposed as gray strips narrow. Large gray end supports remain, together with thin hanging gray strands below the seat. Plausible input use coexists with these surface-quality concerns.'),
  'The blue surface is more exposed after POST, but nearby gray layers and thin hanging supports remain.'),
 (st(C,PL,Y,Y,Y,'The blue upper assembly lies at a plausible seating level inside broad gray end supports. Multiple gray strips cover it and pointed gray pieces hang below. The sections show several close seat-layer traces. The overall role is plausible without certified attachment.'),
  st(C,PL,Y,Y,Y,'The blue seat is more exposed, with very thin gray hanging strands below. A triangular opening is conspicuous in one gray end panel in the surface view. The large supports still define the macro bench, but local panel robustness is uncertain.'),
  'POST exposes the blue surface and reduces gray strips to thin remnants; an end-panel opening remains visible.'))

add('f495bd67af8d9f0cadd1cf65e8dc4de__LEG_SUPPORT_1__n0',
 (st(C,U,'PARTIAL',Y,U,'A long gray seat rests on curved or loop-like gray end supports. The blue leg is partly hidden and extends below the nearby curved gray elements. The section traces cross its projection, leaving the relationship between this leg and the gray support loops ambiguous.'),
  st(P,U,'PARTIAL',Y,Y,'The broad seat survives, but the curved end supports become narrow strips with large visible openings. The blue leg remains under the slab and partly obscured, extending lower than the gray support structure. Projected crossings are not certified intersections.'),
  'POST substantially thins the curved free supports without resolving the input-leg relationship.'),
 (st(C,U,'PARTIAL',Y,U,'The gray body is a broad seat above curved end supports. The blue leg is mostly hidden and extends lower than the curved elements. The side section shows a curved gray trace crossing the blue leg projection; it does not establish either attachment or collision.'),
  st(P,U,'PARTIAL',Y,Y,'The curved gray supports become thin open strips, and the seat is flatter. The blue leg stays in the same partly obscured position beneath the slab. Its coherent relation to the existing loops remains unclear.'),
  'POST turns the curved supports into thinner strips; the partly hidden blue leg remains ambiguous.'))

add('f495bd67af8d9f0cadd1cf65e8dc4de__LEG_SUPPORT_1__n1',
 (st(C,U,'PARTIAL',Y,Y,'The gray bench is rack-like, with a lower slatted seating layer and a raised rear strip. The blue leg projects both above and below the gray seating level, clearly visible in the front section. Its support role is possible, but the upper projection and surrounding layers make use uncertain.'),
  st(C,U,'PARTIAL',Y,Y,'The rack-like shape remains, with thinner gray rails and end details. The blue leg still protrudes above the seating layer. POST does not clarify that upper projection or its junction with the gray seat.'),
  'The height mismatch persists while gray rail details become thinner.'),
 (st(C,U,'PARTIAL',Y,Y,'The long gray rack-like bench has an open slatted lower layer and a rear rail. The blue leg extends above the lower seating plane as well as below it. Gray surfaces partly hide the upper blue portion in one view; the sections still show the height mismatch.'),
  st(C,U,'PARTIAL',Y,Y,'The gray rack and rear rail are flatter, but the blue leg keeps the same upper protrusion through the seating-level region. Thin gray end pieces remain. No change in coherent input use is apparent.'),
  'POST retains the rack-like arrangement and blue upper protrusion, with somewhat thinner gray details.'))

add('f5d370e9fe5b7452dbbc9440457e303e__BACK_SURFACE_1__n0',
 (st(C,PL,Y,U,U,'The blue board forms the broad visible back of a bench with gray seat strips and substantial end supports. No second broad gray back panel is apparent. The side section places the gray seat near the lower part of the blue board, a plausible back-board role.'),
  st(P,PL,Y,U,Y,'The blue back and gray end supports remain, but the gray seating strips become much thinner and more open-looking. The side section has fewer seat traces. Plausible input placement persists while the seat surface looks weaker.'),
  'The blue back role remains plausible, but POST substantially thins the gray seating strips.'),
 (st(C,U,Y,Y,U,'The blue board is in the back region, but a large gray triangular or angled panel covers part of it from the front view. Gray traces at another depth appear around the blue board in the sections. The input has a back role, while this additional back layer makes clean integration uncertain.'),
  st(P,U,Y,Y,Y,'The gray panel remains in front of part of the blue back, and the gray seating strips become very thin. The side section retains nearby blue and gray back traces. POST does not visibly eliminate the extra back surface.'),
  'The gray covering back panel persists, while POST weakens the visible seat strips.'))

add('f5d370e9fe5b7452dbbc9440457e303e__BACK_SURFACE_1__n1',
 (st(C,PL,Y,U,U,'The broad blue board serves as the back of a gray bench with a solid-looking seat and curved end supports. No second broad back panel is visible. The side section places the seat near the blue lower region, supporting a plausible back-board interpretation.'),
  st(P,PL,Y,U,Y,'The broad blue back and gray seat remain, but the curved gray supports turn into thin open strips. The input back role stays plausible; the loss of robust-looking free supports is a separate quality cost.'),
  'POST retains the blue-back arrangement but substantially thins the curved free supports.'),
 (st(C,U,Y,Y,U,'The blue back board is partly covered by a broad gray triangular panel. The gray seat and curved end supports still form a recognizable bench. Both the front and side sections include extra gray back traces, so the role is visible but the layered back arrangement is uncertain.'),
  st(P,U,Y,Y,Y,'The triangular gray back covering remains. The curved gray end supports become open narrow strips, while the main seat survives. The images do not show that the blue/gray back layers have been resolved.'),
  'The extra back layer persists and the curved supports become visibly thinner.'))

add('f5d370e9fe5b7452dbbc9440457e303e__SEAT_SURFACE_1__n0',
 (st(C,U,Y,Y,N,'A recognizable bench with broad gray back rails and supports surrounds the blue seat rectangle. Both sections show the blue plane above a separate segmented gray seat slab, despite its plausible seat-level location. No obvious field of dangling free sheets is visible.'),
  st(C,U,Y,Y,N,'The broad back and supports remain. The lower gray seat slab becomes smoother but stays visibly separated from the blue plane in the sections. The overall shape is robust, while exact integration of the blue seating surface is unresolved.'),
  'The robust macro bench is retained; smoothing leaves the separate gray layer below the blue seat.'),
 (st(C,U,Y,Y,N,'The blue rectangle occupies the seating region of a recognizable bench with broad back rails and supports. Short gray sides descend from the blue plane, but a distinct gray slab remains below it. Those apparent edge approaches do not certify a shared junction.'),
  st(C,U,Y,Y,N,'The same broad bench remains with a flatter lower gray slab. Short gray sides near the blue edges persist, while the layer separation remains visible. No widespread small dangling-sheet appearance is apparent in these views.'),
  'POST smooths the lower slab but retains the layered seat and short gray edge extensions.'))

add('f5d370e9fe5b7452dbbc9440457e303e__SEAT_SURFACE_1__n1',
 (st(C,U,Y,Y,U,'The blue seat rectangle lies within a large gray backed, box-like bench. A lower gray horizontal surface is visibly beneath it, especially in the side section. Small feet support the gray body. The input has a seat-region role but its relation to the lower slab is uncertain.'),
  st(P,U,Y,Y,Y,'The large back and box panels remain, but the small gray feet become thin strands. The blue seat remains above the lower gray surface. The height separation is not resolved by POST.'),
  'The layered seating arrangement persists, with thinner small feet after POST.'),
 (st(C,U,Y,Y,Y,'The blue seat is surrounded by a box-like gray bench, but the rear gray panel has a pronounced folded or diagonal sheet arrangement and a large opening in one surface view. Gray flanges descend near the blue seat edges, still above the lower body. Macro shape is present but local sheets and layers remain.'),
  st(P,U,Y,Y,Y,'The folded rear-panel arrangement and lower gray seat layer remain. The small free feet become very thin. Short gray flanges near the blue seat still do not visibly bridge every gap to the lower box.'),
  'POST retains the folded gray back and layered seat while reducing the visible free-foot thickness.'))

for noise in (0,1):
 add(f'fd355d06238853c7f0d9805ad6c2059c__COMBO_SEAT_SUPPORT__n{noise}',
  (st(C,PL,Y,Y,Y,'The blue seating surface and lower two-leg frame are plausible components of a gray backed bench with side arms. A lower gray seating layer and a narrow light gap near the rear seat are visible. Pointed gray sheets extend down beside the blue leg tops. Macro use is plausible but the surface is not uniformly clean.'),
   st(C,PL,Y,Y,Y,'The blue seat/frame arrangement is retained as gray panels become flatter. The lower gray seat trace, narrow rear gap and pointed gray extensions remain. These images do not certify the panel-to-frame junctions.'),
   'Macro input placement remains plausible; POST mostly flattens panels and leaves the layers and pointed extensions.'),
  (st(C,PL,Y,Y,Y,'The blue seat and lower frame form plausible parts of a bench with gray arms and back. Gray sheets approach the blue seat boundary and cover portions of the blue upper legs, including long pointed triangular extensions. The rear seat gap appears smaller than in the adjacent Native view, but the lower gray layer remains and topology is not certified.'),
   st(C,PL,Y,Y,Y,'The same bench and blue assembly placement remain. The gray back and arms are flatter, while pointed gray surfaces still cover or flank the blue upper legs. The sections retain multiple seat-level traces; apparent proximity is not a legal-connection test.'),
   'POST smooths panels but retains the gray leg-covering sheets and the nearby lower seat layer.'))

add('fd355d06238853c7f0d9805ad6c2059c__LEG_SUPPORT_2__n0',
 (st(C,NO,'NO',Y,U,'The gray backed bench has its own small legs and sits above and offset from the blue lower frame. The rear surface view shows a clear gap above the blue top rail. The C-centered front section contains only the blue frame, while the side section puts the gray seat above and at another depth. No coherent adoption of the frame is visible.'),
  st(C,NO,'NO',Y,U,'The separate gray bench and blue frame remain. Gray surfaces are flatter and small free legs somewhat thinner, but the rear gap and positional offset persist. Missing gray traces in the C-centered front section are not treated as absence of the whole body.'),
  'POST smooths the gray bench without resolving its separation from the blue frame.'),
 (st(C,PL,Y,U,Y,'The blue lower frame lies beneath a gray bench with arms and back, rather than beneath a separate offset bench. Gray surfaces extend down near or over the blue leg tops, and the front section includes gray traces above the blue rail. The macro support role is plausible, although a seat-to-rail gap and pointed gray leg-covering sheets remain.'),
  st(C,PL,Y,U,Y,'The plausible blue-frame/gray-bench arrangement remains after panel smoothing. Gray pointed sheets still cover portions of the blue legs, and the sections retain an offset above the blue rail. This is a visual improvement in overall arrangement relative to the adjacent Native result, not proof of shared topology.'),
  'POST retains the more plausible frame-supported arrangement and its pointed gray extensions; exact joining remains unknown.'))

add('fd355d06238853c7f0d9805ad6c2059c__LEG_SUPPORT_2__n1',
 (st(C,PL,Y,U,Y,'The blue lower frame sits beneath a gray backed bench with arms and is in a plausible support position. The gray seat lies above the blue top rail in the sections, and pointed gray extensions descend near the blue leg tops. The overall arrangement is coherent-looking, with unresolved local gaps and sheets.'),
  st(C,PL,Y,U,Y,'The gray back, seat and arms become slightly flatter. The blue lower frame remains in the same plausible position, while pointed extensions and the seat-to-rail offset persist. The views do not show exact shared connectivity.'),
  'The plausible support arrangement is retained; POST mainly smooths panels and leaves the local offset and pointed sheets.'),
 (st(C,PL,Y,U,Y,'The blue lower frame plausibly supports the gray backed bench. Gray pointed panels extend down over portions of the blue legs, and extra diagonal traces appear near the upper rail in the front section. The macro role is clear, but local continuity and surface overlap cannot be certified visually.'),
  st(C,PL,Y,U,Y,'The blue-frame arrangement remains, with smoother gray panels. Long pointed gray sheets continue alongside or over the blue legs. The seat still lies above the blue rail in the fixed section, so smoothing is not evidence of a fully resolved interface.'),
  'POST preserves the plausible macro role and the gray leg-covering sheets, while simplifying section traces.'))

manifest_path=ROOT/'gallery_manifests'/(NAME+'.json')
manifest=read(manifest_path);completion=read(GALLERY/'completion.json')
pages=sorted(manifest['pages'],key=lambda x:x['request_id'])
assert len(pages)==48 and len(completion['images'])==48
selected=pages[32:48]
assert [x['request_id'] for x in selected]==[x[0] for x in notes]
schema=read(PRIOR)
schema.update(checkpoint=['NATIVE_S15000','SEAM_S15000'],actual_view_count=16,
    scope='Sorted request_id indices32..47, two arms and RAW/POST:16 joint pages,64 stage observations,32 within-arm pairs',
    prior_schema_sha256=sha(PRIOR))
schema['limitations']=[v for v in schema['limitations'] if not v.startswith('No pretraining image baseline')]
schema['limitations'].extend(['All16 requests in this shard are DEV. This is not a full-panel aggregate.',
    'Joint labeled pages are nonblind. Limited macro appearance differences are not proofs of legal topology or whole-model superiority.',
    'Only two fixed noises per condition are shown. These observations do not select a checkpoint or alter stopping policy.'])
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
definition_sha=sha(GALLERY/'definition.json')
rows=[];pairs=[];image_records=[]
for page,(rid,arms) in zip(selected,notes):
    path=GALLERY/(page['page_id']+'.png');receipt=path.with_suffix('.json');ir=read(receipt)
    image_sha=sha(path)
    assert image_sha==ir['image_sha256'] and ir['definition_sha256']==definition_sha
    image_records.append(dict(request_id=rid,path=str(path),sha256=image_sha,receipt_sha256=sha(receipt)))
    for arm,offset in [('NATIVE',0),('SEAM',2)]:
        raw,post,change=arms[arm];label=arm+'_S15000'
        for stage,obs,col in [('RAW',raw,offset),('POST',post,offset+1)]:
            assert page['cells'][col]['label']==label+'_'+stage
            rows.append(dict(review_version=schema['review_version'],reviewer='AI_PROXY',actual_image_viewed=True,
                human_reviewed=False,nonblind=True,reviewed_at_utc=now,label=label,global_step=15000,
                request_id=rid,condition_id=page['condition_id'],parent_id=page['parent_id'],split=page['split'],stage=stage,
                K=page['K'],N=page['N'],part_description=page['kind'],image_path=str(path),image_sha256=image_sha,
                image_receipt_sha256=sha(receipt),gallery_definition_sha256=definition_sha,
                reviewed_column_index_zero_based=col,model_stage_label=page['cells'][col]['label'],
                views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
                body_recognizable_as_bench=True,**obs,exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
                topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY',metric_values_used_for_labels=False,
                source_full_mesh_shown=False,scope='Fixed full-panel image observation; no checkpoint/data/seed selection or training-policy changes'))
        pairs.append(dict(request_id=rid,label=label,split=page['split'],part_description=page['kind'],RAW_to_POST=change,
            image_sha256=image_sha,reviewer='AI_PROXY',actual_image_viewed=True))
assert len(rows)==64 and len(pairs)==32
def write(name,content):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(content)
write('review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
write('raw_post_comparison.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
write('review_schema.json',json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
counts={label:{stage:{field:dict(collections.Counter(r[field] for r in rows if r['label']==label and r['stage']==stage))
    for field in ('body_completeness','C_visual_coherent_use','thin_or_dangling_sheet_appearance')}
    for stage in ('RAW','POST')} for label in ('NATIVE_S15000','SEAM_S15000')}
write('observed_label_counts.json',json.dumps(dict(source='Actual image-review labels only, no geometry metrics',counts=counts),indent=2)+'\n')
md=['# Step15000 full joint image review: part03 (AI_PROXY)','',
    'Actually viewed16 existing joint pages, sorted request_id indices32..47. Each page has both arms, RAW/POST, two surface views and fixed C-centered front/side sections. All16 requests are DEV. Saved64 stage observations and32 within-arm RAW-to-POST comparisons. No metric values were used to assign labels; no models, sampling, rescoring or new renders were run.','',
    'A bounded visual difference appears in fd355 lower-frame n0: Native retains an offset gray bench above the blue frame, whereas SEAM places a backed arm bench more plausibly on that frame. This does not certify a legal interface. Both arms have plausible frame use in n1. Several other requests have similar large-scale arrangements between arms.','',
    'Costs remain visible. The9fd5 seat slat is above the gray main seat in both arms; SEAM adds short gray flanges at C but they do not visibly span the remaining gap. On both f5d3 back requests, SEAM adds a broad gray panel covering part of the blue board, while Native exposes the board more directly. POST often reduces free legs, curved supports or seat strips to thin strands or narrow strips in both arms. These observations do not establish model-wide superiority.','',
    'Source part labels are construction diagnostics, not semantic inputs or a unique-source reconstruction criterion. A plausible visual role can coexist with uncertain joining. Exact shared connectivity is NOT_ASSESSED_FROM_IMAGES and disconnectedness is UNKNOWN_IMAGE_ONLY throughout. Missing or short section traces alone are not topology evidence.','',
    '| Request | Arm | RAW to POST observation |','|---|---|---|']
for p in pairs:md.append('|'+p['request_id']+'|'+p['label']+'|'+p['RAW_to_POST']+'|')
md.extend(['','This shard is not the complete48-request assessment. It selects no checkpoint and makes no stopping-policy change. Counts in observed_label_counts.json summarize the recorded visual labels only.',''])
write('REVIEW.md','\n'.join(md))
write('receipt.json',json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',reviewer='AI_PROXY',reviewed_at_utc=now,
    part='part_03',sorted_request_indices=[32,47],images_viewed=16,image_view_calls_per_page=1,
    stage_rows=64,paired_rows=32,model_calls=0,new_samples=0,rescoring=0,data_audits=0,new_renders=0,
    metric_values_used=False,nonblind=True,protected_gallery_definition_sha256=definition_sha,
    protected_manifest_sha256=sha(manifest_path),protected_completion_sha256=sha(GALLERY/'completion.json'),
    image_records=image_records,files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()}),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',images=16,stage_rows=64,paired_rows=32,counts=counts)))
