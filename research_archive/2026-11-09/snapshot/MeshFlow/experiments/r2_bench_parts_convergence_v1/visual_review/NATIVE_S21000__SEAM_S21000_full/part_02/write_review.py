"""Archive actual image review of final joint pages 17--32; no geometry scoring."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
NAME = 'NATIVE_S21000__SEAM_S21000_full'
GALLERY = ROOT / 'gallery' / NAME
PRIOR = ROOT / 'visual_review/NATIVE_S15000__SEAM_S15000_full/part_03/review_schema.json'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def st(body, use, role, layer, thin, note):
    return dict(body_completeness=body, C_visual_coherent_use=use,
        construction_intended_role_visible=role, visible_offset_or_redundant_layer=layer,
        thin_or_dangling_sheet_appearance=thin, notes=note)
C='COMPLETE_MACRO'; P='PARTIAL'; PL='PLAUSIBLE'; U='UNCERTAIN'; Y='YES'; N='NO_OBVIOUS_IN_VIEWS'
notes=[]
def add(rid, native, seam): notes.append((rid, {'NATIVE':native,'SEAM':seam}))

add('3db1d3904f058b574a9ca2c84bced0dd__BACK_SURFACE_1__n0',
 (st(C,PL,Y,U,U,'The broad blue board is the visible back of a gray bench with parallel seat strips and short slab-like supports. The seat lies near the lower part of the blue board. No additional broad gray back is obvious. The back role is plausible, without certification of attachment.'),
  st(P,PL,Y,U,Y,'The blue back remains exposed, but gray seat strips become very narrow and the seating surface looks more open. Gray support panels remain. The input role is retained while the free seat looks weaker.'),
  'POST retains the plausible blue-back role but markedly thins and opens the gray seating strips.'),
 (st(C,U,Y,Y,U,'The blue board has a back position, while angled gray patches cover parts of it in the two surface views. A gray seat and end supports form a recognizable bench. Those additional patches leave the clean use of the board uncertain.'),
  st(P,U,Y,Y,Y,'The gray patches near the blue back remain, with a changed covering pattern. The gray seat strips become narrow and open-looking. The back role persists, with both layering and a weaker free seat.'),
  'POST thins the free seat but does not remove the visible gray patches covering portions of the blue back.'))

add('3db1d3904f058b574a9ca2c84bced0dd__BACK_SURFACE_1__n1',
 (st(C,U,Y,Y,Y,'A large gray back panel stands in front of and higher than the blue board. The side section clearly separates the blue and gray back traces. Two narrow gray seat bands and angular end supports form a recognizable bench, but the blue board is not the only back surface.'),
  st(P,U,Y,Y,Y,'The separate gray back panel remains. Seat bands narrow further and the end supports become more open-looking. The blue board still occupies a back position, with unresolved layered backs.'),
  'POST weakens the seat and supports while retaining the separated blue and gray back surfaces.'),
 (st(C,U,Y,Y,Y,'The gray bench has a broad back panel offset in front of the blue board. Short gray extensions approach the board near its edges, but the side section retains the depth separation. Narrow seating bands and angular supports are present.'),
  st(P,U,Y,Y,Y,'The blue and gray backs stay offset. Gray seat bands and support details become thinner, while small input-adjacent gray extensions remain. These extensions do not establish a fully joined back.'),
  'The layered-back arrangement remains; POST narrows the free seat and support details.'))

for noise in (0,1):
    rid=f'3db1d3904f058b574a9ca2c84bced0dd__COMBO_PART_ASSEMBLY__n{noise}'
    seam_extra=('The front section has a break in one lower gray seating trace; this is not treated as proof of a mesh hole. ' if noise==1 else '')
    add(rid,
     (st(C,PL,Y,Y,U,'The blue rear board and paired lower leg-and-rail assemblies occupy plausible positions in a bench with a gray seat and front-facing gray back panel. The side section shows closely spaced blue and gray back surfaces. The assembly role is plausible, although layered back surfaces remain.'),
      st(C,PL,Y,Y,Y,'The blue back and leg assemblies remain in a plausible macro arrangement. Gray seat surfaces flatten, while the upper gray cap narrows into a long thin strip and some edges look frayed. The additional front back panel remains. The front section misses C and cannot diagnose its absence or attachment.'),
      'POST retains the macro assembly while thinning the gray upper cap and simplifying the seat; the back remains layered.'),
     (st(C,PL,Y,Y,U,'The blue back and paired lower assemblies form plausible parts of the gray bench. A broad gray surface is in front of the blue back, and the gray seat sits near the blue lower rails. '+seam_extra+'Exact joining cannot be resolved from these views.'),
      st(C,PL,Y,Y,Y,'The overall blue assembly and gray bench remain. The gray upper cap becomes a narrow strip, with thin edge details around the seat. The parallel blue and gray back surfaces persist. '+seam_extra+'The C-missing front section is not evidence of disconnected input.'),
      'POST keeps the plausible assembly and layered back, while narrowing the gray top cap and edge geometry.'))

for noise in (0,1):
    rid=f'459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n{noise}'
    add(rid,
     (st(C,U,Y,Y,Y,'The large blue seat slab and curved end support retain clear seat/support positions. A separate gray folded or ramp-like upper panel sits above the blue slab, and a gray support is present at the other end. Both sections show a height gap to the gray upper surface. The macro bench is present, but the upper layering makes coherent use uncertain.'),
      st(C,U,Y,Y,Y,'The blue slab and curved support remain unchanged. The folded gray upper panel becomes narrower and more open, with thin pointed strips. The opposite gray end support remains. The upper gray layer still sits above the blue seat in the sections.'),
      'POST narrows the folded gray upper layer; the blue seat/support role and vertical separation remain.'),
     (st(C,U,Y,Y,Y,'The blue seat and curved end support are recognizable parts of the bench, but a broad gray folded cover is visibly raised above the seat. A gray support occupies the other end. The sections retain separate blue and gray horizontal levels.'),
      st(C,U,Y,Y,Y,'The gray upper cover becomes thin open strips while the blue assembly and opposite end support remain. The separated upper gray and lower blue seating levels are still visible. No legal junction can be certified from the close projections.'),
      'POST thins the gray folded cover without resolving its height separation from the blue seating slab.'))

native_seat=(
 st(C,U,Y,Y,N,'The blue rectangle lies in the seating region of a recognizable backed bench with substantial gray end supports. The gray seating slab is visibly below it in both sections. The back rail and support structure look robust, but the blue surface and lower gray slab remain separate layers.'),
 st(C,U,Y,Y,N,'The broad back rail, end supports and blue seating rectangle remain. The lower gray slab is flatter but stays below the blue plane. POST smooths the body without visibly resolving the two seat levels.'),
 'The robust macro bench is retained, with a smoother lower gray slab still beneath the blue seat.')
add('459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1__n0', native_seat,
 (st(C,U,Y,Y,N,'The blue rectangle is at a plausible seat location in a backed bench with gray end supports. Short gray flanges descend from the blue edges, while a second broad gray slab remains below. The flanges approach the lower level but do not visibly eliminate the entire gap.'),
  st(C,U,Y,Y,N,'The broad bench remains, with flatter gray surfaces. Short gray flanges at the blue seat edges persist, together with the lower gray seat layer. No exact connection is inferred from those approaches.'),
  'POST smooths the lower slab and keeps the short input-adjacent flanges; the layered seating relation persists.'))
add('459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1__n1', native_seat,
 (st(C,U,'PARTIAL',Y,Y,'A low gray bench has a broad folded upper surface raised well above the blue rectangle, largely hiding the input from the surface views. The side section includes a diagonal gray sheet rising from the blue level and short gray projections below it. The macro body is present, but the input is beneath rather than clearly forming the upper seating surface.'),
  st(C,U,'PARTIAL',Y,Y,'The broad raised gray cover and diagonal internal-looking sheet remain. Thin gray projections extend beneath the blue plane. The gray cover is flatter after POST but still mostly obscures the blue rectangle and preserves the vertical layering.'),
  'POST retains the raised gray cover and diagonal sheet, with the blue plane still below the main visible seating surface.'))

native_leg=(
 st(C,U,'PARTIAL',Y,N,'A simple gray backless bench has a thick-looking seat and short block-like legs. The blue leg surface sits at the outer end, protruding slightly above the gray seat and much lower than the short gray supports. Its leg role is only partly consistent with the generated support proportions.'),
 st(C,U,'PARTIAL',Y,N,'The gray seat becomes smoother with fine longitudinal strips, while the short gray supports remain. The blue leg keeps its upper protrusion and greater downward extent. The sparse fixed section does not establish missing body geometry.'),
 'POST smooths the gray seat but leaves the blue-leg height mismatch and the different short gray supports.')
add('5dda4ef4ab9add9411ee8243f54292d6__LEG_SUPPORT_1__n0', native_leg,
 (st(C,U,'PARTIAL',U,U,'The gray result is a tall backed or box-like bench with a seat and long support legs. The blue leg surface lies along one end but starts below the upper gray rail and stops above the gray leg bottoms. Its support relation is possible, though its vertical span and exact attachment remain uncertain.'),
  st(P,U,'PARTIAL',U,Y,'The gray backed body remains, but several long gray legs become very thin strands. The blue leg keeps its position beside the end panel. POST weakens the visible free supports without resolving the span or junction of the input leg.'),
  'POST markedly thins the long free legs; the blue end-leg placement remains uncertain.'))
add('5dda4ef4ab9add9411ee8243f54292d6__LEG_SUPPORT_1__n1', native_leg,
 (st(C,U,'PARTIAL',Y,N,'A gray backless bench has a long seat and short block-like supports. The blue leg surface is at the outer end, extending above the seat edge and lower than the gray supports. Its support role is possible but not cleanly aligned with the rest of the support geometry.'),
  st(C,U,'PARTIAL',Y,N,'The long gray seat becomes flatter, and the short gray supports remain. The input leg still projects above the seat and below the gray supports. A new or longer section trace alone is not considered evidence of attachment.'),
  'POST flattens the gray seating surface and retains the input-leg height mismatch.'))

add('5dda4ef4ab9add9411ee8243f54292d6__SEAT_SURFACE_1__n0',
 (st(C,U,Y,Y,U,'The blue longitudinal slat lies above the gray seating strips of a low bench with braced legs. Both sections show a gap between the blue slat and the gray seating level. Its seat-like location is visible, but integration with the lower strips is uncertain.'),
  st(C,U,Y,Y,Y,'The blue slat stays above the gray seat. The seat strips and leg details become narrower and frayed-looking, though the bench structure is still recognizable. The section gap is not resolved.'),
  'POST thins the gray strips and support details while preserving the elevated blue slat.'),
 (st(C,U,Y,Y,U,'The blue longitudinal slat sits above the gray seating strips. Short sloped gray flanges descend from its edges, but the side section still shows separation from the main gray seat. A braced bench body is present below.'),
  st(C,U,Y,Y,Y,'The short flanges at the blue slat remain as the gray seat and leg details narrow. The gray main seating level remains lower, and the fixed views do not show a complete bridge around the slat.'),
  'POST retains input-adjacent flanges and the main height gap, while making the free seat and leg details thinner.'))

add('5dda4ef4ab9add9411ee8243f54292d6__SEAT_SURFACE_1__n1',
 (st(C,U,'NO',Y,U,'A gray bench with tall arms and back rails surrounds the blue slat at a rail-like height. The main gray seat is far below it in the sections. An alternative brace role is possible, but the input is not visibly a seat slat in this completion and its exact incorporation remains uncertain.'),
  st(P,U,'NO',Y,Y,'The gray legs, arm rails and back rails become thin. The blue slat remains well above the main gray seat, at a possible brace level. The visual role does not change, and topology is not inferred.'),
  'POST thins the gray frame while preserving the rail-height blue slat above the main seating plane.'),
 (st(C,U,'NO',Y,Y,'The gray bench has tall arms and back rails, with the blue slat high above the seat. Gray sheets descend diagonally from the slat toward the body but end above the main seating plane in the side section. A rail-like role is possible, not the source seat-slat role.'),
  st(P,U,'NO',Y,Y,'The gray frame becomes thinner, while diagonal sheets still descend from the blue slat. The gap to the lower gray seat remains visible. These sheets are not treated as evidence of a legal shared interface.'),
  'POST thins the frame but retains the elevated slat and diagonal gray extensions above the seat.'))

native_wall=(
 st(C,U,Y,Y,U,'A complete-looking gray slatted bench stands in front of the broad blue rear wall. Gray rear supports visibly cover portions of the blue wall, and the side section separates the wall from the slanted gray back slats. The wall has a rear role but is not the only back layer.'),
 st(C,U,Y,Y,Y,'The gray slatted bench and blue rear wall remain. Thin gray rear-support strips and frayed-looking leg details are visible after POST. The layered blue wall and gray back slats are not resolved.'),
 'POST simplifies and thins gray details while retaining the blue rear wall and additional gray back layer.')
add('9d56087c9f98bf90717cca22daef45e7__LEG_SUPPORT_1__n0', native_wall,
 (st(P,PL,Y,Y,Y,'The blue rear wall is exposed and appears to organize a compact gray seat/back assembly in front of it. Gray triangular side panels wrap near the wall edges. The gray seat and downward supports are narrow, leaving a weaker macro bench than the adjacent Native result. The wall role is plausible without certified joining.'),
  st(P,PL,Y,Y,Y,'The blue rear wall and compact front gray assembly persist. The lower gray supports become very thin pointed strands and the seat remains narrow. The plausible wall role coexists with weak free support structure.'),
  'POST retains the wall-centered assembly but further thins its already weak gray supports.'))
add('9d56087c9f98bf90717cca22daef45e7__LEG_SUPPORT_1__n1', native_wall,
 (st(C,U,Y,Y,Y,'The gray slatted bench is in front of the blue rear wall, with substantial seat and support structure. Gray triangular patches cover portions of the wall near lower corners, and rear support strips cross its projection. The wall role is visible but the additional gray back layer leaves coherent integration uncertain.'),
  st(C,U,Y,Y,Y,'The slatted bench remains, with thin pointed gray strips across parts of the blue wall and narrower support details. The side section still separates the blue wall and gray back slats. The larger body is retained.'),
  'POST keeps the full macro bench and layered back, while narrowing gray strips near the blue wall.'))

add('9d56087c9f98bf90717cca22daef45e7__SEAT_SURFACE_1__n0',
 (st(C,PL,Y,U,U,'The blue narrow cap is near the top of a gray slatted bench back. It projects forward from the upper back region; nearby gray top slats extend slightly higher. This is a plausible back-cap role, although the sections do not certify the local join.'),
  st(C,PL,Y,U,U,'The gray slatted bench remains robust at macro scale. The blue cap is more exposed as nearby gray strips simplify. Its upper-back role persists without a visible large-scale mismatch.'),
  'POST slightly exposes the blue cap and simplifies nearby gray strips; the plausible back-cap role remains.'),
 (st(C,U,Y,Y,U,'The blue cap lies near the upper back of a gray slatted bench, but an angled gray panel covers much of it in the front view. The side section shows a gray sloped trace approaching or crossing the cap projection. The role is visible while local layering is uncertain.'),
  st(C,U,Y,Y,Y,'The gray panel still covers part of the blue cap. Nearby gray back strips and support details look thinner, and an angled trace remains near the input in the side section. POST does not visibly remove the covering layer.'),
  'POST thins some gray details but keeps the additional panel covering the blue back cap.'))

add('9d56087c9f98bf90717cca22daef45e7__SEAT_SURFACE_1__n1',
 (st(C,U,'NO',Y,N,'The result has a robust two-tier shelf-like bench form with four posts. The blue cap is mostly hidden underneath the upper gray slab rather than forming an exposed back cap. The sections show the blue plane below that upper surface. The alternative overall style is not itself a failure, but input incorporation is uncertain.'),
  st(C,U,'NO',Y,N,'The two-tier body remains robust with flatter gray surfaces. The blue plane stays largely concealed below the upper slab. No widespread field of thin dangling sheets is obvious, but the blue-to-gray layered relation persists.'),
  'POST preserves the robust two-tier body and the mostly hidden blue plane below the upper slab.'),
 (st(C,U,'NO',Y,N,'A robust two-tier gray body surrounds the mostly hidden blue cap. The sections place the blue plane below the upper slab and show a short gray edge or diagonal near it. No exposed back-cap role is apparent; the alternative tiered style is not scored against a unique source mesh.'),
  st(C,U,'NO',Y,N,'The gray two-tier structure remains, with flatter panels and substantial posts. The blue plane is still underneath the upper slab, with small nearby gray traces. These traces do not establish shared topology.'),
  'POST retains the two-tier structure and concealed blue plane, with small changes to nearby gray edge traces.'))

manifest_path=ROOT/'gallery_manifests'/(NAME+'.json')
manifest=read(manifest_path); completion=read(GALLERY/'completion.json')
pages=sorted(manifest['pages'],key=lambda p:p['request_id'])
assert len(pages)==48 and len(completion['images'])==48
selected=pages[16:32]
assert [p['request_id'] for p in selected]==[n[0] for n in notes]
schema=read(PRIOR)
schema.update(checkpoint=['NATIVE_S21000','SEAM_S21000'],actual_view_count=16,
    scope='Final joint panel, sorted request_id indices16..31:16 pages,64 stage observations,32 within-arm pairs',
    prior_schema_sha256=sha(PRIOR))
schema['limitations']=[v for v in schema['limitations'] if not v.startswith('All16 requests in this shard')]
schema['limitations'].extend(['This shard has4 TRAIN and12 DEV requests, not the complete48-request panel.',
    'No source-role matching requirement is imposed: role labels describe construction intent only.'])
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
definition_sha=sha(GALLERY/'definition.json')
rows=[];pairs=[];image_records=[]
for page,(rid,arms) in zip(selected,notes):
    path=GALLERY/(page['page_id']+'.png');receipt=path.with_suffix('.json');ir=read(receipt)
    image_sha=sha(path)
    assert image_sha==ir['image_sha256'] and ir['definition_sha256']==definition_sha
    image_records.append(dict(request_id=rid,path=str(path),sha256=image_sha,receipt_sha256=sha(receipt)))
    for arm,offset in [('NATIVE',0),('SEAM',2)]:
        raw,post,change=arms[arm];label=arm+'_S21000'
        for stage,obs,col in [('RAW',raw,offset),('POST',post,offset+1)]:
            assert page['cells'][col]['label']==label+'_'+stage
            rows.append(dict(review_version=schema['review_version'],reviewer='AI_PROXY',actual_image_viewed=True,
                human_reviewed=False,nonblind=True,reviewed_at_utc=now,label=label,global_step=21000,
                request_id=rid,condition_id=page['condition_id'],parent_id=page['parent_id'],split=page['split'],stage=stage,
                K=page['K'],N=page['N'],part_description=page['kind'],image_path=str(path),image_sha256=image_sha,
                image_receipt_sha256=sha(receipt),gallery_definition_sha256=definition_sha,
                reviewed_column_index_zero_based=col,model_stage_label=page['cells'][col]['label'],
                views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
                body_recognizable_as_bench=True,**obs,exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
                topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY',metric_values_used_for_labels=False,
                source_full_mesh_shown=False,scope='Fixed final-panel observation; no checkpoint/data/seed selection or training-policy changes'))
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
    for stage in ('RAW','POST')} for label in ('NATIVE_S21000','SEAM_S21000')}
write('observed_label_counts.json',json.dumps(dict(source='Actual image-review labels only, no geometry metrics',counts=counts),indent=2)+'\n')
md=['# Final step21000 joint image review: part02 (AI_PROXY)','',
    'Actually viewed all16 existing joint pages at sorted request_id indices16..31:4 TRAIN and12 DEV requests. Each page contains Native/SEAM RAW/POST, two surface views and fixed C-centered front/side sections. Saved64 stage observations and32 within-arm RAW-to-POST comparisons. Labels were assigned from the images, not metric values. No model, sampling, scoring or rendering was run.','',
    'Local positives coexist with costs. The blue back-and-leg assemblies of3dbCOMBO have plausible macro roles in both arms. Native3dbBACK n0 directly exposes the blue board as the back. Native9d560 back-cap n0 likewise has a plausible upper-back role. These are visual roles, not certifications of exact attachment.','',
    'SEAM sometimes adds input-adjacent gray flanges, as on459 seat n0 and5dda seat requests, while visible height gaps to the main seat remain. On459 seat n1, SEAM instead makes a broad raised folded cover that mostly hides the blue plane. On9d560 rear-wall n0, SEAM organizes a compact assembly around the blue wall but has narrower seat/support structure than Native. These differences do not establish global superiority.','',
    'POST often thins free seat strips, rails or legs. This is clear in3db back requests,5dda seat n1 and SEAM5dda leg n0. Layered backs and separate blue/gray seating levels often remain. Other outputs, such as the two-tier9d560 n1 bodies, retain substantial panels and posts. A different plausible style is not penalized merely for differing from the source.','',
    'Exact shared connectivity is NOT_ASSESSED_FROM_IMAGES; topological disconnectedness is UNKNOWN_IMAGE_ONLY throughout. Short or absent fixed-section traces do not prove holes or disconnected components. The front sections of3dbCOMBO miss C; this is not evidence that C is absent. Source part labels are construction diagnostics and were not model inputs.','',
    '| Request | Arm | RAW to POST observation |','|---|---|---|']
for p in pairs:md.append('|'+p['request_id']+'|'+p['label']+'|'+p['RAW_to_POST']+'|')
md.extend(['','This is one16-request shard of the48-request final panel. Counts summarize visual labels only. No checkpoint selection or stopping-policy change is made.',''])
write('REVIEW.md','\n'.join(md))
write('receipt.json',json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',reviewer='AI_PROXY',reviewed_at_utc=now,
    part='part_02',sorted_request_indices=[16,31],images_viewed=16,image_view_calls_per_page=1,
    stage_rows=64,paired_rows=32,model_calls=0,new_samples=0,rescoring=0,data_audits=0,new_renders=0,
    metric_values_used=False,nonblind=True,protected_gallery_definition_sha256=definition_sha,
    protected_manifest_sha256=sha(manifest_path),protected_completion_sha256=sha(GALLERY/'completion.json'),
    image_records=image_records,files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()}),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',images=16,stage_rows=64,paired_rows=32,counts=counts)))
