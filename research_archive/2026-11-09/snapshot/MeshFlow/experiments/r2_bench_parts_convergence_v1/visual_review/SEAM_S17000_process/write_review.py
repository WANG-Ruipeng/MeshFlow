"""Archive actual SEAM_S17000 image observations; no geometry/model computation."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
LABEL='SEAM_S17000'
NAME='NATIVE_S17000__SEAM_S17000_process'
GALLERY=ROOT/'gallery'/NAME
PRIOR=ROOT/'visual_review/NATIVE_S10000_process/review_schema.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def st(body,use,role,layer,thin,note):return dict(body_completeness=body,C_visual_coherent_use=use,
    construction_intended_role_visible=role,visible_offset_or_redundant_layer=layer,
    thin_or_dangling_sheet_appearance=thin,notes=note)
C='COMPLETE_MACRO';P='PARTIAL';PL='PLAUSIBLE';U='UNCERTAIN';Y='YES'
notes=[
('2d1e5eeb255e79dcab53bf7fa22f3636__BACK_SURFACE_1__n0',
 st(C,U,Y,Y,U,'The blue board is in the back region of a recognizable bench with seat and four gray supports. Gray back panels cover or stand at another depth from parts of the blue board; the blue surface is mainly visible as a long band. The side section contains several nearby blue and gray back traces. A back role is visible, but the layered arrangement makes integration uncertain.'),
 st(P,U,Y,Y,Y,'The seat and broad back remain, while the gray legs and upper back-edge strip become very thin. The blue board stays in the same position with gray surfaces at nearby depths. POST simplifies section traces but does not visibly resolve the layered back.'),
 'POST retains the back-layer arrangement and markedly thins free legs and back-edge details.'),
('459f90aa2162b1f1d46c340938e2ff1c__LEG_SUPPORT_1__n0',
 st(P,U,'PARTIAL',Y,Y,'The blue input supplies a broad seat slab and one curved end support. The free complement includes an opposite support and several folded gray triangular strips above the blue slab. Both sections show a raised gray layer over the seating surface. The blue assembly remains substantial, but the gray complement is visibly layered and irregular.'),
 st(P,U,'PARTIAL',Y,Y,'Folded gray strips remain over the blue seat, with long narrow openings among them. The opposite gray support becomes thinner. The side section retains an upper gray trace separate from the blue slab, so smoothing has not resolved the extra layer.'),
 'The raised folded gray layer persists; POST mostly narrows parts of the free complement.'),
('459f90aa2162b1f1d46c340938e2ff1c__SEAT_SURFACE_1__n0',
 st(C,U,Y,Y,U,'A recognizable bench with broad side supports and an open gray back frame surrounds the blue seat plane. A gray horizontal slab lies below it in both sections. Short gray sloping sides descend near the blue ends but stop above or beside that lower layer in the fixed side section. The input has a seat role, with exact integration unresolved.'),
 st(C,U,Y,Y,U,'The macro bench and broad supports remain. The back rail and lower gray slab are smoother, while the blue seat stays above the lower slab. Short gray edge extensions persist without a visibly complete bridge across the section gap.'),
 'POST preserves the macro bench and short edge extensions but leaves the lower gray seat layer separated.'),
('9fd5b67a2b20b5d1b19874039c8f3e05__LEG_SUPPORT_1__n0',
 st(C,PL,Y,U,U,'The blue U-shaped frame forms one end of a gray backless bench. The gray seat sits inside and below its top beam and extends to visible opposite-end legs. The sections show close but distinct gray seat traces inside the frame. The input is in a plausible support position.'),
 st(P,PL,Y,U,Y,'The gray seat flattens while the opposite free legs become line-like strands. The robust blue frame stays in its plausible support location. A small turned gray trace inside the frame is not treated as proof of a shared junction.'),
 'The frame role remains plausible; POST substantially thins the opposite free supports.'),
('122a480cfcdd742650c626aa72455dae__LEG_SUPPORT_1__n0',
 st(C,PL,Y,U,U,'The gray bench has slatted back and seat surfaces, end supports and a lower rail. The blue tapered leg occupies a plausible support position beneath the seat near one end. Nearby gray supports partly hide it in one view. The fixed sections show a small positional separation near the leg top, without proving a legal connection.'),
 st(C,PL,Y,U,U,'The slatted bench remains recognizable and retains robust-looking supports. Some slat edges become thinner or smoother, while the blue leg stays in the same plausible location. Its upper junction remains partly obscured and unresolved by the sections.'),
 'The plausible leg placement and macro bench remain stable; POST mostly smooths or narrows slat edges.'),
('122a480cfcdd742650c626aa72455dae__SEAT_SURFACE_1__n0',
 st(C,U,'NO',Y,U,'The gray slatted seating surface is well above the narrow blue slat. A broad gray apron grows around the blue element at lower brace height. The side section shows gray sides extending upward from the blue level but still below the upper seat. A brace role is possible; the construction-intended seat-slat role is not visible.'),
 st(P,U,'NO',Y,Y,'The blue lower slat and its gray apron remain below the main gray seating surface. The upper seat strips become more open and narrow, with clear gaps. The height difference is unchanged. This does not rule out a reasonable lower-brace use, but it does not show the intended seating-slat use.'),
 'POST makes the upper gray seat strips more open while retaining the lower blue-slat/apron arrangement.'),
('20222f467d257022c8bbbd98dee48cb__BACK_SURFACE_1__n0',
 st(C,U,Y,Y,U,'The blue open rectangular frame is at back height in a recognizable high-backed gray bench. Angled gray sheets extend above and below it toward a larger rear panel. The side section places these gray sheets close to the blue frame segments, while the rear panel remains at another depth. The back role is apparent, but the layered construction is ambiguous.'),
 st(C,U,Y,Y,U,'The gray body and blue back-frame position remain. The angled gray sheets around the frame and large rear panel persist after smoothing. Apparent approach in the side section does not certify shared topology or removal of the extra back layer.'),
 'POST retains the layered gray back around the blue frame, with no clear image-only change in coherent use.'),
('20222f467d257022c8bbbd98dee48cb__COMBO_SEAT_SUPPORT__n0',
 st(C,U,'PARTIAL',U,U,'The blue rear slat and wedge-shaped end panel lie near the rear seat edge and one support of a recognizable gray bench. The gray bench also has its own narrow end-leg structure. Small offsets remain in the C-centered side section. Both component positions are plausible, but whether the whole blue assembly participates coherently is uncertain.'),
 st(P,U,'PARTIAL',U,Y,'The gray seat and back remain, while free legs become extremely thin. The blue wedge retains its area and the rear slat stays in place. The side section still contains nearby gray traces at slightly different positions, without resolving the assembly junction.'),
 'POST keeps the input placement but strongly thins the free legs; exact assembly integration remains unknown.'),
]
manifest_path=ROOT/'gallery_manifests'/(NAME+'.json')
manifest=read(manifest_path);completion=read(GALLERY/'completion.json')
pages=manifest['pages']
assert len(pages)==8 and len(completion['images'])==8
assert [p['request_id'] for p in pages]==[n[0] for n in notes]
schema=read(PRIOR)
schema.update(checkpoint=LABEL,actual_view_count=8,
    scope='Eight fixed process requests; SEAM columns3/4 of existing joint pages,16 RAW/POST observations',
    prior_schema_sha256=sha(PRIOR))
schema['limitations'].append('Existing joint pages also show labeled NATIVE columns. Review is nonblind; only SEAM columns3/4 are labeled here. No cross-arm or convergence claim is made.')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
definition_sha=sha(GALLERY/'definition.json');rows=[];pairs=[];images=[]
for page,(rid,raw,post,change) in zip(pages,notes):
    path=GALLERY/(page['page_id']+'.png');receipt=path.with_suffix('.json');ir=read(receipt);image_sha=sha(path)
    assert image_sha==ir['image_sha256'] and ir['definition_sha256']==definition_sha
    images.append(dict(request_id=rid,path=str(path),sha256=image_sha,receipt_sha256=sha(receipt),reviewed_column_indices_zero_based=[2,3]))
    for stage,obs,col in [('RAW',raw,2),('POST',post,3)]:
        assert page['cells'][col]['label']==LABEL+'_'+stage
        rows.append(dict(review_version=schema['review_version'],reviewer='AI_PROXY',actual_image_viewed=True,
            human_reviewed=False,nonblind=True,reviewed_at_utc=now,label=LABEL,global_step=17000,
            request_id=rid,condition_id=page['condition_id'],parent_id=page['parent_id'],split=page['split'],stage=stage,
            K=page['K'],N=page['N'],part_description=page['kind'],image_path=str(path),image_sha256=image_sha,
            image_receipt_sha256=sha(receipt),gallery_definition_sha256=definition_sha,reviewed_column_index_zero_based=col,
            model_stage_label=page['cells'][col]['label'],
            views_used=['surface_-60_23','surface_120_23','finite_front_section_at_C_bbox_center','finite_side_section_at_C_bbox_center'],
            body_recognizable_as_bench=True,**obs,exact_shared_connectivity='NOT_ASSESSED_FROM_IMAGES',
            topologically_disconnected_fragment='UNKNOWN_IMAGE_ONLY',metric_values_used_for_labels=False,
            source_full_mesh_shown=False,scope='Fixed process checkpoint observation; no checkpoint/data/seed selection or training-policy changes'))
    pairs.append(dict(request_id=rid,label=LABEL,split=page['split'],part_description=page['kind'],RAW_to_POST=change,
        image_sha256=image_sha,reviewer='AI_PROXY',actual_image_viewed=True))
def write(name,content):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(content)
write('review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
write('raw_post_comparison.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs))
write('review_schema.json',json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
counts={stage:{field:dict(collections.Counter(r[field] for r in rows if r['stage']==stage))
    for field in ('body_completeness','C_visual_coherent_use','thin_or_dangling_sheet_appearance')} for stage in ('RAW','POST')}
write('observed_label_counts.json',json.dumps(dict(source='Actual image-review labels only',counts=counts),indent=2)+'\n')
md=['# SEAM_S17000 process image review (AI_PROXY)','',
    'Actually viewed all eight existing joint process pages, using SEAM columns3/4, two surface views and fixed front/side sections. Saved16 stage observations and8 RAW-to-POST comparisons. Labeled Native columns were visible, so review is nonblind. No metrics were used to assign these labels, and no model, sampling, rescoring or new rendering was performed.','',
    'Plausible support placement is visible for9fd5 end frame and122a tapered leg. The122a seat slat is still below the main gray seating surface and may function as a lower brace; source-intended seating role and all possible coherent uses are not equated. The459f large blue assembly retains a raised folded gray layer, and both back-condition cases retain gray surfaces around or at a different depth from C.','',
    'POST often flattens surfaces while making free legs, rails or seat strips substantially thinner. The effect is clear in2d,9fd5,122a seat and20222 combo. The122a leg and459f seat cases retain a more robust macro structure in these views. This is a checkpoint observation, not a convergence, whole-panel or cross-arm improvement conclusion.','',
    'Exact shared connectivity is NOT_ASSESSED_FROM_IMAGES and disconnectedness is UNKNOWN_IMAGE_ONLY throughout. A short or missing section trace alone is not proof of a disconnected face. Source role labels are construction diagnostics, not semantic inputs or a unique required style.','',
    '| Request | RAW to POST observation |','|---|---|']
for p in pairs:md.append('|'+p['request_id']+'|'+p['RAW_to_POST']+'|')
md.extend(['','No checkpoint, noise, data or training-policy change is made by this review.',''])
write('REVIEW.md','\n'.join(md))
write('receipt.json',json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',reviewer='AI_PROXY',reviewed_at_utc=now,
    images_viewed=8,image_view_calls_per_page=1,stage_rows=16,paired_rows=8,model_calls=0,new_samples=0,
    rescoring=0,data_audits=0,new_renders=0,metric_values_used=False,nonblind=True,
    joint_gallery_name=NAME,reviewed_column_indices_zero_based=[2,3],protected_gallery_definition_sha256=definition_sha,
    protected_manifest_sha256=sha(manifest_path),protected_completion_sha256=sha(GALLERY/'completion.json'),
    image_records=images,files={str(p):sha(p) for p in OUT.iterdir() if p.is_file()}),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='ACTUAL_IMAGE_REVIEW_COMPLETE',images=8,stage_rows=16,paired_rows=8,counts=counts)))
