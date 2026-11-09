"""Transcribe the six actually viewed paired figures; inherit only byte-identical cases."""
from . import run as A

NOTES={
 '73b369b9_n1':'Two views retain the cantilever chair, blue sitting plane and back. The original broad gap between upper and lower folded pieces is still visible. No obvious new missing body/support or loose fragment; micro-contact and hidden layers are UNKNOWN.',
 '73b369b9_n2':'Two views retain the blue seat, high folded back and lower folded support. The upper back slit and the lower/open side gaps remain visible. No clear large-scale visual change; exact contact/hidden overlap cannot be certified.',
 '73b369b9_n3':'Two views retain seat use, polygonal back and cantilever lower support. Existing visible gap under the upper chair remains. No clear new large fragment or support loss; no visual claim of closed seams.',
 '3f741759_n2':'Both paired views retain the slatted rounded back, four legs and blue top seat. The near-seat internal layer is largely hidden by C and the apron; these views do not certify its removal despite numerical I reduction. No obvious new body/support loss.',
 '8c9b2023_n1':'Both views retain blue seat, horizontal back bars and four legs. Seat outline and support silhouette look similar. No obvious new large fragment or missing support; hidden near-seat overlap is UNKNOWN.',
 '8c9b2023_n2':'Both views show the same blue seat, slatted back, legs and existing back-bar gaps. No visible change. Final triangle-array equality independently explains this, although pre-GEO/displacement metadata changed.'}


def main():
    gallery=A.read(A.OUT/'gallery/render_receipt.json');images={r['request_id']:r for r in gallery['rows']}
    prior=A.rows(A.OLD/'evaluation/visual_review_all64.csv')
    prior={r['request_id']:r for r in prior if r['label']=='SEAM_SLOT_C200' and r['stage']=='POST'}
    identity={r['request_id']:r for r in A.rows(A.OUT/'post_geometry_identity.csv')}
    records=[]
    for q in A.read(A.OLD/'eval_manifest.json')['requests']:
        rid=q['request_id'];base=dict(request_id=rid,parent_id=q['parent_id'],noise_index=q['noise_index'],
            stage='POST',baseline='PREDICTED_ORIGINAL_GEO',candidate='ANTI_FLATTEN',reviewer='AI_PROXY_SEAM_EVIDENCE',
            nonblind=True,precise_connection='UNKNOWN_FROM_IMAGES',hidden_overlap='UNKNOWN_FROM_IMAGES')
        if rid in NOTES:
            item=images[rid]
            records.append(dict(**base,review_status='ACTUALLY_VIEWED_PAIRED_TWO_VIEWS',image=item['image'],image_sha256=item['sha256'],
                                C_role='VISIBLE_SEAT_BOTH',obvious_new_body_or_support_loss='NO_VISIBLE',
                                obvious_new_fragments='NO_VISIBLE',overall_visual_delta='NO_CLEAR_LARGE_SCALE_CHANGE',notes=NOTES[rid]))
        else:
            assert identity[rid]['all_POST_arrays_identical']=='True'
            old=prior[rid]
            records.append(dict(**base,review_status='INHERITED_OLD_REVIEW_BYTE_IDENTICAL_ALL_POST_ARRAYS_NOT_NEWLY_VIEWED',
                                image=old['reviewed_image_path'],image_sha256=old['reviewed_image_sha256'],
                                C_role=old['C_adoption'],obvious_new_body_or_support_loss='NONE_BYTE_IDENTICAL',
                                obvious_new_fragments='NONE_BYTE_IDENTICAL',overall_visual_delta='IDENTICAL_MESH',
                                notes=old['notes'],inherited_reviewer=old['reviewer'],inherited_source_review=A.OLD/'evaluation/visual_review_all64.csv'))
    assert len(records)==16 and sum(x['review_status'].startswith('ACTUALLY') for x in records)==6
    A.C.csvwrite(A.OUT/'review.csv',records)
    A.C.write(A.OUT/'gallery/review_completion.json',dict(status='COMPLETE',paired_rows=16,new_actual_pairs=6,
        inherited_byte_identical_pairs=10,actual_mesh_views=24,semantic_labels_generated_from_metrics=False,
        review_sha256=A.C.sha(A.OUT/'review.csv'),model_calls=0,additional_quality_scans=0))


if __name__=='__main__':main()
