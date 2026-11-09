"""Record observations actually made by root from eight paired contact sheets.

This is a transcription of completed AI_PROXY viewing, not an image classifier.
"""
import csv
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[3]/'experiments/r2_seam_occupancy_rollout_v1/gallery'
    rows=list(csv.DictReader((root/'review_template_NOT_REVIEWED.csv').open(encoding='utf-8-sig')))
    out=[]
    for row in rows:
        parent=row['parent_id'][:8]
        if parent not in ('73b369b9','85b16941'):
            continue
        group='RAW' if row['stage']=='RAW' else 'POST'
        images=[root/f'{parent}_{group}_contact_view{i}.png' for i in (1,2)]
        evidence=[dict(path=str(p),sha256=sha(p)) for p in images]
        row.update(reviewer='AI_PROXY_ROOT',review_status='ACTUALLY_VIEWED_NONBLIND',
            C_adoption='YES',C_floating='UNKNOWN',C_replaced_by_other_surface='UNKNOWN',
            C_wrong_layer='NO_VISIBLE_WRONG_LAYER',whole_body_complete='YES_VISUAL',
            support_missing='NO_VISIBLE_MAJOR_LOSS',fragments_or_folded_layers='UNKNOWN',
            visible_join='UNKNOWN_NOT_CERTIFIED_BY_IMAGE',
            reviewed_image_path=str(images[0]),reviewed_image_sha256=sha(images[0]),
            reviewed_images_json=json.dumps(evidence),
            evaluation_boundary='Same-scale two-view contact review; exact topology, hidden overlap and tiny cracks not certified.')
        if parent=='73b369b9':
            row['notes']='All three layouts look identical within this noise. Blue C is the seat; back and folded pedestal/base remain. Existing narrow gaps are still visible, especially around the base and the n2 upper back; no visual repair can be credited to occupancy.'
        elif row['noise_index']=='3' and row['label']!='PREDICTED':
            row['notes']='Blue C remains the seat and the major legs/back are present. Relative to PREDICTED, the lower back panel extends diagonally downward closer to C. Both new layouts look identical. This is a visible shape change, not evidence that cracks or hidden intersections were fixed.'
        else:
            row['notes']='Blue C occupies the seat position; back and major support legs remain recognizable. New layouts preserve overall style with local panel/edge changes. No obvious new whole-leg loss in these views. Small gaps and exact attachment are not certified.'
        out.append(row)
    assert len(out)==72
    dest=root/'review_73_85_AI_PROXY.csv'
    with dest.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    p=root/'3f741759_n2_POST.png'
    note=dict(reviewer='AI_PROXY_ROOT',status='ACTUALLY_VIEWED_NONBLIND',image=str(p),image_sha256=sha(p),
        observation='ALL_ON has extra oblique panels across/near the back openings compared with PREDICTED and SOURCE_ORACLE. All retain recognizable chair body and seat C. Blue-C depth tie/occlusion means the image alone cannot establish removal of the old near-coplanar I tail; use fixed geometric scores separately.',
        main_labels='Recorded separately by 3f/8c reviewer; this note does not overwrite them.')
    with (root/'root_3f_n2_additional_view.json').open('x',encoding='utf-8') as f:json.dump(note,f,indent=2)
    print(json.dumps(dict(status='RECORDED_ACTUAL_VIEWS',rows=72,contact_images=8,additional_full_page=1)))


if __name__=='__main__':main()
