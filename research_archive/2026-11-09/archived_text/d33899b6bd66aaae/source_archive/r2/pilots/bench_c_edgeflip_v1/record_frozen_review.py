"""Transcribe root's actual tool-viewed frozen bcb comparison pages."""
from .common import *

def main():
    rows=[]
    for n in (0,1):
        rid='bcb4f576747e10bdd7f6e86d7d335698__COMPLETE_FLAT_SEAT_TOP_SURFACE__HIGH_JOINT_V1__n'+str(n)
        image=OUT/'gallery'/rid/'comparison.png'
        for label in ('ORIGINAL','EDGE_FLIP'):
            note='Both views show a rounded-end ring frame with blue C as exposed horizontal top. Fixed front and side sections place the blue top on the expected upper contour. Fine gray strokes and gaps cannot by themselves certify detached connected components.'
            if label=='EDGE_FLIP' and n==0:note+=' A short additional gray inner sheet appears under the left/middle C in the front/side sections; broad organization and support remain. No clear global quality improvement.'
            if label=='EDGE_FLIP' and n==1:note+=' Curved frame and blue top remain; lower inner contour/thickness changes. No L-shaped replacement like previous midpoint refinement. No confirmed fragment reduction.'
            rows.append(dict(cell_id=label+'__'+rid,request_id=rid,noise_index=n,label=label,stage='RAW',
                reviewer='AI_PROXY_root',blind=False,actually_viewed=True,tool='tools.view_image(detail=original)',
                whole_shape='REASONABLE',C_adoption='PLAUSIBLE',main_support='PRESENT',
                visible_detached_fragments='UNCERTAIN',layer_gap_or_overlap_concern='UNCERTAIN',observations=note,
                source_image=str(image),source_image_sha256=sha(image)))
    write(OUT/'visual_review/frozen_bcb.json',dict(schema='actual_nonblind_AI_PROXY_v1',rows=rows,actual_pages=2,
        limitation='Only existing fixed views/slices; no exact topology or human validation; no unique-source style criterion.'))
    csvwrite(OUT/'visual_review/frozen_bcb.csv',rows)

if __name__=='__main__':main()
