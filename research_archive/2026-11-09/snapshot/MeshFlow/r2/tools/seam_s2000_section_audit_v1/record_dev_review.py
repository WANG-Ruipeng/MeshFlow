"""Literal observations after the root reviewer actually viewed twelve DEV pages."""
import csv
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[3]/'experiments/r2_seam_s2000_section_audit_v1'
# These judgments are manually supplied, not inferred from metrics or topology.
RECORDS=[
('2d08a64e_n0','YES','YES',[], 'side section, short lines between blue C and the lower gray contour', 'RAW',
 'Several short inner-looking horizontal traces are present in RAW. D0 reduces some rough/overprinted traces; a sloping short trace below C and separated bottom runs remain in POST. The side cutaway still has folded-looking geometry near the back-seat junction.',
 'These traces are visible suspects, not proven detached pieces; the complete faces and their connectivity are not identified in this review.'),
('2d08a64e_n1','UNCERTAIN','YES',[], 'side section, small short segment below C near the front half', 'RAW',
 'A small isolated-looking segment below C appears in all four stages, with broadly similar location. RAW bottom contour is rougher; D0 and POST reduce the many small traces. The cutaway mostly shows the outer apron and leg supports.',
 'Cannot establish that the short line is an unwanted internal sheet rather than a plane crossing a structural face; no face ID assigned.'),
('2d08a64e_n2','UNCERTAIN','YES',[], 'side section, short vertical stubs around rear/front seat ends', 'RAW',
 'RAW already has stubs and a lower horizontal contour. D0 makes the lower contour more continuous; the local stubs persist. In the front cutaway a triangular facet/open region beneath the arm remains across stages.',
 'Some apparent separation is outside the section plane or due to the artificial halfspace cut; no independent floating component is certified.'),
('2d08a64e_n3','UNCERTAIN','UNCERTAIN',[], 'side section lower left and cutaway below C', 'RAW',
 'The lower gray supporting structure and gap beneath C are visible already in RAW. D0 smooths many jagged traces, and the main lower gap persists through RESTORED and POST. Several contour ends exit the fixed local plotting window.',
 'The supporting cantilever shape and an actual loose sheet cannot be separated from these views alone; lines ending at the frame are not counted as proof of disconnection.'),
('610b1621_n0','NO','UNCERTAIN',[], 'front and side sections, vertical pairs near the box sides', 'RAW',
 'Parallel vertical contours and the broad box shell are already present in RAW and remain through POST. D0 makes outlines less rough. No clear new loose interior face is apparent in the two cutaway views.',
 'Contours leave the local plotting bounds; neither apparent line separation nor absence of a visible loose piece certifies whole-mesh connectivity.'),
('610b1621_n1','YES','YES',[], 'side cutaway, triangular tips near rear C/back junction', 'RAW',
 'Triangular tips near the back-seat junction are already visible in RAW and persist in D0, RESTORED and POST. The side section has separated inclined traces. A bright lower apron line becomes less open after D0.',
 'The tips are visually irregular but complete face connectivity is not assigned here; they may be attached flaps rather than free-floating pieces.'),
('610b1621_n2','UNCERTAIN','YES',[], 'front lower tray-like contour, side lower shell separated from C', 'RAW',
 'The lower tray-like gray contour is separate from the short C-side slopes already in RAW. D0 reduces jagged side details without closing the broad separation, and POST retains it. Half-cut views show a larger lower support/shell, not just tiny scraps.',
 'No unique source style is assumed; these views cannot decide whether the lower structure is an intended support or an unwanted extra shell.'),
('610b1621_n3','YES','YES',[], 'side section, short angular fragments below and just above C', 'RAW',
 'Small angular and rectangular-looking section traces occur near the seat/back junction in RAW and persist after D0 and restoration. POST reduces roughness around the back opening, while the short interior-looking traces remain.',
 'A section through a thin support can also create small line loops. Visible suspects are not certified detached triangle components.'),
('608b46b0_n0','UNCERTAIN','YES',[], 'front/side section, interrupted horizontal contours immediately below C', 'RAW',
 'Several underside contour runs are separated in RAW and remain separated after D0, RESTORED and POST. The cutaway exposes a narrow diagonal white gap in the lower seat band throughout the four stages.',
 'The gray faces lie on/near the underside; this is not proof of floating scraps inside a closed volume. Connectivity requires the full mesh.'),
('608b46b0_n1','UNCERTAIN','YES',[], 'side section, long underside trace and short front-bottom segment', 'RAW',
 'RAW already has separated underside traces; D0 joins or overlaps some of the longer visible runs, but POST still has a short separate-looking segment near the front. The broad seat-back shape is stable.',
 'A short isolated cut line need not come from a small triangle, and the cutaway does not prove that it is an independent floating component.'),
('608b46b0_n2','NO','YES',[], 'front section small oblique stubs above C near outer sides', 'RAW',
 'The small oblique side traces and broad box boundaries can be followed from RAW to POST. The half-cut views show the large box/arm surfaces, without a clear loose interior sheet newly appearing after D0.',
 'Isolated traces can be corners of the outer frame crossing the plane; full topology is not inferred from their appearance.'),
('608b46b0_n3','YES','YES',[77,113], 'front section, two separated underside spans beneath C', 'RAW',
 'Separated gray underside spans are already visible in RAW. D0 shifts and straightens them; RESTORED is similar, and POST retains the front-section gap. Pixel/finite-section lookup subsequently identifies the left span as slot77 and the next long span as slot113.',
 'The visual suspicion is a disconnected-looking underside sheet, not necessarily a tiny floating shard. Separate topology records establish point-only attachment for slot77; this is not inferred from the image alone.'),
]

def main():
 fields=['request_id','split','reviewer','actually_viewed','image_path','image_sha256',
         'post_suspect_internal_sheet_visible','post_isolated_section_line_visible','suspect_raw_slots',
         'view_location','first_visible_stage','stage_change_note','uncertainty']
 rows=[]
 for rid,internal,isolated,slots,where,first,note,uncertain in RECORDS:
  p=OUT/'gallery'/(rid+'.png')
  rows.append(dict(zip(fields,[rid,'DEV','AI_PROXY_NONBLIND',True,str(p),hashlib.sha256(p.read_bytes()).hexdigest(),
              internal,isolated,json.dumps(slots),where,first,note,uncertain])))
 assert len(rows)==12 and len({r['request_id'] for r in rows})==12
 with (OUT/'visual_dev.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 print(json.dumps({'actual_main_pages_viewed':12,'status':'RECORDED','automatic_labels':False}))

if __name__=='__main__': main()
