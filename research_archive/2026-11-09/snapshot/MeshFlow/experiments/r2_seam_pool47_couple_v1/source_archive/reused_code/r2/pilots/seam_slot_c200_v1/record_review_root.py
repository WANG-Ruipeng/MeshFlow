"""Record root's actual nonblind review of eight already rendered pages.

No metric-derived semantic labels. All eight pages were opened with view_image
before this file was authored. This preserves the literal assessment and hashes.
"""
from pathlib import Path
import csv, hashlib, json

ROOT=Path(__file__).resolve().parents[3]/'experiments/r2_seam_slot_c200_v1/evaluation/gallery'
NOTES={
 '85b16941_n0':(
  'C is the exposed sitting surface; back and four supports are present. The front apron has a visible triangular opening. No replacement seat is apparent.',
  'C is the exposed sitting surface; the arched back reaches lower and legs are thicker. Front under-seat gaps remain. No replacement seat is apparent.'),
 '85b16941_n1':(
  'C occupies the seat top over the side/front frame. Back and supports are present; central cuts show no separate competing upper seat.',
  'C occupies the seat top; overall shape remains close to the paired ordinary chair. The seat box is slightly deeper. No competing upper seat is visible.'),
 '85b16941_n2':(
  'Blue C forms the seat top; back/frame/supports are coherent at whole-shape scale. The cuts place the sidewall near the C rim.',
  'Blue C forms the seat top; back and supports are present. The underside central cut is interrupted, without an obvious missing main body.'),
 '85b16941_n3':(
  'Blue C forms the top of the sitting region of a thick-legged, arched-back chair. The front/right blue rim overhangs the generated frame slightly.',
  'Blue C forms the seat top of the slimmer open-back chair. The side/front frame lies near its perimeter; no competing seat is visible.'),
 '8c9b2023_n0':(
  'C is the top seat of the ladder-back chair. Four supports and a surrounding frame are visible. Whole views and cuts agree on seat height.',
  'C remains the top seat; ladder back and supports remain. A short sidewall/cut gap exists on one side of the deeper seat box.'),
 '8c9b2023_n1':(
  'C is the top seat above a closed-looking seat frame; back and supports are present. No upper replacement layer is visible.',
  'C is still the top seat. The front cut contains an inward diagonal segment beneath the right C rim; these views do not resolve whether it is a folded face or a legitimate internal surface.'),
 '8c9b2023_n2':(
  'C forms the top sitting surface. Ladder back and legs remain complete at whole-shape scale; no second seat is evident.',
  'C forms the top sitting surface. The lowest back rail/seat junction has a visible small corner gap. This does not remove the main back or supports.'),
 '8c9b2023_n3':(
  'Blue C is the seat top. Whole chair and supports are present; the finite side cut contains a short lower-corner interruption.',
  'Blue C is the seat top with intact main back and supports. Central front/side cuts show several short gaps in the seat box; no obvious replacement layer.')
}

def main():
 with (ROOT/'review_template_NOT_REVIEWED.csv').open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;template=list(reader)
 rows=[]
 for r in template:
  rid=r['request_id']
  if rid not in NOTES:continue
  p=ROOT/(rid+'.png');h=hashlib.sha256(p.read_bytes()).hexdigest()
  assert h==r['reviewed_image_sha256']
  seam=r['label']=='SEAM_SLOT_C200'
  r.update(reviewer='AI_PROXY_ROOT',review_status='ACTUALLY_VIEWED_NONBLIND',
   C_adoption='YES',C_floating='NO',C_replaced_by_other_surface='NO',C_wrong_layer='NO',
   whole_body_complete='YES',support_missing='NO',
   fragments_or_folded_layers='UNCERTAIN' if seam and rid=='8c9b2023_n1' else 'NO',
   visible_join='YES',notes=NOTES[rid][int(seam)]+
   (' POST retains the same visible use/body arrangement.' if r['stage']=='POST' else ' RAW remains visibly triangulated; use judgment does not certify exact closure.')+
   ' visible_join means local contact appearance only, not shared-index or legal-edge certification. Two cuts and two views cannot certify absence of hidden defects.')
  rows.append(r)
 assert len(rows)==32
 dest=ROOT/'review_85_8c_AI_PROXY.csv'
 with dest.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 receipt=dict(status='COMPLETE',actual_pages=8,actual_cells=32,reviewer='AI_PROXY_ROOT',
  method='Each of eight PNG pages opened using view_image; four columns and both whole views and cuts inspected before numeric scoring results were read.',
  blind=False,visible_join='Visual local contact appearance, not topology certification.',
  source_images={k:hashlib.sha256((ROOT/(k+'.png')).read_bytes()).hexdigest() for k in NOTES},
  csv_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
 with (ROOT/'review_85_8c_AI_PROXY.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
 print(json.dumps(dict(status='COMPLETE',cells=len(rows))))

if __name__=='__main__':main()
