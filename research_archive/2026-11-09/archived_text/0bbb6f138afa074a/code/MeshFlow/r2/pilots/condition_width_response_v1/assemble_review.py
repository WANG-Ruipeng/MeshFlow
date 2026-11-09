"""Assemble actual image observations; never infer visual labels from numeric scores."""
import csv,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
P=Path('/mnt/e/MeshFlow/experiments/r2_condition_width_response_v1/evaluation/gallery')
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvread(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
fields=['C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer','whole_body_complete','support_missing','fragments_or_folded_layers','visible_join']
template=csvread(P/'AI_PROXY_review_template.csv');obs=load(P/'review_root_pages.json')
assert len(obs)==16
root=[]
for o in obs:
 matches=[r.copy() for r in template if r['parent_id'].startswith(o['parent']) and r['base_request_id'].endswith('__n'+str(o['noise'])) and r['stage']==o['stage']]
 assert len(matches)==3
 for r in matches:
  vals=o.get('variantLabels',{}).get(r['variant'],o['labels']);assert len(vals)==8
  r.update(dict(zip(fields,vals)))
  r.update(review_status='REVIEWED_AI_PROXY',reviewer='AI_PROXY',blind='False',reviewed_image_sha256=r['image_sha256'],notes=o['notes'])
  assert sha(Path(r['image_path']))==r['image_sha256']
  root.append(r)
def writecsv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(template[0]));w.writeheader();w.writerows(rows)
writecsv(P/'review_root.csv',root)
others=csvread(P/'review_assets.csv')
assert len(root)==len(others)==48
lookup={(r['request_id'],r['stage']):r for r in root+others}
assert len(lookup)==96
merged=[lookup[(t['request_id'],t['stage'])] for t in template]
writecsv(P/'AI_PROXY_visual_review.csv',merged)
receipt=dict(status='COMPLETE',review_csv=str(P/'AI_PROXY_visual_review.csv'),review_csv_sha256=sha(P/'AI_PROXY_visual_review.csv'),utc=datetime.now(timezone.utc).isoformat(),pages_actually_viewed=32,rows=96,observers=['root: first 4 SEAT_HALF parents','scaling_assets: remaining 4 SEAT_FULL/BACK parents'],blind=False,metric_rank_not_read_before_labels=True,review_type='Nonblind AI_PROXY visual observations; not human rating; no index-topology inference from image proximity',uncertainty='YES/NO/UNCERTAIN retained; occlusion is never treated as adoption',notes='Each page displays all three registered alpha values, front/side/oblique/wire. Titles overlap slightly at center; geometry remains readable.')
(P/'output_review_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
