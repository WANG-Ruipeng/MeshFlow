"""Compare historical FULL reviews with the new independent anonymous BASE reviews.
No relabeling, winner selection, image review, model calls or score recomputation.
"""
import csv,io,json
from collections import Counter
from . import data as D

def main():
 root=D.OUT/'evaluation/analysis'
 final=D.read(root/'summary.json')
 if final['status']!='COMPLETE':raise RuntimeError('Final complete review required')
 historical=D.ROOT/'experiments/r2_full_seat_learning_curve_v1/evaluation/review/AI_PROXY_reviews.csv'
 with historical.open(encoding='utf-8-sig',newline='') as f:
  old=[r for r in csv.DictReader(f) if r['label']=='FULL_S6000']
 oldmap={(r['request_id'],r['stage']):r for r in old}
 if len(oldmap)!=48 or len(old)!=48:raise RuntimeError('Expected historical FULL 24 RAW + 24 POST reviews')
 current=[r for r in D.read(root/'per_output.json') if r['label']=='BASE' and r['evaluation_noise_partition']=='OLD']
 if len(current)!=24:raise RuntimeError('Expected current BASE OLD24')
 def norm(x):return x if x in ('YES','NO') else 'UNKNOWN'
 rows=[]
 for r in current:
  for field,stage,oldfield in [('RAW_adoption','RAW','C_adoption'),('POST_adoption','POST','C_adoption'),('POST_body_complete','POST','whole_body_complete')]:
   prior=oldmap[r['request_id'],stage];a,b=norm(prior[oldfield]),norm(r[field])
   rows.append(dict(request_id=r['request_id'],parent_id=r['parent_id'],split=r['split'],field=field,
       original_label=prior[oldfield],original_label_normalized=a,current_blind_label=r[field],
       agreement=a==b,transition=a+'->'+b,prior_reviewer=prior['reviewer'],current_reviewer=r['reviewer'],
       historical_image_sha256=prior.get('image_sha256'),current_image_sha256=r['review_image_sha256']))
 buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 target=root/'historical_BASE_review_consistency.csv';payload=buffer.getvalue().encode('utf-8-sig')
 if target.exists() and target.read_bytes()!=payload:raise RuntimeError('Immutable comparison differs')
 if not target.exists():target.write_bytes(payload)
 summary=dict(status='COMPLETE',rows=72,outputs=24,model_calls=0,labels_changed_by_this_analysis=0,
   groups=[dict(field=f,split=s,rows=len(v),agreement=sum(r['agreement'] for r in v),transitions=dict(Counter(r['transition'] for r in v)))
      for f in ('RAW_adoption','POST_adoption','POST_body_complete') for s in ('ALL','TRAIN','DEV')
      for v in [[r for r in rows if r['field']==f and (s=='ALL' or r['split']==s)]]],
   caveats=['Historical nonblind review and final anonymous review use different panels/views and separate AI_PROXY reviewers.',
     'This is consistency evidence, not human ground truth or a reason to replace the new blinded labels.',
     'All final methods use the same new blinded review protocol; original reports and labels remain unchanged.'],
   source_hashes={str(historical):D.sha(historical),str(root/'per_output.json'):D.sha(root/'per_output.json'),str(__file__):D.sha(__file__)},
   output_csv_sha256=D.sha(target))
 D.write(root/'historical_BASE_review_consistency.json',summary)
 print(json.dumps({k:summary[k] for k in ('status','rows','outputs','groups')}))
if __name__=='__main__':main()
