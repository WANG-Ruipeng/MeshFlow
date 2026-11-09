"""Compile recorded image observations; never infer review labels from scores."""
import csv, json, collections
from . import common as E

G=E.OUT/'evaluation/gallery'
def readcsv(p):
    with E.resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def run():
    obs=E.read(G/'root_visual_observations.json'); reg=E.load_registration();tmpl=readcsv(G/'AI_PROXY_review_template.csv'); indep=readcsv(G/'review_sub4/review_sub4.csv')
    fields=['C_adoption','C_floating','C_replaced_by_other_surface','C_wrong_layer','whole_body_complete','support_missing','fragments_or_folded_layers','visible_join']
    ob={(x['pair'],x['split'],x['n']):x for x in obs};root=[]
    for t in tmpl:
        if t['variant']!='base':continue
        o=ob[t['pair_id'],t['split'],int(t['candidate'])];t=dict(t)
        image=G/'images'/(t['pair_id']+'__n'+t['candidate']+'__TRAIN_DEV.jpg');h=E.sha(image)
        if str(image) not in json.loads(t['image_paths']) or h not in json.loads(t['image_sha256s']):raise ValueError('Review image identity changed')
        t.update(review_status='REVIEWED',reviewer='/root',review_kind='AI_PROXY',blind=False,reviewed_image_sha256=h,actual_review_image_path=str(image),
          C_adoption=o['a'],C_floating=o['floating'],C_replaced_by_other_surface=o['replacement'],C_wrong_layer=o['wrong'],whole_body_complete=o['whole'],support_missing='NO',
          fragments_or_folded_layers='UNCERTAIN',visible_join='UNCERTAIN',notes=o['notes'],review_scope='Whole oblique view and two fixed central cross sections; same-stage labels independently inspected; not a topology, watertightness or no-overlap certificate.')
        root.append(t)
    assert len(root)==64
    rootby={(r['request_id'],r['stage']):r for r in root};differences=[]
    for i in indep:
        if E.sha(i['image_path'])!=i['reviewed_image_sha256']:raise ValueError('Independent review image changed')
        k=i['request_id'],i['stage']
        if k in rootby:
            b=rootby[k]
            for f in fields:
                if b[f]!=i[f]:differences.append(dict(request_id=k[0],stage=k[1],field=f,root_value=b[f],independent_value=i[f],resolution='BOTH_RETAINED_NO_VOTE',root_image=b['actual_review_image_path'],independent_image=i['image_path']))
    main=root+[i for i in indep if i['variant']=='sub4']
    assert len(main)==80 and len({(r['request_id'],r['stage']) for r in main})==80
    for r in main:
        r['has_independent_adoption_disagreement']=any(d['request_id']==r['request_id'] and d['stage']==r['stage'] and d['field']=='C_adoption' for d in differences)
    E.csvwrite(G/'AI_PROXY_root_base_review.csv',root);E.csvwrite(G/'AI_PROXY_review_disagreements.csv',differences);E.csvwrite(G/'AI_PROXY_review_completed.csv',main)
    summary=[]
    for stage in ('RAW','POST'):
      for split in ('TRAIN','DEV'):
       for region in ('SEAT','BACK'):
        for exclusion in (False,True):
         rr=[r for r in main if r['stage']==stage and r['split']==split and r['region_type']==region and r['variant']=='base' and (not exclusion or r['pair_id']!='PAIR04')]
         summary.append(dict(stage=stage,split=split,region=region,exclude_near_duplicate_PAIR04=exclusion,count=len(rr),**dict(collections.Counter(r['C_adoption'] for r in rr)),definition='Primary nonblind AI_PROXY judgment; disagreements saved separately; do not treat as human or general benchmark'))
    E.csvwrite(G/'AI_PROXY_review_summary.csv',summary)
    E.write(G/'review_completion.json',dict(status='ACTUALLY_REVIEWED',unique_output_stage_rows=80,pages_actually_reviewed=24,root_rows=64,independent_rows=32,independent_duplicate_base_rows=16,sub4_primary_rows=16,all_primary_rows_have_notes_and_image_hash=True,adoption_disagreement_rows=sum(d['field']=='C_adoption' for d in differences),all_field_disagreements=len(differences),human_review=False,blind=False,metric_defined_semantic_labels=False))
    print(json.dumps(dict(status='COMPLETE',primary_rows=80,adoption_disagreements=sum(d['field']=='C_adoption' for d in differences),summary=summary),ensure_ascii=False))
if __name__=='__main__':run()