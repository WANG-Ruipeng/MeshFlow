import csv,json
from collections import Counter
from . import common as c

def main():
    reg=c.read(c.OUT/'registration.json');reviews=[]
    for name in ['reviewer_local_scope.csv','reviewer_seat_data.csv','reviewer_root.csv']:
        path=c.OUT/'reviews'/name
        rs=list(csv.DictReader(path.open(encoding='utf-8-sig')))
        for r in rs:
            assert r['review_status']=='REVIEWED'
            assert json.loads(r['reviewed_image_sha256s'])==json.loads(r['image_sha256s'])
            for p,h in zip(json.loads(r['image_paths']),json.loads(r['image_sha256s'])):assert c.sha(p)==h
        reviews+=rs
    assert len(reviews)==44 and len({r['cell_id'] for r in reviews})==44
    assert {r['cell_id'] for r in reviews}=={r['cell_id'] for r in reg['cells']}
    by={r['cell_id']:r for r in reviews};ordered=[by[x['cell_id']] for x in reg['cells']]
    c.csvwrite(c.OUT/'AI_PROXY_layer_review.csv',ordered)
    pairs=[]
    for req in reg['requests']:
        out={k:req[k] for k in ['short_id','request_id','condition_id','parent_id','selection_reason','K','N']}
        for label in ['SMALL_C200','CREF_C200']:
            for stage in ['RAW','POST']:
                row=by[req['short_id']+'__'+label+'__'+stage]
                for field in ['observed_layer_relation','remaining_replacement_surface','sidewall_support','C_participates_in_seat','visible_local_connection','evidence_sufficient','notes']:
                    out[label+'__'+stage+'__'+field]=row[field]
        pairs.append(out)
    c.csvwrite(c.OUT/'paired_layer_review.csv',pairs)
    images=set(h for r in ordered for h in json.loads(r['reviewed_image_sha256s']))
    c.write(c.OUT/'review_completion.json',dict(status='COMPLETE',actual_AI_PROXY_cells=44,unique_request_images=len(images),source_reference_images=6,
        nonblind=True,independent_human_review=False,old_adoption_labels_modified=False,
        label_counts_descriptive_only=dict(Counter(r['observed_layer_relation'] for r in ordered)),
        note='RAW/POST are related observations; result-selected panel is not a new population success rate.'))
    c.event('ACTUAL_IMAGE_REVIEW_COMPLETE',cells=44,request_images=22,source_reference_images=6,reviewer_type='NONBLIND_AI_PROXY')
    print(c.read(c.OUT/'review_completion.json'))
if __name__=='__main__':main()