"""Validate complete actual anonymous reviews, preserve them, then decode aliases."""
from .bootstrap import bind
D,E=bind()
FIELDS=('RAW_adoption','POST_adoption','POST_body_complete')

def run():
    ev=D.OUT/'evaluation';g=ev/'blind_gallery'
    gallery=D.read(g/'manifest.json');review=D.read(g/'reviews_anonymous_actual.json');views=D.read(g/'view_receipts.json')
    assert gallery['status']==review['status']==views['status']=='COMPLETE'
    assert review['gallery_manifest_sha256']==D.sha(g/'manifest.json')
    images={r['request_id']:r for r in gallery['images']}
    assert len(images)==32 and gallery['identity']['cases']==96
    actual={r['request_id']:r for r in views['rows']}
    assert len(views['rows'])==32 and set(actual)==set(images)
    for rid,v in actual.items():
        im=images[rid]
        assert v['viewed_actual'] is True and v['image_sha256']==im['image_sha256']
        assert D.resolve(v['image_path'])==D.resolve(im['image_path'])
        assert D.sha(im['image_path'])==im['image_sha256']
    expected={(rid,a) for rid in images for a in ('A','B','C')}
    rows=review['rows'];seen=set()
    for r in rows:
        key=(r['request_id'],r['label']);assert key in expected and key not in seen;seen.add(key)
        assert r['review_status']=='REVIEWED' and r['reviewer_kind']=='AI_PROXY' and r['blinded'] is True
        assert all(r[f] in ('YES','NO','UNKNOWN') for f in FIELDS)
        assert r['notes'].strip() and r['reviewer'].strip()
        assert r['image_sha256']==images[r['request_id']]['image_sha256']
    assert len(rows)==96 and seen==expected
    # Mapping is read only after all review labels and actual-view receipts passed.
    sealed={str(g/f):D.sha(g/f) for f in ('reviews_anonymous_actual.json','view_receipts.json','manifest.json')}
    D.write(ev/'review_seal_before_unblinding.json',dict(status='PASS',files=sealed,rows=96,pages=32,review_labels_modified=0,model_calls=0))
    mapping=D.read(ev/'private/label_mapping.json')
    maps={r['request_id']:r['aliases'] for r in mapping['rows']}
    result=[]
    for r in rows:
        rr=dict(r);rr['anonymous_label']=r['label'];rr['label']=maps[r['request_id']][r['label']]
        rr['image_path']=images[r['request_id']]['image_path'];result.append(rr)
    assert len({(r['request_id'],r['label']) for r in result})==96
    D.write(ev/'reviews_actual.json',dict(status='COMPLETE',rows=sorted(result,key=lambda r:(r['request_id'],r['label'])),
        input_review_hashes=sealed,mapping_sha256=D.sha(ev/'private/label_mapping.json'),review_labels_modified=0,unblinding_after_seal=True))
    print('ACTUAL_REVIEW_SEALED_AND_DECODED 96 rows/32 pages; labels unchanged',flush=True)

if __name__=='__main__':run()
