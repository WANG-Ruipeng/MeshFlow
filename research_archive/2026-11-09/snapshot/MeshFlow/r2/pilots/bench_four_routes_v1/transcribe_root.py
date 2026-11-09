"""Package explicitly recorded view_image notes; no score-derived labels."""
import argparse, importlib, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('route',choices=['seam','exposure']); args=p.parse_args()
    R=importlib.import_module('r2.pilots.bench_four_routes_v1.'+args.route+'.review_record')
    folder=R.OUT/'visual_review/manual'; source=folder/'root_compact_observations.json'
    notes=R.read(source)['notes']; assignment=R.read(R.OUT/'visual_review/assignments.json')['assignments']['root']
    ids=assignment['page_ids']; protocol=R.read(R.OUT/'visual_review/protocol.json')
    assert [n['index'] for n in notes]==list(range(len(ids)))
    pages=[]
    for pid,n in zip(ids,notes):
        png=R.OUT/'gallery'/(pid+'.png'); receipt=R.read(png.with_suffix('.json'))
        labels=[c['model'] for c in receipt['cells']]
        assert labels==protocol['labels'] and len(n['ratings'])==len(labels)
        ratings={}
        for label,v in zip(labels,n['ratings']):
            assert len(v)==6
            ratings[label]=dict(zip(protocol['axes'],v[:5])); ratings[label]['note']=v[5]
        pages.append(dict(page_id=pid,tool='view_image',view_event_id='root_'+args.route+'_view_'+str(n['index']),
                          viewed_at_utc=n['viewed_at_utc'],actually_viewed=True,views_and_sections_checked=True,
                          viewed_image_sha256=R.sha(png),ratings=ratings))
    obj=dict(reviewer='root',reviewer_kind='AI_PROXY',blind=False,assigned_page_ids=ids,pages=pages,
             explicit_notes_path=str(source),explicit_notes_sha256=R.sha(source))
    target=folder/'root.json'
    raw=json.dumps(obj,ensure_ascii=False,indent=2).encode('utf-8')
    if target.exists(): assert target.read_bytes()==raw
    else: target.write_bytes(raw)
    dest,result=R.compile_notes(target,'root')
    raw=json.dumps(result,ensure_ascii=False,indent=2).encode('utf-8')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): assert dest.read_bytes()==raw
    else: dest.write_bytes(raw)
    print(dict(status=result['status'],pages=result['actual_pages'],cells=result['actual_observations']))

if __name__=='__main__': main()
