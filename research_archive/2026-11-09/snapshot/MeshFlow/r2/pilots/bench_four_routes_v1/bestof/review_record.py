"""Compile explicit real-image observations; variable RAW4/POST3 cell labels."""
from __future__ import annotations
import argparse,collections,datetime as dt,json,re
from .common import *


def require(value,message):
    if not value:raise ValueError(message)


def make_assignments(reviewers):
    require(len(reviewers)==3 and len(set(reviewers))==3,'Three distinct reviewer names required')
    p=OUT/'visual_review/protocol.json';protocol=read(p);verify(protocol['protected'])
    require(protocol['status']=='FROZEN' and protocol['cells']==230,'Complete frozen protocol required')
    ordered=sorted({u['condition_id'] for u in protocol['expected_units']});assignments={}
    for owner,cids in zip(reviewers,(ordered[:8],ordered[8:16],ordered[16:])):
        units=[u for u in protocol['expected_units'] if u['condition_id'] in cids]
        pages=list(dict.fromkeys(u['page_id'] for u in units))
        require(len(pages)==3*len(cids) and len(units)==10*len(cids),'RAW4+BASE3+FINAL3 coverage')
        assignments[owner]=dict(reviewer=owner,condition_ids=cids,condition_count=len(cids),page_ids=pages,
            pages=len(pages),observations=len(units),notes_path=str(OUT/'visual_review/manual'/(owner+'.json')))
    result=dict(schema='bestof_all_candidates_review_assignments_v2',status='FROZEN',protocol_sha256=sha(p),
        assignment_rule='Sorted condition_id; first8/middle8/last7; no candidate quality inspection',
        total_pages=69,total_observations=230,reviewer_kind='AI_PROXY',blind=False,assignments=assignments,
        actual_observations_performed=0,labels_generated=0,metrics_read=0)
    immutable(OUT/'visual_review/assignments.json',result);return result


def compile_notes(path,shard):
    require(re.fullmatch(r'[a-z0-9_]+',shard) is not None,'Safe shard name required')
    path=resolve(path);notes=read(path)
    require(not path.is_relative_to(OUT/'visual_review/shards'),'Notes must live outside final shards')
    protocol_path=OUT/'visual_review/protocol.json';protocol=read(protocol_path);verify(protocol['protected'])
    assignment=read(OUT/'visual_review/assignments.json')
    require(assignment['protocol_sha256']==sha(protocol_path),'Assignment/protocol changed')
    owner=notes.get('reviewer');require(owner in assignment['assignments'],'Unassigned reviewer')
    require(notes.get('reviewer_kind')=='AI_PROXY' and notes.get('blind') is False,'Explicit nonblind AI_PROXY required')
    assigned=assignment['assignments'][owner]['page_ids']
    require(notes.get('assigned_page_ids')==assigned,'Exact frozen assigned_page_ids required')
    pages=notes.get('pages');require(isinstance(pages,list) and len(pages)==len(assigned),'All assigned pages required')
    ids=[p['page_id'] for p in pages];require(len(ids)==len(set(ids)) and set(ids)==set(assigned),'Page coverage mismatch')
    expected=collections.defaultdict(dict)
    for u in protocol['expected_units']:expected[u['page_id']][u['label']]=u
    rows=[];events=[]
    for page in pages:
        pid=page['page_id'];units=expected[pid]
        require(page.get('actually_viewed') is True and page.get('views_and_sections_checked') is True,'Actual complete visual observation required')
        require(page.get('tool')=='view_image','view_image event required')
        require(isinstance(page.get('view_event_id'),str) and page['view_event_id'].strip(),'Real view event ID required')
        stamp=dt.datetime.fromisoformat(page['viewed_at_utc'].replace('Z','+00:00'))
        require(stamp.tzinfo is not None and stamp.utcoffset()==dt.timedelta(0),'Explicit UTC timestamp required')
        require(set(page.get('ratings',{}))==set(units),'All four RAW or all three POST cells must be rated')
        png=OUT/'gallery'/(pid+'.png');receipt_path=png.with_suffix('.json');receipt=read(receipt_path)
        require(receipt['page_id']==pid and receipt['image_sha256']==sha(png),'Sealed image identity differs')
        if 'viewed_image_sha256' in page:require(page['viewed_image_sha256']==sha(png),'Actually viewed image changed')
        for label,u in units.items():
            rating=page['ratings'][label]
            require(set(rating)==set(protocol['axes'])|{'note'},'Exactly five axes and one concrete note required')
            for axis,allowed in protocol['axes'].items():require(rating[axis] in allowed,'Invalid visual axis')
            require(isinstance(rating['note'],str) and rating['note'].strip(),'Per-cell concrete observation required')
            require(any(c['model']==label and c['request_id']==u['request_id'] and c['stage']==u['stage'] for c in receipt['cells']),
                    'Rendered cell not registered')
            rows.append(dict(u,**rating,reviewer=owner,reviewer_kind='AI_PROXY',blind=False,actually_viewed=True,
                views_and_sections_checked=True,view_event_id=page['view_event_id']+':'+label,viewed_at_utc=page['viewed_at_utc'],
                image_sha256=sha(png),image_receipt_sha256=sha(receipt_path),gallery_definition_sha256=receipt['definition_sha256']))
        events.append(dict(page_id=pid,tool='view_image',view_event_id=page['view_event_id'],viewed_at_utc=page['viewed_at_utc'],
            actually_viewed=True,views_and_sections_checked=True,image_sha256=sha(png),image_receipt_sha256=sha(receipt_path)))
    destination=OUT/'visual_review/shards'/(shard+'.json')
    keys={(r['page_id'],r['label']) for r in rows}
    for other in destination.parent.glob('*.json'):
        if other!=destination:require(not keys & {(r['page_id'],r['label']) for r in read(other)['rows']},'Overlapping shard ownership')
    result=dict(schema=protocol['schema'],status='COMPLETE_ASSIGNED_SHARD',reviewer=owner,reviewer_kind='AI_PROXY',blind=False,
        actual_pages=len(pages),actual_observations=len(rows),labels_inferred=0,metrics_read=0,historical_ratings_read=0,
        source_manual_notes=str(path),source_manual_notes_sha256=sha(path),protocol_sha256=sha(protocol_path),
        assignment_sha256=sha(OUT/'visual_review/assignments.json'),interpretation_version=2,historical_label_pooling=False,
        helper_sha256=sha(__file__),view_events=events,rows=rows)
    return destination,result


def main():
    p=argparse.ArgumentParser();p.add_argument('--make-assignments',action='store_true');p.add_argument('--reviewers',nargs=3)
    p.add_argument('--input');p.add_argument('--shard');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    if a.make_assignments:
        require(a.reviewers,'--reviewers is explicit root assignment policy')
        r=make_assignments(a.reviewers);print(json.dumps({k:r[k] for k in ('status','total_pages','total_observations')}));return
    require(a.input and a.shard,'--input and --shard required')
    dest,r=compile_notes(a.input,a.shard)
    if not a.validate_only:immutable(dest,r)
    print(json.dumps(dict(status='VALIDATED_NO_WRITE' if a.validate_only else 'WRITTEN',path=str(dest),
        actual_pages=r['actual_pages'],actual_observations=r['actual_observations'],labels_inferred=0,metrics_read=0)))


if __name__=='__main__':main()
