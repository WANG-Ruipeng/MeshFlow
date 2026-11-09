"""Transcribe explicit actual-view notes into one immutable review shard.

No models, mesh metrics, historical ratings, or automatic labels are read.
The tool validates a reviewer's attestation; it cannot itself prove a view event.

Input JSON shape (all indicated fields must be explicitly supplied):
  {"reviewer":"agent_name", "reviewer_kind":"AI_PROXY", "blind":false,
   "pages":[{"page_id":"<registered page>", "tool":"view_image",
      "view_event_id":"<actual tool-view event>", "viewed_at_utc":"<ISO UTC>",
      "actually_viewed":true, "views_and_sections_checked":true,
      "ratings":{"B":{<all five axes>,"note":"concrete visible evidence"},
                 "S":{<all five axes>,"note":"concrete visible evidence"}}}]}
Required assigned_page_ids must exactly match the frozen reviewer assignment.

Usage: python -B -m r2.pilots.bench_four_routes_v1.seam.review_record
       --input <manual-notes.json> --shard <agent_name> [--validate-only]
Input notes should live outside visual_review/shards; that folder is only for
completed schema-compatible shards. Existing shards are never overwritten.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'experiments/r2_bench_four_routes_v1/seam'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    value=str(path).replace('\\','/')
    for prefix in ('E:/MeshFlow/','/mnt/e/MeshFlow/'):
        if value.startswith(prefix):return ROOT/value[len(prefix):]
    return Path(value).resolve()


def require(condition,message):
    if not condition:raise ValueError(message)


def make_assignments():
    """Freeze condition-level ownership without inspecting outputs."""
    protocol_path=OUT/'visual_review/protocol.json';protocol=read(protocol_path)
    require(protocol['status']=='FROZEN' and protocol['interpretation_version']==2,'Frozen V2 review contract required')
    require(protocol['labels']==['B','S'],'Only B/S has a new visual comparison')
    ordered={}
    for unit in protocol['expected_units']:
        ordered.setdefault(unit['condition_id'],[]).append(unit)
    require(len(ordered)==23,'Exactly23 registered conditions required')
    ids=list(ordered);assignments={};all_pages=[]
    for owner,cids in zip(('backend_runtime','n_range_audit','root'),(ids[:8],ids[8:16],ids[16:])):
        pages=[];requests=[];conditions=[]
        for cid in cids:
            units=ordered[cid]
            require({r['noise_index'] for r in units}=={0,1},'Two fixed noises per C required')
            require({r['stage'] for r in units}=={'RAW','BASE_POST','CLEAN_FINAL_POST'},'All three stages required')
            cpages=list(dict.fromkeys(r['page_id'] for r in units));crequests=list(dict.fromkeys(r['request_id'] for r in units))
            require(len(cpages)==6 and len(crequests)==2 and len(units)==12,'Condition review coverage differs')
            pages.extend(cpages);requests.extend(crequests)
            conditions.append(dict(condition_id=cid,parent_id=units[0]['parent_id'],report_group=units[0]['report_group'],request_ids=crequests))
        assignments[owner]=dict(reviewer=owner,conditions=conditions,condition_ids=cids,condition_count=len(cids),
            request_ids=requests,page_ids=pages,pages=len(pages),observations=2*len(pages),
            notes_path=str(OUT/'visual_review/manual'/(owner+'.json')),shard=owner)
        all_pages.extend(pages)
    require(len(all_pages)==len(set(all_pages))==138,'Assignment overlap or missing pages')
    result=dict(schema='bench_four_routes_seam_actual_review_assignments_v2',status='FROZEN',
        protocol_sha256=sha(protocol_path),interpretation_version=2,labels=['B','S'],
        assignment_rule='Original expected-unit condition order; first8/middle8/last7; no output or metric inspection',
        conditions=23,total_pages=138,total_observations=276,reviewer_kind='AI_PROXY',blind=False,
        assignments=assignments,actual_observations_performed=0,labels_generated=0,metrics_read=0)
    path=OUT/'visual_review/assignments.json'
    if path.exists():require(read(path)==result,'Frozen assignments differ')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    return result


def validate_page(page,axes,labels):
    """Reject omissions, defaults and inferred labels before reading any image."""
    require(page.get('actually_viewed') is True,'Explicit actually_viewed:true required')
    require(page.get('views_and_sections_checked') is True,'Explicit views_and_sections_checked:true required')
    require(page.get('tool')=='view_image','Actual tool must be explicitly view_image')
    for field in ('page_id','view_event_id','viewed_at_utc'):
        require(isinstance(page.get(field),str) and bool(page[field].strip()),'Missing '+field)
    stamp=datetime.datetime.fromisoformat(page['viewed_at_utc'].replace('Z','+00:00'))
    require(stamp.tzinfo is not None and stamp.utcoffset()==datetime.timedelta(0),'View time must explicitly be UTC')
    require(set(page.get('ratings',{}))==set(labels),'Explicit B/S ratings required')
    for label in labels:
        rating=page['ratings'][label]
        require(isinstance(rating,dict) and set(rating)==set(axes)|{'note'},'Each cell needs exactly five axes and note')
        for axis,allowed in axes.items():require(rating[axis] in allowed,'Missing/invalid '+label+' '+axis)
        require(isinstance(rating['note'],str) and bool(rating['note'].strip()),'Concrete per-cell note required')


def compile_notes(input_path,shard):
    require(re.fullmatch(r'[a-z0-9_]+',shard) is not None,'Safe lowercase shard name required')
    input_path=resolve(input_path);notes=read(input_path)
    require(not input_path.is_relative_to(OUT/'visual_review/shards'),'Manual input must stay outside completed-shards directory')
    protocol_path=OUT/'visual_review/protocol.json';protocol=read(protocol_path)
    require(protocol['status']=='FROZEN' and protocol['interpretation_version']==2,'Review V2 protocol not frozen')
    require(protocol['labels']==['B','S'],'Only B/S actual ratings permitted')
    require(notes.get('reviewer_kind')=='AI_PROXY' and notes.get('blind') is False,'Explicit AI_PROXY and blind:false required')
    require(isinstance(notes.get('reviewer'),str) and bool(notes['reviewer'].strip()),'Explicit reviewer identity required')
    pages=notes.get('pages');require(isinstance(pages,list) and len(pages)>0,'Nonempty actual page records required')
    ids=[p.get('page_id') for p in pages];require(len(ids)==len(set(ids)),'Duplicate page records')
    assignments=read(OUT/'visual_review/assignments.json')
    require(assignments['protocol_sha256']==sha(protocol_path),'Assignment/protocol mismatch')
    require(notes['reviewer'] in assignments['assignments'],'Reviewer not assigned')
    assigned=assignments['assignments'][notes['reviewer']]['page_ids']
    require(notes.get('assigned_page_ids')==assigned,'Explicit frozen assigned_page_ids required')
    require(len(assigned)==len(set(assigned)) and set(ids)==set(assigned),'Assigned page coverage differs')
    expected={}
    for unit in protocol['expected_units']:expected.setdefault(unit['page_id'],{})[unit['label']]=unit
    require(not (set(ids)-set(expected)),'Unregistered page identity')
    rows=[];events=[]
    for page in pages:
        validate_page(page,protocol['axes'],protocol['labels'])
        pid=page['page_id'];png=OUT/'gallery'/(pid+'.png');receipt_path=png.with_suffix('.json')
        receipt=read(receipt_path);image_hash=sha(png);receipt_hash=sha(receipt_path)
        require(receipt['page_id']==pid and receipt['image_sha256']==image_hash,'Rendered image/receipt identity mismatch')
        if 'viewed_image_sha256' in page:
            require(page['viewed_image_sha256']==image_hash,'Viewed bytes differ from sealed image')
        for label in protocol['labels']:
            unit=expected[pid][label];rating=page['ratings'][label]
            require(any(c['model']==label and c['request_id']==unit['request_id'] and c['stage']==unit['stage'] for c in receipt['cells']),
                    'Rendered cell identity differs from protocol')
            row=dict(unit,**rating,reviewer=notes['reviewer'],reviewer_kind=notes['reviewer_kind'],blind=notes['blind'],
                     actually_viewed=page['actually_viewed'],views_and_sections_checked=page['views_and_sections_checked'],
                     view_event_id=page['view_event_id']+':'+label,viewed_at_utc=page['viewed_at_utc'],
                     image_sha256=image_hash,image_receipt_sha256=receipt_hash,
                     gallery_definition_sha256=receipt['definition_sha256'])
            require(set(protocol['required_fields'])<=set(row),'Required output field absent')
            rows.append(row)
        events.append({**{k:page[k] for k in ('page_id','tool','view_event_id','viewed_at_utc','actually_viewed','views_and_sections_checked')},
                       'image_sha256':image_hash,'image_receipt_sha256':receipt_hash})
    dest=OUT/'visual_review/shards'/(shard+'.json')
    keys={(r['request_id'],r['label'],r['stage']) for r in rows}
    for other in dest.parent.glob('*.json'):
        if other==dest:continue
        overlap=keys & {(r['request_id'],r['label'],r['stage']) for r in read(other)['rows']}
        require(not overlap,'Duplicate review ownership with '+other.name)
    result=dict(schema=protocol['schema'],status='COMPLETE_ASSIGNED_SHARD',
                reviewer=notes['reviewer'],reviewer_kind=notes['reviewer_kind'],blind=notes['blind'],
                actual_pages=len(pages),actual_observations=len(rows),
                labels_inferred=0,metrics_read=0,historical_ratings_read=0,
                source_manual_notes=str(input_path),source_manual_notes_sha256=sha(input_path),
                protocol_sha256=sha(protocol_path),assignment_sha256=sha(OUT/'visual_review/assignments.json'),
                interpretation_version=2,historical_label_pooling=False,
                helper_sha256=sha(__file__),view_events=events,rows=rows)
    return dest,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input');parser.add_argument('--shard')
    parser.add_argument('--make-assignments',action='store_true')
    parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    if args.make_assignments:
        result=make_assignments();print(json.dumps({k:result[k] for k in ('status','conditions','total_pages','total_observations')}));return
    require(args.input and args.shard,'--input and --shard required for explicit actual notes')
    dest,result=compile_notes(args.input,args.shard)
    if args.validate_only:status='VALIDATED_NO_WRITE'
    elif dest.exists():
        require(read(dest)==result,'Existing actual-review shard differs; refusing overwrite')
        status='EXISTING_IDENTICAL'
    else:
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
        status='WRITTEN'
    print(json.dumps(dict(status=status,path=str(dest),actual_pages=result['actual_pages'],
                         actual_observations=result['actual_observations'],labels_inferred=0,metrics_read=0),indent=2))


if __name__=='__main__':main()
