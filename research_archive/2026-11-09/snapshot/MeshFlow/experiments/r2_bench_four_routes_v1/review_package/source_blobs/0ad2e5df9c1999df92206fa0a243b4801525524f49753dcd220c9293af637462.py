"""Compile explicit actual-image observations, never create visual labels."""
from . import common as C
import argparse

LABELS=('RAW','M_D0','HYBRID')
AXES={'overall_shape':('REASONABLE','INCOMPLETE','UNCERTAIN'),
      'C_use':('PLAUSIBLE','REPLACED_OR_IGNORED','UNCERTAIN'),
      'support_retention':('PRESENT','MISSING_OR_DAMAGED','UNCERTAIN'),
      'visible_fragment_or_gap':('YES','NO_OBVIOUS','UNCERTAIN'),
      'layering_or_fold':('YES','NO_OBVIOUS','UNCERTAIN')}

def prepare():
    qs=[t['request'] for t in C.read(C.OUT/'registration.json')['tasks']]
    definition=C.read(C.OLD/'visual_review/v2_definition.json')
    C.write(C.OUT/'visual_review/v2_definition.json',definition)
    pages=['CLEAN_FINAL_POST__'+q['request_id'] for q in qs]
    protocol=dict(status='FROZEN',version='HYBRID_V2_actual_nonblind_review_v1',labels=list(LABELS),axes=AXES,
        definition_sha256=C.sha(C.OUT/'visual_review/v2_definition.json'),definition=definition,
        pages=pages,expected_cells=138,primary_stage='CLEAN_FINAL_POST',RAW_reference_is_same_generated_RAW=True,
        all_labels_freshly_reviewed=True,historical_labels_reused=False,reviewer_kind='AI_PROXY',blind=False,
        instructions=['Actually open every assigned main PNG and inspect all three columns, both views and fixed sections.',
            'Apply V2 support PRESENT only when major support surfaces are present, not merely lower endpoints or remnant needles.',
            'Five axes independent; retain uncertainty. No unique source-style target. RAW reference, original M FINAL and hybrid FINAL separately rated.',
            'No label derived from metric, seed count, file name, or old rating. BASE supplemental optional; actually viewed sources recorded separately.'],
        record_schema=dict(reviewer='backend_data or root',assigned_page_ids='complete assigned list',records=[dict(
            page_id='CLEAN_FINAL_POST__request_id',image_sha256='...',image_receipt_sha256='...',
            actually_viewed=True,view_event_id='actual view_image event identifier',viewed_at_utc='UTC ISO',views_and_sections_checked=True,
            cells=[dict(label='RAW/M_D0/HYBRID',overall_shape='enum',C_use='enum',support_retention='enum',visible_fragment_or_gap='enum',layering_or_fold='enum',note='specific observed evidence')])]))
    C.write(C.OUT/'visual_review/protocol.json',protocol)
    assignments=dict(root=pages[:14],backend_data=pages[14:])
    C.write(C.OUT/'visual_review/assignments.json',assignments)
    return protocol

def compile_records(path,shard):
    protocol=prepare();doc=C.read(path);assigned=C.read(C.OUT/'visual_review/assignments.json')[shard]
    if doc['reviewer']!=shard or doc['assigned_page_ids']!=assigned:raise ValueError('Assignment differs')
    if len(doc['records'])!=len(assigned) or {r['page_id'] for r in doc['records']}!=set(assigned):raise ValueError('Missing or duplicate reviewed page')
    rows=[];sources={str(C.resolve(path)):C.sha(path)}
    for record in doc['records']:
        page=record['page_id'];png=C.OUT/'gallery'/(page+'.png');receipt=png.with_suffix('.json');rr=C.read(receipt)
        if not record['actually_viewed'] or not record['views_and_sections_checked'] or not record['view_event_id'] or not record['viewed_at_utc']:raise ValueError('Actual view evidence absent')
        if record['image_sha256']!=C.sha(png) or record['image_receipt_sha256']!=C.sha(receipt):raise ValueError('Viewed image identity differs')
        if rr['image_sha256']!=record['image_sha256']:raise ValueError('Image receipt differs')
        if len(record['cells'])!=3 or {r['label'] for r in record['cells']}!=set(LABELS):raise ValueError('Three required independent cells absent')
        for cell in record['cells']:
            if any(cell[key] not in enums for key,enums in AXES.items()) or not cell['note'].strip():raise ValueError('Missing visual observation/enum')
            rows.append(dict(**{k:v for k,v in record.items() if k!='cells'},**cell,request_id=rr['request_id'],parent_id=rr['parent_id'],report_group=rr['report_group'],
                reviewer=shard,reviewer_kind='AI_PROXY',blind=False,stage='RAW' if cell['label']=='RAW' else 'CLEAN_FINAL_POST'))
        sources[str(png)]=C.sha(png);sources[str(receipt)]=C.sha(receipt)
    result=dict(status='COMPLETE_ACTUAL_VIEW',reviewer=shard,pages=len(assigned),cells=len(rows),records=rows,
        protocol_sha256=C.sha(C.OUT/'visual_review/protocol.json'),source_files=sources,labels_generated_from_metrics=False)
    C.write(C.OUT/'visual_review/shards'/(shard+'.json'),result);return dict(status=result['status'],pages=len(assigned),cells=len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--shard');a=p.parse_args()
    print(compile_records(a.input,a.shard) if a.input else prepare())
