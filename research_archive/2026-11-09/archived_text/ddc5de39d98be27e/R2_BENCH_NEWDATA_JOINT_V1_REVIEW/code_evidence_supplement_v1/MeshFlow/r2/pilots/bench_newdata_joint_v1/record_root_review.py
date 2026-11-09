"""Serialize root's actual, explicit visual observations; never infer labels."""
from r2.pilots.bench_newdata_joint_v1 import visual_review as v
import datetime,csv
def run():
    notes=v.read(v.REVIEW/'root_manual_observations.json')
    schema=v.read(v.REVIEW/'schema.json')
    assignments=v.read(v.REVIEW/'assignments.json')
    pages=assignments['tasks']['a_new_and_flat']['expected_pages']
    index={(p['request_id'],p['stage']):p for p in pages}
    assert len(notes)==len(pages)==24
    path=v.REVIEW/'review_a_new_and_flat.csv'
    assert not list(csv.DictReader(path.open(encoding='utf-8-sig')))
    rows=[]
    for i,n in enumerate(notes):
        p=index[(n['prefix'],n['stage'])]
        ts=datetime.datetime.strptime(n['viewed_at_utc'],'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc).isoformat()
        for label,obs in zip(v.LABELS,n['observations']):
            assert len(obs)==6
            row={k:p[k] for k in ('review_group','request_id','condition_id','parent_id','noise_index','eval_group','split','part_type','stage','page_id','image_path')}
            row.update(schema=v.SCHEMA,protocol_sha256=v.sha(v.REVIEW/'protocol.md'),assignments_sha256=v.sha(v.REVIEW/'assignments.json'),
                reviewer='/root',reviewer_kind='AI_PROXY',blind='false',review_round='root_A_v1',
                view_event_id=f'root_A_v1_page_{i+1:02d}',viewed_at_utc=ts,label=label,
                image_sha256=v.sha(p['image_path']),image_receipt_sha256=v.sha(p['image_receipt']),
                gallery_definition_sha256=v.sha(v.OUT/'gallery/definition.json'),
                actual_view_source='functions.exec -> tools.view_image: actual paired PNG inspection',
                actually_viewed='true',views_and_sections_checked='true',note=obs[5])
            row.update(zip(v.AXES,obs[:5]));rows.append(row)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=v.FIELDS);w.writeheader();w.writerows(rows)
    print({'actual_view_cells':len(rows),'pages':len(notes)})
if __name__=='__main__':run()
