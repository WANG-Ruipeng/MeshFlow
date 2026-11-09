"""Expand root's explicit tool-view observations; never infer visual labels."""
from .hybrid import common as C
from .hybrid import review_record as R

def main():
    source=C.OUT/'visual_review/manual/root_compact_observations.json'
    notes=C.read(source)['notes']
    pages=C.read(C.OUT/'visual_review/assignments.json')['root']
    assert [n['index'] for n in notes]==list(range(len(pages)))
    records=[]
    for page,note in zip(pages,notes):
        png=C.OUT/'gallery'/(page+'.png');receipt=png.with_suffix('.json')
        cells=[]
        for label,rating in zip(R.LABELS,note['ratings']):
            assert len(rating)==6
            cells.append(dict(label=label,**dict(zip(R.AXES,rating[:5])),note=rating[5]))
        records.append(dict(page_id=page,image_sha256=C.sha(png),image_receipt_sha256=C.sha(receipt),
            actually_viewed=True,view_event_id='root_hybrid_view_image_'+str(note['index']),
            viewed_at_utc=note['viewed_at_utc'],views_and_sections_checked=True,cells=cells))
    target=C.OUT/'visual_review/manual/root.json'
    C.write(target,dict(reviewer='root',assigned_page_ids=pages,records=records,explicit_notes_sha256=C.sha(source)))
    print(R.compile_records(target,'root'))

if __name__=='__main__':main()
