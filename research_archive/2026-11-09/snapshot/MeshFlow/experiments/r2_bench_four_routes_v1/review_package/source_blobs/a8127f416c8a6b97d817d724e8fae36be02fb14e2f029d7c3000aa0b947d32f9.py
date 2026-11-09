"""Attach immutable page hashes to explicitly authored image observations."""
from . import common as C
from .review_record import AXES, LABELS


def run():
    pages=C.read(C.OUT/'visual_review/assignments.json')['backend_data']
    observations=[]
    for path in sorted((C.OUT/'visual_review/manual').glob('backend_data_observations_*.json')):
        observations.extend(C.read(path))
    assert len(observations)==len(pages)
    assert sorted(r['index'] for r in observations)==list(range(len(pages)))
    records=[]
    for observation in sorted(observations,key=lambda x:x['index']):
        page=pages[observation['index']]
        png=C.OUT/'gallery'/(page+'.png')
        receipt=png.with_suffix('.json')
        assert len(observation['cells'])==3
        cells=[]
        for label,values in zip(LABELS,observation['cells']):
            assert len(values)==6
            cell=dict(label=label,**dict(zip(AXES,values[:5])),note=values[5])
            assert all(cell[key] in enums for key,enums in AXES.items())
            cells.append(cell)
        records.append(dict(page_id=page,image_sha256=C.sha(png),image_receipt_sha256=C.sha(receipt),
            actually_viewed=True,view_event_id=observation['event'],viewed_at_utc=observation['utc'],
            views_and_sections_checked=True,cells=cells))
    doc=dict(reviewer='backend_data',assigned_page_ids=pages,records=records,
        provenance='Labels and notes explicitly authored after real view_image calls; this script only attaches page identity and formats already recorded observations.')
    C.write(C.OUT/'visual_review/manual/backend_data.json',doc)
    print({'status':'EXPLICIT_OBSERVATIONS_ASSEMBLED','pages':len(records)})


if __name__=='__main__':run()
