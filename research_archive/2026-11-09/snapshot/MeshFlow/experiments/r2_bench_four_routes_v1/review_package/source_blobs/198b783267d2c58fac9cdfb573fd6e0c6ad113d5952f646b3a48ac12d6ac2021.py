"""Package explicit root view notes into the frozen Bestof review schema."""
from .bestof import common as C
from .bestof import review_record as R

def main():
    folder=C.OUT/'visual_review/manual'
    sources=[folder/'root_raw_observations.json',folder/'root_post_observations.json']
    notes=[n for p in sources for n in C.read(p)['notes']]
    assignment=C.read(C.OUT/'visual_review/assignments.json')['assignments']['root']
    pages=assignment['page_ids'];protocol=C.read(C.OUT/'visual_review/protocol.json')
    assert [n['index'] for n in notes]==list(range(len(pages)))
    observations=[]
    for pid,note in zip(pages,notes):
        png=C.OUT/'gallery'/(pid+'.png');receipt=C.read(png.with_suffix('.json'))
        labels=[c['model'] for c in receipt['cells']]
        assert len(labels)==len(note['ratings']) and len(set(labels))==len(labels)
        ratings={}
        for label,values in zip(labels,note['ratings']):
            assert len(values)==6
            ratings[label]=dict(zip(protocol['axes'],values[:5]));ratings[label]['note']=values[5]
        observations.append(dict(page_id=pid,tool='view_image',view_event_id='root_bestof_view_image_'+str(note['index']),
            viewed_at_utc=note['viewed_at_utc'],actually_viewed=True,views_and_sections_checked=True,
            viewed_image_sha256=C.sha(png),ratings=ratings))
    target=folder/'root.json'
    C.immutable(target,dict(reviewer='root',reviewer_kind='AI_PROXY',blind=False,
        assigned_page_ids=pages,pages=observations,explicit_notes={str(p):C.sha(p) for p in sources}))
    dest,result=R.compile_notes(target,'root');C.immutable(dest,result)
    print({'status':result['status'],'pages':result['actual_pages'],'cells':result['actual_observations']})

if __name__=='__main__':main()
