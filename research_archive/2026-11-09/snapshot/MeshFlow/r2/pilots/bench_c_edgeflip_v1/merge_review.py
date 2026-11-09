"""Validate and collect the completed human-visible AI_PROXY observations."""
from collections import Counter,defaultdict
from .common import *

def main():
    paths=[OUT/'visual_review'/('candidates_'+p+'.json') for p in ('459','bcb','f495')]
    cells=read(OUT/'candidate_evaluation/tasks.json')+read(OUT/'candidate_evaluation/baseline.json')
    ids={c['cell_id']:c for c in cells};expected={(c,s) for c in ids for s in ('RAW','CLEAN_FINAL_POST')}
    rows=[]
    for path in paths:rows+=read(path)['records']
    assert len(rows)==60 and {(r['cell_id'],r['stage']) for r in rows}==expected
    for r in rows:
        assert r['actually_viewed'] and r['reviewer']=='AI_PROXY' and r['blind'] is False
        assert sha(r['source_image'])==r['source_image_sha256']
        c=ids[r['cell_id']]
        r.update(label=c['label'],representation=c['representation'],request_id=c['q']['request_id'],parent_id=c['q']['parent_id'],split=c['q']['split'])
        r['three_axis_intersection']=r['adoption']=='PLAUSIBLE' and r['body']=='REASONABLE' and r['support']=='PRESENT'
    rows.sort(key=lambda r:(r['representation'],r['request_id'],r['stage'],r['label']))
    assert len({r['source_image_sha256'] for r in rows})==20
    by=defaultdict(list)
    for r in rows:by[r['representation'],r['stage'],r['label']].append(r)
    summary=[]
    for (rep,s,m),rr in sorted(by.items()):
        x=dict(representation=rep,stage=s,label=m,n=len(rr),three_axis_intersection=sum(r['three_axis_intersection'] for r in rr))
        for field in ('adoption','body','support','fragment_evidence'):
            for value,num in Counter(r[field] for r in rr).items():x[field+'__'+value]=num
        summary.append(x)
    write(OUT/'visual_review/candidates_actual.json',dict(status='COMPLETE',reviewer='AI_PROXY',blind=False,reviewed_cells=60,reviewed_pages=20,
        source_reviews={str(p):sha(p) for p in paths},records=rows,
        limits='Nonblind model-assisted visual observations, not independent humans. Indexed graphs and section endpoints do not certify visible fragments. Historical labels are not overwritten.'))
    csvwrite(OUT/'visual_review/candidates_actual.csv',rows)
    csvwrite(OUT/'visual_review/summary.csv',summary)
    print(summary)

if __name__=='__main__':main()
