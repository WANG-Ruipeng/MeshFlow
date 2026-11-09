"""Read-only aggregation of completed zipper cells and selected face identities."""
from __future__ import annotations
import argparse,csv,json
from . import hinge as H


def table(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def collect():
    records=[];files={};rows=[];paired=[];summaries=[]
    for arm in H.ARMS:
        completion=H.OUT/(arm+'_completion.json');r=H.C.read(completion);H.B.verify(r['files'])
        files[str(completion)]=H.C.sha(completion)
        folder=H.OUT/'aggregate'/arm;reg=H.C.read(folder/'registry.json')
        records.extend(reg['records'])
        for name,target in [('per_output',rows),('paired_metrics',paired),('summary',summaries)]:target.extend(table(folder/(name+'.csv')))
    assert len(records)==168 and len({(r['label'],r['request_id']) for r in records})==168
    H.C.write(H.OUT/'registry.json',dict(version=H.VERSION,records=records,completion_files=files))
    for name,data in [('per_output',rows),('paired_metrics',paired),('summary',summaries)]:H.C.csvwrite(H.OUT/(name+'.csv'),data)
    result=dict(status='COMPLETE',cells=len(records),files={str(H.OUT/name):H.C.sha(H.OUT/name) for name in ('registry.json','per_output.csv','paired_metrics.csv','summary.csv')},source_completions=files)
    H.C.write(H.OUT/'completion.json',result);return result


def cases(arm):
    from . import topology as T
    rows=[];sources={}
    for rid,slots in [('608b46b0_n3',[77,113]),('3f741759_n2',[21])]:
        q=next(q for q in H.requests() if q['request_id']==rid)
        for d in (None,*H.DELTAS):
            name=arm if d is None else H.label(arm,d)
            if d is None:
                p=H.paths(arm,q);mesh=p['post'];score=p['score'];edge=score.parent/'valid_edges.csv';candidate=None
            else:
                dest=H.OUT/'cells'/name/rid;mesh=dest/'post.npz';edge=dest/'valid_edges.csv';score=dest/'score.json';candidate=H.C.read(dest/'candidate_records.json')
            for p in (mesh,edge,score):sources[str(p)]=H.C.sha(p)
            _,faces=T.measure(H.C.arrays(mesh),int(q['K']),T.edges_file(edge),H.C.L)
            for slot in slots:
                found=[r for r in faces if r['source_raw_slot']==slot]
                rows.append(dict(label=name,request_id=rid,source_raw_slot=slot,identity_scope='Within one generated RAW slot; cross-arm semantic counterpart not asserted',
                    status='RETAINED' if found else 'DELETED_BY_ORIGINAL_GEO',face=found[0] if found else None,
                    direct_candidate_records=[r for r in candidate if slot in r['source_raw_slots']] if candidate is not None else None,
                    affected_candidate_records=[r for r in candidate if slot in r.get('affected_source_raw_slots',[])] if candidate is not None else None))
    p=H.OUT/'cases'/(arm+'.json');H.C.write(p,dict(status='COMPLETE',records=rows,source_hashes=sources));return dict(path=str(p),records=len(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('collect','cases'));p.add_argument('--arm',choices=H.ARMS,default='PLAIN_S2000');a=p.parse_args()
    result=collect() if a.stage=='collect' else cases(a.arm);print(json.dumps(result),flush=True)
