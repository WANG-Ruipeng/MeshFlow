"""Aggregate frozen actual image judgments, without rewriting observations."""
from . import common as C
from .review_record import AXES, LABELS
from collections import Counter


def run():
    done = C.OUT/'visual_review/summary.json'
    if done.exists():
        result = C.read(done)
        C.verify(result['files'])
        return result
    assignments = C.read(C.OUT/'visual_review/assignments.json')
    rows = []
    inputs = {}
    for reviewer in assignments:
        path = C.OUT/'visual_review/shards'/(reviewer+'.json')
        shard = C.read(path)
        assert shard['status'] == 'COMPLETE_ACTUAL_VIEW'
        assert shard['protocol_sha256'] == C.sha(C.OUT/'visual_review/protocol.json')
        assert shard['pages'] == len(assignments[reviewer])
        C.verify(shard['source_files'])
        rows.extend(shard['records'])
        inputs[str(path)] = C.sha(path)
    assert len(rows) == 138
    assert len({(r['request_id'], r['label']) for r in rows}) == 138
    assert all(r['actually_viewed'] and r['views_and_sections_checked'] for r in rows)
    def joint(r):
        return r['overall_shape']=='REASONABLE' and r['C_use']=='PLAUSIBLE' and r['support_retention']=='PRESENT'
    summary=[]
    for group in sorted({r['report_group'] for r in rows})+['ALL']:
        for label in LABELS:
            rs=[r for r in rows if r['label']==label and (group=='ALL' or r['report_group']==group)]
            row=dict(report_group=group,label=label,requests=len(rs),parents=len({r['parent_id'] for r in rs}),three_axis_joint=sum(joint(r) for r in rs))
            for axis,enums in AXES.items():
                counts=Counter(r[axis] for r in rs)
                row.update({axis+'__'+value:counts[value] for value in enums})
            summary.append(row)
    index={(r['request_id'],r['label']):r for r in rows}
    pairs=[]
    for rid in sorted({r['request_id'] for r in rows}):
        for ref,candidate in (('M_D0','HYBRID'),('RAW','HYBRID')):
            a,b=index[rid,ref],index[rid,candidate]
            row=dict(request_id=rid,parent_id=a['parent_id'],report_group=a['report_group'],reference=ref,candidate=candidate,
                reference_joint=joint(a),candidate_joint=joint(b),joint_fate='RETAINED' if joint(a) and joint(b) else 'GAINED' if joint(b) else 'LOST' if joint(a) else 'NEITHER')
            for axis in AXES:row.update({axis+'__reference':a[axis],axis+'__candidate':b[axis]})
            pairs.append(row)
    C.csvwrite(C.OUT/'visual_review/counts.csv',summary)
    C.csvwrite(C.OUT/'visual_review/paired_labels.csv',pairs)
    C.csvwrite(C.OUT/'visual_review/actual_records.csv',rows)
    result=dict(status='COMPLETE',actual_pages=46,actual_cells=138,reviewer_kind='AI_PROXY',blind=False,
        historical_ratings_reused=False,counts=summary,
        limitation='Three-axis intersection is a visual body/C-use/support judgment, not certification of no fragments, no intersections, shared topology, or independent human review.',
        files={**inputs,**{str(C.OUT/'visual_review'/name):C.sha(C.OUT/'visual_review'/name) for name in ('counts.csv','paired_labels.csv','actual_records.csv')}})
    C.write(done,result)
    return result


if __name__=='__main__':
    result=run()
    print({k:result[k] for k in ('status','actual_pages','actual_cells')})
