"""Read-only table reductions, with explicit closed-C interface applicability.

No models, geometry modifications, rescoring, or replacement of original fields.
"""
import collections
import json
import statistics
from .delivery import OUT, read, rows, sha, csvwrite, write


def run():
    sources = [OUT/'results/metric_rows.json', OUT/'data/eval_manifest.json',
               OUT/'results/analysis_v1/primary_parent_macro.csv']
    data = read(sources[0]); requests = {q['request_id']: q for q in read(sources[1])['requests']}
    reduced = []
    groups = collections.defaultdict(list)
    for r in data:
        groups[r['label'], r['stage'], r['report_group']].append(r)
        assert int(r['C_boundary_edges']) == int(requests[r['request_id']]['boundary_edges'])
    for (label, stage, group), rr in sorted(groups.items()):
        eligible = [r for r in rr if int(r['C_boundary_edges']) > 0]
        by = collections.defaultdict(list)
        for r in eligible:
            by[r['parent_id']].append(float(r['C_F_has_valid_edge']))
        reduced.append(dict(label=label, stage=stage, report_group=group,
            all_requests=len(rr), open_C_requests=len(eligible),
            closed_C_NA_requests=len(rr)-len(eligible),
            open_C_parents=len(by),
            open_C_requests_with_legal_edge=sum(bool(r['C_F_has_valid_edge']) for r in eligible),
            open_C_parent_macro_has_legal_edge=statistics.fmean(statistics.fmean(v) for v in by.values()) if by else None,
            scope='Descriptive existing legal edge; open boundary does not imply every boundary must be closed'))
    perparent=[]
    keys = ('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction',
            'ConflictCov_C','free_total_surface_area','actual_free_faces','deleted_free_faces',
            'free_minimum_angle_lt5_fraction','C_F_valid_edge_total_length',
            'hinge_diag_legal_opposed_edge_C_reached_free_area_fraction','nonmanifold_edges')
    metrics=rows(sources[2])
    for r in metrics:
        if r['metric'] not in keys: continue
        for parent, value in json.loads(r['parent_values']).items():
            perparent.append(dict(label=r['label'],stage=r['stage'],report_group=r['report_group'],
                                  parent_id=parent,metric=r['metric'],value=value))
    dest=OUT/'results/delivery_v1'
    csvwrite(dest/'interface_applicability.csv', reduced)
    csvwrite(dest/'selected_per_parent.csv', perparent)
    write(dest/'APPLICABILITY.md',
        '原177字段逐值保留。本文件只把预先规定的闭合C接口NA合同用于交付摘要，不修改评分。'
        '77aa与617没有开放C边，不能将未获得共边计作接续失败，也不能把空边界计为成功。'
        '其余开放C上的合法共边为描述性存在统计，不要求整个边界闭合，不等于用途合理或主体完整。'
        'I仍只汇总原PASS支持域；没有把LOW_SUPPORT改成零。\n')
    receipt=dict(status='COMPLETE',definition='DELIVERY_APPLICABILITY_V1',
        model_calls=0,scoring_calls=0,original_metrics_modified=False,
        source_hashes={str(p):sha(p) for p in sources},
        output_hashes={str(dest/n):sha(dest/n) for n in
                      ('interface_applicability.csv','selected_per_parent.csv','APPLICABILITY.md')})
    write(dest/'readout_receipt.json',receipt)
    print(json.dumps({'status':'COMPLETE','interface_rows':len(reduced),'parent_rows':len(perparent)}))


if __name__ == '__main__': run()
