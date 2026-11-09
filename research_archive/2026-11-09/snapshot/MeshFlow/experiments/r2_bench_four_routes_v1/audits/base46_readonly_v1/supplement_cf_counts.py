"""Read the same frozen BASE rows for CF counts and I support identity only."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[3]
sources = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    sources[str(path)] = sha(path)
    return value


def csvwrite(name, rows):
    with (DEST/name).open('x', encoding='utf-8', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run():
    if (DEST/'cf_counts_receipt.json').exists():
        raise ValueError('Supplement exists; no overwrite')
    original = read(DEST/'source_identities.json')['files']
    reg_path = ROOT/'experiments/r2_bench_four_routes_v1/hybrid/registration.json'
    reg=read(reg_path)
    if sources[str(reg_path)] != original[str(reg_path)]:
        raise ValueError('Registration changed since BASE46 audit')
    rows=[]; supports=[]
    identity_fields=('I_status','ConflictCov_C_status','ConflictCov_C_interior_samples',
                     'ConflictCov_C_interior_weight_fraction','C_query_weight_denominator')
    for task in reg['tasks']:
        q=task['request'];rid=q['request_id']
        paths={'M':Path(task['paths']['BASE_POST_row']),
               'HYBRID':ROOT/'experiments/r2_bench_four_routes_v1/hybrid/cpu/HYBRID'/rid/'BASE_POST/scores/row.json'}
        pair={label:read(path) for label,path in paths.items()}
        if any(sources[str(p)] != original[str(p)] for p in paths.values()):
            raise ValueError('Row changed since BASE46 audit')
        row={k:q[k] for k in ('request_id','condition_id','parent_id','noise_index','report_group')}
        row['stage']='BASE_POST'
        for label,r in pair.items():
            if r['bridge_face_count'] != 0 or r['bridge_CF_illegal_affected_count'] != 0:
                raise ValueError('BASE unexpectedly contains bridge role; fixed-parent count no longer whole free count')
            if r['fixed_parent_free_face_denominator'] != r['actual_free_faces'] or r['free_face_denominator'] != r['actual_free_faces']:
                raise ValueError('Original free denominators differ')
            count=r['fixed_parent_CF_illegal_affected_count'];n=r['actual_free_faces']
            if abs(count/n-r['CF_illegal_affected_free_fraction'])>1e-12:
                raise ValueError('Existing affected count/fraction disagree')
            if r['CF_illegal_pair_count'] != r['fixed_parent_CF_illegal_unique_pair_count']:
                raise ValueError('Existing pair counters disagree')
            row[label+'_CF_affected_free_count']=count
            row[label+'_CF_illegal_pair_count']=r['CF_illegal_pair_count']
            row[label+'_actual_free_faces']=n
            row[label+'_CF_affected_free_fraction']=r['CF_illegal_affected_free_fraction']
        for key in ('CF_affected_free_count','CF_illegal_pair_count','actual_free_faces','CF_affected_free_fraction'):
            row['delta_'+key]=row['HYBRID_'+key]-row['M_'+key]
        rows.append(row)
        checks={key:pair['M'][key]==pair['HYBRID'][key] for key in identity_fields}
        supports.append(dict(request_id=rid,parent_id=q['parent_id'],report_group=q['report_group'],
                             common_PASS=pair['M']['I_status']==pair['HYBRID']['I_status']=='PASS',
                             M_I_status=pair['M']['I_status'],HYBRID_I_status=pair['HYBRID']['I_status'],
                             field_identity=checks,
                             M_support={key:pair['M'][key] for key in identity_fields},
                             HYBRID_support={key:pair['HYBRID'][key] for key in identity_fields},
                             frozen_C_query_sha256=q['query_descriptor']['sha256']))
    if not all(all(r['field_identity'].values()) for r in supports):
        raise ValueError('I support identity changed')
    metrics=[k for k in rows[0] if k.startswith(('M_','HYBRID_','delta_'))]
    parents=defaultdict(list)
    for r in rows:parents[r['parent_id']].append(r)
    if len(rows)!=46 or len(parents)!=23 or any(len(v)!=2 for v in parents.values()):
        raise ValueError('Expected46 requests/23 parents')
    parent_rows=[dict(parent_id=pid,report_group=rs[0]['report_group'],stage='BASE_POST',
                      **{key:mean(r[key] for r in rs) for key in metrics}) for pid,rs in parents.items()]
    group_rows=[]
    for group in ['ALL']+list(dict.fromkeys(r['report_group'] for r in rows)):
        selected=[r for r in rows if group=='ALL' or r['report_group']==group]
        ps=[p for p in parent_rows if group=='ALL' or p['report_group']==group]
        for metric in metrics:
            group_rows.append(dict(report_group=group,stage='BASE_POST',metric=metric,requests=len(selected),
                                   parents=len(ps),parent_macro_mean=mean(p[metric] for p in ps),
                                   request_total=sum(r[metric] for r in selected) if not metric.endswith('_fraction') else None))
    summary=dict(status='PASS_BASE_ONLY_COUNT_AND_SUPPORT_SUPPLEMENT',utc=datetime.now(timezone.utc).isoformat(),
                 stage='BASE_POST',rows=46,parents=23,
                 count_definition='Read fixed_parent_CF_illegal_affected_count directly. Both BASE outputs have zero BRIDGE faces/affected bridge faces, and fixed-parent denominator equals all actual free faces.',
                 pair_definition='Read CF_illegal_pair_count directly, checked equal to fixed_parent_CF_illegal_unique_pair_count for all92 rows.',
                 aggregation='Parent macro averages the two fixed noise counts within each parent. Request totals are separately summed; no mesh pooling is relabeled as parent macro.',
                 I_support=dict(all46_request_status_and_support_fields_identical=True,checked_fields=identity_fields,
                                common_PASS_requests=sum(r['common_PASS'] for r in supports),
                                common_PASS_parents=len({r['parent_id'] for r in supports if r['common_PASS']}),
                                status_counts=dict(Counter(r['M_I_status'] for r in supports)),
                                excluded_parent_ids=sorted({r['parent_id'] for r in supports if not r['common_PASS']}),
                                requests=supports),
                 all_group_values=[r for r in group_rows if r['report_group']=='ALL'],
                 interpretation='Lower parent-mean CF fraction is assessed alongside affected-face and pair counts. This supplement does not assert a causal denominator-only explanation or final-stage behavior.',
                 model_calls=0,scoring_calls=0,geometry_calls=0,final_chain_read=False,visual_labels_read=False)
    for path,digest in sources.items():
        if sha(path)!=digest:raise ValueError('Source changed during supplement')
    csvwrite('cf_counts_per_request.csv',rows)
    csvwrite('cf_counts_per_parent.csv',parent_rows)
    csvwrite('cf_counts_group_summary.csv',group_rows)
    with (DEST/'cf_counts_support.json').open('x',encoding='utf-8') as handle:json.dump(summary,handle,indent=2)
    files={str(DEST/name):sha(DEST/name) for name in ('cf_counts_per_request.csv','cf_counts_per_parent.csv','cf_counts_group_summary.csv','cf_counts_support.json')}
    receipt=dict(status='PASS_BASE46_CF_COUNTS_AND_SUPPORT',source_rows_unchanged_from_BASE46=True,
                 prior_receipt_sha256=sha(DEST/'receipt.json'),script_sha256=sha(__file__),files=files,
                 source_files=sources,model_calls=0,scoring_calls=0,geometry_calls=0,final_result_claim=False)
    with (DEST/'cf_counts_receipt.json').open('x',encoding='utf-8') as handle:json.dump(receipt,handle,indent=2)
    print(json.dumps(dict(status=summary['status'],I_support=summary['I_support']['status_counts'],
                          all_group_values=summary['all_group_values'])))


if __name__=='__main__':run()
