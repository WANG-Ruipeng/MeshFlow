"""Read-only cross-branch audit of reused meshes, scores and actual ratings.

No scorer, geometry loader, model, or image-labeling function is imported.
Original ratings are never reconciled. Each run writes a new named snapshot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_bench_four_routes_v1'
AXES = ('overall_shape', 'C_use', 'support_retention',
        'visible_fragment_or_gap', 'layering_or_fold')
STAGES = ('RAW', 'BASE_POST', 'CLEAN_FINAL_POST')
INPUTS = {}
HASHES = {}


def resolve(value):
    text = str(value).replace('\\', '/')
    if text.startswith('/mnt/e/MeshFlow/'):
        return ROOT / text[len('/mnt/e/MeshFlow/'):]
    if text.lower().startswith('e:/meshflow/'):
        return ROOT / text[len('e:/MeshFlow/'):]
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def digest(path):
    p = resolve(path)
    if str(p) not in HASHES:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        HASHES[str(p)] = h.hexdigest()
    return HASHES[str(p)]


def read(path):
    p = resolve(path)
    INPUTS[str(p)] = digest(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))


def equal(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def indexed(rows):
    d = {}
    for r in rows:
        key = (r['request_id'], r['stage'])
        if key in d:
            raise AssertionError(('Duplicate scientific identity', key))
        d[key] = r
    return d


def gallery_assets(manifests, labels):
    out = {}
    for manifest in manifests:
        for page in manifest['pages']:
            q = page.get('request', page)
            for c in page['cells']:
                if c['model'] not in labels:
                    continue
                key = (c.get('request_id', q['request_id']), c['stage'])
                item = dict(path=c['path'], page_id=page['page_id'],
                            receipt=c.get('receipt'),
                            condition_id=q['condition_id'], parent_id=q['parent_id'],
                            N=q['N'], K=q['K'])
                if key in out and resolve(out[key]['path']) != resolve(item['path']):
                    raise AssertionError(('Multiple different baseline meshes', key))
                out[key] = item
    return out


def visual(branch, labels):
    out = []
    for p in sorted((OUT / branch / 'visual_review/shards').glob('*.json')):
        d = read(p)
        for r in d.get('rows', d.get('records', [])):
            if r['label'] not in labels:
                continue
            if not (r['actually_viewed'] and r['views_and_sections_checked']
                    and r['reviewer_kind'] == 'AI_PROXY' and r['blind'] is False):
                raise AssertionError(('Not an actual nonblind review', p, r))
            out.append(dict(r, shard_path=str(p)))
    return indexed(out)


def joint(r):
    return (r['overall_shape'] == 'REASONABLE' and r['C_use'] == 'PLAUSIBLE'
            and r['support_retention'] == 'PRESENT')


def main(snapshot):
    dest = OUT / 'audits/baseline_consistency' / snapshot
    if dest.exists():
        raise FileExistsError('Snapshots are immutable; choose a new --snapshot')
    sm = read(OUT / 'seam/data/eval_manifest.json')
    em = read(OUT / 'exposure/data/eval_manifest.json')
    bm = read(OUT / 'bestof/data/manifest.json')
    hr = read(OUT / 'hybrid/registration.json')
    protocols={b:read(OUT/b/'visual_review/protocol.json')
               for b in ('seam','bestof','exposure','hybrid')}
    definitions={b:d['definition']['definitions'] for b,d in protocols.items()
                 if 'definition' in d}
    definitions['bestof']=read(OUT/'bestof/visual_review/v2_definition.json')['definitions']
    definitions_equal=all(d==definitions['seam'] for d in definitions.values())
    assert definitions_equal, 'Do not treat different rating definitions as repeated judgments'
    assert all(d['axes']==protocols['seam']['axes'] for d in protocols.values())
    fields = sm['metric_fields']
    assert len(fields) == 177 and set(fields) == set(em['metric_fields']) == set(bm['metric_fields'])
    sgal = read(OUT / 'seam/gallery/manifest.json')
    egal = read(OUT / 'exposure/gallery/manifest.json')
    bgal = [read(OUT / f'bestof/gallery/{x}_manifest.json') for x in ('raw', 'post')]
    hgal = read(OUT / 'hybrid/gallery/manifest.json')
    protected = {}
    for d in [sgal, egal, *bgal, hr]:
        for p, sha in d.get('protected', {}).items():
            p = str(resolve(p))
            if p in protected and protected[p] != sha:
                raise AssertionError(('Different frozen identity', p))
            protected[p] = sha
    hybrid_collected = OUT / 'hybrid/results/metric_rows.json'
    hm = []
    inherited_row_metadata=[]
    for task in hr['tasks']:
        for stage in STAGES:
            row = read(task['paths'][stage + '_row'])
            inherited_row_metadata.append(dict(path=task['paths'][stage+'_row'],
                original_label=row.get('label'),original_stage=row.get('stage'),
                registered_stage=stage,request_id=task['request']['request_id']))
            # Legacy direct scorer rows use a display-label stage tag. The
            # frozen registration's stage-specific path disambiguates it,
            # exactly as the branch collector does; no metric is decorated.
            row=dict(row,label='M',stage=stage)
            hm.append(row)
    collected_check = []
    if hybrid_collected.exists():
        collected = indexed([r for r in read(hybrid_collected) if r['label'] == 'M'])
        for key, row in indexed(hm).items():
            for f in fields:
                if f not in row or f not in collected[key] or not equal(row[f], collected[key][f]):
                    collected_check.append(dict(request_id=key[0], stage=key[1], field=f))
        assert not collected_check, collected_check
    configurations = [
        dict(name='B_SEAM_vs_BESTOF_FIXED', left='seam:B', right='bestof:FIXED_N0_N1',
             left_q=sm['requests'], right_q=[q for q in bm['candidates'] if q['noise_index'] < 2],
             left_rows=[r for r in read(OUT / 'seam/results/metric_rows.json') if r['label'] == 'B'],
             right_rows=[r for r in read(OUT / 'bestof/results/metric_rows.json') if r['label'] in ('FIXED_N0','FIXED_N1')],
             left_assets=gallery_assets([sgal], {'B'}),
             right_assets=gallery_assets(bgal, {'CANDIDATE_N0','CANDIDATE_N1','FIXED_N0','FIXED_N1'}),
             left_visual=visual('seam', {'B'}),
             right_visual=visual('bestof', {'CANDIDATE_N0','CANDIDATE_N1','FIXED_N0','FIXED_N1'}),
             expected_visual_stages=STAGES),
        dict(name='M_EXPOSURE_vs_HYBRID', left='exposure:M', right='hybrid:M_D0_and_RAW',
             left_q=em['requests'], right_q=[t['request'] for t in hr['tasks']],
             left_rows=[r for r in read(OUT / 'exposure/results/metric_rows.json') if r['label'] == 'M'],
             right_rows=hm, left_assets=gallery_assets([egal], {'M'}),
             right_assets=gallery_assets([hgal], {'RAW','M_D0'}),
             left_visual=visual('exposure', {'M'}), right_visual=visual('hybrid', {'RAW','M_D0'}),
             expected_visual_stages=('RAW','CLEAN_FINAL_POST')),
    ]
    assets, metrics, score_differences, observations, visual_differences, identities, summaries = [], [], [], [], [], [], []
    identity_fields = ('request_id','condition_id','parent_id','noise_index','N','K',
                       'inference_npz','inference_npz_sha256','metadata_npz','metadata_npz_sha256',
                       'noise_npz','noise_sha256','query_descriptor')
    for c in configurations:
        ql = {q['request_id']:q for q in c['left_q']}
        qr = {q['request_id']:q for q in c['right_q']}
        assert len(ql) == len(qr) == 46 and ql.keys() == qr.keys()
        for rid in ql:
            a,b = ql[rid],qr[rid]
            diff = [f for f in identity_fields if a.get(f) != b.get(f)]
            checks = {}
            for f, hf in [('inference_npz','inference_npz_sha256'), ('metadata_npz','metadata_npz_sha256'), ('noise_npz','noise_sha256')]:
                checks[f] = digest(a[f]) == a[hf]
                INPUTS[str(resolve(a[f]))] = digest(a[f])
            checks['query_descriptor'] = digest(a['query_descriptor']['path']) == a['query_descriptor']['sha256']
            INPUTS[str(resolve(a['query_descriptor']['path']))] = digest(a['query_descriptor']['path'])
            identities.append(dict(comparison=c['name'],request_id=rid,identity_fields_equal=not diff,
                                   difference_fields=diff,current_input_hashes_match=all(checks.values())))
            assert not diff and all(checks.values()), identities[-1]
        ml,mr = indexed(c['left_rows']),indexed(c['right_rows'])
        assert len(ml) == len(mr) == 138 and ml.keys() == mr.keys()
        for key in ml:
            a,b = ml[key],mr[key]
            rid,stage = key
            assert all(a.get(k)==b.get(k) for k in ('request_id','condition_id','parent_id','noise_index','N','K','stage'))
            aa,bb = c['left_assets'][key],c['right_assets'][key]
            pa,pb = resolve(aa['path']),resolve(bb['path'])
            ha,hb = digest(pa),digest(pb)
            INPUTS[str(pa)],INPUTS[str(pb)] = ha,hb
            frozen_a,frozen_b = protected.get(str(pa)),protected.get(str(pb))
            # Sample/postprocess receipts bind generated outputs; gallery
            # protected maps primarily bind definitions and input assets.
            for item in (aa,bb):
                if item.get('receipt'):
                    receipt=read(item['receipt'])
                    for p,h in receipt.get('files',{}).items():
                        p=str(resolve(p))
                        if p in protected and protected[p]!=h:
                            raise AssertionError(('Receipt disagrees with frozen identity',p))
                        protected[p]=h
            frozen_a,frozen_b = protected.get(str(pa)),protected.get(str(pb))
            assert frozen_a is not None and frozen_b is not None, ('Missing registered asset hash',pa,pb)
            assert ha==frozen_a and hb==frozen_b
            same = pa==pb and ha==hb
            assert same and all(aa[k]==bb[k] for k in ('condition_id','parent_id','N','K'))
            assets.append(dict(comparison=c['name'],request_id=rid,stage=stage,left_path=str(pa),
                               right_path=str(pb),left_sha256=ha,right_sha256=hb,
                               same_resolved_path=pa==pb,same_bytes=ha==hb,registered_hashes_match=True))
            differences=[]
            for f in fields:
                assert f in a and f in b, ('Missing metric',key,f)
                if not equal(a[f],b[f]):
                    differences.append(f)
                    score_differences.append(dict(comparison=c['name'],request_id=rid,stage=stage,
                                                  metric=f,left=a[f],right=b[f]))
            metrics.append(dict(comparison=c['name'],request_id=rid,stage=stage,fields_compared=177,
                                exact_equal=not differences,changed_fields=differences))
        vl,vr = c['left_visual'],c['right_visual']
        expected = {(rid,stage) for rid in ql for stage in c['expected_visual_stages']}
        common = sorted(expected & vl.keys() & vr.keys())
        missing = sorted(expected - set(common))
        for key in common:
            a,b = vl[key],vr[key]
            rid,stage=key
            diff=[x for x in AXES if a[x]!=b[x]]
            meta=dict(comparison=c['name'],request_id=rid,condition_id=ql[rid]['condition_id'],
                      parent_id=ql[rid]['parent_id'],report_group=ql[rid]['report_group'],stage=stage,
                      left_branch=c['left'],right_branch=c['right'],left_joint=joint(a),right_joint=joint(b),
                      joint_disagreement=joint(a)!=joint(b),different_axes=diff,
                      same_reviewer=a['reviewer']==b['reviewer'])
            for side,r in [('left',a),('right',b)]:
                for f in (*AXES,'note','reviewer','page_id','image_sha256','image_receipt_sha256',
                          'view_event_id','viewed_at_utc','shard_path'):
                    meta[side+'_'+f]=r[f]
            observations.append(meta)
            for x in diff:
                visual_differences.append(dict(meta,axis=x,left_value=a[x],right_value=b[x],
                    involves_uncertainty='UNCERTAIN' in (a[x],b[x])))
        stage_counts=[]
        for stage in c['expected_visual_stages']:
            rr=[r for r in observations if r['comparison']==c['name'] and r['stage']==stage]
            stage_counts.append(dict(stage=stage,paired=len(rr),expected=46,
                any_axis_disagreement=sum(bool(r['different_axes']) for r in rr),
                differing_axis_total=sum(len(r['different_axes']) for r in rr),
                joint_disagreement=sum(r['joint_disagreement'] for r in rr),
                left_joint=sum(r['left_joint'] for r in rr),right_joint=sum(r['right_joint'] for r in rr),
                axis_disagreements={x:sum(x in r['different_axes'] for r in rr) for x in AXES}))
        summaries.append(dict(comparison=c['name'],identity_requests=46,asset_pairs=138,metric_pairs=138,
            metric_fields=177,field_comparisons=138*177,
            metric_differences=sum(r['comparison']==c['name'] for r in score_differences),
            expected_visual=len(expected),actual_paired_visual=len(common),missing_visual=missing,
            visual_status='COMPLETE' if not missing else 'PARTIAL',stages=stage_counts))
    # Ensure no consumed source file changed during the audit; hash again without cache.
    for path, before in INPUTS.items():
        HASHES.pop(path, None)
        assert digest(path)==before, ('Source changed while reading',path)
    result=dict(schema='CROSS_BRANCH_REUSED_BASELINE_AUDIT_V1',
        status='COMPLETE' if not any(s['missing_visual'] for s in summaries) and hybrid_collected.exists() else 'PARTIAL',
        utc=datetime.now(timezone.utc).isoformat(),summaries=summaries,
        hybrid_reference_scores_source='registered original per-output row.json',
        inherited_row_metadata=inherited_row_metadata,
        hybrid_collected_table_available=hybrid_collected.exists(),
        hybrid_collected_reference_matches_registered_rows=not collected_check if hybrid_collected.exists() else None,
        five_axis_allowed_values_and_definitions_equal=definitions_equal,
        objective_assets_and_metrics_pass=not score_differences,
        visual_interpretation='Repeated nonblind AI_PROXY judgments in different comparison-page contexts. Disagreements are preserved uncertainty, not new ratings or adjudication. Do not pool baselines or use these counts as independent human agreement.',
        visual_BASE_POST_M='Not required by HYBRID main review; NA, not an absent-quality or agreement claim.',
        models=0,scorer_calls=0,new_labels=0,original_files_modified=0,
        source_inputs=INPUTS,code_sha256=digest(Path(__file__)))
    dest.mkdir(parents=True)
    tables=dict(request_identity=identities,asset_identity=assets,metric_comparison=metrics,
                metric_differences=score_differences,visual_pairs=observations,visual_disagreements=visual_differences)
    for name,rows in tables.items():
        (dest/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (dest/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    (dest/'complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','summaries','objective_assets_and_metrics_pass')},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True)
    args=parser.parse_args()
    if not args.snapshot.replace('_','').isalnum():
        raise ValueError('Use an alphanumeric snapshot name')
    main(args.snapshot)
