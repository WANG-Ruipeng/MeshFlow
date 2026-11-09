#!/usr/bin/env python3
"""Recompute saved MINI training-source arithmetic only; stdlib, no geometry/model."""
import argparse, csv, hashlib, json, math
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

ROLES = ('COMMON_SMALL32', 'EXP_ONLY35')
ILLEGAL = {'proper', 'coplanar_positive_overlap'}
FILES = ('conditions.csv', 'source_quality_and_interfaces.csv', 'parents.csv',
         'exposure_by_condition.csv', 'source_events.json')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def key(row):
    return (row['parent_uid'], row['condition_id'])

def js(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def csv_read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def csv_write(path, rows):
    assert rows, path
    fields = list(rows[0])
    assert all(set(r) == set(fields) for r in rows), path
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def json_write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def mean(xs):
    return sum(xs, Fraction(0)) / len(xs)

def fraction_columns(name, value):
    return {name: repr(float(value)), name + '_exact': str(value),
            name + '_percent': repr(float(value * 100))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mini', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    mini, out = args.mini.resolve(), args.out.resolve()
    if mini == out or mini in out.parents or out in mini.parents:
        raise SystemExit('Input MINI and output must be separate, non-nested directories.')
    if mini.is_symlink() or args.out.is_symlink():
        raise SystemExit('Symlinks are not accepted.')
    out.mkdir(parents=True, exist_ok=True)
    outputs = ('per_condition.csv','per_parent.csv','deduplicated_FF_events.csv',
        'exposure_association.csv','exposure_summary.csv','summary.json',
        'source_hashes.json','comparison_checks.json','output_manifest.json')
    if any((out / p).exists() for p in outputs):
        raise SystemExit('Refusing to overwrite existing arithmetic results; use a fresh --out.')
    manifest_path = mini / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    registered = {r['path']:r for r in manifest['files']}
    sources = []
    for name in FILES:
        p = mini / name
        if p.is_symlink():
            raise SystemExit(f'Symlink source rejected: {p}')
        actual = sha(p)
        expected = registered[name]['sha256']
        assert actual == expected, (name, 'MINI manifest SHA mismatch')
        sources.append({'file': name,'path':str(p),'bytes':p.stat().st_size,
            'existing_MINI_manifest_sha256':expected,'this_read_sha256':actual,
            'manifest_match':True})
    conditions_all = csv_read(mini / 'conditions.csv')
    qualities_all = csv_read(mini / 'source_quality_and_interfaces.csv')
    parents_all = csv_read(mini / 'parents.csv')
    exposure = csv_read(mini / 'exposure_by_condition.csv')
    saved = json.loads((mini / 'source_events.json').read_text(encoding='utf-8-sig'),
                       parse_float=Decimal)['records']
    conditions = sorted((r for r in conditions_all if r['parent_role'] in ROLES),key=key)
    qualities = {key(r):r for r in qualities_all}
    parents = {r['parent_uid']:r for r in parents_all if r['parent_role'] in ROLES}
    assert len(conditions) == 194 and len(parents) == 67
    assert len({key(r) for r in conditions}) == len(conditions)
    assert len(qualities) == len(qualities_all)
    assert len({(r['arm'],*key(r)) for r in exposure}) == len(exposure) == 285
    condition_rows, parent_rows, event_rows = [], [], []
    by_parent, per_key, events, rates = defaultdict(list), {}, {}, {}
    checks, source_records = [], {}
    for c in conditions:
        q = qualities[key(c)]
        assert q['parent_role'] == c['parent_role']
        assert q['quality_source_id'] == c['quality_source_id']
        source_id = c['quality_source_id']
        r = saved[source_id]
        assert r['parent_uid'] == c['parent_uid']
        assert r['variant_uid'] == c['variant_uid']
        assert r['source_existing_sha256'] == r['source_this_read_sha256'] == q['quality_existing_sha256']
        source_records[source_id] = {k:r[k] for k in ('parent_uid','variant_uid','source_id',
            'source_path','source_existing_sha256','source_this_read_sha256')}
        pf = {row[0]:dict(zip(r['per_face_columns'],row)) for row in r['per_face']}
        C, Y = set(json.loads(c['source_C_face_ids'])), set(json.loads(c['source_Y_face_ids']))
        N, K = int(c['N']), int(c['K'])
        assert C.isdisjoint(Y) and C | Y == set(pf) and len(C) == K
        assert len(Y) == N-K == int(c['Y_face_count_DERIVED']) and N == len(pf)
        ids = {'illegal': [],'unknown': []}
        nproper = nco = 0
        for raw in r['contacts']:
            e = dict(zip(r['contact_columns'], raw))
            i,j = sorted((e['source_i'], e['source_j']))
            if not {i,j} <= Y or e['category'] not in ILLEGAL | {'UNKNOWN'}:
                continue
            kind = 'illegal' if e['category'] in ILLEGAL else 'unknown'
            eid = e['event_id_DERIVED']
            ids[kind].append(eid)
            nproper += e['category'] == 'proper'
            nco += e['category'] == 'coplanar_positive_overlap'
            # Source ID names may change on extraction; source parent/variant/pair is the dedup key.
            ek = (c['parent_uid'], c['variant_uid'], i, j)
            if ek not in events:
                events[ek] = {'parent_uid':c['parent_uid'],'parent_role':c['parent_role'],
                    'variant_uid':c['variant_uid'],'quality_source_id':source_id,
                    'event_id_in_MINI':eid,'source_i':i,'source_j':j,
                    'original_face_ids':js(e['original_face_ids']),
                    'category':e['category'],'classification':e['classification'],
                    'certainty':'CONFIRMED_ILLEGAL' if kind=='illegal' else 'UNKNOWN',
                    'condition_ids':set()}
            assert events[ek]['category'] == e['category'] and events[ek]['event_id_in_MINI'] == eid
            events[ek]['condition_ids'].add(c['condition_id'])
        assert len(ids['illegal']) == len(set(ids['illegal']))
        assert len(ids['unknown']) == len(set(ids['unknown']))
        dups = sorted(sorted(set(g)&Y) for g in r['exact_duplicate_groups'] if len(set(g)&Y)>=2)
        valid = {i:pf[i]['minimum_angle_degrees'] for i in Y
                 if pf[i]['minimum_angle_degrees'] is not None
                 and Decimal(str(pf[i]['minimum_angle_degrees'])).is_finite()}
        lt5 = sorted(i for i,v in valid.items() if v < 5)
        lt10 = sorted(i for i,v in valid.items() if v < 10)
        for field, got in (
            ('FF_confirmed_illegal_pairs',len(ids['illegal'])),
            ('FF_unknown_pairs',len(ids['unknown'])),
            ('FF_proper_pairs_DERIVED',nproper),
            ('FF_coplanar_positive_overlap_pairs_DERIVED',nco),
            ('Y_angle_valid_denominator_DERIVED',len(valid)),
            ('Y_small_angle_lt10_faces_DERIVED',len(lt10)),
            ('Y_exact_duplicate_group_count_DERIVED',len(dups)),
            ('Y_duplicate_involved_face_count_DERIVED',len(set(sum(dups,[]))))):
            assert int(q[field]) == got, (key(c),field,q[field],got)
        assert set(json.loads(q['FF_problem_event_ids_DERIVED'])) == set(sum(ids.values(),[]))
        assert sorted(json.loads(q['Y_exact_duplicate_groups_DERIVED'])) == dups
        assert sorted(json.loads(q['Y_small_angle_source_face_ids_DERIVED'])) == lt10
        assert math.isclose(float(q['Y_small_angle_lt10_fraction_DERIVED']),len(lt10)/len(Y),abs_tol=1e-15)
        checks.append({'parent_uid':c['parent_uid'],'condition_id':c['condition_id'],
            'saved_FF_counts_and_event_ids_match':True,'saved_duplicate_groups_match':True,
            'saved_lt10_counts_ids_and_rate_match':True,'source_partition_ID_sets_match':True})
        row = {k:c[k] for k in ('parent_uid','condition_id','parent_role','condition_type','variant_uid')}
        row.update({'quality_source_id':source_id,'source_quality_label':q['source_quality'],
            'N':N,'K':K,'Y_faces_denominator':len(Y),
            'confirmed_illegal_FF_pairs':len(ids['illegal']),'UNKNOWN_FF_pairs':len(ids['unknown']),
            'proper_FF_pairs':nproper,'coplanar_positive_overlap_FF_pairs':nco,
            'confirmed_illegal_FF_event_ids':js(sorted(ids['illegal'])),
            'UNKNOWN_FF_event_ids':js(sorted(ids['unknown'])),
            'Y_has_exact_duplicate_group':bool(dups),'Y_exact_duplicate_groups':js(dups),
            'Y_exact_duplicate_group_count':len(dups),
            'Y_duplicate_involved_face_count':len(set(sum(dups,[]))),
            'Y_valid_angle_count':len(valid),'Y_angle_unknown_count':len(Y)-len(valid),
            'Y_small_angle_lt5_count':len(lt5),'Y_small_angle_lt10_count':len(lt10),
            'Y_small_angle_lt5_source_face_ids':js(lt5),
            'Y_small_angle_lt10_source_face_ids':js(lt10)})
        for threshold, num in ((5,len(lt5)),(10,len(lt10))):
            value = Fraction(num,len(Y))
            row.update(fraction_columns(f'Y_small_angle_lt{threshold}_fraction',value))
            rates[(key(c),threshold)] = value
        row['derivation'] = 'DERIVED_FROM_SAVED_SOURCE_RECORDS; no geometric recomputation'
        condition_rows.append(row)
        by_parent[c['parent_uid']].append(row)
        per_key[key(c)] = row
    for ek, e in sorted(events.items()):
        r = dict(e)
        r['condition_ids'] = js(sorted(e['condition_ids']))
        r['condition_occurrences'] = len(e['condition_ids'])
        r['derivation'] = 'DERIVED_UNION_OVER_TRAINING_CONDITIONS; same parent/variant/unordered face pair once'
        event_rows.append(r)
    parent_rates = {}
    for uid, rows in sorted(by_parent.items()):
        p = parents[uid]
        assert {r['parent_role'] for r in rows} == {p['parent_role']}
        assert {r['condition_id'] for r in rows} == set(json.loads(p['condition_ids_DERIVED']))
        local_events = [e for e in event_rows if e['parent_uid']==uid]
        src = saved[rows[0]['quality_source_id']]
        source_problem = [dict(zip(src['contact_columns'],x)) for x in src['contacts']]
        source_problem = [e for e in source_problem if e['category'] in ILLEGAL | {'UNKNOWN'}]
        unique_source = {(min(e['source_i'],e['source_j']),max(e['source_i'],e['source_j'])):e for e in source_problem}
        assert len(unique_source) == len(source_problem)
        pr = {'parent_uid':uid,'parent_role':p['parent_role'],'variant_uid':p['variant_uid'],
            'condition_count':len(rows),
            'source_quality_labels':js(sorted({r['source_quality_label'] for r in rows})),
            'has_ILLEGAL_label':any(r['source_quality_label']=='ILLEGAL' for r in rows),
            'deduplicated_confirmed_illegal_FF_pairs':sum(e['certainty']=='CONFIRMED_ILLEGAL' for e in local_events),
            'deduplicated_UNKNOWN_FF_pairs':sum(e['certainty']=='UNKNOWN' for e in local_events),
            'all_M_saved_confirmed_illegal_pairs':sum(e['category'] in ILLEGAL for e in source_problem),
            'all_M_saved_UNKNOWN_pairs':sum(e['category']=='UNKNOWN' for e in source_problem),
            'has_Y_exact_duplicate_condition':any(r['Y_has_exact_duplicate_group'] for r in rows),
            'Y_exact_duplicate_condition_count':sum(r['Y_has_exact_duplicate_group'] for r in rows),
            'Y_angle_unknown_condition_count':sum(r['Y_angle_unknown_count']>0 for r in rows),
            'condition_ids':js(sorted(r['condition_id'] for r in rows))}
        for t in (5,10):
            value = mean([rates[(key(r),t)] for r in rows])
            parent_rates[(uid,t)] = value
            pr.update(fraction_columns(f'condition_mean_Y_lt{t}_fraction',value))
        pr['derivation'] = 'DERIVED; each parent mean gives equal weight to its frozen conditions'
        parent_rows.append(pr)
    exposure_rows = []
    for e in sorted(exposure,key=lambda r:(r['arm'],*key(r))):
        c = per_key[key(e)]
        assert e['parent_role']==c['parent_role'] and int(e['N'])==c['N'] and int(e['K'])==c['K']
        z = {k:e[k] for k in ('arm','parent_uid','condition_id','parent_role','condition_type')}
        for field in ('actual_exposures','clean_exposures','width_exposures','cumulative_free_scalars',
                      'clean_free_scalars','width_free_scalars'):
            z[field] = int(e[field])
        assert z['actual_exposures']==z['clean_exposures']+z['width_exposures']
        assert z['cumulative_free_scalars']==z['actual_exposures']*c['Y_faces_denominator']*9
        assert z['clean_free_scalars']==z['clean_exposures']*c['Y_faces_denominator']*9
        assert z['width_free_scalars']==z['width_exposures']*c['Y_faces_denominator']*9
        z.update({k:c[k] for k in ('N','K','Y_faces_denominator','source_quality_label',
            'confirmed_illegal_FF_pairs','UNKNOWN_FF_pairs','Y_has_exact_duplicate_group',
            'Y_exact_duplicate_groups','Y_small_angle_lt5_count','Y_small_angle_lt10_count')})
        z['duplicate_Y_exposures'] = z['actual_exposures'] if c['Y_has_exact_duplicate_group'] else 0
        z['duplicate_Y_clean_exposures'] = z['clean_exposures'] if c['Y_has_exact_duplicate_group'] else 0
        z['duplicate_Y_width_exposures'] = z['width_exposures'] if c['Y_has_exact_duplicate_group'] else 0
        z['duplicate_Y_free_scalars'] = z['cumulative_free_scalars'] if c['Y_has_exact_duplicate_group'] else 0
        z['inherited_exposure_source_ids'] = e['source_ids']
        z['derivation'] = 'DERIVED_JOIN_OF_SAVED_CONDITION_EXPOSURE; not gradient contribution'
        exposure_rows.append(z)
    exp_summaries = []
    for arm in ('USER_SMALL','USER_EXPANDED_67'):
        for role in (*ROLES,'ALL_TRAIN'):
            rows = [r for r in exposure_rows if r['arm']==arm and (role=='ALL_TRAIN' or r['parent_role']==role)]
            if not rows:
                continue
            n = sum(r['actual_exposures'] for r in rows)
            row = {'arm':arm,'parent_role':role,'parents':len({r['parent_uid'] for r in rows}),
                'conditions':len(rows),'actual_exposures':n,
                'clean_exposures':sum(r['clean_exposures'] for r in rows),
                'width_exposures':sum(r['width_exposures'] for r in rows),
                'cumulative_free_scalars':sum(r['cumulative_free_scalars'] for r in rows)}
            for label, pred in (
                ('duplicate_Y',lambda r:r['Y_has_exact_duplicate_group']),
                ('confirmed_illegal_FF',lambda r:r['confirmed_illegal_FF_pairs']>0),
                ('UNKNOWN_FF',lambda r:r['UNKNOWN_FF_pairs']>0),
                ('original_ILLEGAL_label',lambda r:r['source_quality_label']=='ILLEGAL')):
                selected = [r for r in rows if pred(r)]
                selected_n = sum(r['actual_exposures'] for r in selected)
                row[label+'_exposures'] = selected_n
                row[label+'_clean_exposures'] = sum(r['clean_exposures'] for r in selected)
                row[label+'_width_exposures'] = sum(r['width_exposures'] for r in selected)
                row[label+'_free_scalars'] = sum(r['cumulative_free_scalars'] for r in selected)
                row.update(fraction_columns(label+'_exposure_fraction',Fraction(selected_n,n)))
            row['derivation'] = 'DERIVED; actual condition draws; risk labels are not gradient shares'
            exp_summaries.append(row)
        assert sum(r['actual_exposures'] for r in exposure_rows if r['arm']==arm)==48000
    cohort_summaries = []
    for role in ROLES:
        rows = [r for r in parent_rows if r['parent_role']==role]
        z = {'parent_role':role,'parents':len(rows),'conditions':sum(r['condition_count'] for r in rows),
            'parents_with_ILLEGAL_label':sum(r['has_ILLEGAL_label'] for r in rows),
            'parents_with_Y_exact_duplicate_condition':sum(r['has_Y_exact_duplicate_condition'] for r in rows),
            'deduplicated_confirmed_illegal_FF_pairs':sum(r['deduplicated_confirmed_illegal_FF_pairs'] for r in rows),
            'deduplicated_UNKNOWN_FF_pairs':sum(r['deduplicated_UNKNOWN_FF_pairs'] for r in rows),
            'all_M_saved_confirmed_illegal_pairs':sum(r['all_M_saved_confirmed_illegal_pairs'] for r in rows),
            'all_M_saved_UNKNOWN_pairs':sum(r['all_M_saved_UNKNOWN_pairs'] for r in rows)}
        for t in (5,10):
            z.update(fraction_columns(f'parent_equal_mean_Y_lt{t}_fraction',mean([parent_rates[(r['parent_uid'],t)] for r in rows])))
        cohort_summaries.append(z)
    claim_checks = []
    def claim(name,actual,expected):
        claim_checks.append({'claim':name,'actual':actual,'expected_from_requested_review':expected,'match':actual==expected})
    for row, illegal, unknown, dupparents in zip(cohort_summaries,(721,840),(2,3),(9,15)):
        claim(row['parent_role']+'_confirmed_illegal',row['deduplicated_confirmed_illegal_FF_pairs'],illegal)
        claim(row['parent_role']+'_UNKNOWN',row['deduplicated_UNKNOWN_FF_pairs'],unknown)
        claim(row['parent_role']+'_duplicate_Y_parents',row['parents_with_Y_exact_duplicate_condition'],dupparents)
    for row in exp_summaries:
        if row['parent_role']=='ALL_TRAIN':
            claim(row['arm']+'_duplicate_Y_exposures',row['duplicate_Y_exposures'],
                  13500 if row['arm']=='USER_SMALL' else 17193)
    csv_write(out/'per_condition.csv',condition_rows)
    csv_write(out/'per_parent.csv',parent_rows)
    csv_write(out/'deduplicated_FF_events.csv',event_rows)
    csv_write(out/'exposure_association.csv',exposure_rows)
    csv_write(out/'exposure_summary.csv',exp_summaries)
    json_write(out/'comparison_checks.json',{'status':'PASS','conditions_checked':len(checks),'checks':checks})
    unknown_angles = sum(r['Y_angle_unknown_count'] for r in condition_rows)
    summary = {'schema':'SOURCE_TRAINING_ARITHMETIC_V1','status':'COMPLETE',
        'created_utc':datetime.now(timezone.utc).isoformat(),'DERIVED':True,
        'scope':'194 frozen training conditions / 67 independent parents; no output metric rescoring',
        'model_calls':0,'geometry_calls':0,'raw_training_log_rereads':0,
        'cohorts':cohort_summaries,'exposure':exp_summaries,'requested_claim_checks':claim_checks,
        'condition_angle_unknown_occurrences':unknown_angles,
        'definitions':{
            'confirmed_illegal':'Saved category proper OR coplanar_positive_overlap; UNKNOWN excluded.',
            'deduplication':'Union over condition Y sets, key parent_uid, variant_uid, unordered source face pair.',
            'duplicates':'Saved exact_duplicate_groups intersect Y; at least 2 source faces remain in Y.',
            'exposure':'Count actual saved training condition draws where Y has >=1 exact duplicate group, not duplicate face multiplicity.',
            'small_angle':'Saved minimum_angle_degrees strictly <5 or <10. Per-condition denominator all Y=N-K. Decimal threshold comparisons; exact rational averaging.',
            'parent_equal':'Within each parent equal mean of frozen condition fractions; across cohort equal mean over parents.',
            'no_causal_claim':'Risk exposure is data exposure, not loss/gradient contribution or proof of causing output defects.',
            'UNKNOWN':'Unresolved detector event, not certified legal and not confirmed illegal.',
            'source_vs_generated':'All counts here concern retained source completion Y, not generated outputs.',
            'no_clean_certification':'No detected saved pair problem does not certify a source mesh is clean.'},
        'limitations':['No geometry or detector rerun. Inherits saved detector and angle values.',
            'Two thresholds characterize different tails; they do not order overall mesh quality.',
            'Any missing saved angle stays unknown; not fabricated as >=threshold.']}
    json_write(out/'summary.json',summary)
    json_write(out/'source_hashes.json',{'manifest_path':str(manifest_path),
        'manifest_this_read_sha256':sha(manifest_path),'files':sources,
        'referenced_original_quality_records_not_reread':list(source_records.values())})
    # Verify original MINI bytes stayed unchanged after all arithmetic, not just before reading.
    assert all(sha(mini/r['file'])==r['this_read_sha256'] for r in sources)
    paths = [out/name for name in outputs if name!='output_manifest.json']
    script = Path(__file__).resolve()
    files = [{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p),
              'DERIVED':True} for p in paths]
    json_write(out/'output_manifest.json',{'schema':'ARITHMETIC_OUTPUT_MANIFEST_V1','status':'COMPLETE',
        'created_utc':datetime.now(timezone.utc).isoformat(),'script':str(script),'script_sha256':sha(script),
        'files':files,'row_counts':{'per_condition.csv':len(condition_rows),'per_parent.csv':len(parent_rows),
            'deduplicated_FF_events.csv':len(event_rows),'exposure_association.csv':len(exposure_rows),
            'exposure_summary.csv':len(exp_summaries)},'MINI_after_read_hashes_unchanged':True,
        'self_hash_policy':'This manifest is hashed by the parent delivery manifest; no self-hash.'})
    print(json.dumps({'status':'COMPLETE','cohorts':cohort_summaries,
        'all_requested_counts_match':all(r['match'] for r in claim_checks),
        'unknown_angle_occurrences':unknown_angles,'rows':{p['path']:p['bytes'] for p in files}},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
