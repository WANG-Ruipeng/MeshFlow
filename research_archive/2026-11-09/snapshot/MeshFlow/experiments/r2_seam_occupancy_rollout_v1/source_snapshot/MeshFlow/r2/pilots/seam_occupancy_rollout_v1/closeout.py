"""CPU-only closeout of completed evidence. Never imports model/geometry code."""
from pathlib import Path
import argparse
import collections
import csv
import difflib
import hashlib
import html
import io
import json
import statistics
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_seam_occupancy_rollout_v1'
CODE = Path(__file__).resolve().parent
ZIP = OUT / 'R2_SEAM_OCCUPANCY_ROLLOUT_REVIEW.zip'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csvwrite(p, values):
    keys = list(dict.fromkeys(k for r in values for k in r))
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(values)


def local(s):
    s = str(s).replace('\\', '/')
    if s.startswith('E:/') and str(ROOT).startswith('/mnt/e/'):
        return Path('/mnt/e/' + s[3:])
    if s.startswith('/mnt/e/') and not str(ROOT).startswith('/mnt/e/'):
        return Path('E:/' + s[7:])
    return Path(s)


def budget():
    counts = collections.Counter()
    timestamps = collections.defaultdict(list)
    for p in sorted((OUT / 'evaluation').glob('*/*/*/calls.jsonl')):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            counts[r['event']] += 1
            timestamps[r['event']].append(r['unix'])
    expected = {'FORWARD_STARTED': 1600, 'FORWARD_RETURNED': 1600,
                'D0_FORWARD_STARTED': 32, 'D0_FORWARD_RETURNED': 32}
    assert dict(counts) == expected, counts
    assert min(timestamps['D0_FORWARD_STARTED']) > max(timestamps['FORWARD_RETURNED'])
    return dict(counts), timestamps


def assemble():
    if (OUT / 'completion.json').exists():
        return verify()
    audit = read(OUT / 'audits/audit_final.json')
    assert audit['status'] == 'PASS'
    assert read(OUT / 'analysis/score_completion.json')['cells'] == 192
    count, times = budget()
    review = []
    enum_map = {'YES_VISUAL': 'YES', 'NO_VISIBLE_MAJOR_LOSS': 'NO',
                'NO_VISIBLE_WRONG_LAYER': 'NO', 'UNKNOWN_NOT_CERTIFIED_BY_IMAGE': 'UNKNOWN'}
    review_fields = ['C_adoption', 'C_floating', 'C_replaced_by_other_surface', 'C_wrong_layer',
                     'whole_body_complete', 'support_missing', 'fragments_or_folded_layers', 'visible_join']
    originals = ['review_73_85_AI_PROXY.csv', 'review_3f_8c_AI_PROXY_v2.csv']
    for name in originals:
        for r in rows(OUT / 'gallery' / name):
            r['original_review_file'] = name
            for k in review_fields:
                r[k] = enum_map.get(r[k], r[k])
                assert r[k] in ('YES', 'NO', 'UNKNOWN'), (k, r[k])
            for e in json.loads(r['reviewed_images_json']):
                assert sha(local(e['path'])) == e['sha256']
            review.append(r)
    assert len(review) == len({(r['request_id'], r['label'], r['stage']) for r in review}) == 144
    review.sort(key=lambda r: (r['request_id'], r['label'], r['stage']))
    csvwrite(OUT / 'gallery/actual_visual_review.csv', review)
    write(OUT / 'gallery/review_completion.json', {
        'status': 'COMPLETE_ACTUAL_AI_PROXY_NONBLIND_REVIEW', 'cells': 144,
        'source_records': originals, 'independent_second_review_claimed': False,
        'method': 'Actual two-view contact-sheet observation; extra detailed pages recorded in source records; exact and occluded interfaces may remain UNKNOWN',
        'format_only_normalization': enum_map, 'model_calls': 0})

    outputs = rows(OUT / 'analysis/per_output.csv')
    compact = []
    for stage in ['RAW', 'PRE_GEO', 'POST', 'CAPACITY_POST']:
        for label in ['PREDICTED', 'ALL_ON', 'SOURCE_ORACLE']:
            rr = [r for r in outputs if r['label'] == label and r['stage'] == stage]
            assert len(rr) == 16
            v = dict(label=label, stage=stage, n=16)
            for metric in ['CF_illegal_affected_free_fraction', 'FF_illegal_affected_free_fraction',
                           'ConflictCov_C', 'FF_common_bulk_affected_fraction']:
                v[metric + '_mean_percent'] = 100 * statistics.mean(float(r[metric]) for r in rr)
            for metric in ['FF_illegal_pair_count', 'FF_common_bulk_illegal_pair_count']:
                v[metric + '_sum'] = sum(int(r[metric]) for r in rr)
            if stage in ('POST', 'CAPACITY_POST'):
                v['legal_CF_edges_sum'] = sum(int(r['C_F_valid_edge_count']) for r in rr)
                reach = [float(r['free_only_reached_bulk_area_fraction']) for r in rr]
                v.update(bulk_reach_mean_percent=100*statistics.mean(reach),
                         bulk_reach_nonzero=sum(x > 0 for x in reach), bulk_reach_ge30=sum(x >= .3 for x in reach))
            compact.append(v)
    csvwrite(OUT / 'analysis/key_results.csv', compact)

    phases = []
    for p in sorted((OUT / 'execution').glob('*/*/completion.json')):
        v = read(p)
        v['arm'] = p.parent.parent.name
        v['phase'] = p.parent.name
        v['stage_directory_created_unix'] = int(p.parent.name.rsplit('_', 1)[1]) / 1e9
        v['receipt'] = str(p.relative_to(OUT))
        phases.append(v)
    write(OUT / 'execution_summary.json', {
        'budget': count, 'training_updates': 0, 'retries': 0,
        'first_generator_call_utc': datetime.fromtimestamp(min(times['FORWARD_STARTED']), timezone.utc).isoformat(),
        'last_D0_return_utc': datetime.fromtimestamp(max(times['D0_FORWARD_RETURNED']), timezone.utc).isoformat(),
        'all_raw_to_first_D0_gap_seconds': min(times['D0_FORWARD_STARTED'])-max(times['FORWARD_RETURNED']),
        'work_segments': phases, 'maximum_dispatched_GPU_workers': 2,
        'exact_process_VRAM': 'NA_WSL_WDDM_process_query_empty; allocator and device samples retained',
        'CPU_score_workers': 2, 'score_wall_seconds': read(OUT/'analysis/score_completion.json')['seconds'],
        'timing_limit': 'Concurrent segments and model-loading overhead; do not sum or compare as isolated throughput. First model call is not first preparation action.'})

    sources = {}
    for doc, key in [('registration.json', 'code_hashes'), ('analysis/registration.json', 'source_hashes')]:
        for name, digest in read(OUT / doc)[key].items():
            assert sha(local(name)) == digest, name
            sources[name] = digest
    for p in CODE.glob('*.py'):
        sources[str(p)] = sha(p)
    source_rows = []
    for name, digest in sorted(sources.items()):
        p = local(name)
        rel = p.relative_to(Path('/mnt/e')) if str(p).startswith('/mnt/e/') else Path(str(p).replace('E:/', '').replace('E:\\', ''))
        dest = OUT / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        source_rows.append(dict(source=name, sha256=digest, snapshot=str(dest.relative_to(OUT))))
    csvwrite(OUT / 'source_hashes.csv', source_rows)
    diff = ''
    for p in sorted(CODE.glob('*.py')):
        rel = str(p.relative_to(ROOT))
        diff += ''.join(difflib.unified_diff([], p.read_text().splitlines(keepends=True), fromfile='/dev/null', tofile='b/'+rel))
    (OUT / 'code.diff').write_text(diff, encoding='utf-8')

    contacts = sorted((OUT / 'gallery').glob('*_contact_view*.png'))
    assert len(contacts) == 16
    content = '<!doctype html><meta charset="utf-8"><title>Occupancy rollout review</title><style>body{font:16px sans-serif;max-width:1600px;margin:auto}img{width:100%}h2{margin-top:40px}</style><h1>Frozen occupancy rollout</h1><p>Blue C. See REPORT.md and actual_visual_review.csv. SOURCE_ORACLE is diagnostic only. All contacts, no selected-best filtering.</p>'
    for p in contacts:
        content += '<h2>'+html.escape(p.stem)+'</h2><img src="'+p.name+'">'
    (OUT / 'gallery/review_contacts.html').write_text(content, encoding='utf-8')

    payload = [OUT / n for n in ['REPORT.md', 'RUN_AND_RESUME.md', 'registration.json', 'eval_manifest.json',
                                'DISPATCH_PLAN.json', 'execution_summary.json', 'source_hashes.csv', 'code.diff']]
    for pattern in ['analysis/*.csv', 'analysis/*.md', 'analysis/*.json', 'audits/*.json',
                    'gallery/*AI_PROXY*.csv', 'gallery/actual_visual_review.csv', 'gallery/review_completion.json',
                    'gallery/root_3f_n2_additional_view.json', 'source_snapshot/**/*.py', 'execution/**/*.json', 'execution/**/*.jsonl']:
        payload.extend(OUT.glob(pattern))
    payload += contacts + [OUT / 'gallery/review_contacts.html', OUT / 'gallery/3f741759_n2_POST.png', OUT / 'gallery/8c9b2023_n1_RAW.png']
    payload += sorted((OUT / 'evaluation').glob('*/*/raw/receipt.json'))
    payload += sorted((OUT / 'evaluation').glob('*/*/raw/audit.json'))
    payload += sorted((OUT / 'evaluation').glob('*/*/raw/calls.jsonl'))
    payload += sorted((OUT / 'evaluation').glob('*/*/post/*receipt.json'))
    payload += sorted((OUT / 'evaluation').glob('*/*/post/calls.jsonl'))
    payload += sorted((OUT / 'capacity/evaluation').glob('*/*/post/receipt.json'))
    payload = sorted(set(payload))
    evidence = {str(p.relative_to(OUT)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in payload}
    write(OUT / 'evidence_manifest.json', evidence)
    completion = {'status': 'COMPLETE', 'experiment': 'R2_SEAM_OCCUPANCY_ROLLOUT_V1',
                  'closed_utc': datetime.now(timezone.utc).isoformat(), 'independent_audit': 'PASS',
                  'new_rollouts': 32, 'new_D0': 32, 'training_updates': 0, 'retries': 0,
                  'score_cells': 192, 'actual_review_cells': 144, 'new_default': False,
                  'source_layout': 'DIAGNOSTIC_ONLY', 'commit_or_push': False,
                  'evidence_manifest_sha256': sha(OUT / 'evidence_manifest.json')}
    write(OUT / 'completion.json', completion)
    payload += [OUT / 'completion.json', OUT / 'evidence_manifest.json']
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in payload:
            z.write(p, str(p.relative_to(OUT)))
    assert ZIP.stat().st_size < 10_000_000, ZIP.stat().st_size
    write(OUT / 'package.json', {'path': ZIP.name, 'bytes': ZIP.stat().st_size, 'sha256': sha(ZIP),
                               'files': len(payload), 'weights_and_raw_meshes_included': False})
    return verify()


def verify():
    c = read(OUT / 'completion.json')
    assert c['status'] == 'COMPLETE'
    assert sha(OUT / 'evidence_manifest.json') == c['evidence_manifest_sha256']
    for name, r in read(OUT / 'evidence_manifest.json').items():
        assert sha(OUT / name) == r['sha256'], name
    counts, _ = budget()
    assert read(OUT / 'audits/audit_final.json')['status'] == 'PASS'
    assert len(rows(OUT / 'gallery/actual_visual_review.csv')) == 144
    package = read(OUT / 'package.json')
    assert sha(ZIP) == package['sha256']
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == package['files']
        for name in z.namelist():
            assert hashlib.sha256(z.read(name)).hexdigest() == sha(OUT / name), name
    print(json.dumps({'status': 'PASS', 'model_calls_in_this_command': 0, 'registered_calls': counts,
                      'zip_bytes': ZIP.stat().st_size, 'zip_files': package['files']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify() if args.verify else assemble()
