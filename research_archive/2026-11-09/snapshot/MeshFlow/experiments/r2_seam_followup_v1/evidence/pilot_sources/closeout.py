"""Verify and seal saved evidence. No model import, inference, or rescoring."""
import collections
import csv
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/r2_seam_followup_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csvwrite(path, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(records)


def local(path):
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def main():
    if (OUT / 'completion.json').exists():
        raise SystemExit('Already sealed; read the existing completion.json.')
    combined = []
    for kind in ('binding', 'layout'):
        for split in ('train', 'dev'):
            p = OUT / 'reviews' / f'{kind}_{split}.csv'
            rr = rows(p)
            assert len(rr) == (64 if split == 'train' else 48)
            for r in rr:
                assert r['review_status'] == 'ACTUALLY_VIEWED'
                image = local(r['image_path'])
                assert sha(image) == (r.get('image_sha256') or r.get('sha'))
                combined.append(dict(r, kind=kind, split=split.upper(),
                                     baseline=r.get('baseline') or r.get('base_arm'),
                                     candidate=r.get('candidate') or r.get('variant_label')))
    assert len(combined) == 224
    pages = {(r['kind'], r['page_id']) for r in combined}
    assert len(pages) == 112
    assert all(n == 2 for n in collections.Counter((r['kind'], r['page_id']) for r in combined).values())
    csvwrite(OUT / 'reviews/visual_review.csv', combined)
    summary = []
    for key in sorted({(r['kind'], r['split'], r['baseline'], r['candidate']) for r in combined}):
        group = [r for r in combined if (r['kind'], r['split'], r['baseline'], r['candidate']) == key]
        for metric in ('whole_body_preserved_vs_base', 'C_use_preserved_vs_base',
                       'visible_crack_change', 'fragment_change', 'visible_join_change'):
            for value, n in sorted(collections.Counter(r[metric] for r in group).items()):
                summary.append(dict(zip(('kind', 'split', 'baseline', 'candidate'), key),
                                    judgment=metric, value=value, count=n, denominator=len(group)))
    csvwrite(OUT / 'reviews/summary.csv', summary)
    write(OUT / 'reviews/completion.json', dict(status='COMPLETE_ACTUALLY_VIEWED',
          pages=112, paired_rows=224, model_calls=0, review_type='AI_PROXY_NONBLIND',
          limits='Relative visual comparisons, not new absolute adoption or topology certification.',
          files={str(p.relative_to(OUT)): sha(p) for p in sorted((OUT/'reviews').glob('*.csv'))}))

    registration = read(OUT / 'registration.json')
    assert sha(OUT/'PROTOCOL.md') == registration['protocol_sha256']
    assert sha(OUT/'USER_ATTACHMENT.txt') == registration['attachment_sha256']
    old = ROOT / 'experiments/r2_seam_abc_v1'
    # Locate the exact frozen source manifest/completion by registered digest,
    # without guessing a different filename or recomputing any old results.
    digests = {sha(p) for p in old.glob('*.json')}
    digests |= {sha(p) for p in (old/'evaluation').glob('*.json')}
    assert registration['source_completion_sha256'] in digests
    assert registration['source_manifest_sha256'] in digests
    la = read(OUT/'layout/audit.json')
    ba = read(OUT/'binding/array_integrity_audit.json')
    assert la['status'] == ba['status'] == 'PASS'
    assert (la['generator_forwards'], la['D0_forwards'], la['training_updates'], la['head_forwards']) == (1400, 28, 0, 0)
    assert la['all_RAW_before_D0'] and la['source_and_code_hashes_unchanged']
    assert ba['cells'] == 112 and ba['model_calls'] == 0
    aa = read(OUT/'analysis/completion.json')
    assert aa['outputs'] == 336 and aa['numeric_fields'] == 124
    assert len(rows(OUT/'analysis/per_output.csv')) == 336
    assert all(r['status'] == 'PASS' for r in rows(OUT/'analysis/summary.csv'))

    evidence = OUT / 'evidence'
    assert (evidence/'source_hashes.csv').exists(), 'Wait for actual scoring-source archive'
    code_files = sorted(Path(__file__).parent.glob('*.py'))
    code_rows, diff = [], []
    for p in code_files:
        target = evidence/'pilot_sources'/p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        code_rows.append(dict(source=str(p), sha256=sha(p), snapshot=str(target.relative_to(OUT))))
        diff.extend(difflib.unified_diff([], p.read_text(encoding='utf-8-sig').splitlines(keepends=True),
                                        fromfile='/dev/null', tofile='b/'+str(p.relative_to(ROOT))))
    csvwrite(evidence/'pilot_hashes.csv', code_rows)
    (evidence/'code.diff').write_text(''.join(diff), encoding='utf-8')
    for s in ('REPORT.md', 'interpretation_review.md'):
        text = (OUT/s).read_text(encoding='utf-8-sig')
        assert '\ufffd' not in text
    files = [OUT/'REPORT.md', OUT/'PROTOCOL.md', OUT/'registration.json',
             OUT/'reviews/completion.json', OUT/'analysis/completion.json',
             OUT/'layout/audit.json', OUT/'binding/array_integrity_audit.json',
             evidence/'source_hashes.csv', evidence/'pilot_hashes.csv', evidence/'code.diff']
    write(OUT/'completion.json', dict(status='COMPLETE', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          budget_actual=registration['budget'], saved_metric_rows=336, original_numeric_fields=124,
          paired_visual_rows=224, actually_viewed_pages=112, no_automatic_continuation=True,
          ideas_3_4='DESIGN_REVIEW_ONLY_NO_TRAINING',
          engineering_event='One pre-forward scalar-type setup failure, 0 forwards, disclosed and repaired.',
          audit='PASS', files={str(p.relative_to(OUT)): sha(p) for p in files}))
    print(json.dumps(read(OUT/'completion.json'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
