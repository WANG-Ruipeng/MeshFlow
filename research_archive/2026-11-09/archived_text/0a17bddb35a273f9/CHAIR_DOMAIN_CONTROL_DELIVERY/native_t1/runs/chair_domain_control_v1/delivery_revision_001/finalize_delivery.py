"""Append audited delivery notes without replaying any scientific operation."""
from pathlib import Path
import argparse
import csv
import difflib
import json
import shutil
import time
from experiments.mf_h1_local0.chair_domain_control_v1.common import read_json, digest, atomic_json, append_event
from experiments.mf_h1_local0.chair_domain_control_v1.run_all import worker_status, code_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    revision = out/'delivery_revision_001'
    stage = dict(module='reporting', args=[])
    assert read_json(out/'run_state.json')['status'] == 'DONE'
    worker_status(out, stage)
    code_snapshot(out, verify=True)
    if (revision/'revision.json').exists():
        print('Verified existing delivery revision; no scientific work dispatched.')
        return
    assert digest(out/'registration/requested_protocol.md') == '135c5dc2fb44f64dae76def4cbd5b40c06906229d9c3ae9065c2a888e6bf3bfc'
    for source, dest in [('CHAIR_DOMAIN_CONTROL_REPORT.md','automatic_report.md'),
                         ('reporting_status.json','automatic_reporting_status.json')]:
        target = revision/dest
        if target.exists():
            assert digest(target) == digest(out/source)
        else:
            shutil.copy2(out/source, target)
    report = read_json(out/'reporting_status.json')
    layout = read_json(out/'figures/layout_v2/manifest.json')
    assert layout['status'] == 'PASS' and len(layout['figures']) == 48
    assert layout['camera_extent_and_dependencies_identical']
    for row in layout['figures']:
        assert digest(out/row['path']) == row['sha256']
        assert digest(out/row['original_path']) == row['original_sha256']
    events = [json.loads(line) for line in (out/'execution_ledger.jsonl').read_text().splitlines() if line]
    pending, numbers, attempts = {}, {}, []
    before = read_json(out/'incidents/001_atomic_replace_permission/ledger.json')
    after = read_json(out/'evaluation/phase0/Full/ledger.json')
    for event in events:
        key = event.get('stage')
        if event['event'] == 'STAGE_BEGIN':
            assert key not in pending
            numbers[key] = numbers.get(key, 0)+1
            pending[key] = event
        elif event['event'] == 'STAGE_END':
            started = pending.pop(key)
            row = dict(stage=key, attempt=numbers[key], status=event['status'],
                returncode=event['returncode'], started_unix=started['time_unix'],
                completed_unix=event['completed_unix'], parent_elapsed_seconds=event['elapsed_seconds'],
                worker_seconds='NA', model_load_seconds='NA', sampling_forward_attempts='NA',
                teacher_forward_attempts='NA', note='Outer worker attempt; nested timing is separate in resource_summary.csv')
            if key == 'eval_Full_0':
                book = before if row['attempt'] == 1 else after
                row.update(worker_seconds=book.get('worker_seconds','NA'), model_load_seconds=book['model_load_seconds'],
                    sampling_forward_attempts=2000 if row['attempt']==1 else 0,
                    teacher_forward_attempts=0 if row['attempt']==1 else 24,
                    note='I/O stop after complete raw; no generation replay' if row['attempt']==1 else
                        'Sampling SKIPPED_COMPLETED; old request peaks retained, not new sampling peaks; 24 teacher + 40 CPU scores')
            attempts.append(row)
    assert not pending
    assert len(attempts) == 24 and sum(row['status']=='PASS' for row in attempts)==23
    special = [row for row in attempts if row['stage']=='eval_Full_0']
    assert len(special) == 2 and [row['status'] for row in special] == ['STOPPED','PASS']
    with (out/'execution_attempts.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(attempts[0])); writer.writeheader(); writer.writerows(attempts)
    diff = []
    for name in ('render_layout.py','finalize_delivery.py'):
        path = revision/name
        diff.extend(difflib.unified_diff([], path.read_text().splitlines(True), fromfile='/dev/null',
            tofile='b/native_t1/runs/chair_domain_control_v1/delivery_revision_001/'+name))
    (out/'delivery_only_code.diff').write_text(''.join(diff), encoding='utf-8')
    notes_path = out/'DELIVERY_NOTES.md'
    notes = notes_path.read_text().replace('display_only_code.diff','delivery_only_code.diff')
    notes = notes.replace('\u4ec5\u6392\u7248\u4ee3\u7801diff','\u4ea4\u4ed8\u4fee\u8ba2\u4ee3\u7801diff')
    notes_path.write_text(notes, encoding='utf-8')
    original = (revision/'automatic_report.md').read_text()
    title, budget, remaining = original.split('\n\n', 2)
    (out/'CHAIR_DOMAIN_CONTROL_REPORT.md').write_text(title+'\n\n'+budget+'\n\n'+notes+'\n\n'+remaining, encoding='utf-8')
    change = dict(status='PASS', schema='chair_domain_delivery_annotation_v1', time_unix=time.time(),
        original_report_sha256=digest(revision/'automatic_report.md'),
        original_reporting_status_sha256=digest(revision/'automatic_reporting_status.json'),
        revised_report_sha256=digest(out/'CHAIR_DOMAIN_CONTROL_REPORT.md'),
        frozen_scientific_code_diff_sha256=digest(out/'code.diff'),
        rendering_manifest_sha256=digest(out/'figures/layout_v2/manifest.json'),
        rendering_seconds=layout['elapsed_seconds'],
        IO_stage_parent_seconds_all_attempts=sum(row['parent_elapsed_seconds'] for row in special),
        worker_attempts=len(attempts), scientific_stages=23, model_forwards=0, optimizer_updates=0,
        OT_calls=0, independent_geometry_score_calls=0, scientific_results_unchanged=True,
        note='Archived automatic report/status. Added transparent I/O accounting and 48 layout-only copies; original plots retained.')
    atomic_json(revision/'revision.json', change)
    extra = ['DELIVERY_NOTES.md','execution_attempts.csv','delivery_only_code.diff','figures/layout_v2/manifest.json',
             'delivery_revision_001/render_layout.py','delivery_revision_001/finalize_delivery.py',
             'delivery_revision_001/automatic_report.md','delivery_revision_001/automatic_reporting_status.json',
             'delivery_revision_001/revision.json']
    extra += [row['path'] for row in layout['figures']]
    index = {item['path']:item for item in report['artifacts']}
    for name in ['CHAIR_DOMAIN_CONTROL_REPORT.md']+extra:
        index[name] = dict(path=name, sha256=digest(out/name))
    report['artifacts'] = [index[name] for name in sorted(index)]
    report['delivery_revision'] = 'delivery_revision_001/revision.json'
    atomic_json(out/'reporting_status.json', report)
    worker_status(out, stage)
    code_snapshot(out, verify=True)
    append_event(out/'execution_ledger.jsonl', dict(event='DELIVERY_ANNOTATION', status='PASS',
        revision_path='delivery_revision_001/revision.json', revision_sha256=digest(revision/'revision.json'),
        model_forwards=0, optimizer_updates=0, OT_calls=0, score_calls=0))
    print(json.dumps(dict(status='PASS', artifacts=len(report['artifacts']), parent_attempts=len(attempts),
                          scientific_stages=23, added_model_forwards=0, added_score_calls=0)))


if __name__ == '__main__':
    main()
