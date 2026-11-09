#!/usr/bin/env python3
"""Idempotent CPU-only source delivery finalizer for the fixed schedule40 run.

Reads and verifies registered source bytes. Never imports delivered Python modules,
executes scientific code, touches weights/RAW, or changes original hash manifests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

DEFAULT = '/mnt/e/MeshFlow-Control-artifacts/experiments/chair_early_alignment_schedule40_v1'
DEFAULT_REPO = '/mnt/e/MeshFlow-Control'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked_bytes(path, expected=None):
    path = Path(path)
    data = path.read_bytes()
    actual = sha(data)
    if expected is not None and actual != expected:
        raise ValueError(f'Source identity differs: {path}: expected {expected}, actual {actual}')
    return data


def preserve_or_create(path, data):
    """Existing bytes are immutable; repeat execution performs only comparison."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refusing to overwrite an existing differing artifact: ' + str(path))
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(data)
    return True


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')


def added_file_diff(relative, raw):
    """Full addition, including exact BOM/CRLF/no-final-newline source bytes."""
    raw.decode('utf-8')
    lines = raw.splitlines(keepends=True)
    header = (f'diff --git a/{relative} b/{relative}\nnew file mode 100644\n'
              f'--- /dev/null\n+++ b/{relative}\n').encode('utf-8')
    if not lines:
        return header
    count = len(lines)
    target_range = '1' if count == 1 else f'1,{count}'
    chunks = [header, f'@@ -0,0 +{target_range} @@\n'.encode('utf-8')]
    for line in lines:
        chunks.append(b'+' + line)
        if not line.endswith(b'\n'):
            chunks.append(b'\n\\ No newline at end of file\n')
    block = b''.join(chunks)
    # Independently extract the single full-addition hunk and compare source bytes.
    body = block.split(b'@@\n', 1)[1]
    recovered = []
    for line in body.splitlines(keepends=True):
        if line.startswith(b'+'):
            recovered.append(line[1:])
        elif line == b'\\ No newline at end of file\n':
            if not recovered or not recovered[-1].endswith(b'\n'):
                raise AssertionError('Malformed no-final-newline marker')
            recovered[-1] = recovered[-1][:-1]
        else:
            raise AssertionError('Unexpected addition hunk content')
    if b''.join(recovered) != raw:
        raise AssertionError('Full addition diff did not preserve source bytes: ' + relative)
    return block


def finalize(root, repo):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    audit, snapshot_root = root/'audit', root/'source_code'
    original_path = audit/'source_hashes_before.json'
    new_path = audit/'new_source_hashes.json'
    analysis_path = audit/'postfreeze_analysis_hashes.json'
    protected_paths = [original_path, new_path, analysis_path]
    protected_before = {str(path.relative_to(root)): sha(path.read_bytes()) for path in protected_paths}
    original, new, analysis = load(original_path), load(new_path), load(analysis_path)
    if (len(original), len(new), len(analysis)) != (63, 12, 3):
        raise ValueError('Unexpected source-manifest scope; refusing implicit expansion')
    sources = {}

    def register(relative, live, expected, group, authority=None):
        destination = snapshot_root/relative
        if '..' in Path(relative).parts or destination.resolve().parent != snapshot_root.resolve() and snapshot_root.resolve() not in destination.resolve().parents:
            raise ValueError('Destination leaves source_code: ' + relative)
        raw = checked_bytes(live, expected)
        record = sources.get(relative)
        if record:
            if record['sha256'] != sha(raw):
                raise ValueError('Conflicting source registrations: ' + relative)
            record['groups'].append(group)
            return
        sources[relative] = dict(live_path=str(Path(live).resolve()),
                                 snapshot_path=str(destination), sha256=sha(raw), bytes=len(raw),
                                 groups=[group], authority=authority, raw=raw)

    for relative, expected in original.items():
        register(relative, repo/relative, expected, 'original_frozen_63', 'audit/source_hashes_before.json')
    for relative, expected in new.items():
        register(relative, repo/relative, expected, 'existing_new_repo_12', 'audit/new_source_hashes.json')
    for relative, record in analysis.items():
        register(relative, record['live_path'], record['sha256'], 'existing_postfreeze_analysis_3', 'audit/postfreeze_analysis_hashes.json')

    r1 = load(audit/'dispatch_r1/registration.json')
    resource = load(audit/'analyze_actual_resources_preparation.json')
    guard = load(audit/'serial_tail_guard/source_identity.json')
    environment = load(audit/'environment.json')
    guard_live = audit/'guard_serial_tail.py'
    checked_bytes(audit/'serial_tail_guard/source/guard_serial_tail.py', guard['sha256'])
    additions = [
        ('tools/run_schedule40_dispatch_r1.py', repo/'tools/run_schedule40_dispatch_r1.py', r1['r1_sha256'], 'audit/dispatch_r1/registration.json'),
        ('tests/test_schedule40_dispatch_r1.py', repo/'tests/test_schedule40_dispatch_r1.py', r1['test_sha256'], 'audit/dispatch_r1/registration.json'),
        ('experiment_audit/guard_serial_tail.py', guard_live, guard['sha256'], 'audit/serial_tail_guard/source_identity.json'),
        ('experiment_audit/analyze_actual_resources.py', audit/'analyze_actual_resources.py', resource['source_sha256'], 'audit/analyze_actual_resources_preparation.json'),
        ('experiment_audit/capture_environment.py', audit/'capture_environment.py', environment['source_sha256'], 'audit/environment.json'),
        ('experiment_audit/finalize_source_delivery.py', Path(__file__).resolve(), None, 'self: SHA256 registered by this CPU-only archive run')
    ]
    for relative, live, expected, authority in additions:
        register(relative, live, expected, 'independent_engineering_addition_6', authority)

    # Preflight every existing snapshot before creating any new snapshot.
    for relative, record in sources.items():
        target = snapshot_root/relative
        if target.exists():
            checked_bytes(target, record['sha256'])
        elif 'independent_engineering_addition_6' not in record['groups']:
            raise ValueError('Previously registered snapshot missing: ' + str(target))

    prior_diff = root/'code.diff'
    archive = audit/'engineering/code_diff_before_delivery.diff'
    if not archive.exists():
        preserve_or_create(archive, prior_diff.read_bytes())
    archived_bytes = archive.read_bytes()
    delta_paths = list(new) + list(analysis) + [item[0] for item in additions]
    if len(delta_paths) != 21 or len(set(delta_paths)) != 21:
        raise ValueError('Unexpected full-addition diff manifest')
    final_diff = b'\n'.join(added_file_diff(relative, sources[relative]['raw']) for relative in delta_paths)
    if prior_diff.read_bytes() not in (archived_bytes, final_diff):
        raise ValueError('Top-level code.diff differs from both archived baseline and expected delivery')

    copied_now = []
    for relative, record in sources.items():
        if preserve_or_create(snapshot_root/relative, record['raw']):
            copied_now.append(relative)
    # Verify all archived code using reads only, including the original 63 dependencies.
    for relative, record in sources.items():
        checked_bytes(snapshot_root/relative, record['sha256'])
        checked_bytes(record['live_path'], record['sha256'])

    if prior_diff.read_bytes() != final_diff:
        # Only this separately archived review diff may be replaced. Scientific source
        # files, original manifests, old reports, weights and RAW remain untouched.
        temporary = prior_diff.with_name('code.diff.delivery.tmp')
        preserve_or_create(temporary, final_diff)
        temporary.replace(prior_diff)
    checked_bytes(prior_diff, sha(final_diff))
    if protected_before != {str(path.relative_to(root)): sha(path.read_bytes()) for path in protected_paths}:
        raise ValueError('A protected original source manifest changed during finalization')

    public_sources = {relative:{key:value for key,value in record.items() if key != 'raw'}
                      for relative,record in sources.items()}
    provenance_files = [audit/'dispatch_r1/registration.json', audit/'serial_tail_guard/source_identity.json',
                        audit/'analyze_actual_resources_preparation.json', audit/'environment.json']
    manifest = dict(schema='schedule40_delivery_sources_v1', status='PASS',
                    original_frozen_dependencies=63, existing_new_repo_sources=12,
                    existing_postfreeze_analysis_sources=3, independent_engineering_additions=6,
                    unique_source_snapshots=len(sources), full_addition_diff_files=len(delta_paths),
                    all_live_and_snapshot_sources_match=True, original_manifests_unchanged=True,
                    protected_original_manifest_sha256=protected_before,
                    independent_authority_sha256={str(p.relative_to(root)):sha(p.read_bytes()) for p in provenance_files},
                    archived_previous_diff=dict(path=str(archive.relative_to(root)),sha256=sha(archived_bytes),bytes=len(archived_bytes)),
                    final_diff=dict(path='code.diff',sha256=sha(final_diff),bytes=len(final_diff),
                                    paths=delta_paths,each_full_addition_round_trip_byte_verified=True),
                    scope='Source copying, hashes and documentation only; no imported scientific code, GPU work, scoring, weights or RAW.',
                    sources=public_sources)
    index = [
        '# Schedule40 source delivery index', '',
        'This delivery separates the original frozen scientific dependencies from independently registered engineering additions. '
        'All live sources and copied source bytes were checked by SHA256. This index is not a claim that every planned experimental phase has completed.', '',
        '- Original frozen dependency manifest: [audit/source_hashes_before.json](audit/source_hashes_before.json), 63 files, unchanged.',
        '- Existing experiment code manifest: [audit/new_source_hashes.json](audit/new_source_hashes.json), 12 files, unchanged; 5 also belong to the original frozen set.',
        '- Existing later CPU-analysis manifest: [audit/postfreeze_analysis_hashes.json](audit/postfreeze_analysis_hashes.json), 3 files, unchanged.',
        '- Independent engineering additions: 6 files, separately registered below; they do not replace the original frozen 63-file manifest.',
        f'- Unique copied sources: {len(sources)}. Full-addition review diff: {len(delta_paths)} files.',
        '- Complete identities: [audit/delivery_source_hashes.json](audit/delivery_source_hashes.json).',
        '- Previous review diff retained exactly: [code_diff_before_delivery.diff](audit/engineering/code_diff_before_delivery.diff).',
        '- Current [code.diff](code.diff) contains the 12 existing repository sources, 3 existing audit sources and 6 engineering additions as full additions; every added source round-trips to its exact bytes.', '',
        '## Independent additions and limits', '',
        '| Source | Purpose and boundary |', '| --- | --- |',
        '| tools/run_schedule40_dispatch_r1.py | Independent dispatch recovery correction. Clears a surviving worker\'s old parallel-retry eligibility; preserves bounded same-tick double-OOM handling. Original controller and scientific definitions remain frozen. |',
        '| tests/test_schedule40_dispatch_r1.py | CPU mock regressions for the dispatch boundary; no real GPU workers. |',
        '| experiment_audit/guard_serial_tail.py | Existing verified controller-only SIGSTOP/SIGCONT guard. Does not pause or signal the GPU worker; copied identity matches its own original guard snapshot. |',
        '| experiment_audit/analyze_actual_resources.py | CPU log audit assigns phase by timed worker identity, including training/sampling name reuse, and adds guard resource logs. Does not replace the original accounting output. |',
        '| experiment_audit/capture_environment.py | Existing environment-capture source, archived without executing it again. |',
        '| experiment_audit/finalize_source_delivery.py | This idempotent source-copy/index/diff finalizer; no scientific execution. |', '',
        '## Complete source inventory', '',
        'Group tags: F63 = original frozen dependencies; R12 = existing new repository sources; A3 = existing analysis scripts; E6 = separate engineering additions.', '',
        '| Copied source | Groups | Bytes | SHA256 |', '| --- | --- | ---: | --- |'
    ]
    labels = dict(original_frozen_63='F63',existing_new_repo_12='R12',existing_postfreeze_analysis_3='A3',independent_engineering_addition_6='E6')
    for relative,record in sorted(public_sources.items()):
        tags = ', '.join(labels[group] for group in record['groups'])
        index.append(f'| [{relative}](source_code/{relative}) | {tags} | {record["bytes"]} | `{record["sha256"]}` |')
    index.extend(['', '## Re-run', '', '```bash',
                  f'/mnt/e/MeshFlow-Control-artifacts/venv/bin/python -B {audit/"finalize_source_delivery.py"}',
                  '```', '',
                  'Repeated runs verify the same identities and preserve identical existing bytes. A changed or missing previously registered source blocks finalization instead of silently replacing it. '
                  'Only the review diff may be updated after its exact previous bytes are archived. No weights, RAW meshes or experiment results are copied by this tool.', ''])
    preserve_or_create(root/'SOURCE_INDEX.md', '\n'.join(index).encode('utf-8'))
    preserve_or_create(audit/'delivery_source_hashes.json', json_bytes(manifest))
    return dict(status='PASS', unique_source_snapshots=len(sources), original_frozen_checked=63,
                newly_copied_this_execution=copied_now, full_addition_diff_files=len(delta_paths),
                source_sha256=sha(Path(__file__).read_bytes()), final_diff_sha256=sha(final_diff),
                delivery_manifest_sha256=sha((audit/'delivery_source_hashes.json').read_bytes()),
                original_manifests_unchanged=True, gpu_or_scoring_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(DEFAULT))
    parser.add_argument('--repo', type=Path, default=Path(DEFAULT_REPO))
    args=parser.parse_args()
    print(json.dumps(finalize(args.out,args.repo),sort_keys=True))


if __name__ == '__main__':
    main()