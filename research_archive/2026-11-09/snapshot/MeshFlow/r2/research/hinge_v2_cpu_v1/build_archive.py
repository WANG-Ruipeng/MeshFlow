"""Copy the completed HINGE V2 CPU experiment into its publication archive.

This is file-only work: no model imports, mesh operations, scoring, rendering,
downloads or Git operations. Source and evidence payloads are byte-identical.
Existing different destinations are refused. Manifests are archive metadata.
Run from the original local workspace; fresh clones intentionally lack meshes.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD = HERE.parent / 'hinge_opposite_v1'
OUT = REPO / 'experiments/r2_hinge_v2_cpu_v1'
COMPARISON = REPO / 'experiments/r2_hinge_v2_comparison_20261018'
MAX_FILE_BYTES = 10 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def put(data, relative):
    path = HERE / relative
    if not path.resolve().is_relative_to(HERE.resolve()):
        raise ValueError('ARCHIVE_PATH_ESCAPE: ' + relative)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError('ARCHIVE_FILE_EXCEEDS_10_MIB: ' + relative)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError('REFUSE_CHANGED_DESTINATION: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return dict(archive_relative_path=relative, sha256=digest(data), bytes=len(data))


def copy(source, relative, **extra):
    if not source.is_file():
        raise FileNotFoundError('BLOCKED_MISSING_ARCHIVE_SOURCE: ' + str(source))
    return dict(source=source.as_posix(), **put(source.read_bytes(), relative), **extra)


def main():
    # Reuse the already-published exact dependency closure. This does not depend
    # on whether an unrelated working-tree dependency has since been edited.
    old_manifest = json.loads((OLD / 'source_manifest.json').read_text(encoding='utf-8-sig'))
    sources = {}
    for old in old_manifest['source_files']:
        data = (OLD / old['archive_relative_path']).read_bytes()
        if digest(data) != old['sha256']:
            raise ValueError('OLD_ARCHIVE_HASH_MISMATCH: ' + old['archive_relative_path'])
        record = dict(old)
        record.update(put(data, old['archive_relative_path']))
        record['copied_from_published_archive'] = 'r2/research/hinge_opposite_v1/' + old['archive_relative_path']
        sources[old['archive_relative_path']] = record
    additions = sorted((REPO / 'r2/tools/hinge_v2_cpu_v1').glob('*.py'))
    additions += [REPO / 'r2/tools/hinge_v2_cpu_v1/README.md', COMPARISON / 'export_comparison.py']
    for source in additions:
        relative = 'source/repository/' + source.relative_to(REPO).as_posix()
        sources[relative] = copy(source, relative, kind='HINGE_V2_FROZEN_SOURCE')
    frozen_sources = json.loads((OUT / 'source_hashes.json').read_text(encoding='utf-8-sig'))['files']
    for frozen in frozen_sources:
        original = frozen['source']
        if original.startswith('/mnt/e/MeshFlow/'):
            relative = 'source/repository/' + original.removeprefix('/mnt/e/MeshFlow/')
        elif original.startswith('/mnt/e/MeshFlow-Control/'):
            relative = 'source/legacy/' + original.removeprefix('/mnt/e/MeshFlow-Control/')
        else:
            raise ValueError('UNKNOWN_FROZEN_SOURCE_LOCATION: ' + original)
        if relative not in sources or sources[relative]['sha256'] != frozen['sha256']:
            raise ValueError('COMPLETED_EXPERIMENT_SOURCE_HASH_MISMATCH: ' + original)
    comparison_receipt = json.loads((COMPARISON / 'receipt.json').read_text(encoding='utf-8-sig'))
    exporter_path = 'source/repository/experiments/r2_hinge_v2_comparison_20261018/export_comparison.py'
    exporter_frozen = comparison_receipt['outputs']['/mnt/e/MeshFlow/experiments/r2_hinge_v2_comparison_20261018/export_comparison.py']
    if sources[exporter_path]['sha256'] != exporter_frozen:
        raise ValueError('COMPLETED_COMPARISON_EXPORTER_HASH_MISMATCH')
    write_json(HERE / 'source_manifest.json', dict(
        status='FINAL_ACTUAL_SOURCE_SNAPSHOT', version='hinge_v2_cpu_actual_source_archive_v1',
        source_files=sorted(sources.values(), key=lambda r: r['archive_relative_path']),
        source_file_count=len(sources), source_bytes=sum(r['bytes'] for r in sources.values()),
        completed_experiment_frozen_hash_checks=len(frozen_sources), comparison_exporter_frozen_hash_check='PASS',
        copy_policy='Exact original bytes; existing different destinations refused; old assets unmodified',
        dependency_scope='Published HINGE closure plus finite-alpha, exact-cycle bridge, scoring and all completed V2 audit/render/report code',
        assets='Frozen meshes, raw noise, model weights and dense pair/query caches remain local and are not distributed',
        runtime='Use replay_array.py with explicit arrays; original full-run scripts preserve historical local asset paths',
        validation='See separate validation receipts; file-only build does not assert an execution check'))

    evidence = []
    top = [
        'REPORT.md', 'protocol.json', 'geometry_complete.json', 'geometry_summary.csv',
        'metric_summary.csv', 'paired_per_output.csv', 'paired_per_parent.csv', 'paired_summary.csv',
        'all_regressions.csv', 'actual_vs_fixed_parent.csv', 'bulk_reach_identities.json',
        'categorical_status.csv', 'connection_audit_summary.json', 'core_metric_summary.csv',
        'core_paired_summary.csv', 'analysis_complete.json', 'analysis_metric_policy.json',
        'independent_audit.json', 'independent_score_pair_audit.json', 'independent_key_metrics_audit.json',
        'review_dev.csv', 'RUN_COMPLETE.json', 'source_hashes.json', 'code.diff', 'ledger.jsonl', 'dispatch.json',
    ]
    for name in top:
        evidence.append(copy(OUT / name, 'evidence/' + name))
    for name in ('per_output.csv', 'completion.json', 'protocol.json', 'dispatch.json'):
        evidence.append(copy(OUT / 'scores' / name, 'evidence/scores/' + name))
    # Original fixed views and actual AI_PROXY review records are preserved as
    # evidence. No new image selection or re-review is performed by this build.
    folders = ('gallery', 'local_geometry', 'independent_audit', 'diagnostics',
               'holes_census', 'preparation', 'visual')
    for folder in folders:
        for source in sorted((OUT / folder).iterdir()):
            if not source.is_file() or source.suffix not in ('.png', '.json', '.csv', '.html'):
                continue
            # Raw process listings are not publication evidence; original files
            # remain protected locally, with their hashes in completion records.
            if source.name.startswith('resources') or source.name == 'before_inventory.json':
                continue
            evidence.append(copy(source, 'evidence/' + folder + '/' + source.name))
    for name in ('FULL_COMPARISON.md', 'all_179_metrics.csv', 'per_parent_comparison.csv',
                 'per_output_two_stages.csv', 'receipt.json'):
        evidence.append(copy(COMPARISON / name, 'evidence/comparison/' + name))

    cell_names = ('candidates.csv', 'complete.json', 'records.json', 'started.json', 'summary.json')
    score_names = ('valid_edges.csv', 'shape_per_face.json', 'seam_bulk_faces.csv',
                   'illegal_pairs_by_role.csv', 'fixed_parent_detail.json', 'bridge_shape.json',
                   'row.json', 'score.json', 'complete.json')
    cells = sorted(p for p in (OUT / 'cells').glob('*/*') if p.is_dir())
    score_arms = ('BASELINE', 'MIDPOINT_REPLAY', 'ALPHA', 'HOLE', 'ALPHA_HOLE')
    scores = sorted(p for arm in score_arms for p in (OUT / 'scores' / arm).iterdir() if p.is_dir())
    if len(cells) != 112 or len(scores) != 140:
        raise ValueError(('UNEXPECTED_FROZEN_PANEL_SIZE', len(cells), len(scores)))
    members = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for directories, names in ((cells, cell_names), (scores, score_names)):
            for directory in directories:
                for name in names:
                    path = directory / name
                    if not path.is_file():
                        raise FileNotFoundError('MISSING_LEDGER: ' + str(path))
                    data = path.read_bytes()
                    relative = path.relative_to(OUT).as_posix()
                    info = zipfile.ZipInfo(relative, date_time=(2026, 10, 18, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, data, compresslevel=9)
                    members.append(dict(path=relative, sha256=digest(data), bytes=len(data)))
    zipped = put(buffer.getvalue(), 'evidence/per_cell_ledgers.zip')
    evidence.append(dict(source='Deterministic compression of exact original per-cell and per-score ledgers', **zipped))
    ledger_index = HERE / 'evidence/per_cell_ledger_manifest.json'
    write_json(ledger_index, dict(status='COMPLETE', geometry_cells=112, score_outputs=140,
        zip_path='evidence/per_cell_ledgers.zip', files=members,
        excluded='All NPZ arrays, dense pairs.json caches, raw/noise/model files; originals remain protected locally'))
    evidence.append(dict(source='Archive metadata indexing exact original ZIP members',
                         archive_relative_path='evidence/per_cell_ledger_manifest.json',
                         sha256=digest(ledger_index.read_bytes()), bytes=ledger_index.stat().st_size))
    write_json(HERE / 'evidence_manifest.json', dict(
        status='FINAL', version='hinge_v2_cpu_evidence_archive_v1', files=evidence,
        evidence_file_count=len(evidence), evidence_bytes=sum(r['bytes'] for r in evidence),
        original_assets='NOT_MODIFIED', new_scientific_execution='NONE_BY_ARCHIVE_BUILD',
        metrics='140 outputs, full 179 numeric fields, full per-output/per-parent pairings and regressions',
        fixed_gallery_images=28, explanatory_local_images=9,
        actual_AI_PROXY_reviewed_requests=dict(TRAIN=16, DEV=12, blind=False),
        models='NOT_DISTRIBUTED', raw_mesh_arrays='NOT_DISTRIBUTED',
        ledger_zip='evidence/per_cell_ledgers.zip', ledger_manifest='evidence/per_cell_ledger_manifest.json'))
    print(json.dumps(dict(status='PASS_FILE_ONLY_ARCHIVE_BUILD', source_files=len(sources),
        source_bytes=sum(r['bytes'] for r in sources.values()), evidence_files=len(evidence),
        evidence_bytes=sum(r['bytes'] for r in evidence), ledger_members=len(members),
        compressed_ledger_bytes=len(buffer.getvalue())), indent=2))


if __name__ == '__main__':
    main()
