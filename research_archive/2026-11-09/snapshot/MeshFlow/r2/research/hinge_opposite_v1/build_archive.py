"""Rebuild the explicit evidence archive from immutable local experiment files.

No model, scorer, GPU, geometry execution, downloads or Git operations. Copies
are byte-identical. Existing different destinations are refused. Preparation
manifests are regenerated only inside this archive directory.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / 'experiments/r2_hinge_opposite_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def copy(source, relative):
    destination = HERE / relative
    if not source.is_file():
        raise FileNotFoundError('BLOCKED_MISSING_ARCHIVE_SOURCE: ' + str(source))
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError('REFUSE_CHANGED_DESTINATION: ' + str(destination))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return dict(source=str(source), archive_relative_path=relative,
                sha256=sha(destination), bytes=destination.stat().st_size)


def main():
    manifest = json.loads((HERE/'source_manifest.json').read_text(encoding='utf-8-sig'))
    files = {r['archive_relative_path']: r for r in manifest['source_files']}
    additions = [
        'r2/pilots/seam_hinge_quality_v1/hinge_audit.py',
        'r2/pilots/seam_followup_v1/gallery.py',
        'r2/pilots/seam_abc_v1/render.py',
        'r2/pilots/regular_patch_control_v1/depth_gallery.py',
        'r2/pilots/regular_patch_control_v1/common.py',
        'r2/tools/cref_layer_audit_v1/render.py',
    ]
    additions += ['r2/tools/hinge_opposite_v1/'+n for n in (
        '__init__.py','engine.py','run.py','test_contract.py','audit.py',
        'analyze.py','report.py','gallery.py','metric_policy.json','README.md')]
    for relative in list(additions):
        p = (REPO/relative).parent
        while p != REPO:
            if (p/'__init__.py').is_file():
                additions.append((p/'__init__.py').relative_to(REPO).as_posix())
            p = p.parent
    for relative in sorted(set(additions)):
        target = 'source/repository/'+relative
        record = copy(REPO/relative, target)
        record['kind'] = ('NEW_EXPERIMENT_SOURCE' if '/hinge_opposite_v1/' in relative
                          else 'REPOSITORY_CPU_DEPENDENCY')
        files[target] = record
    for record in files.values():
        assert sha(HERE/record['archive_relative_path']) == record['sha256']
    manifest.update(status='FINAL_ACTUAL_SOURCE_SNAPSHOT',
                    source_files=sorted(files.values(), key=lambda r:r['archive_relative_path']),
                    source_file_count=len(files), source_bytes=sum(r['bytes'] for r in files.values()),
                    import_check='PASS_ISOLATED_ARCHIVE_ENGINE_IMPORT',
                    array_decode_check='PASS_608b46b0_n0_015_ALL_63_ARRAYS_BYTE_IDENTICAL',
                    new_code_scope='Final engine, runner, independent audit, analysis, report, renderer, policy and contract tests')
    write_json(HERE/'source_manifest.json', manifest)

    evidence=[]
    top_names=[
        'REPORT.md','retained_profile.json','combined_per_output.csv','metric_summary.csv',
        'paired_summary.csv','paired_per_output.csv','paired_per_parent.csv','all_regressions.csv',
        'new_per_output.csv','connection_ledger.csv','rejection_funnel.csv','rejection_totals.json',
        'code.diff','protocol.json','engineering_check.json','execution_complete.json',
        'analysis_complete.json','independent_audit.json','independent_statistics_validation.json',
        'dispatch.json','ledger.jsonl','resources_engineering_before.json','resources_engineering_after.json']
    for name in top_names:
        evidence.append(copy(OUT/name,'evidence/'+name))
    for folder in ('gallery','independent_audit'):
        for p in sorted((OUT/folder).iterdir()):
            if p.is_file() and p.suffix in ('.png','.json','.csv','.html'):
                evidence.append(copy(p,'evidence/'+folder+'/'+p.name))
    evidence.append(copy(REPO/'experiments/r2_hinge_baseline_metric_review_v1/metric_catalog.csv',
                         'evidence/metric_catalog_original84.csv'))
    evidence.append(copy(OUT/'archive_decode_check/608b46b0_n0_015/receipt.json',
                         'validation/portable_decode_receipt.json'))

    # Preserve exact source records, including every candidate/rejection and edge
    # identity, without shipping raw meshes or the large per-query pair cache.
    cell_names=('baseline_edge_fates.csv','candidates.csv','candidate_records.json',
                'complete.json','hinge.json','illegal_pair_fates.csv','illegal_pairs_by_role.csv',
                'new_edges.csv','paired_ledger.json','valid_edges.csv','shape_per_face.json',
                'per_face_connectivity.csv','seam_bulk_faces.csv')
    cells=sorted((OUT/'cells').glob('HINGE_OPP*/*'))
    assert len(cells)==56, len(cells)
    zpath=HERE/'evidence/per_cell_ledgers.zip'
    entries=[]
    # Deterministic ZIP entry metadata; this is archive-only byte compression.
    import io
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for cell in cells:
            for name in cell_names:
                p=cell/name
                assert p.is_file(),p
                arc=p.relative_to(OUT).as_posix()
                info=zipfile.ZipInfo(arc,date_time=(2026,10,18,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(info,p.read_bytes(),compresslevel=9)
                entries.append(dict(path=arc,sha256=sha(p),bytes=p.stat().st_size))
    data=buffer.getvalue()
    if zpath.exists() and zpath.read_bytes()!=data:
        raise RuntimeError('REFUSE_CHANGED_CELL_LEDGER_ARCHIVE')
    zpath.write_bytes(data)
    ledger_index=HERE/'evidence/per_cell_ledger_manifest.json'
    write_json(ledger_index,dict(status='COMPLETE',cells=56,files=entries,
                               excluded='NPZ, dense pairs.json/query payload and connectivity cache; originals remain protected locally'))
    for p in (zpath,ledger_index):
        evidence.append(dict(source='deterministic byte-preserving archive of original cell ledgers',
                             archive_relative_path=p.relative_to(HERE).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    write_json(HERE/'evidence_manifest.json',dict(status='FINAL',files=evidence,
                 evidence_file_count=len(evidence),evidence_bytes=sum(r['bytes'] for r in evidence),
                 original_assets='NOT_MODIFIED',models='NOT_DISTRIBUTED',
                 generated_images=28,actual_AI_PROXY_reviewed_requests=7,
                 metrics='140 rows x 135 numeric fields; original metric catalog records prior 84-row coverage'))
    print(json.dumps(dict(status='PASS_ARCHIVE_BUILD',source_files=len(files),
                          source_bytes=manifest['source_bytes'],evidence_files=len(evidence),
                          evidence_bytes=sum(r['bytes'] for r in evidence),
                          compressed_cell_ledgers=len(data)),indent=2))


if __name__=='__main__':
    main()
