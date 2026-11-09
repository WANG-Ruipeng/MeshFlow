"""Seal or verify the completed CPU check. Never decode, score, or load a model."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .register import OUT, ROOT, read, sha, write, resources


def count(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return sum(1 for _ in csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    target = OUT / 'completion.json'
    if not args.seal:
        receipt = read(target)
        for path, expected in receipt['deliverable_and_code_sha256'].items():
            assert sha(ROOT / path) == expected, path
        print(json.dumps({'status': 'VERIFIED', 'hashed_files': len(receipt['deliverable_and_code_sha256']), 'geometry_calls': 0, 'model_calls': 0}))
        return
    if target.exists():
        raise FileExistsError('Already sealed; run without --seal to verify.')
    stages = {
        'hinge': OUT/'hinge/completion.json',
        'authoritative_tables': OUT/'hinge/tables_v2/completion.json',
        'deletion': OUT/'deletion/completion.json',
        'point_contact': OUT/'point_contact/completion.json',
        'point_review': OUT/'audits/point_contact_independent_review.json',
        'visual_review': OUT/'visual_review.json',
    }
    receipts = {key: read(path) for key, path in stages.items()}
    for key, result in receipts.items():
        assert result['status'] in ('COMPLETE', 'PASS'), key
    assert receipts['hinge']['cells'] == 96
    assert receipts['authoritative_tables']['rows'] == 192
    assert receipts['authoritative_tables']['numeric_field_count'] == 135
    assert count(OUT/'point_contact/per_output.csv') == 96
    assert count(OUT/'deletion/per_output.csv') == 96
    assert count(OUT/'deletion/per_face.csv') == 18688
    assert receipts['visual_review']['pages'] == 3
    report = OUT/'REPORT.md'
    assert report.exists()
    # Read-only phase-end resource snapshot. nvidia-smi inspection is not GPU work.
    resource_path = OUT/'resources_after.json'
    if not resource_path.exists():
        write(resource_path, resources())
    selected = set(stages.values()) | {report, resource_path, OUT/'registration.json', OUT/'resources_before.json'}
    for folder, patterns in (
        (OUT/'hinge/tables_v2', ('*.csv',)),
        (OUT/'hinge/tables', ('connectivity_ledger.csv', 'illegal_pairs_by_role.csv')),
        (OUT/'hinge/cached_followup', ('*.json', '*.csv')),
        (OUT/'point_contact', ('*.csv', '*.json')),
        (OUT/'deletion', ('*.csv', 'FINDINGS.md', 'minimum_edge_area_summary.json')),
        (OUT/'gallery/RAW_PRE_GEO_POST', ('*.png', '*.html')),
        (Path(__file__).parent, ('*.py', 'README.md')),
    ):
        for pattern in patterns:
            selected.update(folder.glob(pattern))
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(selected) if p.is_file()}
    result = dict(status='COMPLETE', version='bench_post_rescue_diagnostic_v1',
        fixed_requests=48, checkpoints=2, hinge_outputs=96, candidate_original_scores=96,
        reused_original_scores=96, numeric_fields=135, point_contact_outputs=96,
        deletion_outputs=96, deletion_source_slot_rows=18688, new_visual_pages_actually_reviewed=3,
        training_updates=0, neural_forwards=0, GPU_compute_calls=0, new_noise_arrays=0, new_query_arrays=0,
        authoritative_tables='hinge/tables_v2', old_asset_hashes_verified_by_hinge=receipts['hinge']['source_assets_verified'],
        hinge_workers=receipts['hinge']['workers'], hinge_wall_seconds=receipts['hinge']['wall_seconds'],
        scope='Completed both authorized CPU checks; no default changes, cleanup, commit, or push.',
        verification_scope='This entry verifies cached deliverables and code only; individual stage receipts preserve input/code hashes and engineering corrections.',
        deliverable_and_code_sha256=hashes)
    write(target, result)
    print(json.dumps({key: result[key] for key in ('status', 'hinge_outputs', 'numeric_fields', 'neural_forwards', 'GPU_compute_calls')}))


if __name__ == '__main__':
    main()
