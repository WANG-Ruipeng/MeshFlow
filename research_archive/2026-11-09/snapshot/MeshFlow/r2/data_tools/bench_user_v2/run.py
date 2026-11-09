"""Recoverable CPU-only entry for the already frozen Bench v2 batch.

No new parents, neural network, training, CUDA work or default-profile changes.
"""
from __future__ import annotations
import argparse
from . import build


def verify():
    from .finalize import verify as verify_data
    verify_data()
    ready=build.read(build.OUT/'DATASET_READY.json')
    assert build.sha(ready['manifest_path'])==ready['manifest_sha256']
    assert build.sha(ready['validation_receipt'])==ready['validation_receipt_sha256']
    assert build.read(ready['validation_receipt'])['status']=='PASS'
    for path,h in build.read(build.OUT/'code_hashes.json').items():assert build.sha(path)==h,path
    for path,h in build.read(build.OUT/'artifact_hashes.json').items():assert build.sha(build.OUT/path)==h,path
    original=build.read(build.OUT/'protected_original_assets_verified.json')
    for path,h in original['files'].items():assert build.sha(path)==h,path
    print('BENCH_V2_SEALED_DELIVERY_PASS; 0 new OT, 0 models, 0 GPU, 0 training')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('build','render','finalize','validate','closeout','verify'),default='verify',nargs='?')
    a=p.parse_args()
    if a.stage=='build':
        if not (build.OUT/'audit/latest.json').exists():
            raise RuntimeError('Run audit_existing first; do not substitute unaudited sources')
        build.freeze();build.targets(4);build.conditions(4)
    elif a.stage=='render':
        if (build.OUT/'gallery/render_completion.json').exists():
            receipt=build.read(build.OUT/'gallery/review_v1/manifest_all.json')
            assert receipt['manifest_sha256']==build.sha(build.OUT/'candidate_manifest.json')
            for row in receipt['pages']:
                expected=row.get('image_sha256') or row.get('sha256')
                assert expected and build.sha(row['image_path'])==expected
            print('EXISTING_RENDER_VERIFIED_NO_RERENDER')
        else:
            from .render import run
            run(build.OUT/'candidate_manifest.json',mode='all',threads=2,output_name='review_v1')
    elif a.stage=='finalize':
        if not (build.OUT/'visual_review_notes.json').exists():
            raise RuntimeError('ACTUAL_IMAGE_REVIEW_REQUIRED; no automatic semantic admission')
        from .finalize import seal_review_notes,finalize
        seal_review_notes();finalize()
    elif a.stage=='validate':
        from .validate import validate
        result=validate(build.OUT/'manifest.json')
        if result['status']!='PASS':raise SystemExit(2)
    elif a.stage=='closeout':
        if (build.OUT/'DATASET_READY.json').exists():verify()
        else:
            from .closeout import main as closeout
            closeout()
    else:verify()


if __name__=='__main__':main()
