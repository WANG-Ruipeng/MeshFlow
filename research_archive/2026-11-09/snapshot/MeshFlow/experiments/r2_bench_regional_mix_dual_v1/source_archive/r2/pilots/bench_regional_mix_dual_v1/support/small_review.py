"""Lossless smaller review: omit only six auxiliary PRE-selection PNG pages."""
import hashlib
import io
import json
from pathlib import PurePosixPath
import zipfile
from r2.pilots.bench_regional_mix_dual_v1.common import OUT, read, sha, write, utc
from .delivery import compress, encoded, digest


def run():
    done = read(OUT / 'delivery/complete.json')
    full = OUT / 'delivery/REVIEW.zip'
    assert done['status'] == 'COMPLETE' and sha(full) == done['zip_sha256']
    target = OUT / 'delivery/REVIEW_SMALL.zip'
    receipt = OUT / 'delivery/small_complete.json'
    if receipt.exists():
        old = read(receipt)
        assert sha(target) == old['zip_sha256']
        return old
    assert not target.exists(), 'Partial small package: do not overwrite'
    with zipfile.ZipFile(full) as z:
        files = {name: z.read(name) for name in z.namelist()}
    omitted = []
    for name in list(files):
        if name.startswith('panel/selection_images/') and name.endswith('.png'):
            data = files.pop(name)
            omitted.append(dict(path=name, bytes=len(data), sha256=digest(data),
                reason='Auxiliary pre-output assisted-selection image only; retained locally and in full REVIEW.zip. Exact region IDs/rules and all 24 actual result PNG pages remain in this small package.'))
    assert len(omitted) == 6
    assert len([p for p in files if p.startswith('runs/') and p.endswith('/gallery/CLEAN_FINAL_POST.png')]) == 24
    original = json.loads(files.pop('PACKAGE_MANIFEST.json'))
    artifact = json.loads(files['delivery/final_artifact_manifest.json'])
    omitted_names = {x['path'] for x in omitted}
    for r in artifact['files']:
        if r['relative_path'] in omitted_names:
            r['included_in_review'] = False
            r['small_review_omission'] = 'Only pre-selection auxiliary PNG; in full ZIP/local original'
    artifact['small_review_omissions'] = omitted
    files['delivery/final_artifact_manifest.json'] = encoded(artifact)
    files['SMALL_PACKAGE_SCOPE.json'] = encoded(dict(
        full_zip_sha256=sha(full), full_zip_bytes=full.stat().st_size,
        omitted=omitted, all_24_result_PNGs_preserved=True,
        complete_177_parent_output_slot_selection_connectivity_tables_preserved=True,
        actual_visual_review_and_source_preserved=True, geometry_or_model_calls=0))
    files['README_REVIEW.txt'] += b'\nSMALL PACKAGE: only six auxiliary pre-selection PNGs are omitted. All 24 FINAL comparison PNGs, all original metric and ledger tables, actual visual reviews and code remain. See SMALL_PACKAGE_SCOPE.json for exact omitted hashes.\n'
    original['small_review_omissions'] = omitted
    original['members'] = [dict(path=name, bytes=len(data), sha256=digest(data)) for name, data in sorted(files.items())]
    files['PACKAGE_MANIFEST.json'] = encoded(original)
    data = compress(files, zipfile.ZIP_DEFLATED)
    method = 'DEFLATE'
    if len(data) >= 10_000_000:
        alternative = compress(files, zipfile.ZIP_LZMA)
        if len(alternative) < len(data):
            data, method = alternative, 'LZMA'
    with target.open('xb') as h:
        h.write(data)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert all(digest(z.read(n)) == digest(d) for n, d in files.items())
    result = dict(status='COMPLETE', utc=utc(), zip_path=str(target), zip_sha256=sha(target),
        bytes=len(data), decimal_MB=len(data) / 1_000_000, below_10MB=len(data) < 10_000_000,
        compression=method, original_gallery_pages=24, omitted_auxiliary_pages=6,
        full_zip_sha256=sha(full), complete_evidence_except_auxiliary_PRE_selection_PNG=True,
        scientific_calls=0, new_visual_labels=0)
    write(receipt, result)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False))
