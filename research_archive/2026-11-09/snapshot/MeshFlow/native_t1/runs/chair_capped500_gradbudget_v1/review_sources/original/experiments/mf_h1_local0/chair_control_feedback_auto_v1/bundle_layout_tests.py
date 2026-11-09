"""Small CPU-only semantic tests for exact duplicate archive reconstruction."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
from . import bundle_layout as layout
from . import assets


def entry(data, category='source'):
    return dict(data=data, sha256=hashlib.sha256(data).hexdigest(), source_path='/synthetic/source', category=category)


def run(out=assets.OUT):
    results=[]
    def check(name, func):
        func(); results.append(dict(name=name, status='PASS'))
    a='source_code/current/test.py'; b='source_code/executed/stage/test.py'
    def source_alias():
        physical, receipt=layout.compact({a:entry(b'one\n'),b:entry(b'one\n')})
        assert a in physical and b not in physical
        assert receipt['index']['exact_duplicate_aliases'][0]['physical_archive_path']==a
    check('Identical executed source aliases current with exact identity', source_alias)
    def different_source():
        physical, receipt=layout.compact({a:entry(b'one\n'),b:entry(b'two\n')})
        assert a in physical and b in physical and not receipt['index']['exact_duplicate_aliases']
    check('Changed executed source is retained separately', different_source)
    def current_and_scoring_physical():
        scoring='source_scoring/legacy/scoring.py'
        physical, receipt=layout.compact({a:entry(b'same\n'),scoring:entry(b'same\n'),b:entry(b'same\n')})
        assert a in physical and scoring in physical and b not in physical
    check('Current and original scoring files both remain physical', current_and_scoring_physical)
    def csv_preserved():
        names=['review/evaluation/a/per_output.csv','review/evaluation/b/per_parent.csv','review/evaluation/c/per_noise.csv']
        physical, receipt=layout.compact({name:entry(b'id,value\nx,1\n','complete_CSV') for name in names})
        assert all(name in physical for name in names)
        assert not receipt['index']['exact_duplicate_aliases']
    check('Every evaluation/parent/noise CSV remains physically present', csv_preserved)
    def proof_alias():
        original='review/decisions/H3_control_equivalence_freeze/per_step.csv'
        repeated='review/cap_control_equivalence/final/per_step.csv'
        physical, receipt=layout.compact({original:entry(b'step,ok\n1,True\n'),repeated:entry(b'step,ok\n1,True\n')})
        assert original in physical and repeated not in physical
        assert receipt['index']['proof_CSV_aliases']==[repeated]
    check('Byte-identical H3 proof CSV is fully recoverable by alias', proof_alias)
    def bad_hash():
        value=entry(b'data');value['sha256']='wrong'
        try:layout.compact({a:value})
        except AssertionError:return
        raise AssertionError('Invalid payload hash accepted')
    check('Payload/hash mismatch is rejected', bad_hash)
    namespace={'__name__':'layout_restore_test'};exec(layout.RESTORE_SOURCE,namespace)
    def roundtrip():
        physical, receipt=layout.compact({a:entry(b'one\n'),b:entry(b'one\n')})
        with tempfile.TemporaryDirectory(prefix='chair_bundle_alias_') as temporary:
            root=Path(temporary)
            for name,value in physical.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value['data'])
            assert namespace['restore'](root)==dict(restored=1,already_identical=0)
            assert (root/b).read_bytes()==b'one\n'
            assert namespace['restore'](root)==dict(restored=0,already_identical=1)
            (root/b).write_bytes(b'changed')
            try:namespace['restore'](root)
            except FileExistsError:pass
            else:raise AssertionError('Different pre-existing file overwritten')
    check('Round-trip, idempotency and overwrite refusal', roundtrip)
    def traversal():
        with tempfile.TemporaryDirectory(prefix='chair_bundle_path_') as temporary:
            try:namespace['resolve_inside'](Path(temporary).resolve(),'../escape.py')
            except ValueError:return
            raise AssertionError('Path traversal accepted')
    check('Reconstruction cannot escape extraction directory', traversal)
    def corruption():
        physical,_=layout.compact({a:entry(b'one\n'),b:entry(b'one\n')})
        with tempfile.TemporaryDirectory(prefix='chair_bundle_corrupt_') as temporary:
            root=Path(temporary)
            for name,value in physical.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value['data'])
            (root/a).write_bytes(b'corrupted')
            try:namespace['restore'](root)
            except ValueError:return
            raise AssertionError('Canonical corrupt bytes accepted')
    check('Reconstruction verifies canonical bytes before writing', corruption)
    def full_manifest_roundtrip():
        name='review/visuals/example/manifest.json'
        data=json.dumps({'render_receipts':['same unchanged receipt']*50000}).encode()
        assert len(data)>1_000_000
        physical, receipt=layout.compact({name:entry(data,'full_gallery_manifest')})
        assert name not in physical and name+'.xz' in physical
        assert len(receipt['index']['lossless_payloads'])==1
        with tempfile.TemporaryDirectory(prefix='chair_manifest_xz_') as temporary:
            root=Path(temporary)
            for key,value in physical.items():
                path=root/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value['data'])
            assert namespace['restore'](root)==dict(restored=1,already_identical=0)
            assert (root/name).read_bytes()==data
            assert namespace['restore'](root)==dict(restored=0,already_identical=1)
    check('Full manifest XZ reconstructs identical original bytes', full_manifest_roundtrip)
    def review_json_plain():
        name='review/visuals/reviews/actual_observations.json'
        physical, receipt=layout.compact({name:entry(b'x'*1_100_000,'visual_review_and_selection')})
        assert name in physical and not receipt['index']['lossless_payloads']
    check('Actual full/partial review JSON remains directly readable', review_json_plain)
    receipt=dict(status='PASS', tests=results, test_count=len(results), model_calls=0, scoring_calls=0, ZIP_files_created=0,
                 source_sha256=hashlib.sha256(Path(layout.__file__).read_bytes()).hexdigest(),
                 tests_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    destination=assets.resolve(out)/'delivery/cpu_tests/layout_receipt.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(receipt,sort_keys=True));return receipt


if __name__=='__main__':
    run()
