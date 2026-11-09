"""Inert ZIP/NPY header and pickle-opcode identity inspection; never unpickle."""
import ast
import hashlib
import json
from pathlib import Path
import pickletools
import re
import struct
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        assert members == ['allow_pickle.npy', 'arr_0.npy'], members
        raw = archive.read('arr_0.npy')
    assert raw[:6] == b'\x93NUMPY'
    version = tuple(raw[6:8])
    assert version in ((1, 0), (2, 0), (3, 0)), version
    start = 10 if version == (1, 0) else 12
    size = struct.unpack('<H' if start == 10 else '<I', raw[8:start])[0]
    header = ast.literal_eval(raw[start:start + size].decode('latin1'))
    assert header['descr'] == '|O' and len(header['shape']) == 1, header
    # The payload's GLOBAL/REDUCE opcodes and tensor storage bytes are never
    # executed or deserialized. Track only scalar string memo entries so each
    # UID can be attributed to its literal/referenced object_id dictionary key.
    memo, previous, uids, key_count = {}, None, [], 0
    string_ops = {'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8',
                  'STRING', 'BINSTRING', 'SHORT_BINSTRING'}
    for opcode, arg, position in pickletools.genops(raw[start + size:]):
        if opcode.name in {'PUT', 'BINPUT', 'LONG_BINPUT'}:
            memo[int(arg)] = previous
            continue
        if opcode.name == 'MEMOIZE':
            memo[len(memo)] = previous
            continue
        if opcode.name in {'GET', 'BINGET', 'LONG_BINGET'}:
            value = memo.get(int(arg))
        elif opcode.name in string_ops:
            value = arg
        else:
            previous = None
            continue
        if value == 'object_id':
            key_count += 1
        if isinstance(value, str) and re.fullmatch(r'[0-9a-f]{20,32}', value):
            assert previous == 'object_id', ('Unattributed UID-like string', position)
            uids.append(value)
        previous = value
    assert len(uids) == key_count == header['shape'][0]
    assert len(set(uids)) == len(uids)
    return {'zip_members': members, 'npy_version': version, 'record_header': header,
            'object_id_key_count': key_count, 'train_uids': sorted(uids)}


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    archive = here / '03691459_train.npz'
    audit = root / '.worktrees/a-layout-value-20260924/output/a_condition_data_audit_20260925_01/AUDIT.json'
    detail = inspect(archive)
    local = json.loads(audit.read_text(encoding='utf-8-sig'))
    train = {uid for uid, row in local['speaker_rows'].items()
             if row['original_split'] == 'train'}
    assert len(train) == 376
    official = set(detail['train_uids'])
    report = {
        'status': 'VERIFIED_IDENTICAL_TRAIN_PARENT_UID_SET',
        'source_url': 'https://huggingface.co/datasets/CH3COOK/MeshXL-shapenet-data/resolve/main/03691459_train.npz',
        'download_path': str(archive), 'download_bytes': archive.stat().st_size,
        'download_sha256': sha(archive), 'local_audit_path': str(audit),
        'local_audit_sha256': sha(audit), 'script_sha256': sha(Path(__file__)),
        'official_train_record_count': detail['record_header']['shape'][0],
        'official_train_unique_parent_uids': len(official),
        'local_meshflow_rebuttal2_train_parent_uids': len(train),
        'overlap_count': len(official & train),
        'new_relative_to_local_train_count': len(official - train),
        'local_train_missing_from_meshxl_count': len(train - official),
        'new_uids': sorted(official - train), 'missing_uids': sorted(train - official),
        'identity_details': detail,
        'method': 'ZIP read, ast.literal_eval NPY header, inert pickletools opcode scan with string memo tracking and object_id key attribution',
        'pickle_execution': False, 'numpy_load_used': False,
        'tensor_deserialization': False, 'geometry_arrays_inspected': False,
        'validation_or_test_downloaded': False, 'new_geometry_screen_or_model_run': False,
        'limits': 'UID identity only; mesh values and preprocessing equivalence were not compared. A ZIP member named allow_pickle.npy is a stored scalar, not permission to unpickle.',
    }
    with (here / 'MESHXL_TRAIN_UID_CHECK.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('status', 'official_train_unique_parent_uids',
        'local_meshflow_rebuttal2_train_parent_uids', 'overlap_count', 'new_relative_to_local_train_count',
        'download_sha256')}))


if __name__ == '__main__':
    main()
