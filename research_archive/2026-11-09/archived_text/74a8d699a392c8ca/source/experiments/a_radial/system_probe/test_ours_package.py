"""Archive-integrity negative controls; no production or GPU execution."""
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from . import ours_package as p


def test_manifest_rejects_tamper_addition_and_missing(tmp_path):
    f=tmp_path/'result.bin';f.write_bytes(b'unchanged');manifest=p.files_manifest(tmp_path)
    assert p.verify_files(tmp_path,manifest)==1
    f.write_bytes(b'changed')
    with pytest.raises(ValueError):p.verify_files(tmp_path,manifest)
    f.write_bytes(b'unchanged');extra=tmp_path/'extra';extra.write_bytes(b'extra')
    with pytest.raises(ValueError):p.verify_files(tmp_path,manifest)
    extra.unlink();f.unlink()
    with pytest.raises(ValueError):p.verify_files(tmp_path,manifest)


def test_zip_paths_and_duplicate_members_rejected_before_extraction(tmp_path):
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:z.writestr('../escaped',b'no')
    with pytest.raises(ValueError,match='traversal'):p.extract_and_verify(bad,tmp_path/'extract1')
    assert not (tmp_path/'escaped').exists()
    duplicate=tmp_path/'duplicate.zip'
    with zipfile.ZipFile(duplicate,'w') as z:
        z.writestr('same',b'a')
        with pytest.warns(UserWarning):z.writestr('same',b'b')
    with pytest.raises(ValueError,match='Duplicate'):p.extract_and_verify(duplicate,tmp_path/'extract2')


def test_saved_array_comparison_preserves_dtype_and_values():
    a=np.array([1,2],np.int32);p.compare(a,a.copy())
    with pytest.raises(ValueError):p.compare(a,a.astype(np.int64))
    with pytest.raises(ValueError):p.compare(a,np.array([1,3],np.int32))


def test_source_whitelist_excludes_parallel_experiments_and_has_transitive_helpers():
    names=set(p.SOURCE_NAMES)
    assert len(names)==len(p.SOURCE_NAMES)
    assert {'system_probe/ours_timing.py','system_probe/ours_metrics.py','legacy/exact_triangles.py',
        'official_contract.py','official_train.py','verify_repo.py','layout_student/train.py'} <= names
    assert not any(any(x in n for x in ('fm_tail','system_formal','learned_outlets','layout_longfit','layout_balanced','warm_start')) for n in names)
    root=Path(p.__file__).parents[1]
    assert all((root/n).is_file() for n in names)


def test_full_zip_crc_hash_and_portable_saved_arrays(tmp_path):
    payload=tmp_path/'payload';run=payload/'run';run.mkdir(parents=True)
    (run/'RUN_PROTOCOL.json').write_text('{}')
    values={k:np.array([1,2],np.float32) for k in ('noise','z_standardized','z','nodes')}
    for seed in (10,11):
        folder=run/f'seed{seed}';(folder/'samples').mkdir(parents=True);records=[]
        for sid in p.metrics.expected_ids():
            sample=folder/'samples'/f'{sid}.npz';np.savez(sample,**values)
            records.append({'sample_id':sid,'path':sample.name,'array_hash':p.array_hash(*values.values()),'nodes_hash':p.array_hash(values['nodes'])})
        (folder/'samples/SAMPLES.json').write_text(json.dumps({'records':records}))
        for arm in p.metrics.ARMS:
            out=folder/'exports';out.mkdir(exist_ok=True);mesh=out/f'{arm}.npz';np.savez(mesh,vertices=np.eye(3),faces=[[0,1,2]])
            rows=[{'status':'ok','sample_id':r['sample_id'],'mesh_path':mesh.name,'mesh_sha256':p.file_hash(mesh),
                'source_nodes_sha256':r['nodes_hash']} for r in records]
            (out/f'{arm}_MANIFEST.json').write_text(json.dumps({'records':rows}))
    (payload/'FILE_MANIFEST.json').write_text(json.dumps({'files':p.files_manifest(payload)}))
    archive=tmp_path/'complete.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in payload.rglob('*'):
            if f.is_file():z.write(f,str(f.relative_to(payload)))
    result=p.extract_and_verify(archive,tmp_path/'fresh')
    assert result['samples']==128 and result['outputs']==384 and result['all_file_SHA256_and_size_match']
    sample=tmp_path/'fresh/run/seed10/samples/00000.npz';sample.write_bytes(b'tampered')
    with pytest.raises(ValueError):p.verify_archive_payload(tmp_path/'fresh')
