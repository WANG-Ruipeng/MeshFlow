from pathlib import Path
import json,zipfile
import numpy as np
import pytest
from . import package_results as pkg


def test_copy_rejects_copy_corruption(tmp_path,monkeypatch):
    src=tmp_path/'source';src.write_bytes(b'original');out=tmp_path/'copied'
    real=pkg.shutil.copy2
    def wrong(a,b):real(a,b);Path(b).write_bytes(b'changed')
    monkeypatch.setattr(pkg.shutil,'copy2',wrong)
    with pytest.raises(ValueError,match='Copied'):pkg.copy_bound_files({'a.bin':src},out)
    assert src.read_bytes()==b'original'


def test_parent_selection_is_explicit(tmp_path):
    names=['RUN_PROTOCOL.json','seed10/samples/00000.npz','seed10/ae/step00003200.pt','seed10/fm/step00008000.pt','seed10/ae/step00000400.pt','timing/RESULTS.json','failed/FAILED.json']
    for n in names:p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x')
    assert set(pkg.selected_parent_files(tmp_path))==set(names)-{'seed10/ae/step00000400.pt','timing/RESULTS.json'}


def test_source_whitelist_is_bounded():
    repo=Path(pkg.__file__).resolve().parents[3];sources=pkg.source_paths(repo)
    assert all(p.is_file() for p in sources.values())
    assert 'experiments/a_radial/fm_tail/package_results.py' in sources
    assert not any(any(f'/{n}/' in k for n in ['system_formal','layout_balanced','layout_longfit']) or k.endswith('learned_outlets.py') for k in sources)


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_zip_rejects_traversal(tmp_path,name):
    z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as f:f.writestr(name,'bad')
    with pytest.raises(ValueError,match='escapes'):pkg.extract_archive(z,tmp_path/'extract','unused')


def test_manifest_bytes_are_bound(tmp_path):
    z=tmp_path/'badmanifest.zip'
    with zipfile.ZipFile(z,'w') as f:f.writestr('FILE_MANIFEST.json','{}')
    with pytest.raises(ValueError,match='manifest differs'):pkg.extract_archive(z,tmp_path/'extract','wrong')


def fixture_arrays(root):
    run=root/'run';run.mkdir(parents=True)
    (run/'RUN_PROTOCOL.json').write_text('{}');(run/'ARTIFACT_PROTOCOL.json').write_text('{}')
    rows=[]
    for seed in (10,11):
        base=run/f'artifacts/seed{seed}'
        for sub,steps in [('samples',32),('auxiliary_e128/samples',128)]:
            folder=base/sub;folder.mkdir(parents=True);records=[]
            for i in range(64):
                sid=f'{i:05d}';noise=np.array([seed,i],dtype=np.float32);nodes=np.array([i/100],dtype=np.float64)
                np.savez(folder/f'{sid}.npz',noise=noise,z_standardized=noise,z=noise,nodes=nodes)
                records.append({'sample_id':sid,'path':sid+'.npz','array_hash':pkg.array_hash(noise,noise,noise,nodes),'nodes_hash':pkg.array_hash(nodes)})
            (folder/'SAMPLES.json').write_text(json.dumps({'fm_steps':steps,'records':records}))
        exports=base/'exports';exports.mkdir()
        for arm in ('fixed','a','qem'):
            folder=exports/arm;folder.mkdir();records=[]
            for i in range(64):
                sid=f'{i:05d}';v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=np.float64);f=np.array([[0,1,2]],dtype=np.int32)
                path=folder/f'{sid}.npz';np.savez(path,vertices=v,faces=f);nodes=np.array([i/100],dtype=np.float64)
                records.append({'sample_id':sid,'status':'ok','mesh_path':f'{arm}/{sid}.npz','mesh_sha256':pkg.file_hash(path),'source_nodes_sha256':pkg.array_hash(nodes)})
                rows.append({'seed':seed,'sample_id':sid,'arm':arm,'mesh_npz_sha256':pkg.file_hash(path),'source_nodes_sha256':pkg.array_hash(nodes),'stored_array_hash':pkg.array_hash(v,f),'canonical_geometry_hash':pkg.array_hash(v,f.astype(np.int64)),'quality':{'face_count':1}})
            (exports/f'{arm}_MANIFEST.json').write_text(json.dumps({'records':records}))
    p=run/'artifacts/metrics';p.mkdir();(p/'RESULTS.json').write_text(json.dumps({'rows':rows}));return root


def test_saved_mesh_dtype_tamper_cannot_hide_behind_canonical_hash(tmp_path):
    root=fixture_arrays(tmp_path/'payload');assert pkg.portable_arrays(root)['raw_outlets']==384
    mesh=root/'run/artifacts/seed10/exports/a/00000.npz'
    with np.load(mesh) as z:v=z['vertices'];f=z['faces'].astype(np.int64)
    np.savez(mesh,vertices=v,faces=f)
    # Even if file hashes were updated, original typed array identity must reject it.
    ledger=mesh.parents[1]/'a_MANIFEST.json';data=json.loads(ledger.read_text());data['records'][0]['mesh_sha256']=pkg.file_hash(mesh);ledger.write_text(json.dumps(data))
    results=root/'run/artifacts/metrics/RESULTS.json';data=json.loads(results.read_text())
    next(r for r in data['rows'] if r['seed']==10 and r['arm']=='a' and r['sample_id']=='00000')['mesh_npz_sha256']=pkg.file_hash(mesh)
    results.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='stored mesh dtype'):pkg.portable_arrays(root)
