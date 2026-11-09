"""Storage integrity regressions: dtype clarification never changes geometry."""
import json
from pathlib import Path
import numpy as np
import pytest
from ..exporter import RadialSupport
from ..io_utils import array_hash,file_hash
from ..legacy.geometry_task import fibonacci,hull_faces
from .integrity import mesh_identity,generate_integrity,validate_integrity,collect_integrity


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))


def write_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as out:np.savez(out,**arrays)


def test_mesh_identity_storage_dtype_roundtrip_preserves_logical_hash(tmp_path):
    v=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]],dtype=np.float64)
    f=np.array([[0,1,2]],dtype=np.int32)
    path=tmp_path/'mesh.npz';write_npz(path,vertices=v,faces=f.astype(np.int64))
    before=path.read_bytes();identity=mesh_identity(path,f)
    assert identity['mesh_hash']==array_hash(v,f)
    assert identity['stored_array_hash']==array_hash(v,f.astype(np.int64))
    assert identity['mesh_hash']!=identity['stored_array_hash']
    assert identity['mesh_npz_sha256']==file_hash(path)
    assert identity['stored_faces_dtype']=='int64' and identity['logical_faces_dtype']=='int32'
    assert path.read_bytes()==before


def test_mesh_identity_rejects_noninteger_and_changed_connectivity(tmp_path):
    v=np.zeros((3,3),dtype=np.float64);f=np.array([[0,1,2]],dtype=np.int32)
    path=tmp_path/'mesh.npz'
    write_npz(path,vertices=v,faces=f.astype(np.float64))
    with pytest.raises(ValueError,match='integer dtype'):mesh_identity(path,f)
    write_npz(path,vertices=v,faces=np.array([[0,2,1]],dtype=np.int64))
    with pytest.raises(ValueError,match='connectivity values'):mesh_identity(path,f)
    with pytest.raises(ValueError,match='must be int32'):mesh_identity(path,f.astype(np.int64))


@pytest.fixture
def complete_run(tmp_path):
    run=tmp_path/'run';run.mkdir();uids=[f'uid_{i:02d}' for i in range(16)]
    u=fibonacci(258);faces=hull_faces(u).astype(np.int32);logr=np.zeros(2050)
    radius=RadialSupport().query(logr,u);v=u*radius[:,None]
    policies={'fixed':['raw'],'original_a':['raw','quality_fallback'],
        'teacher_reference':['offline_teacher'],'student_10':['raw','quality_fallback'],
        'student_11':['raw','quality_fallback']}
    manifest={};rows=[]
    for uid in uids:
        data=run/'data'/f'{uid}.npz';write_npz(data,faces=faces,logr=logr)
        dh=file_hash(data);manifest[f'data/{uid}.npz']=dh
        for model,pp in policies.items():
            attempt_rows=[]
            for policy in pp:
                mesh_file=f'attempts/{uid}/{model}/{policy}.npz'
                stored_faces=faces if model=='teacher_reference' else faces.astype(np.int64)
                write_npz(run/mesh_file,vertices=v,faces=stored_faces,u=u,r=radius)
                row={'uid':uid,'model':model,'policy':policy,'status':'success',
                    'mesh_file':mesh_file,'mesh_hash':array_hash(v,faces),
                    'data_file_sha256':dh,'mainfield_hash':array_hash(logr)}
                attempt_rows.append(row);rows.append(row)
            write_json(run/'attempts'/uid/model/'result.json',
                {'uid':uid,'model':model,'status':'success','rows':attempt_rows})
    write_json(run/'results.json',{'rows':rows,'attempt_count':80,'errors':[]})
    write_json(run/'RUN_PROTOCOL.json',{'uids':uids})
    write_json(run/'DATA_MANIFEST.json',{'files':manifest})
    return run


def test_complete_dual_hash_sidecar_generation_and_exclusive_write(complete_run):
    run=complete_run
    original={p:file_hash(p) for p in run.rglob('*') if p.is_file()}
    result=generate_integrity(run)
    assert result['rows_checked']==128 and result['dtype_only_hash_difference_count']==112
    assert result['all_accepted_direction_requeries_exact'] and not result['geometry_arrays_modified']
    assert validate_integrity(run)==result
    for p,h in original.items():assert file_hash(p)==h
    with pytest.raises(FileExistsError):generate_integrity(run)


def test_stale_ledger_and_npz_byte_change_are_rejected(complete_run):
    run=complete_run;generate_integrity(run)
    ledger=run/'results.json';old_ledger=ledger.read_bytes();data=json.loads(old_ledger)
    data['new_unbound_metadata']='changed';write_json(ledger,data)
    with pytest.raises(ValueError,match='stale'):validate_integrity(run)
    ledger.write_bytes(old_ledger)
    row=data['rows'][0];path=run/row['mesh_file']
    with np.load(path) as z:arrays={k:np.array(z[k]) for k in z.files}
    unchanged_surface_hash=array_hash(arrays['vertices'],arrays['faces'])
    arrays['extra_storage_marker']=np.array([42]);write_npz(path,**arrays)
    assert array_hash(arrays['vertices'],arrays['faces'])==unchanged_surface_hash
    with pytest.raises(ValueError,match='stale'):validate_integrity(run)


def test_geometry_corruption_and_duplicate_rows_are_rejected(complete_run):
    run=complete_run;generate_integrity(run)
    ledger=run/'results.json';original=ledger.read_bytes();data=json.loads(original)
    row=data['rows'][0];path=run/row['mesh_file'];old_mesh=path.read_bytes()
    with np.load(path) as z:arrays={k:np.array(z[k]) for k in z.files}
    arrays['vertices'][0,0]+=1e-7;write_npz(path,**arrays)
    with pytest.raises(ValueError,match='ledger mesh hash mismatch'):validate_integrity(run)
    path.write_bytes(old_mesh)
    data['rows'][1]=data['rows'][0];write_json(ledger,data)
    with pytest.raises(ValueError,match='128 unique'):validate_integrity(run)


def test_unhashed_radius_corruption_is_rejected(complete_run):
    run=complete_run;generate_integrity(run)
    row=json.loads((run/'results.json').read_text())['rows'][0];path=run/row['mesh_file']
    with np.load(path) as z:arrays={k:np.array(z[k]) for k in z.files}
    arrays['r'][0]+=1e-10;write_npz(path,**arrays)
    with pytest.raises(ValueError,match='radius differs'):validate_integrity(run)


def test_future_declared_dual_hash_fields_are_checked(complete_run):
    run=complete_run;ledger=run/'results.json';data=json.loads(ledger.read_text());row=data['rows'][0]
    with np.load(run/'data'/(row['uid']+'.npz')) as d:
        identity=mesh_identity(run/row['mesh_file'],d['faces'])
    row.update(identity)
    attempt_path=run/'attempts'/row['uid']/row['model']/'result.json'
    attempt=json.loads(attempt_path.read_text());attempt['rows'][0].update(identity)
    write_json(attempt_path,attempt);write_json(ledger,data)
    assert collect_integrity(run)['rows_checked']==128
    row['stored_array_hash']='0'*64;attempt['rows'][0]['stored_array_hash']='0'*64
    write_json(attempt_path,attempt);write_json(ledger,data)
    with pytest.raises(ValueError,match='Declared row identity field mismatch'):
        collect_integrity(run)
