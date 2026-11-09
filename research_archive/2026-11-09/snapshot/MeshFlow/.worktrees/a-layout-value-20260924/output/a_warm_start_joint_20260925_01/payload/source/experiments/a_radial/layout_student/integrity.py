"""Distinguish preserved template-index mesh hashes from NPZ storage hashes.

The first evaluation hashed int32 template connectivity while deployment saved
identical index values as int64. Neither vertices nor connectivity values are
rewritten by this module. Both representations are verified explicitly.
"""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import numpy as np

from ..exporter import RadialSupport
from ..io_utils import array_hash, file_hash, write_json

MESH_HASH_SCHEMA = 'array_hash(vertices_as_saved, original_data_faces_int32); dtype_and_shape_included'
STORED_HASH_SCHEMA = 'array_hash(vertices_as_saved, faces_as_saved); dtype_and_shape_included'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe_path(run, relative):
    run=Path(run).resolve();path=(run/relative).resolve()
    require(path.is_relative_to(run),'Artifact path escapes run: '+str(relative))
    return path


def mesh_identity(mesh_path, template_faces):
    """Read-back identity of saved arrays; never writes or canonicalizes an NPZ.

    `mesh_hash` preserves the original ledger's int32-template convention.
    `stored_array_hash` binds the actual saved connectivity dtype and bytes.
    """
    mesh_path=Path(mesh_path)
    template_faces=np.asarray(template_faces)
    require(template_faces.dtype==np.dtype('int32'),'Logical template faces must be int32')
    require(template_faces.ndim==2 and template_faces.shape[1]==3,'Template faces must be triangular')
    before=file_hash(mesh_path)
    with np.load(mesh_path,allow_pickle=False) as z:
        vertices=np.array(z['vertices']);faces=np.array(z['faces'])
    require(np.issubdtype(faces.dtype,np.integer),'Saved faces must have integer dtype')
    require(np.array_equal(faces,template_faces),'Saved connectivity values differ from original template')
    require(vertices.ndim==2 and vertices.shape[1]==3 and np.isfinite(vertices).all(),'Invalid saved vertices')
    require(file_hash(mesh_path)==before,'Mesh file changed during identity read')
    return {'mesh_hash':array_hash(vertices,template_faces),'mesh_hash_schema':MESH_HASH_SCHEMA,
            'stored_array_hash':array_hash(vertices,faces),'mesh_npz_sha256':before,
            'stored_faces_dtype':str(faces.dtype),'logical_faces_dtype':str(template_faces.dtype)}


def collect_integrity(run):
    """Recheck every planned row, with no writes and no dropped error records."""
    run=Path(run).resolve()
    ledger_path=run/'results.json';ledger_sha=file_hash(ledger_path);ledger=read_json(ledger_path)
    protocol_path=run/'RUN_PROTOCOL.json';protocol_sha=file_hash(protocol_path);protocol=read_json(protocol_path)
    manifest_path=run/'DATA_MANIFEST.json';manifest_sha=file_hash(manifest_path);manifest=read_json(manifest_path)['files']
    for relative,expected in manifest.items():
        require(file_hash(safe_path(run,relative))==expected,'Data manifest mismatch: '+relative)
    uids=sorted(protocol['uids'])
    require(len(uids)==len(set(uids))==16,'Expected 16 unique UIDs')
    policies={'fixed':['raw'],'original_a':['raw','quality_fallback'],
              'teacher_reference':['offline_teacher'],'student_10':['raw','quality_fallback'],
              'student_11':['raw','quality_fallback']}
    expected={(uid,model,policy) for uid in uids for model in policies for policy in policies[model]}
    rows=ledger['rows'];key=lambda row:(row['uid'],row['model'],row['policy'])
    keys=[key(row) for row in rows]
    require(len(keys)==len(set(keys))==128 and set(keys)==expected,'Expected exactly 128 unique predeclared rows')
    require(ledger['attempt_count']==80 and ledger['errors']==[],'Expected completed 80-attempt ledger without omitted error outputs')
    support=RadialSupport();records=[];mismatch=0;attempts={}
    for row in sorted(rows,key=key):
        uid=row['uid'];data_relative='data/'+uid+'.npz';data_path=safe_path(run,data_relative)
        data_sha=file_hash(data_path)
        require(data_sha==manifest[data_relative]==row['data_file_sha256'],'Row data file hash mismatch')
        with np.load(data_path,allow_pickle=False) as z:
            faces=np.array(z['faces']);logr=np.array(z['logr'])
        require(array_hash(logr)==row['mainfield_hash'],'Row mainfield hash mismatch')
        mesh_path=safe_path(run,row['mesh_file']);identity=mesh_identity(mesh_path,faces)
        require(identity['mesh_hash']==row['mesh_hash'],'Original ledger mesh hash mismatch')
        for field,value in identity.items():
            if field in row:
                require(row[field]==value,'Declared row identity field mismatch: '+field)
        with np.load(mesh_path,allow_pickle=False) as z:
            vertices=np.array(z['vertices']);u=np.array(z['u']);r=np.array(z['r']);stored_faces=np.array(z['faces'])
        require(vertices.dtype==np.dtype('float64') and vertices.shape==(258,3),'Unexpected vertex storage contract')
        require(stored_faces.shape==(512,3),'Unexpected face budget')
        require(u.shape==(258,3) and np.isfinite(u).all(),'Invalid accepted direction array')
        require(np.allclose(np.linalg.norm(u,axis=1),1,rtol=0,atol=1e-12),'Accepted directions are not unit vectors')
        radius=support.query(logr,u)
        require(np.array_equal(r,radius),'Saved radius differs from accepted-direction requery')
        require(np.array_equal(vertices,u*radius[:,None]),'Saved vertices differ from accepted-direction requery')
        attempt_relative=f'attempts/{uid}/{row["model"]}/result.json'
        if attempt_relative not in attempts:
            attempt_path=safe_path(run,attempt_relative)
            attempts[attempt_relative]=(read_json(attempt_path),file_hash(attempt_path))
        attempt,attempt_sha=attempts[attempt_relative]
        require(attempt['status']=='success' and row in attempt['rows'],'Aggregate row differs from original attempt ledger')
        differs=identity['stored_array_hash']!=identity['mesh_hash']
        if differs:
            require(identity['stored_faces_dtype']!=identity['logical_faces_dtype'],'Hash difference not explained by face dtype')
            mismatch+=1
        records.append({'uid':uid,'model':row['model'],'policy':row['policy'],
            'mesh_file':row['mesh_file'],**identity,
            'stored_array_hash_schema':STORED_HASH_SCHEMA,
            'vertices_dtype':str(vertices.dtype),'data_file':data_relative,
            'data_file_sha256':data_sha,'attempt_result_file':attempt_relative,
            'attempt_result_sha256':attempt_sha,'connectivity_values_identical':True,
            'accepted_direction_radius_requery_exact':True,
            'stored_vs_logical_hash_differs_only_by_face_dtype':differs})
    require(len(attempts)==80,'Incorrect distinct attempt count')
    require(file_hash(ledger_path)==ledger_sha and file_hash(protocol_path)==protocol_sha
            and file_hash(manifest_path)==manifest_sha,'Run metadata changed during verification')
    return {'schema_version':1,'status':'PASS','results_sha256':ledger_sha,
        'run_protocol_sha256':protocol_sha,'data_manifest_sha256':manifest_sha,
        'rows_checked':128,'attempt_ledgers_checked':80,
        'dtype_only_hash_difference_count':mismatch,
        'mesh_hash_schema':MESH_HASH_SCHEMA,'stored_array_hash_schema':STORED_HASH_SCHEMA,
        'reason':'Original evaluation mesh_hash used int32 data.faces; deployed NPZ stored equal index values as int64. Teacher NPZ retained int32.',
        'geometry_arrays_modified':False,'original_npz_and_ledgers_preserved':True,
        'all_accepted_direction_requeries_exact':True,'rows':records}


def generate_integrity(run):
    run=Path(run)
    target=run/'MESH_INTEGRITY.json'
    if target.exists():
        raise FileExistsError(target)
    result=collect_integrity(run)
    result['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(target,result)
    return result


def validate_integrity(run):
    """Reject stale sidecars by recomputing all file and logical/stored hashes."""
    run=Path(run);path=run/'MESH_INTEGRITY.json'
    require(path.is_file(),'Required MESH_INTEGRITY.json is missing')
    recorded=read_json(path)
    require('created_utc' in recorded,'Integrity sidecar lacks creation provenance')
    observed=collect_integrity(run)
    expected={k:v for k,v in recorded.items() if k!='created_utc'}
    require(expected==observed,'MESH_INTEGRITY.json is stale or inconsistent with saved files/ledger')
    return recorded


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--verify',action='store_true');a=p.parse_args()
    result=validate_integrity(a.run) if a.verify else generate_integrity(a.run)
    print(json.dumps({'status':result['status'],'rows_checked':result['rows_checked'],
        'dtype_only_hash_difference_count':result['dtype_only_hash_difference_count'],
        'geometry_arrays_modified':result['geometry_arrays_modified']},indent=2),flush=True)


if __name__=='__main__':main()
