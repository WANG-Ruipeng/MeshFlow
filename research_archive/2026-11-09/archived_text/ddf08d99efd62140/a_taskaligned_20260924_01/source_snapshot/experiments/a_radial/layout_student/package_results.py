"""Seal the task-aligned A experiment, then verify a fresh CPU replay.

This is intentionally an exclusive, one-shot operation. Run after all reports
and figures are complete. All failed attempts remain in the planned denominator.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import torch

from ..exporter import ASSETS, RadialSupport
from ..io_utils import array_hash, file_hash, write_json
from ..verify_repo import verify
from .deployment import named_quality, qualification
from .prepare import split_uids
from .train import model_sha256
from .integrity import validate_integrity


REPLAY_SCRIPT = r'''"""Replay only extracted code, input, and the predeclared final student."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--run',required=True,type=Path)
p.add_argument('--report',required=True,type=Path)
a=p.parse_args();run=a.run.resolve()
sys.path.insert(0,str(run/'source_snapshot'))
import numpy as np
import torch
from experiments.a_radial import exporter
from experiments.a_radial.layout_student import deployment
from experiments.a_radial.layout_student.train import model_sha256
from experiments.a_radial.layout_student.integrity import validate_integrity
from experiments.a_radial.legacy.networks import DomainVelocity
from experiments.a_radial.io_utils import array_hash,file_hash
assert Path(exporter.__file__).resolve().is_relative_to(run/'source_snapshot')
assert Path(deployment.__file__).resolve().is_relative_to(run/'source_snapshot')
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
protocol=json.loads((run/'RUN_PROTOCOL.json').read_text())
integrity=validate_integrity(run)
integrity_rows={(r['uid'],r['model'],r['policy']):r for r in integrity['rows']}
uid=sorted(protocol['uids'])[0]
checkpoint=run/'training/seed_10/final.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert state['update']==3000 and state['is_predeclared_final']
assert state['model_sha256']==model_sha256(state['model'])
model=DomainVelocity().cpu();model.load_state_dict(state['model'],strict=True)
model.eval().requires_grad_(False)
with np.load(run/'data'/f'{uid}.npz',allow_pickle=False) as z:
    y=z['logr'];base=z['base_u'];faces=z['faces']
support=exporter.RadialSupport(run/'source_snapshot/experiments/a_radial/assets/support.npz')
result=deployment.deploy(model,y,support,base,faces)
rows=json.loads((run/'attempts'/uid/'student_10/result.json').read_text())['rows']
checks={}
for policy,prefix in [('raw','raw'),('quality_fallback','delivered')]:
    row=next(x for x in rows if x['policy']==policy)
    mesh=result[prefix+'_mesh'];audit=result[prefix+'_audit'];quality=result[prefix+'_quality']
    stored=integrity_rows[(uid,'student_10',policy)]
    observed_hash=array_hash(mesh['vertices'],faces)
    observed_stored_hash=array_hash(mesh['vertices'],mesh['faces'])
    assert np.array_equal(mesh['faces'],faces)
    assert observed_hash==row['mesh_hash']==stored['mesh_hash'],(policy,observed_hash,row['mesh_hash'])
    assert observed_stored_hash==stored['stored_array_hash']
    assert file_hash(run/row['mesh_file'])==stored['mesh_npz_sha256']
    q=deployment.qualification(quality,audit,result['floor'])
    assert q['qualified']==row['qualified'] and q['failure_reasons']==row['failure_reasons']
    assert audit['valid']==row['audit']['valid']
    assert audit['exact_executed'] and audit['exact_requested']
    assert audit['selfintersection_count']==row['audit']['selfintersection_count']
    with np.load(run/row['mesh_file'],allow_pickle=False) as saved:
        assert np.array_equal(mesh['vertices'],saved['vertices'])
        assert np.array_equal(mesh['faces'],saved['faces'])
        assert np.array_equal(mesh['u'],saved['u'])
    checks[policy]={'mesh_hash':observed_hash,'stored_array_hash':observed_stored_hash,
        'logical_and_stored_hashes_match':True,'mesh_npz_sha256_matches_integrity_sidecar':True,'matches_saved_hash':True,
        'vertices_faces_directions_bitwise_equal':True,'qualified':q['qualified'],
        'exact_endpoint_audit_matches':True}
assert result['delivery']['fallback_to_fixed']==next(x for x in rows if x['policy']=='quality_fallback')['fallback_to_fixed']
report={'status':'PASS','uid':uid,'model':'student_10','training_updates':3000,
    'device':'cpu','guard_steps':32,'network_dtype':'float32','guard_geometry_dtype':'float64',
    'checkpoint_sha256':file_hash(checkpoint),'model_sha256':state['model_sha256'],
    'code_loaded_from_fresh_extraction':True,'deployment_module':str(Path(deployment.__file__).resolve()),
    'teacher_label_read_for_prediction':False,'checks':checks,
    'fallback_to_fixed':result['delivery']['fallback_to_fixed']}
with a.report.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report),flush=True)
'''


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def inside(root, relative):
    root=Path(root).resolve()
    target=(root/relative).resolve()
    require(target.is_relative_to(root), 'Artifact path escapes its root: '+str(relative))
    return target


def verify_manifest(root, manifest):
    for relative, expected in manifest.items():
        path=inside(root,relative)
        require(path.is_file() and not path.is_symlink(), 'Missing/nonregular manifest file: '+relative)
        require(file_hash(path)==expected, 'Artifact changed: '+str(path))
    return len(manifest)


def mesh_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {key:np.array(z[key]) for key in z.files}


def write_obj(path, vertices, faces):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as out:
        out.write('# F512 task-aligned A experiment; see paired ledger for qualification.\n')
        for point in vertices:
            out.write('v '+' '.join(f'{x:.17g}' for x in point)+'\n')
        for face in faces:
            out.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')


def verify_teacher_arrays(source_mesh, reference_mesh, template_faces,
                          source_mesh_hash, logical_mesh_hash, stored_array_hash):
    """Compare immutable source/reference geometry before comparing typed hashes.

    Historical optimized teachers store int64 indices; the new reference stores
    the original int32 template. The two hashes intentionally retain their own
    dtype and are never treated as interchangeable.
    """
    sv,sf,su=[source_mesh[k] for k in ('vertices','faces','u')]
    v,f,u=[reference_mesh[k] for k in ('vertices','faces','u')]
    require(np.issubdtype(sf.dtype,np.integer) and np.issubdtype(f.dtype,np.integer),
            'Teacher connectivity must have integer dtype')
    require(array_hash(sv,sf)==source_mesh_hash,'Original teacher source array hash mismatch')
    require(np.array_equal(v,sv) and v.dtype==sv.dtype,'Teacher reference vertices changed')
    require(np.array_equal(u,su) and u.dtype==su.dtype,'Teacher reference directions changed')
    require(np.array_equal(f,sf),'Teacher reference connectivity values changed')
    require(np.array_equal(f,template_faces),'Teacher reference differs from original template')
    require(array_hash(v,template_faces)==logical_mesh_hash,'Teacher reference logical hash mismatch')
    require(array_hash(v,f)==stored_array_hash,'Teacher reference storage hash mismatch')
    return {'source_mesh_hash':source_mesh_hash,'reference_logical_mesh_hash':logical_mesh_hash,
        'reference_stored_array_hash':stored_array_hash,'source_faces_dtype':str(sf.dtype),
        'reference_faces_dtype':str(f.dtype),'source_vertices_dtype':str(sv.dtype),
        'reference_vertices_dtype':str(v.dtype),'vertices_exactly_equal':True,
        'directions_exactly_equal':True,'connectivity_values_exactly_equal':True,
        'face_dtype_differs':sf.dtype!=f.dtype,
        'source_and_reference_hashes_keep_original_dtype_semantics':True}


def audit_run(run, smoke_run):
    protocol=read_json(run/'RUN_PROTOCOL.json')
    integrity=validate_integrity(run)
    integrity_rows={(r['uid'],r['model'],r['policy']):r for r in integrity['rows']}
    require(protocol['experiment']=='A-TASKALIGNED-v1','Wrong experiment')
    uids=sorted(protocol['uids']);train=protocol['train_uids'];check=protocol['internal_check_uids']
    require((train,check)==split_uids(uids),'Parent split changed')
    require(len(train)==12 and len(check)==4 and not set(train)&set(check),'Invalid split')
    require(protocol['training']['updates']==3000 and protocol['training']['seeds']==[10,11], 'Budget changed')
    config_hash=hashlib.sha256(json.dumps(protocol['training'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(config_hash==protocol['hashes']['config_sha256'],'Frozen training config hash mismatch')
    data_manifest=read_json(run/'DATA_MANIFEST.json')['files']
    require(file_hash(run/'DATA_MANIFEST.json')==protocol['hashes']['data_manifest_sha256'],'Data manifest changed')
    data_checked=verify_manifest(run,data_manifest)
    require(file_hash(ASSETS/'original_A_10.pt')==protocol['hashes']['original_a_sha256'],'Original A changed')
    for rel,expected in protocol['fixed_source_hashes'].items():
        require(file_hash(inside(ASSETS.parent,rel))==expected,'Frozen source changed: '+rel)
    with np.load(run/'train_labels.npz',allow_pickle=False) as a:
        require(a['train_uids'].tolist()==train,'Training archive UID mismatch')
        require(not set(a['train_uids'].tolist())&set(check),'Internal-check label leakage')
    old=Path(protocol['source_run']).resolve()
    require(file_hash(old/'MANIFEST_SHA256.json')==protocol['hashes']['source_run_manifest_sha256'], 'Old source manifest changed')
    old_manifest=read_json(old/'MANIFEST_SHA256.json')['files']
    old_checked=verify_manifest(old,old_manifest)
    provenance=read_json(run/'SOURCE_PROVENANCE.json')
    for rel,expected in provenance['verified_files'].items():
        require(old_manifest.get(rel)==expected,'Teacher provenance disagrees with original source manifest')
    identity=verify(Path(__file__).resolve().parents[3])
    require(identity['pass'] and identity['protected_files_checked']==11,'Official source identity failed')
    models=['fixed','original_a','teacher_reference','student_10','student_11']
    policies={'fixed':['raw'],'original_a':['raw','quality_fallback'],
              'teacher_reference':['offline_teacher'],'student_10':['raw','quality_fallback'],
              'student_11':['raw','quality_fallback']}
    expected_attempts={(uid,model) for uid in uids for model in models}
    expected_rows={(uid,model,policy) for uid in uids for model in models for policy in policies[model]}
    attempts=[read_json(p) for p in sorted((run/'attempts').glob('*/*/result.json'))]
    attempt_ids=[(a['uid'],a['model']) for a in attempts]
    require(len(attempt_ids)==len(set(attempt_ids))==80 and set(attempt_ids)==expected_attempts,'Missing or duplicate attempts')
    ledger=read_json(run/'results.json');rows=ledger['rows']
    row_ids=[(r['uid'],r['model'],r['policy']) for r in rows]
    require(len(row_ids)==len(set(row_ids)),'Duplicate result rows')
    flattened=[r for attempt in attempts for r in attempt['rows']]
    key=lambda r:(r['uid'],r['model'],r['policy'])
    require(sorted(flattened,key=key)==sorted(rows,key=key),'Aggregate ledger lost or changed attempt rows')
    errors=[a for a in attempts if a['status']!='success']
    require(sorted(errors,key=lambda a:(a['uid'],a['model']))==sorted(ledger['errors'],key=lambda a:(a['uid'],a['model'])), 'Aggregate lost failed attempts')
    failed_slots={(a['uid'],a['model'],policy) for a in errors for policy in policies[a['model']]}
    require(not set(row_ids)&failed_slots,'Failed attempt also contains successful rows')
    require(set(row_ids)|failed_slots==expected_rows,'Unaccounted predeclared result slots')
    require(len(expected_rows)==128 and ledger['attempt_count']==80,'Wrong planned denominator')
    original_state=torch.load(ASSETS/'original_A_10.pt',map_location='cpu',weights_only=True)
    model_hashes={'original_a':model_sha256(original_state)}
    checkpoint_hashes={'original_a':file_hash(ASSETS/'original_A_10.pt')}
    training_checks={}
    for seed in (10,11):
        folder=run/'training'/f'seed_{seed}';path=folder/'final.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        report=read_json(folder/'TRAIN_REPORT.json')
        conf=state['config'];trace=state['trace']
        require(state['update']==3000 and state['is_predeclared_final'], 'Not predeclared final checkpoint')
        require(conf['seed']==seed and conf['train_uids']==train,'Training seed or split mismatch')
        require(conf['total_updates']==3000 and conf['parameter_count']==29907,'Architecture or training budget changed')
        require(conf['training_labels_sha256']==protocol['hashes']['training_labels_sha256'], 'Training data mismatch')
        require(state['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json'),'Training protocol mismatch')
        require(state['initial_checkpoint_sha256']==protocol['hashes']['original_a_sha256'],'Training initialization changed')
        require(conf['source_sha256']['train.py']==file_hash(Path(__file__).parent/'train.py'),'Training source changed')
        require(conf['source_sha256']['legacy/networks.py']==file_hash(ASSETS.parent/'legacy/networks.py'),'Network source changed')
        require(len(trace)==3000 and [t['update'] for t in trace]==list(range(1,3001)),'Training updates missing')
        exposures=Counter()
        for step in trace:
            require(len(step['uids'])==4 and set(step['uids'])<=set(train),'Nontraining UID in training trace')
            require(np.isfinite(step['loss']),'Nonfinite training trace')
            exposures.update(step['uids'])
        require(dict(exposures)=={uid:1000 for uid in train},'Unequal parent exposure')
        require(state['sampler']['exposures']==[1000]*12,'Saved sampler exposure mismatch')
        require(set(['optimizer','scheduler','rng','sampler'])<=state.keys(),'Incomplete resumable checkpoint')
        require(len(state['rng']['torch_cuda'])>0,'Missing GPU RNG state')
        mh=model_sha256(state['model']);fh=file_hash(path)
        require(mh==state['model_sha256']==report['model_sha256'],'Final model hash mismatch')
        require(fh==report['checkpoint_sha256'],'Final checkpoint hash mismatch')
        require(report['selection_uses_internal_check'] is False and report['internal_check_teacher_files_opened']==0, 'Training/check separation report failed')
        model_hashes[f'student_{seed}']=mh;checkpoint_hashes[f'student_{seed}']=fh
        training_checks[str(seed)]={'updates':3000,'model_sha256':mh,'checkpoint_sha256':fh,
            'all_trace_uids_training_only':True,'exposures_per_train_parent':1000,
            'full_optimizer_scheduler_rng_sampler_saved':True}
    smoke=read_json(smoke_run/'GPU_SMOKE.json')
    require(smoke['status']=='PASS' and smoke['formal_protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json'),'GPU smoke mismatch')
    require(smoke['train_source_sha256']==file_hash(Path(__file__).parent/'train.py'),'GPU tested source changed')
    require(file_hash(smoke_run/'GPU_SMOKE_TEST.py')==smoke['smoke_source_sha256'],'GPU smoke source changed')
    teachers={r['uid']:r for r in read_json(run/'TEACHERS.json')['rows']}
    support=RadialSupport();meshes={};fallback_count=0;unchanged_baseline_count=0;teacher_identity_records=[]
    for uid in uids:
        data=mesh_arrays(run/'data'/f'{uid}.npz');y=data['logr'];f=data['faces'];base=data['base_u']
        fv=base*support.query(y,base)[:,None];fq=named_quality(fv,f);fixed_hash=array_hash(fv,f)
        require(array_hash(y)==teachers[uid]['mainfield_hash'],'Teacher mainfield identity mismatch')
        for row in (r for r in rows if r['uid']==uid):
            require(row['status']=='success','Unexpected row status')
            require(row['split']==('train' if uid in train else 'internal_check'),'Row split mismatch')
            require(row['mainfield_hash']==array_hash(y),'Mixed mainfields in output')
            require(row['data_file_sha256']==data_manifest[f'data/{uid}.npz'],'Row data identity mismatch')
            mesh=mesh_arrays(inside(run,row['mesh_file']));v=mesh['vertices'];mf=mesh['faces'];u=mesh['u']
            stored_identity=integrity_rows[(uid,row['model'],row['policy'])]
            require(array_hash(v,f)==row['mesh_hash']==stored_identity['mesh_hash'],'Original template-index mesh hash mismatch')
            require(array_hash(v,mf)==stored_identity['stored_array_hash'],'Actual NPZ storage array hash mismatch')
            require(file_hash(inside(run,row['mesh_file']))==stored_identity['mesh_npz_sha256'],'NPZ file hash mismatch')
            require(np.array_equal(mf,f) and len(mf)==512,'Output connectivity/face count changed')
            require(np.allclose(np.linalg.norm(u,axis=1),1,rtol=0,atol=1e-12),'Output direction is not unit')
            radius=support.query(y,u)
            require(np.allclose(v,u*radius[:,None],rtol=1e-12,atol=1e-12),'Stale accepted-direction radii')
            require(np.allclose(mesh['r'],radius,rtol=1e-12,atol=1e-12),'Saved radii mismatch')
            quality=named_quality(v,mf)
            for qkey in ('min_triangle_angle_deg','triangle_min_angle_p05_deg'):
                require(abs(quality[qkey]-row['quality'][qkey])<=1e-10,'Saved quality inconsistent')
                require(abs(fq[qkey]-row['quality_floor'][qkey])<=1e-10,'Fixed floor inconsistent')
            require(row['audit']['exact_executed'] and row['audit']['exact_requested'],'Missing actual exact endpoint audit')
            q=qualification(quality,row['audit'],fq)
            require(q['qualified']==row['qualified'] and q['failure_reasons']==row['failure_reasons'],'Qualification inconsistent')
            if row['model'] in model_hashes:
                require(row['identity']['model_sha256']==model_hashes[row['model']],'Output uses a different model')
                require(row['identity']['file_sha256']==checkpoint_hashes[row['model']],'Output uses a different checkpoint')
            if row['policy']=='raw' and row['model'] in ('fixed','original_a'):
                old_arm='fixed' if row['model']=='fixed' else 'a'
                old_row=read_json(old/'attempts'/uid/old_arm/'result.json')
                require(old_row['mesh_hash']==row['mesh_hash'],'Frozen baseline output changed from original experiment')
                unchanged_baseline_count+=1
            if row['model']=='fixed':
                require(row['mesh_hash']==fixed_hash and np.array_equal(u,base),'Fixed mesh changed')
            if row['model']=='teacher_reference':
                teacher=teachers[uid]
                source_result_path=old/'attempts'/uid/teacher['chosen_arm']/'result.json'
                source_row=read_json(source_result_path)
                source_mesh_path=inside(old,source_row['mesh_file'])
                source_mesh=mesh_arrays(source_mesh_path)
                require(source_row['mesh_hash']==teacher['teacher_mesh_hash'],'Teacher metadata disagrees with original source ledger')
                teacher_identity=verify_teacher_arrays(source_mesh,mesh,f,
                    teacher['teacher_mesh_hash'],row['mesh_hash'],stored_identity['stored_array_hash'])
                teacher_identity_records.append({'uid':uid,'chosen_arm':teacher['chosen_arm'],
                    'source_mesh_file':source_row['mesh_file'],'source_mesh_file_sha256':file_hash(source_mesh_path),
                    'source_result_file':source_result_path.relative_to(old).as_posix(),
                    'source_result_file_sha256':file_hash(source_result_path),
                    'reference_mesh_file':row['mesh_file'],
                    'reference_mesh_npz_sha256':stored_identity['mesh_npz_sha256'],**teacher_identity})
            if row['fallback_to_fixed']:
                require(row['policy']=='quality_fallback','Fallback hidden in raw row')
                require(row['mesh_hash']==fixed_hash and np.array_equal(v,fv) and np.array_equal(u,base),'Fallback is not same-object Fixed')
                fallback_count+=1
            metrics=row['metrics'];index=uids.index(uid)
            require(len(metrics)==2,'Missing independent evaluation repeat')
            require([m['seed'] for m in metrics]==[192401+index,292401+index],'Evaluation seed changed')
            require(all(m['points_per_direction']==8192 for m in metrics),'Evaluation sample count changed')
            meshes[(uid,row['model'],row['policy'])]=mesh
    row_map={key(r):r for r in rows}
    for uid,model,policy in row_ids:
        if policy=='quality_fallback':
            raw=row_map[(uid,model,'raw')];delivered=row_map[(uid,model,policy)]
            require(delivered['fallback_to_fixed']==(not raw['qualified']),'Fallback decision used another criterion')
            if raw['qualified']:
                require(delivered['mesh_hash']==raw['mesh_hash'],'Qualified prediction changed by delivery')
    return {'planned_attempts':80,'observed_attempts':len(attempts),'planned_row_slots':128,
        'saved_success_rows':len(rows),'error_attempts_retained':len(errors),
        'error_accounted_row_slots':len(failed_slots),'all_attempts_and_slots_accounted':True,
        'saved_mesh_hashes_checked':len(meshes),
        'logical_and_storage_hashes_checked_separately':True,
        'mesh_integrity_sidecar_sha256':file_hash(run/'MESH_INTEGRITY.json'),
        'dtype_only_logical_storage_hash_differences':integrity['dtype_only_hash_difference_count'],
        'same_connectivity_and_accepted_direction_requeries_checked':len(meshes),
        'fallback_meshes_identical_to_same_uid_fixed':fallback_count,'data_manifest_files_checked':data_checked,
        'old_source_manifest_files_checked_unchanged':old_checked,'old_source_manifest_sha256':file_hash(old/'MANIFEST_SHA256.json'),
        'train_parent_count':12,'internal_check_parent_count':4,'training_checkpoints':training_checks,
        'gpu_resume_smoke':'PASS','original_A_hash_unchanged':True,
        'fixed_and_original_A_mesh_hashes_unchanged_from_previous_run':unchanged_baseline_count,
        'frozen_source_files_checked':len(protocol['fixed_source_hashes']),
        'teacher_reference_source_identity_checks':len(teacher_identity_records),
        'teacher_source_vs_reference_face_dtype_differences':sum(r['face_dtype_differs'] for r in teacher_identity_records),
        'teacher_source_reference_identity_records':teacher_identity_records,'upstream':identity,
        'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},rows,meshes,old,provenance


def package_run(run,smoke_run):
    run=Path(run).resolve();smoke_run=Path(smoke_run).resolve()
    archive=run.with_name(run.name+'_results.zip')
    fresh=run.with_name(run.name+'_fresh_extract')
    archive_report=archive.with_suffix('.zip.json')
    fresh_report=archive.with_suffix('.zip.fresh.json')
    destinations=[run/'FINAL_AUDIT.json',run/'MANIFEST_SHA256.json',run/'meshes_obj',
        run/'source_snapshot',run/'gpu_smoke_evidence',run/'teacher_evidence',run/'REPLAY_FRESH.py',
        archive,archive_report,fresh,fresh_report]
    for path in destinations:
        require(not path.exists(),'Exclusive packaging destination already exists: '+str(path))
    audit,rows,meshes,old,provenance=audit_run(run,smoke_run)
    for row in rows:
        key=(row['uid'],row['model'],row['policy']);mesh=meshes[key]
        write_obj(run/'meshes_obj'/row['uid']/(row['model']+'__'+row['policy']+'.obj'),mesh['vertices'],mesh['faces'])
    source=Path(__file__).resolve().parents[1]
    ignored=shutil.ignore_patterns('__pycache__','.pytest_cache','build','*.pyc','*.nbi','*.nbc')
    shutil.copytree(source,run/'source_snapshot/experiments/a_radial',ignore=ignored)
    shutil.copytree(smoke_run,run/'gpu_smoke_evidence',ignore=ignored)
    teacher_dest=run/'teacher_evidence';teacher_dest.mkdir()
    shutil.copy2(old/'MANIFEST_SHA256.json',teacher_dest/'MANIFEST_SOURCE_FULL.json')
    for rel,expected in provenance['verified_files'].items():
        src=inside(old,rel);dest=inside(teacher_dest,rel)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        require(file_hash(dest)==expected,'Copied teacher evidence changed')
    write_json(teacher_dest/'COPIED_FILES.json',{'source_run':str(old),
        'source_manifest_sha256':file_hash(old/'MANIFEST_SHA256.json'),
        'files':provenance['verified_files'],
        'note':'MANIFEST_SOURCE_FULL retains original complete manifest; only referenced teacher evidence files are copied here.'})
    with (run/'REPLAY_FRESH.py').open('x',encoding='utf-8') as out:
        out.write(REPLAY_SCRIPT)
    audit['exported_obj_count']=len(rows)
    audit['source_snapshot_files']=sum(p.is_file() for p in (run/'source_snapshot').rglob('*'))
    audit['teacher_evidence_referenced_files_copied']=len(provenance['verified_files'])
    write_json(run/'FINAL_AUDIT.json',audit)
    manifest={p.relative_to(run).as_posix():file_hash(p) for p in sorted(run.rglob('*')) if p.is_file()}
    write_json(run/'MANIFEST_SHA256.json',{'files':manifest})
    members={**manifest,'MANIFEST_SHA256.json':file_hash(run/'MANIFEST_SHA256.json')}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipped:
        for rel in sorted(members):
            zipped.write(run/rel,arcname=(PurePosixPath(run.name)/rel).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.testzip() is None,'ZIP CRC check failed')
        expected_names={(PurePosixPath(run.name)/rel).as_posix() for rel in members}
        require(len(zipped.namelist())==len(expected_names) and set(zipped.namelist())==expected_names,'Unexpected archive member')
        for rel,expected in members.items():
            require(hashlib.sha256(zipped.read((PurePosixPath(run.name)/rel).as_posix())).hexdigest()==expected,'Archive member hash mismatch')
        fresh.mkdir()
        for name in zipped.namelist():
            inside(fresh,name)
        zipped.extractall(fresh)
    extracted=fresh/run.name
    require(read_json(extracted/'MANIFEST_SHA256.json')['files']==manifest,'Fresh extracted manifest mismatch')
    fresh_checked=verify_manifest(extracted,members)
    env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
    command=[sys.executable,'-I','-B',str(extracted/'REPLAY_FRESH.py'),
             '--run',str(extracted),'--report',str(fresh_report)]
    replay=subprocess.run(command,cwd=fresh,env=env,capture_output=True,text=True)
    if replay.returncode:
        failure=archive.with_suffix('.zip.replay_failed.json')
        write_json(failure,{'status':'FAIL','returncode':replay.returncode,
            'stdout':replay.stdout,'stderr':replay.stderr,'fresh_manifest_passed':True})
        raise RuntimeError('Fresh extracted CPU replay failed; evidence saved in '+str(failure))
    require(read_json(fresh_report)['status']=='PASS','Fresh replay did not pass')
    # Replaying may not modify either the extracted artifacts or the sealed run.
    verify_manifest(extracted,members);verify_manifest(run,members)
    result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':file_hash(archive),
        'crc_verified':True,'member_hashes_verified_including_manifest':len(members),
        'fresh_extracted_manifest_files_verified':fresh_checked,
        'fresh_extract_directory':str(fresh),'fresh_cpu_replay':'PASS','fresh_replay_report':str(fresh_report),
        'sealed_run_unchanged_after_replay':True,'fresh_artifacts_unchanged_after_replay':True,
        'saved_meshes':len(rows),'exported_obj_count':len(rows),'failed_attempts_retained':audit['error_attempts_retained']}
    write_json(archive_report,result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--smoke-run',required=True,type=Path)
    args=parser.parse_args();package_run(args.run,args.smoke_run)


if __name__=='__main__':
    main()
