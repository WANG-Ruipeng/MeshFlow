"""Immutable portable joint archive of warm-start results and initializer controls.

Audits saved artifacts only. No optimization, new Monte Carlo evaluation or GPU.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
import torch
from ..io_utils import file_hash,array_hash,write_json,new_dir
from ..exporter import RadialSupport
from ..layout_student.deployment import named_quality,qualification
from ..layout_student.integrity import mesh_identity
from ..layout_student.train import model_sha256

OLD_MODELS=('fixed','longfit_10','longfit_11','balanced_10','balanced_11')
NEW_MODELS=('original_A_10','early_v1_10')
KINDS=('raw','snapshot_0','snapshot_1','snapshot_4','snapshot_16','optimizer_final')
SOURCE_NAMES=('system_probe/warm_start.py','system_probe/warm_start_controls.py','system_probe/warm_start_report.py',
    'system_probe/warm_start_package.py','system_probe/test_warm_start.py','system_probe/test_warm_start_controls.py',
    'system_probe/test_warm_start_package.py','layout_value/optimizer.py','layout_value/geometry.py',
    'layout_student/deployment.py','layout_student/integrity.py','layout_student/evaluate.py','layout_student/train.py',
    'legacy/deform.py','legacy/chart.py','legacy/networks.py','legacy/common.py','legacy/exact_accel.py',
    'legacy/geometry_task.py','legacy/exact_triangles.py','legacy/audit_fast.py','mesh_ops.py','exporter.py','io_utils.py',
    'assets/support.npz','assets/original_A_10.pt','__init__.py','legacy/__init__.py','layout_value/__init__.py','layout_student/__init__.py','system_probe/__init__.py')


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def inside(root,name):
    require(isinstance(name,str) and name and '\\' not in name and ':' not in name,'Nonportable archive name')
    p=PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts,'Path escapes archive')
    root=Path(root).resolve();dest=(root/name).resolve();require(dest.is_relative_to(root),'Path escapes archive')
    return dest


def inventory(root,exclude=()):
    root=Path(root);result={}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(),'Symlink prohibited')
        if path.is_file():
            rel=path.relative_to(root).as_posix()
            if rel not in exclude:result[rel]={'sha256':file_hash(path),'bytes':path.stat().st_size}
    require(len(result)==len({x.casefold() for x in result}),'Case-insensitive filename collision')
    return result


def verify_files(root,manifest,exclude=()):require(inventory(root,exclude)==manifest,'File set/size/hash changed: '+str(root))


def bound_name(original,digest):
    origin=hashlib.sha256(str(Path(original).resolve()).encode()).hexdigest()
    return 'bound/'+digest+'/'+origin+'/'+Path(original).name


def copy_checked(source,dest,digest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);require(not dest.exists(),'Copy destination exists')
    require(file_hash(source)==digest,'Source changed before copy');shutil.copyfile(source,dest)
    require(file_hash(dest)==file_hash(source)==digest,'Source/copy changed during copy')


def check_slots(attempts,p,models):
    require(tuple(p['models'])==models and len(p['uids'])==len(set(p['uids']))==4,'Registered study differs')
    keys=[(a['uid'],a['model']) for a in attempts];expected={(u,m) for u in p['uids'] for m in models}
    require(len(keys)==len(set(keys)) and set(keys)==expected,'Missing/duplicate task')
    rows=[r for a in attempts for r in a['rows']];keys=[(r['uid'],r['model'],r['kind']) for r in rows]
    require(len(keys)==len(set(keys)) and set(keys)=={(u,m,k) for u,m in expected for k in KINDS},'Missing/duplicate primary row')
    return rows


def readback_study(run,resolve_bound,models):
    run=Path(run);p=read(run/'RUN_PROTOCOL.json');result=read(run/'RESULTS.json');support=RadialSupport()
    require(result['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json'),'Ledger protocol hash differs')
    rows=check_slots(result['attempts'],p,models);checked=0;failures=[];data={};floors={}
    require(p['max_steps']==16 and p['snapshot_steps']==[0,1,4,16] and p['evaluation_seed_bases']==[1593501,1693501]
        and p['optimizer_seed_base']==1493501 and p['threads']==4,'Fixed warm-start budget/seeds changed')
    for uid in p['uids']:
        entry=p['data'][uid];path=resolve_bound(entry['path']);require(file_hash(path)==entry['sha256'],'Portable data hash differs')
        with np.load(path,allow_pickle=False) as z:d={k:np.array(z[k]) for k in z.files}
        v,f=support.mesh(d['logr']);require(np.array_equal(v,d['ref_vertices']) and np.array_equal(f,d['ref_faces']),'Reference is not same full field')
        data[uid]=d;fixed=d['base_u']*support.query(d['logr'],d['base_u'])[:,None];floors[uid]=named_quality(fixed,d['faces'])
    for name,identity in p['model_bindings'].items():
        path=resolve_bound(identity['path']);require(file_hash(path)==identity['file_sha256'],'Initializer file differs')
        state=torch.load(path,map_location='cpu',weights_only=False);model=state if name=='original_A_10' else state['model']
        require(model_sha256(model)==identity['model_sha256'] and sum(x.numel() for x in model.values())==29907,'Initializer model differs')
    for attempt in result['attempts']:
        folder=run/'attempts'/attempt['uid']/attempt['model']
        original_task={k:v for k,v in attempt.items() if k not in ('extra_raw_identity_setup_seconds','extra_raw_identity_record')}
        require(read(folder/'TASK.json')==original_task,'Task ledger differs')
        if 'extra_raw_identity_record' in attempt:
            identity=read(resolve_bound(attempt['extra_raw_identity_record']))
            reference=folder.parents[2]/'raw_identity_reference'/attempt['uid']/attempt['model']/'raw.npz'
            require(file_hash(reference)==identity['raw_file_sha256'],'Control raw identity evidence changed')
            with np.load(reference,allow_pickle=False) as a,np.load(folder/'RAW_INITIALIZER.npz',allow_pickle=False) as b:
                require(set(a.files)==set(b.files) and all(a[k].dtype==b[k].dtype and np.array_equal(a[k],b[k]) for k in a.files),'Control raw exact setup replay differs')
        optimizer=read(folder/'OPTIMIZER.json') if (folder/'OPTIMIZER.json').exists() else None
        if optimizer:
            require(file_hash(folder/'OPTIMIZER_ARRAYS.npz')==optimizer['arrays_sha256'],'Optimizer saved arrays differ')
            steps=optimizer['stats']['iterations'];require(0<=steps<=16 and [x['iteration'] for x in optimizer['trace']]==list(range(steps+1)),'Optimizer trace budget differs')
        for row in attempt['rows']+attempt.get('extra_terminal_snapshots',[]):
            row_path=folder/row['kind']/'RESULT.json'
            if row_path.exists():require(read(row_path)==row,'Row ledger differs')
            if row.get('status')!='success':failures.append({k:row.get(k) for k in ('uid','model','kind','status','error')})
            if row.get('status')=='success':
                require(all(k in row for k in ('mesh_file','quality','audit','metrics','GT_reference_hash','qualified','failure_reasons')),'Incomplete successful endpoint')
            if 'mesh_file' not in row:continue
            d=data[row['uid']];path=inside(run,row['mesh_file']);identity=mesh_identity(path,d['faces'])
            require(all(row[k]==v for k,v in identity.items()),'Typed saved-mesh identity differs')
            with np.load(path,allow_pickle=False) as z:mesh={k:np.array(z[k]) for k in z.files}
            radius=support.query(d['logr'],mesh['u']);vertices=mesh['u']*radius[:,None]
            require(np.array_equal(radius,mesh['r']) and np.array_equal(vertices,mesh['vertices']),'Accepted-radius requery differs')
            quality=named_quality(vertices,mesh['faces'])
            if 'quality' in row:require(quality==row['quality'],'Actual quality differs')
            if 'audit' in row:
                q=qualification(quality,row['audit'],floors[row['uid']]);require(all(row[k]==v for k,v in q.items()),'Recorded qualification differs')
            require(row['mainfield_hash']==array_hash(d['logr']),'Mainfield identity differs')
            if 'GT_reference_hash' in row:require(row['GT_reference_hash']==array_hash(d['ref_vertices'],d['ref_faces']),'Reference identity differs')
            if 'metrics' in row:
                seeds=[b+p['all16_sorted_uids'].index(row['uid']) for b in p['evaluation_seed_bases']]
                require([m['seed'] for m in row['metrics']]==seeds and all(m['points_per_direction']==8192 for m in row['metrics']),'MC evidence budget differs')
                cacheid=array_hash(vertices,mesh['faces'],d['ref_vertices'],d['ref_faces'])
                for seed,metric in zip(seeds,row['metrics']):
                    cached=read(run/'metric_cache'/row['uid']/f'{cacheid}_{seed}_8192.json')
                    require(cached['mesh_reference_hash']==cacheid and cached['metrics']==metric,'Saved MC evidence differs')
            if row['kind']=='raw':
                with np.load(folder/'RAW_INITIALIZER.npz',allow_pickle=False) as z:
                    require(np.array_equal(z['u'],mesh['u']) and np.array_equal(z['vertices'],vertices),'Raw saved initializer differs')
            elif optimizer:
                stem='final' if row['kind']=='optimizer_final' else 'snapshot_'+str(row['actual_iteration'])
                with np.load(folder/'OPTIMIZER_ARRAYS.npz',allow_pickle=False) as z:
                    require(np.array_equal(z[stem+'_u'],mesh['u']) and np.array_equal(z[stem+'_vertices'],vertices),'Optimizer snapshot geometry differs')
            checked+=1
    return {'tasks':len(result['attempts']),'primary_slots':len(rows),'saved_meshes_verified':checked,'failures':failures,
        'model_initializers_verified':len(p['model_bindings']),'MC_recomputed':False,'optimization_rerun':False,
        'exact_audit_evidence_preserved_not_recomputed':True,'same_field_reference_arrays_exact':True}


def plan(parent,control,review,correction):
    roots={'parent':Path(parent).resolve(),'controls':Path(control).resolve(),'parent_review':Path(review).resolve()}
    docfix=roots['controls'].with_name(roots['controls'].name+'_docfix_01')
    if docfix.is_dir():roots['control_docfix']=docfix
    driver=roots['parent'].with_name(roots['parent'].name+'_driver')
    if driver.is_dir():roots['parent_driver']=driver
    correction=Path(correction).resolve();code=Path(__file__).resolve().parents[1];bound={};p={}
    for name in ('parent','controls'):
        p[name]=read(roots[name]/'RUN_PROTOCOL.json')
        for original,digest in p[name]['bound_files_sha256'].items():
            require(original not in bound or bound[original]==digest,'Conflicting source binding');bound[original]=digest
        require((roots[name]/'INTERPRETATION_zh.md').is_file(),'Final reviewed interpretation missing: '+name)
    require(p['controls']['uids']==p['parent']['uids'] and p['controls']['all16_sorted_uids']==p['parent']['all16_sorted_uids'],'Control changed registered parents')
    require(p['controls']['parent_protocol_sha256']==file_hash(roots['parent']/'RUN_PROTOCOL.json')
        and p['controls']['parent_results_sha256']==file_hash(roots['parent']/'RESULTS.json'),'Control parent binding changed')
    parent_manifest=read(roots['controls']/'PARENT_FILE_MANIFEST.json')
    require(file_hash(roots['controls']/'PARENT_FILE_MANIFEST.json')==p['controls']['parent_file_manifest_sha256'],'Bound parent manifest changed')
    actual_parent={str(roots['parent']/rel):v['sha256'] for rel,v in inventory(roots['parent']).items()}
    require(actual_parent==parent_manifest['files'],'Frozen parent inventory changed')
    reviewer=read(roots['parent_review']/'REVIEW.json')
    require(reviewer['results_sha256']==file_hash(roots['parent']/'RESULTS.json') and reviewer['all_requested_rows_accounted'],'Original review stale')
    require(reviewer['source_sha256']==file_hash(roots['parent_review']/'EXECUTED_SOURCE.py')==file_hash(code/'system_probe/warm_start_report.py'),'Original review source changed')
    copies={};paths={};source_meta={}
    root_manifest={name:inventory(root) for name,root in roots.items()}
    def add(original,relative,digest):
        original=Path(original).resolve();key=str(original)
        require(file_hash(original)==digest,'Source changed: '+key)
        if key in paths:require(paths[key]['sha256']==digest,'Conflicting portable source');return
        require(relative not in copies,'Portable path collision')
        paths[key]={'relative':relative,'sha256':digest};copies[relative]={'source':key,'sha256':digest}
    for name,root in roots.items():
        for rel,entry in root_manifest[name].items():add(root/rel,'runs/'+name+'/'+rel,entry['sha256'])
    for name in SOURCE_NAMES:
        path=code/name;add(path,'source/experiments/a_radial/'+name,file_hash(path))
        source_meta[name]={'original_path':str(path),'sha256':file_hash(path)}
    for original,digest in bound.items():
        path=Path(original)
        if path.is_relative_to(code):require(path.relative_to(code).as_posix() in SOURCE_NAMES,'Bound source outside explicit whitelist')
        add(path,bound_name(path,digest),digest)
    for failure_dir in sorted(correction.parent.glob('a_warm_start_joint_audit_failure_*')):
        if failure_dir.is_dir():
            for path in failure_dir.rglob('*'):
                if path.is_file():add(path,'audit_history/'+failure_dir.name+'/'+path.relative_to(failure_dir).as_posix(),file_hash(path))
    for path in sorted(correction.parent.glob('a_warm_start_joint_PACKAGE_TESTS*.xml')):
        add(path,'test_evidence/'+path.name,file_hash(path))
    add(correction,'REFERENCE_SEMANTICS_CORRECTION_zh.md',file_hash(correction))
    correction_run=correction.parent/'a_reference_semantics_20260925_01'
    for path in correction_run.rglob('*'):
        if path.is_file():add(path,'reference_audit/'+path.relative_to(correction_run).as_posix(),file_hash(path))
    stats={name:readback_study(roots[name],lambda path:Path(path),models) for name,models in [('parent',OLD_MODELS),('controls',NEW_MODELS)]}
    require(stats['parent']['primary_slots']+stats['controls']['primary_slots']==168,'Combined row denominator differs')
    return {'roots':{k:str(v) for k,v in roots.items()},'original_manifests':root_manifest,'copy_plan':copies,
        'portable_paths':paths,'source_whitelist':source_meta,'readback':stats,'omitted_parallel_modules':True,
        'historical_absolute_paths_preserved_in_original_protocols':True,'reference_correction_added_outside_frozen_runs':True}


def portable_readback(root):
    root=Path(root).resolve();manifest=read(root/'BUNDLE_MANIFEST.json')['files'];verify_files(root,manifest,('BUNDLE_MANIFEST.json',))
    mapping=read(root/'PORTABLE_PATHS.json')['files']
    def resolve(original):
        require(original in mapping,'Unbound original path');entry=mapping[original];path=inside(root,entry['relative'])
        require(file_hash(path)==entry['sha256'],'Portable source hash differs');return path
    for original in mapping:resolve(original)
    stats={name:readback_study(root/'runs'/name,resolve,models) for name,models in [('parent',OLD_MODELS),('controls',NEW_MODELS)]}
    require(not torch.cuda.is_initialized(),'Portable verification initialized GPU')
    require(Path(__file__).resolve().is_relative_to(root/'source'),'Source import escaped extracted bundle')
    return {'status':'PASS','readback':stats,'files':len(manifest),'original_absolute_paths_opened':False,
        'optimization_updates':0,'new_MC_evaluations':0,'GPU_initialized':False}


def extract_checked(archive,out,members):
    out=new_dir(out)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();require(len(names)==len(set(names))==len(members) and set(names)==set(members),'ZIP member set differs')
        for name in names:inside(out,name)
        require(z.testzip() is None,'ZIP CRC failure')
        for name,digest in members.items():require(hashlib.sha256(z.read(name)).hexdigest()==digest,'ZIP member SHA differs')
        z.extractall(out)
    return out


def package(parent,control,review,correction,out):
    require(not torch.cuda.is_initialized(),'CPU archive only');torch.set_num_threads(4)
    out=Path(out).resolve();require(not out.exists(),'Archive destination already exists')
    audit=plan(parent,control,review,correction) # All source assertions precede any sealing write.
    out=new_dir(out);started=time.perf_counter()
    try:
        write_json(out/'AUDIT.json',audit);payload=new_dir(out/'payload')
        for rel,entry in audit['copy_plan'].items():copy_checked(entry['source'],inside(payload,rel),entry['sha256'])
        write_json(payload/'PORTABLE_PATHS.json',{'files':audit['portable_paths'],'original_JSON_absolute_paths_preserved':True})
        write_json(payload/'SOURCE_WHITELIST.json',{'files':audit['source_whitelist'],'unrelated_parallel_experiments_excluded':True})
        write_json(payload/'AUDIT.json',audit)
        (payload/'FRESH_VERIFY.py').write_text("from pathlib import Path\nimport os,sys,json\nos.environ['CUDA_VISIBLE_DEVICES']=''\nroot=Path(__file__).resolve().parent\nsys.path.insert(0,str(root/'source'))\nfrom experiments.a_radial.system_probe import warm_start_package as p\np.torch.set_num_threads(4)\nprint(json.dumps(p.portable_readback(root)),flush=True)\n")
        (payload/'README.md').write_text('Frozen warm-start20tasks and initializer-control8tasks: all168 primary slots and failures retained. The geometric reference is the same frozen radial-field surface, not the original raw mesh. See the external correction and both interpretations. No optimization or MC is rerun by fresh verification. Original JSON paths remain as provenance; PORTABLE_PATHS maps every bound file to this package.\n')
        manifest=inventory(payload);write_json(payload/'BUNDLE_MANIFEST.json',{'files':manifest,'self_excluded':True})
        members={name:v['sha256'] for name,v in manifest.items()};members['BUNDLE_MANIFEST.json']=file_hash(payload/'BUNDLE_MANIFEST.json')
        archive=out/'WARM_START_AND_INITIALIZER_CONTROLS.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in sorted(members):z.write(inside(payload,name),name)
        fresh=extract_checked(archive,out/'fresh_extract',members)
        require(file_hash(fresh/'BUNDLE_MANIFEST.json')==members['BUNDLE_MANIFEST.json'],'Extracted manifest itself differs')
        proc=subprocess.run([sys.executable,'-I','-B',str(fresh/'FRESH_VERIFY.py')],cwd=fresh,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(out/'fresh_numba_cache')),
            capture_output=True,text=True)
        write_json(out/'FRESH_CPU_READBACK.json',{'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr})
        require(proc.returncode==0,'Fresh CPU readback failed');fresh_result=json.loads(proc.stdout.strip().splitlines()[-1])
        require(fresh_result['status']=='PASS','Fresh readback did not pass')
        verify_files(payload,manifest,('BUNDLE_MANIFEST.json',));verify_files(fresh,manifest,('BUNDLE_MANIFEST.json',))
        for name,root in audit['roots'].items():verify_files(root,audit['original_manifests'][name])
        for entry in audit['copy_plan'].values():require(file_hash(entry['source'])==entry['sha256'],'Original source changed after archive')
        result={'status':'SEALED','archive':str(archive),'bytes':archive.stat().st_size,'sha256':file_hash(archive),
            'members_including_manifest':len(members),'CRC_all_member_hashes_and_fresh_manifest_pass':True,
            'original_runs_unchanged':True,'source_copy_original_fresh_equal':True,'fresh_CPU_readback':fresh_result,
            'no_new_optimization_MC_or_GPU':True,'elapsed_seconds':time.perf_counter()-started}
        write_json(out/'SEALED.json',result);print(json.dumps(result,indent=2),flush=True);return result
    except BaseException as exc:
        write_json(out/'FAILED.json',{'error':repr(exc),'partial_outputs_retained':True});raise


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--parent-run',required=True);a.add_argument('--control-run',required=True)
    a.add_argument('--parent-review',required=True);a.add_argument('--reference-correction',required=True);a.add_argument('--out');a.add_argument('--audit-only',action='store_true');args=a.parse_args()
    torch.set_num_threads(4)
    if args.audit_only:print(json.dumps(plan(args.parent_run,args.control_run,args.parent_review,args.reference_correction)['readback'],indent=2))
    else:
        require(args.out is not None,'--out required');package(args.parent_run,args.control_run,args.parent_review,args.reference_correction,args.out)


if __name__=='__main__':main()
