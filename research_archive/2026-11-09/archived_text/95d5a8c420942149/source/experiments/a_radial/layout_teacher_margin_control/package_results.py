"""Joint immutable archive of strict-margin and paired original-floor diagnostics."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,subprocess,sys,time,zipfile,shutil
from pathlib import Path,PurePosixPath
import numpy as np
import torch
from ..io_utils import file_hash,new_dir,write_json
from ..exporter import RadialSupport
from ..layout_student.deployment import named_quality,qualification
from ..layout_student.integrity import mesh_identity
from ..legacy.chart import dets
from ..layout_teacher_margin.review import audit as strict_audit,reconstruct
from .review import audit as control_audit


def require(ok,message):
    if not ok:raise ValueError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def inside(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    require(path.is_relative_to(root),'Path escapes artifact root')
    return path


def inventory(root,exclude=()):
    root=Path(root);result={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'Symlink not permitted')
        if p.is_file():
            rel=p.relative_to(root).as_posix()
            if rel not in exclude:result[rel]={'sha256':file_hash(p),'bytes':p.stat().st_size}
    return result


def verify_files(root,expected,exclude=()):
    require(inventory(root,exclude)==expected,'File set/size/hash mismatch: '+str(root))
    return len(expected)


def copy_checked(source,target,expected):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    require(not target.exists(),'Do not overwrite copy')
    shutil.copyfile(source,target)
    require(file_hash(source)==file_hash(target)==expected,'Source/copy changed during snapshot')


def source_map(strict,control):
    root=Path(__file__).resolve().parents[1];paths={}
    for run in (strict,control):
        for name,digest in read(run/'RUN_PROTOCOL.json')['source_sha256'].items():
            original=Path(name).resolve();require(original.is_relative_to(root),'Unexpected source root')
            require(file_hash(original)==digest,'Frozen implementation changed')
            paths[original.relative_to(root).as_posix()]=original
    extras=['__init__.py','legacy/__init__.py','legacy/exact_triangles.py','legacy/audit_fast.py',
        'layout_value/__init__.py','layout_student/__init__.py','system_probe/__init__.py',
        'layout_teacher_margin/review.py','layout_teacher_margin/visualize.py','layout_teacher_margin/test_experiment.py',
        'layout_teacher_margin_control/review.py','layout_teacher_margin_control/visualize.py',
        'layout_teacher_margin_control/test_experiment.py','layout_teacher_margin_control/package_results.py',
        'layout_teacher_margin_control/test_package_results.py']
    for name in extras:paths[name]=root/name
    require(all(p.is_file() for p in paths.values()),'Source whitelist dependency missing')
    require(not any('cpu_c0_cost' in name or name.startswith(('layout_balanced/','fm_tail/','system_formal/')) for name in paths),
        'Unrelated experiment entered whitelist')
    return paths


def write_obj(path,vertices,faces):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='ascii') as f:
        for v in vertices:f.write('v %.17g %.17g %.17g\n'%tuple(v))
        for tri in faces:f.write('f %d %d %d\n'%tuple(np.asarray(tri,dtype=np.int64)+1))


def portable_readback(root):
    """Only this extracted payload is opened; no optimizer, GT MC, or external paths."""
    root=Path(root).resolve();manifest=read(root/'BUNDLE_MANIFEST.json')['files']
    verify_files(root,manifest,exclude=('BUNDLE_MANIFEST.json',));support=RadialSupport();meshes=0;noise_count=0
    mapping=read(root/'SOURCE_WHITELIST.json')['files']
    for relative,record in mapping.items():
        require(file_hash(inside(root,'source/experiments/a_radial/'+relative))==record['sha256'],'Whitelisted source identity differs')
    for kind in ('strict','control'):
        run=root/'runs'/kind;p=read(run/'RUN_PROTOCOL.json');result=read(run/'RESULTS.json');noise=read(run/'SENSITIVITY.json')
        expected_rows=16 if kind=='strict' else 4;expected_noise=1344 if kind=='strict' else 448
        require(len(result['rows'])==expected_rows and len(noise['rows'])==expected_noise,'Artifact denominator differs')
        require(result['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json'),'Portable protocol binding differs')
        for original,digest in p['source_sha256'].items():
            matches=[rel for rel,v in mapping.items() if v['original_path']==original]
            require(len(matches)==1 and mapping[matches[0]]['sha256']==digest,'Portable frozen source binding missing')
        centers={};datasets={};floors={}
        for uid in p['uids']:
            with np.load(run/'data'/f'{uid}.npz',allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
            datasets[uid]=data;v=data['base_u']*support.query(data['logr'],data['base_u'])[:,None]
            floors[uid]=named_quality(v,data['faces'])
        for row in result['rows']:
            if 'mesh_file' not in row:continue
            uid=row['uid'];data=datasets[uid];path=inside(run,row['mesh_file'])
            identity=mesh_identity(path,data['faces']);require(all(row[k]==v for k,v in identity.items()),'Portable typed mesh hash differs')
            with np.load(path,allow_pickle=False) as z:
                u=np.array(z['u']);saved_r=np.array(z['r']);saved_v=np.array(z['vertices'])
            radius=support.query(data['logr'],u);vertices=u*radius[:,None]
            require(np.array_equal(saved_r,radius) and np.array_equal(saved_v,vertices),'Portable accepted radius differs')
            q=named_quality(vertices,data['faces']);require(q==row['quality'],'Portable actual quality differs')
            floor=floors[uid];strict={k:floor[k]+.1 for k in ('min_triangle_angle_deg','triangle_min_angle_p05_deg')}
            require(row['qualified_original']==qualification(q,row['audit'],floor)
                and row['qualified_strict']==qualification(q,row['audit'],strict),'Portable gates differ')
            centers[(uid,row['endpoint'])]=u;meshes+=1
        for row in noise['rows']:
            uid=row['uid'];name=row['endpoint'] if kind=='strict' else 'teacher_start_original_floor'
            if row['status']!='success' or (uid,name) not in centers:continue
            data=datasets[uid];u0=centers[(uid,name)];energy=float(np.square(data['teacher_u']-data['base_u']).mean())
            seed=1393501+16*p['all_train_uids'].index(uid)+row['trial']
            require(seed==row['seed'] and energy==row['original_teacher_energy'],'Portable noise identity differs')
            ambient=np.random.default_rng(seed).normal(size=u0.shape);u=reconstruct(u0,ambient,energy,row['fraction'])
            q=named_quality(u*support.query(data['logr'],u)[:,None],data['faces'])
            margins={k:q[k]-floors[uid][k] for k in ('min_triangle_angle_deg','triangle_min_angle_p05_deg')}
            require(all(v>=-1e-6 for v in margins.values())==row['angular_pass'],'Portable noise gate differs')
            require(max(abs(margins[k]-row['margins_vs_fixed_deg'][k]) for k in margins)<=1e-10,'Portable noise angles differ')
            require(abs(float(np.square(u-u0).mean())-row['actual_direction_MSE_to_endpoint'])<=1e-18,'Portable direction MSE differs')
            require(bool(dets(u,data['faces']).min()>0)==row['direction_determinants_positive'],'Portable direction sign differs')
            noise_count+=1
    require(not torch.cuda.is_initialized(),'Unexpected GPU context')
    return {'status':'PASS','files':len(manifest),'saved_endpoint_meshes_verified':meshes,
        'noise_records_reconstructed':noise_count,'CPU_only':True,'GPU_initialized':False,
        'optimization_updates':0,'new_GT_MC_evaluations':0,'original_paths_opened':False}


def extract_checked(archive,target,expected):
    target=new_dir(target)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();require(len(names)==len(set(names))==len(expected) and set(names)==set(expected),'ZIP member set differs')
        for name in names:inside(target,name)
        require(z.testzip() is None,'ZIP CRC failure')
        for name,digest in expected.items():require(hashlib.sha256(z.read(name)).hexdigest()==digest,'ZIP member hash differs')
        z.extractall(target)
    return target


def audit_pair(strict,control):
    strict=Path(strict).resolve();control=Path(control).resolve()
    a=strict_audit(strict);b=control_audit(control)
    require(a['status']==b['status']=='PASS','Independent source review failed')
    for run in (strict,control):
        for name in ('INTERPRETATION_zh.md','REVIEW_zh.md','VISUAL_QA.json','figures/FIGURE_MANIFEST.json'):
            require((run/name).is_file(),'Missing reviewed document: '+str(run/name))
        require(read(run/'VISUAL_QA.json')['status']=='PASS','Figure visual QA missing')
        fig=read(run/'figures/FIGURE_MANIFEST.json')
        for name,digest in fig['files'].items():require(file_hash(run/'figures'/name)==digest,'Reviewed figure changed')
    sources=source_map(strict,control)
    return {'strict':a,'control':b,'original_run_manifests':{'strict':inventory(strict),'control':inventory(control)},
        'source_whitelist':{name:{'original_path':str(path),'sha256':file_hash(path),'bytes':path.stat().st_size} for name,path in sources.items()},
        'no_new_optimization_or_GT_MC':True,'science_success_not_required_for_archive':True},sources


def package(strict,control,out):
    require(not torch.cuda.is_initialized(),'CPU-only package')
    strict=Path(strict).resolve();control=Path(control).resolve();out=Path(out).resolve()
    require(not out.exists(),'Archive destination must not exist')
    # Complete all read-only audit assertions before creating a sealing directory.
    audit,sources=audit_pair(strict,control);out=new_dir(out);started=time.perf_counter()
    try:
        write_json(out/'AUDIT.json',audit);payload=new_dir(out/'payload')
        for kind,original in [('strict',strict),('control',control)]:
            shutil.copytree(original,payload/'runs'/kind)
            verify_files(payload/'runs'/kind,audit['original_run_manifests'][kind])
            verify_files(original,audit['original_run_manifests'][kind])
        for name,path in sources.items():copy_checked(path,payload/'source/experiments/a_radial'/name,audit['source_whitelist'][name]['sha256'])
        write_json(payload/'SOURCE_WHITELIST.json',{'files':audit['source_whitelist'],'scope':'explicit actual dependencies; CPU cost/parallel research excluded'})
        shutil.copyfile(out/'AUDIT.json',payload/'AUDIT.json')
        testroot=Path(__file__).resolve().parents[3]/'output'
        for name in ('a_teacher_margin_20260925_PREPARE_TESTS.xml','a_teacher_margin_control_20260925_PREPARE_TESTS.xml','a_teacher_margin_joint_PACKAGE_TESTS.xml'):
            source=testroot/name;copy_checked(source,payload/'test_evidence'/name,file_hash(source))
        for kind in ('strict','control'):
            run=payload/'runs'/kind
            for row in read(run/'RESULTS.json')['rows']:
                if 'mesh_file' not in row:continue
                with np.load(inside(run,row['mesh_file']),allow_pickle=False) as z:
                    write_obj(payload/'objs'/kind/row['uid']/(row['endpoint']+'.obj'),z['vertices'],z['faces'])
        write_json(payload/'PORTABLE_PATHS.json',{'original_strict_run':str(strict),'strict_run':'runs/strict',
            'original_control_run':str(control),'control_run':'runs/control','original_JSON_absolute_paths_preserved':True,
            'source':'source','parent_longfit_archive_not_nested':'its hash-bound receipts and source evidence are preserved inside strict run'})
        (payload/'README_zh.md').write_text('四训练对象的质量余量可行性与唯一原floor控制联合封存。\n\n两run逐字节复制、原目录不改；16+4端点、1344+448扰动及全部失败槽保留。源代码按显式白名单复制。三个原floor端点停滞、严格floor触发恢复的局部机制不代表网络/泛化优势。零位移教师的随机扰动尺度为零，不能作稳健性结论。\n\n新目录验证只用包内源码/数据，重查末端半径、质量、mesh哈希和固定角条件噪声，不重新优化或GT MC。上游longfit大包不嵌套；其清单、归档收据与四对象实际数据保留。\n',encoding='utf-8')
        (payload/'FRESH_VERIFY.py').write_text("from pathlib import Path\nimport os,sys,json\nos.environ['CUDA_VISIBLE_DEVICES']=''\nroot=Path(__file__).resolve().parent\nsys.path.insert(0,str(root/'source'))\nfrom experiments.a_radial.layout_teacher_margin_control import package_results as p\nassert Path(p.__file__).resolve().is_relative_to(root/'source')\nprint(json.dumps(p.portable_readback(root)),flush=True)\n",encoding='utf-8')
        manifest=inventory(payload);write_json(payload/'BUNDLE_MANIFEST.json',{'files':manifest,'self_excluded':True})
        members={name:v['sha256'] for name,v in manifest.items()};members['BUNDLE_MANIFEST.json']=file_hash(payload/'BUNDLE_MANIFEST.json')
        archive=out/'TEACHER_MARGIN_AND_CONTROL.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in sorted(members):z.write(inside(payload,name),name)
        fresh=extract_checked(archive,out/'fresh_extract',members)
        require(file_hash(fresh/'BUNDLE_MANIFEST.json')==file_hash(payload/'BUNDLE_MANIFEST.json'),'Fresh manifest itself differs')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(out/'fresh_numba_cache'))
        completed=subprocess.run([sys.executable,'-I','-B',str(fresh/'FRESH_VERIFY.py')],cwd=fresh,env=env,capture_output=True,text=True)
        write_json(out/'FRESH_CPU_READBACK.json',{'returncode':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr})
        require(completed.returncode==0,'Fresh CPU readback failed')
        readback=json.loads(completed.stdout.strip().splitlines()[-1]);require(readback['status']=='PASS','Readback did not pass')
        verify_files(payload,manifest,exclude=('BUNDLE_MANIFEST.json',));verify_files(fresh,manifest,exclude=('BUNDLE_MANIFEST.json',))
        for kind,original in [('strict',strict),('control',control)]:verify_files(original,audit['original_run_manifests'][kind])
        for name,path in sources.items():require(file_hash(path)==audit['source_whitelist'][name]['sha256'],'Source changed during sealing')
        result={'status':'SEALED','archive':str(archive),'bytes':archive.stat().st_size,'sha256':file_hash(archive),
            'members_including_manifest':len(members),'CRC_and_all_member_SHA256_pass':True,'manifest_itself_payload_zip_fresh_equal':True,
            'fresh_CPU_readback':readback,'strict_and_control_original_runs_unchanged':True,'source_copy_origin_fresh_hashes_equal':True,
            'no_new_optimizer_or_MC_runs':True,'source_whitelist_file_count':len(sources),'elapsed_sealing_seconds':time.perf_counter()-started}
        write_json(out/'SEALED.json',result);print(json.dumps(result,indent=2),flush=True);return result
    except BaseException as exc:
        write_json(out/'FAILED.json',{'error':repr(exc),'partial_outputs_retained':True});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--strict-run',required=True);p.add_argument('--control-run',required=True)
    p.add_argument('--out');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if a.audit_only:
        audit,_=audit_pair(a.strict_run,a.control_run);print(json.dumps({'status':'PASS','strict_files':len(audit['original_run_manifests']['strict']),
            'control_files':len(audit['original_run_manifests']['control']),'source_files':len(audit['source_whitelist'])}),flush=True)
    else:
        require(a.out is not None,'--out required');package(a.strict_run,a.control_run,a.out)


if __name__=='__main__':main()
