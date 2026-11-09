"""Independent CPU-only archive of fixed FM-tail64k and its saved evidence.
Never runs inference, optimization, geometry audit, or sampling. Original JSON paths
remain byte-identical; PORTABLE_PATHS records where their copies can be found.
"""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from ..io_utils import array_hash,file_hash,new_dir,write_json
from ..system_probe.ours_package import SOURCE_NAMES as ORIGINAL_SOURCES
from ..system_probe.ours_package import compare,files_manifest,read,require,verify_files
from . import artifacts
from .smoke import semantic

TAIL_SOURCES=('__init__.py','train.py','smoke.py','artifacts.py','metrics.py','visualize.py',
    'compare_figures.py','test_train.py','test_artifacts.py','package_results.py','test_package_results.py')
FAILURE_NAMES=('a_fm_tail_20260925_01','a_fm_tail_20260925_02','a_fm_tail_gpu_smoke_20260925_01')
REVIEW_RECORDS={'PROBE_FIELDS_REVIEW.json':'review_probe_fields.py','METRICS_REVIEW.json':'review_metrics.py',
    'DISTRIBUTION_REVIEW.json':'review_distribution.py'}


def inside(root,name):
    root=Path(root).resolve();p=(root/name).resolve()
    require(p.is_relative_to(root),'Path escapes package root: '+str(name));return p


def source_paths(repo):
    names=[f'experiments/a_radial/{n}' for n in ORIGINAL_SOURCES]
    names += [f'experiments/a_radial/fm_tail/{n}' for n in TAIL_SOURCES]
    require(len(names)==len(set(names)),'Duplicate source whitelist')
    return {n:Path(repo)/n for n in names}


def selected_parent_files(parent):
    """Parent is separately fully sealed; omit only unrelated timing and intermediate weights."""
    selected={}
    for p in sorted(Path(parent).rglob('*')):
        require(not p.is_symlink(),'Parent symlink')
        if not p.is_file():continue
        rel=p.relative_to(parent)
        if rel.parts[0]=='timing':continue
        if p.suffix=='.pt' and p.name not in ('step00003200.pt','step00008000.pt'):continue
        selected[rel.as_posix()]=p
    return selected


def copy_bound_files(sources,destination):
    before={n:{'sha256':file_hash(p),'bytes':p.stat().st_size} for n,p in sources.items()}
    for name,src in sources.items():
        dst=inside(destination,name);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        compare({'sha256':file_hash(dst),'bytes':dst.stat().st_size},before[name],'Copied '+name)
        compare({'sha256':file_hash(src),'bytes':src.stat().st_size},before[name],'Source changed '+name)
    verify_files(destination,before);return before


def validate_reviews(review,repo,run):
    records={}
    for record,script in REVIEW_RECORDS.items():
        r=read(review/record);require(r['source_sha256']==file_hash(review/script),'Review source changed '+record)
        bound=r.get('bound_reviewed_files',r.get('bound_files_sha256',{}))
        for name,expected in bound.items():
            p=Path(name) if Path(name).is_absolute() else repo/name
            require(file_hash(p)==expected,'Reviewed file changed '+name)
        records[record]={'sha256':file_hash(review/record),'source_sha256':file_hash(review/script),'status':r['status']}
    pr=read(review/'PROBE_FIELDS_REVIEW.json');mr=read(review/'METRICS_REVIEW.json');dr=read(review/'DISTRIBUTION_REVIEW.json')
    require(pr['primary_E32_registered_fields']==pr['aux_E128_registered_fields']==128,'Field denominator')
    require(pr['training_protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json') and pr['artifact_protocol_sha256']==file_hash(run/'ARTIFACT_PROTOCOL.json'),'Review protocol')
    require(mr['rows']==384 and mr['metric_failures']==0 and mr['results_sha256']==file_hash(run/'artifacts/metrics/RESULTS.json'),'Review metrics identity')
    require(dr['status']=='DISTRIBUTION_SAVED_MATRIX_AND_DENOMINATOR_REVIEW_PASS' and dr['per_sample_rows']==384 and dr['group_matrices_recomputed']==24,'Distribution review incomplete')
    require((review/'REVIEW_zh.md').is_file(),'Missing factual review');return records


def training_gate_audit(run,protocol):
    gate=read(run/'GPU_GATE.json');gate_path=Path(gate['out']).resolve()
    require(gate['status']=='PASS' and gate['actual_new_optimizer_updates']==36,'CUDA gate incomplete')
    require(file_hash(gate_path/'GPU_GATE.json')==file_hash(run/'GPU_GATE.json'),'External CUDA gate differs')
    checks=[]
    for entry in gate['seeds']:
        folder=gate_path/f"seed_{entry['seed']}"
        for name,sha in entry['checkpoints_sha256'].items():require(file_hash(folder/name)==sha,'Gate checkpoint changed')
        whole=torch.load(folder/'continuous8.pt',map_location='cpu',weights_only=True)
        resumed=torch.load(folder/'resumed3plus5.pt',map_location='cpu',weights_only=True)
        compare(semantic(whole),semantic(resumed),'Saved CUDA continuous8 vs3plus5 all semantic states')
        checks.append({'seed':entry['seed'],'saved_semantic_states_exact':True,'gate_only_not_formal_weights':True})
    identities={str(s):artifacts.final_inputs(run,s) for s in (10,11)}
    return gate_path,{'fixed_formal_step':64000,'new_updates_total':112000,'parent_exposure_each':512000,
        'final_checkpoint_full_state_validation':True,'gate_saved_state_readback':checks,'identities':identities,
        'new_GPU_execution':False,'manual_next_step_evidence_is_historical_not_reexecuted':True}


def portable_arrays(root):
    root=Path(root);run=root/'run';p=read(run/'RUN_PROTOCOL.json');ap=read(run/'ARTIFACT_PROTOCOL.json')
    rows=read(run/'artifacts/metrics/RESULTS.json')['rows'];mi={(r['seed'],r['arm'],r['sample_id']):r for r in rows}
    require(len(rows)==len(mi)==384,'Portable metric slots')
    primary=auxiliary=outlets=0
    for seed in (10,11):
        base=run/f'artifacts/seed{seed}';arrays_by_kind={};expected={f'{i:05d}' for i in range(64)}
        for kind,folder,steps in [('primary',base/'samples',32),('auxiliary',base/'auxiliary_e128/samples',128)]:
            sm=read(folder/'SAMPLES.json');records=sm['records'];require(len(records)==64 and {r['sample_id'] for r in records}==expected,'Portable sample slots')
            require(sm['fm_steps']==steps,'Primary/auxiliary isolation')
            arrays_by_kind[kind]={}
            for r in records:
                f=inside(folder,r['path'])
                with np.load(f,allow_pickle=False) as z:
                    aa={k:z[k].copy() for k in z.files}
                require(array_hash(aa['noise'],aa['z_standardized'],aa['z'],aa['nodes'])==r['array_hash'] and array_hash(aa['nodes'])==r['nodes_hash'],'Portable sample arrays')
                arrays_by_kind[kind][r['sample_id']]=(aa,r)
                if kind=='primary':primary+=1
                else:auxiliary+=1
        require(not (base/'auxiliary_e128/exports').exists(),'Auxiliary used as main geometry')
        for sid in sorted(expected):
            aa,rr=arrays_by_kind['primary'][sid];bb,_=arrays_by_kind['auxiliary'][sid]
            compare(aa['noise'],bb['noise'],'Primary/aux exact epsilon')
        for arm in ('fixed','a','qem'):
            ledger=read(base/'exports'/f'{arm}_MANIFEST.json')['records']
            require(len(ledger)==64 and {r['sample_id'] for r in ledger}==expected,'Portable outlet slots')
            for r in ledger:
                row=mi[(seed,arm,r['sample_id'])];m=inside(base/'exports',r['mesh_path'])
                require(r['status']=='ok' and file_hash(m)==r['mesh_sha256']==row['mesh_npz_sha256'],'Portable outlet file')
                require(r['source_nodes_sha256']==arrays_by_kind['primary'][r['sample_id']][1]['nodes_hash']==row['source_nodes_sha256'],'Portable same mainfield')
                with np.load(m,allow_pickle=False) as z:v=z['vertices'];f=z['faces']
                require(array_hash(v,f)==row['stored_array_hash'],'Portable stored mesh dtype and values')
                require(array_hash(v.astype(np.float64),f.astype(np.int64))==row['canonical_geometry_hash'],'Portable canonical geometry')
                require(len(f)==row['quality']['face_count'],'Portable actual face denominator');outlets+=1
    return {'primary_E32':primary,'auxiliary_E128':auxiliary,'raw_outlets':outlets,'saved_arrays_checked':True,'new_model_forward':False}


def verify_payload(root):
    root=Path(root).resolve();manifest=read(root/'FILE_MANIFEST.json')
    verify_files(root,manifest['files'],exclude=('FILE_MANIFEST.json',))
    return {'files':len(manifest['files']),'SHA256_and_bytes_all_match':True,**portable_arrays(root)}


def extract_archive(archive,target,expected_manifest_sha):
    target=new_dir(Path(target).resolve())
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();require(len(names)==len(set(names)),'Duplicate ZIP members')
        for name in names:inside(target,name)
        require(z.testzip() is None,'ZIP CRC failure');z.extractall(target)
    require(file_hash(target/'FILE_MANIFEST.json')==expected_manifest_sha,'Extracted manifest differs from payload manifest')
    return verify_payload(target)


def package(run,review,out):
    require(not torch.cuda.is_initialized(),'CPU-only package');torch.set_num_threads(1)
    run=Path(run).resolve();review=Path(review).resolve();repo=Path(__file__).resolve().parents[3]
    out=new_dir(Path(out).resolve());started=time.perf_counter()
    try:
        require((run/'INTERPRETATION_zh.md').is_file(),'Wait for root final interpretation/freeze')
        originals=files_manifest(run);write_json(out/'ORIGINAL_RUN_BEFORE.json',originals)
        p,ap,_=artifacts.artifact_contract(run);review_identity=validate_reviews(review,repo,run)
        gate,training=training_gate_audit(run,p);parent=Path(p['source_run']).resolve();data=Path(ap['evaluation_data']).resolve()
        for rel,sha in p['source_data_hashes'].items():require(file_hash(data/rel)==sha,'Data identity')
        for uid,sha in ap['raw_mesh_sha256'].items():require(file_hash(data/'raw'/f'{uid}.npz')==sha,'Raw GT identity')
        # Every evidence/source copy is tied to its pre-copy digest, its destination, and later fresh extraction.
        payload=out/'payload';payload.mkdir();copies={};sources_before={}
        trees={'run':run,'data':data,'evidence/fm_tail_GPU_gate':gate,'evidence/independent_review':review}
        for name in FAILURE_NAMES:trees['evidence/prior_failures/'+name]=repo/'output'/name
        for dst,src in trees.items():
            require(src.is_dir(),'Missing required evidence '+str(src));before=files_manifest(src)
            shutil.copytree(src,payload/dst);verify_files(src,before);verify_files(payload/dst,before)
            copies[dst]={'source':str(src),'files':before};sources_before[dst]=(src,before)
        sources=source_paths(repo);source_records=copy_bound_files(sources,payload/'source')
        parent_sources=selected_parent_files(parent);parent_records=copy_bound_files(parent_sources,payload/'evidence/source8k')
        # Keep the full earlier archive identity/receipt without duplicating its huge ZIP inside this ZIP.
        original_package=repo/'output/a_ae_fm_closed16_20260925_01_package_01'
        parent_receipts={n:original_package/n for n in ('SEALED_AFTER_RECOVERY.json','POSTSEAL_COPY_AUDIT.json','FAILED.json','RECOVERY_INVOCATION_ERRORS.json')}
        copy_bound_files(parent_receipts,payload/'evidence/source8k_full_archive_receipts')
        receipt=read(original_package/'SEALED_AFTER_RECOVERY.json')
        audit={'scope':'IDENTITY_SAVED_ARRAY_AND_ARCHIVE_AUDIT_NOT_NEW_NUMERICAL_EXPERIMENT',
            'training':training,'review_records':review_identity,'copies':copies,'source_files':source_records,
            'source8k_selected_files':parent_records,'original8k_full_archive_receipt':receipt,
            'source8k_omissions':'timing and intermediate .pt weights; complete original archive remains separately sealed',
            'formal_primary_samples':128,'auxiliary_samples_excluded_from_primary':128,'all_outlet_attempts':384,
            'new_geometry_sampling_or_exact_audit':False,'new_GPU_execution':False,'original_run_modified':False,
            'earlier_preparation_and_zero_update_smoke_failures_fully_preserved':True}
        write_json(out/'AUDIT.json',audit);write_json(payload/'AUDIT.json',audit)
        write_json(payload/'SOURCE_WHITELIST.json',{'files':source_records,'explicit_only':True,
            'parallel_system_formal_learned_longfit_balanced_modules_excluded':True})
        write_json(payload/'PORTABLE_PATHS.json',{'original_run':str(run),'run':'run','original_source8k':str(parent),'source8k':'evidence/source8k',
            'original_data':str(data),'data':'data','original_repo':str(repo),'source':'source','original_gate':str(gate),'gate':'evidence/fm_tail_GPU_gate',
            'original_review':str(review),'review':'evidence/independent_review','original_absolute_JSON_paths_preserved':True})
        (payload/'README_zh.md').write_text('FM-tail 固定64k双种子封存。run 保留全部正式/观察 checkpoint、128 Euler32主场、128隔离Euler128主场、384出口及全部失败；两个前序准备目录与0-update失败门禁完整保留。\n\n数据、原始A权重、显式源码白名单、独立审查与36更新GPU门禁证据均在包内。门禁权重不是正式结果。source8k保留比较所需完整样本/出口/指标/诊断与final模型；原8k中间模型和计时归其独立完整ZIP，SHA与封存/恢复收据另存。\n\n64k是额外56k更新加声明的常数尾学习率，不能视为纯预算消融。原A共同质量0/128，QEM相交失败2/128均保留，不能宣称类别泛化或外部基线胜出。\n\n本次归档只核身份、保存数组、既有评测及独立复算记录、全量SHA/CRC/fresh解压；没有新GPU、网格采样或384份几何再计算。原绝对路径不改，PORTABLE_PATHS提供副本位置。可从source以python -B -m experiments.a_radial.fm_tail.package_results verify --payload <解压目录>验证。\n',encoding='utf-8')
        require(not torch.cuda.is_initialized(),'Unexpected CUDA')
        verify_files(run,originals);manifest=files_manifest(payload)
        write_json(payload/'FILE_MANIFEST.json',{'schema':'FMTAIL64K-ARCHIVE-v1','files':manifest,'self_excluded':True})
        msha=file_hash(payload/'FILE_MANIFEST.json');archive=out/'FMTAIL64K_CLOSED16_FULL.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for name in sorted([*manifest,'FILE_MANIFEST.json']):z.write(payload/name,name)
        fresh=extract_archive(archive,out/'fresh_extract',msha)
        code='from pathlib import Path; import experiments.a_radial.fm_tail.package_results as p; import experiments.a_radial.fm_tail.artifacts as a; import experiments.a_radial.fm_tail.metrics as m; root=Path.cwd()/"source"; assert all(Path(x.__file__).resolve().is_relative_to(root.resolve()) for x in [p,a,m]); print("FRESH_SOURCE_IMPORT_PASS")'
        completed=subprocess.run([sys.executable,'-B','-c',code],cwd=out/'fresh_extract',
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str((out/'fresh_extract/source').resolve())),capture_output=True,text=True)
        require(completed.returncode==0,'Fresh source import: '+completed.stderr)
        # Imported modules must leave archive unchanged; compare external source copies once more.
        verify_payload(out/'fresh_extract');require(file_hash(out/'fresh_extract/FILE_MANIFEST.json')==msha,'Manifest mutated')
        for dst,(src,before) in sources_before.items():verify_files(src,before);verify_files(out/'fresh_extract'/dst,before)
        for rel,record in source_records.items():require(file_hash(sources[rel])==record['sha256']==file_hash(out/'fresh_extract/source'/rel),'Source snapshot changed')
        for rel,record in parent_records.items():require(file_hash(parent_sources[rel])==record['sha256']==file_hash(out/'fresh_extract/evidence/source8k'/rel),'Parent selected file changed')
        verify_files(run,originals)
        result={'status':'SEALED','archive':archive.name,'archive_sha256':file_hash(archive),'archive_bytes':archive.stat().st_size,
            'file_manifest_sha256':msha,'all_members_CRC_pass':True,'fresh_extraction':fresh,
            'fresh_source_import':completed.stdout.strip(),'all_source_copy_and_fresh_hashes_match':True,
            'run_files':len(originals),'original_run_unchanged':True,'all_original_run_files_preserved':True,
            'new_GPU_execution':False,'new_384_geometric_reproduction':False,'elapsed_seconds':time.perf_counter()-started}
        write_json(out/'SEALED.json',result);print(json.dumps(result),flush=True);return result
    except BaseException as exc:
        write_json(out/'FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),'partial_files_retained':True,'original_run_never_modified':True});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('package');a.add_argument('--run',required=True);a.add_argument('--review',required=True);a.add_argument('--out',required=True)
    a=s.add_parser('verify');a.add_argument('--payload',required=True);a=p.parse_args()
    if a.command=='package':package(a.run,a.review,a.out)
    else:print(json.dumps(verify_payload(a.payload)),flush=True)
if __name__=='__main__':main()
