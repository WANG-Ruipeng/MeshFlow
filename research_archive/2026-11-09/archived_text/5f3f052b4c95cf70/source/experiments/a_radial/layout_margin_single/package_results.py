"""External immutable archive for the completed paired CPU single-parent study."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,shutil,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from ..io_utils import array_hash,file_hash,write_json,new_dir
from ..exporter import RadialSupport
from ..layout_student.deployment import named_quality,qualification
from ..layout_student.integrity import mesh_identity
from ..layout_student.train import model_sha256
from ..system_probe.warm_start_package import inventory,verify_files,inside,copy_checked,extract_checked
from .prepare import assert_contract,UIDS,verify
from .train import validate_state,step_sequence_hash
from .evaluate import slots,summarize_slots,UPDATES,checkpoint_path
from .candidate_gate import classify,POLICY_SHA256

SOURCE_NAMES=tuple('layout_margin_single/'+name for name in (
    '__init__.py','prepare.py','train.py','smoke.py','run_all.py','evaluate.py','report.py','review.py',
    'candidate_gate.py','follow_evaluation.py','package_results.py','README.md','test_prepare.py','test_train.py',
    'test_driver.py','test_evaluate.py','test_report.py','test_review.py','test_candidate_gate.py','test_package_results.py'))+(
    'layout_fitdiag/train.py','layout_fitdiag/diagnose.py','layout_fitdiag/__init__.py',
    'layout_endpoint/train.py','layout_endpoint/rollout.py','layout_endpoint/__init__.py',
    'layout_student/train.py','layout_student/deployment.py','layout_student/integrity.py','layout_student/evaluate.py','layout_student/__init__.py',
    'layout_value/optimizer.py','layout_value/geometry.py','layout_value/__init__.py',
    'legacy/networks.py','legacy/deform.py','legacy/chart.py','legacy/common.py','legacy/geometry_task.py',
    'legacy/exact_accel.py','legacy/exact_triangles.py','legacy/audit_fast.py','legacy/__init__.py',
    'exporter.py','mesh_ops.py','io_utils.py','assets/support.npz','__init__.py',
    'system_probe/warm_start_package.py','system_probe/__init__.py')
ANGLES=('min_triangle_angle_deg','triangle_min_angle_p05_deg')


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def compare_prefix(initial,middle,final):
    require(initial['update']==0 and middle['update']==5000 and final['update']==20000,'Observation budgets changed')
    require(initial['config']==middle['config']==final['config'],'Checkpoint configuration differs')
    require(initial['trace']==[] and middle['trace']==final['trace'][:5000],'Saved checkpoint trace prefix differs')
    require(initial['model_sha256']==initial['config']['initial_model_sha256'],'Initial model differs from shared v1')


def readback(run,source_root):
    """Saved-state/array verification only: no forward, optimization, MC or exact rerun."""
    run=Path(run).resolve();source_root=Path(source_root).resolve();p=read(run/'RUN_PROTOCOL.json');assert_contract(p)
    protocol_sha=file_hash(run/'RUN_PROTOCOL.json');manifest=read(run/'DATA_MANIFEST.json')['files']
    require(len(manifest)==27,'Prepared 27-input denominator changed')
    require(file_hash(run/'DATA_MANIFEST.json')==p['hashes']['data_manifest_sha256'],'Input manifest differs')
    for rel,digest in manifest.items():require(file_hash(inside(run,rel))==digest,'Prepared input differs')
    with np.load(run/'train_labels.npz',allow_pickle=False) as z:labels={k:np.array(z[k]) for k in z.files}
    require(set(labels)=={'train_uids','directions0','directions1_old','directions1_margin','proxy_points','point_features','faces'}
        and labels['train_uids'].tolist()==list(UIDS),'Frozen train archive differs')
    model_records={};sequences={};states_read=0
    for name,job in p['jobs'].items():
        states=[];identities={}
        for update in UPDATES:
            path=checkpoint_path(run,name,update);s=torch.load(path,map_location='cpu',weights_only=False);c=s['config'];validate_state(s,c)
            require(s['protocol_sha256']==protocol_sha and c['job_id']==c['job']==name and c['uid']==job['uid']
                and c['target_kind']==job['target_kind'] and c['seed']==10 and c['active_training_uids']==[job['uid']]
                and c['total_updates']==20000 and c['config_sha256']==p['hashes']['config_sha256'],'Wrong checkpoint identity')
            require(c['initial_checkpoint_sha256']==job['initial_checkpoint_sha256'] and c['initial_model_sha256']==job['initial_model_sha256'],'Wrong common initializer')
            require(sum(t.numel() for t in s['model'].values())==29907 and all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in s['model'].values()),'Invalid model parameters')
            for rel,digest in c['source_sha256'].items():require(file_hash(inside(source_root,rel))==digest,'Frozen trainer source differs')
            states.append(s);identities[update]={'file_sha256':file_hash(path),'model_sha256':s['model_sha256']};states_read+=1
        compare_prefix(*states);final=states[-1];folder=run/'training'/name
        require(read(folder/'TRAIN_TRACE.json')==final['trace'] and read(folder/'TRAIN_TIMING.json')==final['timing_trace'],'Complete training trace differs')
        report=read(folder/'TRAIN_REPORT.json');require(report['status']=='COMPLETE' and report['checkpoint_sha256']==identities[20000]['file_sha256']
            and report['model_sha256']==identities[20000]['model_sha256'] and report['parent_exposures']=={job['uid']:80000},'Final training report differs')
        source=read(folder/'SOURCE_IMPORT.json');require(source['optimizer_inherited'] is False and source['source_rng_inherited'] is False
            and source['scheduler_inherited'] is False,'Fresh CPU initialization contract changed')
        sequences[name]=step_sequence_hash(final['trace']);model_records[name]=identities
    require(all(sequences['old_'+u[:8]]==sequences['margin_'+u[:8]] for u in UIDS),'Paired k draw sequences differ')
    pairing=read(run/'PAIRING_AUDIT.json')
    for name,identity in model_records.items():
        require(pairing['rows'][name]['checkpoint_sha256']==identity[20000]['file_sha256']
            and pairing['rows'][name]['step_sequence_sha256']==sequences[name],'Pairing audit stale')
    diagnostic=read(run/'TRAIN_DIAGNOSTICS.json');result=read(run/'results.json')
    for ledger,kind in ((diagnostic,'diagnostic'),(result,'final')):
        require(ledger['protocol_sha256']==protocol_sha,'Ledger protocol differs')
        require(ledger['evaluate_source_sha256']==file_hash(source_root/'layout_margin_single/evaluate.py'),'Evaluation source differs')
        fresh=summarize_slots(p,ledger['all_attempts'],kind)
        require(fresh['recorded_slots']==len(slots(p,kind)) and not fresh['unattempted'],'Planned attempt missing')
        require(all(ledger[k]==v for k,v in fresh.items()),'Ledger denominator/failed slot differs')
    diagnostic_arrays={}
    for row in diagnostic['all_attempts']:
        path=run/'diagnostics'/row['job']/str(row['update'])/'RESULT.json';require(read(path)==row,'Diagnostic attempt differs')
        if row['status']!='success':continue
        identity=model_records[row['job']][row['update']]
        require(row['identity']['checkpoint_file_sha256']==identity['file_sha256'] and row['identity']['model_sha256']==identity['model_sha256'],'Diagnostic used another checkpoint')
        require(file_hash(inside(run,row['directions_file']))==row['directions_file_sha256'],'Saved diagnostic file differs')
        with np.load(inside(run,row['directions_file']),allow_pickle=False) as z:u=np.array(z['u']);alpha=np.array(z['alpha'])
        require(u.shape==(258,3) and alpha.shape==(32,) and array_hash(u,alpha)==row['direction_alpha_hash'],'Saved diagnostic direction/alpha differs')
        diagnostic_arrays[(row['job'],row['update'])]=(u,alpha)
    support=RadialSupport();endpoints=0
    for row in result['all_attempts']:
        uid=row['uid'];i=list(UIDS).index(uid);require(read(run/'endpoints'/uid/row['endpoint']/'RESULT.json')==row,'Endpoint attempt differs')
        if row['status']!='success':continue
        with np.load(run/'data'/f'{uid}.npz',allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
        rv,rf=support.mesh(data['logr']);require(np.array_equal(rv,data['ref_vertices']) and np.array_equal(rf,data['ref_faces']),'Reference is not same full radial field')
        path=inside(run,row['mesh_file']);identity=mesh_identity(path,data['faces']);require(all(row[k]==v for k,v in identity.items()),'Typed mesh identity differs')
        with np.load(path,allow_pickle=False) as z:mesh={k:np.array(z[k]) for k in z.files}
        r=support.query(data['logr'],mesh['u']);v=mesh['u']*r[:,None]
        require(np.array_equal(r,mesh['r']) and np.array_equal(v,mesh['vertices']),'Accepted-radius requery differs')
        if row['endpoint'].endswith('_final'):
            name=row['endpoint'].split('_')[0]+'_'+uid[:8];u,alpha=diagnostic_arrays[(name,20000)]
            require(np.array_equal(u,mesh['u']) and np.array_equal(alpha,mesh['alpha']),'Final/diagnostic arrays differ')
        else:
            key={'fixed':'directions0','old_teacher':'directions1_old','margin_teacher':'directions1_margin'}[row['endpoint']]
            require(np.array_equal(mesh['u'],labels[key][i]),'Frozen baseline/teacher direction differs')
        q=named_quality(v,data['faces']);require(q==row['quality'],'Actual endpoint quality differs')
        base=labels['directions0'][i];fixedq=named_quality(base*support.query(data['logr'],base)[:,None],data['faces']);strict={k:fixedq[k]+.1 for k in ANGLES}
        require(row['qualified_original']==qualification(q,row['audit'],fixedq) and row['qualified_strict']==qualification(q,row['audit'],strict),'Saved quality gate differs')
        expected=[b+p['all16_sorted_uids'].index(uid) for b in (1593501,1693501)];require(row['metric_seeds']==expected,'MC seed changed')
        require(len(row['metrics'])==2,'Both MC observations must remain present')
        cacheid=array_hash(v,data['faces'],rv,rf)
        for seed,metric in zip(expected,row['metrics']):
            cache=read(run/'metric_cache'/uid/f'{cacheid}_{seed}_8192.json');require(cache['mesh_reference_hash']==cacheid and cache['metrics']==metric,'MC cache differs')
        endpoints+=1
    summary=read(run/'SUMMARY.json');require(summary['results_sha256']==file_hash(run/'results.json')
        and summary['mechanism_sha256']==file_hash(run/'TRAIN_DIAGNOSTICS.json') and summary['protocol_sha256']==protocol_sha,'Summary stale')
    gate=read(run/'EXPLORATORY_JOINT_GATE.json');computed=classify(result['all_attempts'])
    require(all(gate[k]==v for k,v in computed.items()) and gate['policy_sha256']==POLICY_SHA256 and gate['candidate_go'] is False,'Predeclared NO-GO gate differs')
    previous=read(run/'READBACK_REVIEW.json');require(previous['status']=='PASS','Prior numerical replay missing')
    for rel,digest in previous['ledger_sha256'].items():require(file_hash(inside(run,rel))==digest,'Prior numerical replay stale')
    require(previous['readback_source_sha256']==file_hash(source_root/'layout_margin_single/review.py'),'Prior numerical replay source differs')
    figs=read(run/'figures/FIGURE_MANIFEST.json');qa=read(run/'VISUAL_QA.json');require(qa['status']=='PASS' and qa['files_sha256']==figs['files'],'Figure QA mismatch')
    for rel,digest in figs['files'].items():require(file_hash(inside(run,rel))==digest,'Scientific figure changed')
    require(not torch.cuda.is_initialized(),'Archive readback initialized GPU')
    return {'status':'PASS','checkpoint_fullstates':states_read,'jobs':len(model_records),'total_training_updates':160000,
        'paired_k_sequences_equal':True,'diagnostic_slots':24,'endpoint_slots':20,'saved_endpoint_arrays_verified':endpoints,
        'diagnostic_errors':len(diagnostic['errors']),'endpoint_errors':len(result['errors']),'candidate_go':False,
        'new_model_forward':False,'new_optimization':False,'new_MC':False,'new_exact_audit':False,
        'prior_original_CPU_numerical_replay_bound':True,'GPU_initialized':False}


def plan(run,review):
    run=Path(run).resolve();review=Path(review).resolve();verify(run);code=Path(__file__).resolve().parents[1]
    gate=read(run/'TRAINING_GATE.json');smoke_file=Path(gate['smoke_path']);smoke=read(smoke_file)
    require(file_hash(smoke_file)==gate['smoke_sha256'] and smoke['status']=='PASS' and smoke['actual_optimizer_updates']==144
        and smoke['full_semantic_state_bitwise_equal'] and smoke['manual_frozen_C0_next_bitwise_equal']
        and smoke['paired_k_sequences_equal'] and smoke['cuda_context_initialized'] is False,'Actual CPU gate mismatch')
    require(gate['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json')==smoke['formal_protocol_sha256'],'Gate run identity differs')
    for rel,digest in gate['training_source_hashes'].items():require(file_hash(code/rel)==digest,'Gated training source changed')
    require(file_hash(code/'layout_margin_single/run_all.py')==gate['driver_sha256'],'Training driver changed')
    review_result=read(review/'REVIEW.json');require(review_result['status']=='PASS_SAVED_RESULT_REVIEW'
        and file_hash(review/'review_results.py')==review_result['source_sha256'],'Independent review source mismatch')
    for rel,digest in review_result['input_sha256'].items():require(file_hash(inside(run,rel))==digest,'Independent review stale')
    roots={'run':run,'evidence/independent_review':review,'evidence/CPU144_smoke':smoke_file.parent}
    semantic=run.parent/'a_reference_semantics_20260925_01'
    if semantic.is_dir():roots['evidence/reference_geometry_audit']=semantic
    manifests={rel:inventory(path) for rel,path in roots.items()};sources={name:code/name for name in SOURCE_NAMES}
    require(all(path.is_file() for path in sources.values()),'Explicit source dependency absent')
    source_hashes={name:file_hash(path) for name,path in sources.items()}
    extras={}
    for name in ('REFERENCE_SEMANTICS_CORRECTION_20260925.md','MARGIN_JOINT_CANDIDATE_20260925.md'):
        path=run.parent/name;require(path.is_file(),'External policy/correction absent');extras['evidence/'+name]=path
    require(file_hash(extras['evidence/MARGIN_JOINT_CANDIDATE_20260925.md'])==POLICY_SHA256,'Pre-final policy changed')
    for path in sorted(run.parent.glob('a_margin_single*.xml')):extras['test_evidence/'+path.name]=path
    numerical=readback(run,code)
    return {'roots':{k:str(v) for k,v in roots.items()},'original_manifests':manifests,
        'source_files':{name:{'original':str(path),'sha256':source_hashes[name]} for name,path in sources.items()},
        'extra_files':{name:{'original':str(path),'sha256':file_hash(path)} for name,path in extras.items()},
        'readback':numerical,'CPU_smoke_actual_updates':144,'independent_review_sha256':file_hash(review/'REVIEW.json'),
        'parallel_interpolation_and_new_research_excluded':True}


def portable(root):
    root=Path(root).resolve();manifest=read(root/'BUNDLE_MANIFEST.json')['files'];verify_files(root,manifest,('BUNDLE_MANIFEST.json',))
    require(Path(__file__).resolve().is_relative_to(root/'source'),'Readback source escaped fresh bundle')
    outcome=readback(root/'run',root/'source/experiments/a_radial')
    gate=read(root/'run/TRAINING_GATE.json');smoke=read(root/'evidence/CPU144_smoke/CPU_SMOKE.json')
    require(file_hash(root/'evidence/CPU144_smoke/CPU_SMOKE.json')==gate['smoke_sha256'] and smoke['actual_optimizer_updates']==144,'Portable CPU gate differs')
    require(file_hash(root/'evidence/MARGIN_JOINT_CANDIDATE_20260925.md')==POLICY_SHA256,'Portable policy changed')
    return {**outcome,'files':len(manifest),'original_absolute_paths_opened':False,'source_executed_from_fresh_extraction':True}


def package(run,review,out):
    torch.set_num_threads(4);require(not torch.cuda.is_initialized(),'CPU archive only');out=Path(out).resolve();require(not out.exists(),'Do not overwrite archive destination')
    audit=plan(run,review);out=new_dir(out);started=time.perf_counter()
    try:
        write_json(out/'AUDIT.json',audit);payload=new_dir(out/'payload')
        for rel,original in audit['roots'].items():
            shutil.copytree(original,inside(payload,rel));verify_files(inside(payload,rel),audit['original_manifests'][rel]);verify_files(original,audit['original_manifests'][rel])
        for rel,entry in audit['source_files'].items():copy_checked(entry['original'],inside(payload,'source/experiments/a_radial/'+rel),entry['sha256'])
        for rel,entry in audit['extra_files'].items():copy_checked(entry['original'],inside(payload,rel),entry['sha256'])
        write_json(payload/'AUDIT.json',audit);write_json(payload/'SOURCE_WHITELIST.json',{'files':audit['source_files']})
        write_json(payload/'PORTABLE_PATHS.json',{'prepared_run':'run','CPU_smoke':'evidence/CPU144_smoke','independent_review':'evidence/independent_review',
            'source':'source','original_absolute_paths_retained_only_as_provenance':True,
            'prepared_training_inputs_and_full_states_are_self_contained':True,'upstream_archives_bound_by_existing_receipts_not_nested':True})
        (payload/'README.md').write_text('Completed CPU paired single-parent diagnosis: 8 jobs x20000 updates, all24 fixed observations, all20 final endpoints and failures retained. Joint expansion NO-GO. One seed10 and four seen parents, no generalization or pure scalar-margin causal claim. Same frozen radial-field reference, not raw object GT. Full CPU states/RNG/optimizer/k traces, original CPU144-step gate and independent reviews are included. Fresh verification reads saved states/arrays and existing metric evidence; it runs no new forward, training, optimization, exact audit or MC. The earlier original CPU numerical replay is preserved and hash-bound in READBACK_REVIEW.json. Original JSON paths remain provenance only; prepared run/data and all model states are portable.\n')
        (payload/'FRESH_VERIFY.py').write_text("from pathlib import Path\nimport sys,os,json\nos.environ['CUDA_VISIBLE_DEVICES']=''\nroot=Path(__file__).resolve().parent\nsys.path.insert(0,str(root/'source'))\nfrom experiments.a_radial.layout_margin_single import package_results as p\np.torch.set_num_threads(4)\nprint(json.dumps(p.portable(root)),flush=True)\n")
        manifest=inventory(payload);write_json(payload/'BUNDLE_MANIFEST.json',{'files':manifest,'self_excluded':True})
        members={k:v['sha256'] for k,v in manifest.items()};members['BUNDLE_MANIFEST.json']=file_hash(payload/'BUNDLE_MANIFEST.json')
        archive=out/'MARGIN_SINGLE_CPU_FULL.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in sorted(members):z.write(inside(payload,name),name)
        fresh=extract_checked(archive,out/'fresh_extract',members);require(file_hash(fresh/'BUNDLE_MANIFEST.json')==members['BUNDLE_MANIFEST.json'],'Manifest itself differs')
        proc=subprocess.run([sys.executable,'-I','-B',str(fresh/'FRESH_VERIFY.py')],cwd=fresh,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=str(out/'fresh_numba_cache')),capture_output=True,text=True)
        write_json(out/'FRESH_CPU_READBACK.json',{'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr})
        require(proc.returncode==0,'Fresh readback failed');result=json.loads(proc.stdout.strip().splitlines()[-1]);require(result['status']=='PASS','Fresh did not pass')
        verify_files(payload,manifest,('BUNDLE_MANIFEST.json',));verify_files(fresh,manifest,('BUNDLE_MANIFEST.json',))
        for rel,original in audit['roots'].items():verify_files(original,audit['original_manifests'][rel])
        for entry in list(audit['source_files'].values())+list(audit['extra_files'].values()):require(file_hash(entry['original'])==entry['sha256'],'Source evidence changed after archive')
        final={'status':'SEALED','archive':str(archive),'bytes':archive.stat().st_size,'sha256':file_hash(archive),
            'members_including_manifest':len(members),'CRC_all_member_hashes_and_fresh_manifest_pass':True,
            'fresh_CPU_readback':result,'original_run_smoke_review_and_sources_unchanged':True,'elapsed_seconds':time.perf_counter()-started}
        write_json(out/'SEALED.json',final);print(json.dumps(final,indent=2),flush=True);return final
    except BaseException as exc:
        write_json(out/'FAILED.json',{'error':repr(exc),'partial_outputs_retained':True});raise


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--run',required=True);a.add_argument('--review',required=True);a.add_argument('--out');a.add_argument('--audit-only',action='store_true');args=a.parse_args()
    torch.set_num_threads(4)
    if args.audit_only:print(json.dumps(plan(args.run,args.review)['readback'],indent=2))
    else:
        require(args.out is not None,'--out required');package(args.run,args.review,args.out)


if __name__=='__main__':main()
