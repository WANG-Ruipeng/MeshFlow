"""Independent CPU read-back, source binding and original-deform replay; no new MC."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .prepare import verify,read,require,UIDS
from .evaluate import configure,load_model,load_labels,slots,summarize_slots,ANGLES
from ..io_utils import array_hash,file_hash,write_json
from ..legacy.deform import deform
from ..exporter import RadialSupport
from ..layout_student.deployment import named_quality,qualification
from ..layout_student.integrity import mesh_identity
from ..layout_value.geometry import audit_mesh
from ..layout_fitdiag.diagnose import velocity_probe,endpoint_probe


def without_timing(value):
    if isinstance(value,dict):return {k:without_timing(v) for k,v in value.items() if not k.endswith('seconds')}
    if isinstance(value,list):return [without_timing(v) for v in value]
    return value


def review(run):
    configure();run=Path(run).resolve();p=verify(run);manifest=read(run/'DATA_MANIFEST.json')['files']
    before={key:file_hash(run/key) for key in ('RUN_PROTOCOL.json','DATA_MANIFEST.json','TRAIN_DIAGNOSTICS.json','results.json','PAIRING_AUDIT.json')}
    parent=read(run/'source_evidence/strict_RUN_PROTOCOL.json')
    for name,digest in parent['source_sha256'].items():require(file_hash(name)==digest,'Sealed deployment/geometry dependency changed: '+name)
    data=load_labels(run,p);support=RadialSupport();diagnostic=read(run/'TRAIN_DIAGNOSTICS.json');final=read(run/'results.json')
    for ledger,kind in ((diagnostic,'diagnostic'),(final,'final')):
        require(ledger['protocol_sha256']==before['RUN_PROTOCOL.json'],'Ledger protocol changed')
        require(ledger['evaluate_source_sha256']==file_hash(Path(__file__).with_name('evaluate.py')),'Evaluation source changed')
        fresh=summarize_slots(p,ledger['all_attempts'],kind)
        require(fresh['recorded_slots']==len(slots(p,kind)) and not fresh['unattempted'],'Not all planned slots were attempted')
        for key in fresh:require(fresh[key]==ledger[key],'Aggregate ledger mismatch: '+key)
        for row in ledger['all_attempts']:
            path=(run/'diagnostics'/row['job']/str(row['update'])/'RESULT.json') if kind=='diagnostic' else (run/'endpoints'/row['uid']/row['endpoint']/'RESULT.json')
            require(read(path)==row,'Aggregate/per-attempt mismatch')
    probes=[]
    for row in diagnostic['rows']:
        job=p['jobs'][row['job']];i=job['index'];model,identity=load_model(run,p,row['job'],row['update'])
        require(identity==row['identity'],'Diagnostic checkpoint identity changed')
        require(file_hash(run/row['directions_file'])==row['directions_file_sha256'],'Saved diagnostic file changed')
        u,_,alpha,_=deform(model,data['proxy_points'][i],data['directions0'][i],data['faces'],True,32)
        with np.load(run/row['directions_file'],allow_pickle=False) as z:
            require(np.array_equal(z['u'],u) and np.array_equal(z['alpha'],alpha),'Independent original CPU replay differs')
        require(array_hash(u,alpha)==row['direction_alpha_hash'],'Diagnostic array hash changed')
        target=data[job['target_key']][i]
        require(velocity_probe(model,data['directions0'][i],target,data['point_features'][i])==row['velocity'],'Velocity probe differs')
        require(endpoint_probe(u,data['directions0'][i],target,data['faces'],p['endpoint_scale_squared'])==row['endpoint'],'Endpoint probe differs')
        probes.append({'job':row['job'],'update':row['update'],'original_CPU_replay_u_and_alpha_exact':True})
    endpoints=[]
    for row in final['rows']:
        uid=row['uid'];i=list(UIDS).index(uid)
        with np.load(run/'data'/f'{uid}.npz',allow_pickle=False) as z:obj={k:np.array(z[k]) for k in z.files}
        rv,rf=support.mesh(obj['logr']);require(np.array_equal(rv,obj['ref_vertices']) and np.array_equal(rf,obj['ref_faces']),'Reference is not same frozen full field')
        path=run/row['mesh_file'];identity=mesh_identity(path,obj['faces'])
        require(all(row[k]==v for k,v in identity.items()),'Mesh logical/storage/file identity mismatch')
        with np.load(path,allow_pickle=False) as z:mesh={k:np.array(z[k]) for k in z.files}
        name=row['endpoint']
        if name.endswith('_final'):
            job=name.split('_')[0]+'_'+uid[:8]
            recorded=read(run/'diagnostics'/job/'20000/RESULT.json')
            require(recorded['status']=='success' and recorded['identity']==row['identity'],'Final diagnostic identity differs')
            with np.load(run/recorded['directions_file'],allow_pickle=False) as z:
                require(np.array_equal(z['u'],mesh['u']) and np.array_equal(z['alpha'],mesh['alpha']),'Final direction differs from original CPU replay')
        else:
            key={'fixed':'directions0','old_teacher':'directions1_old','margin_teacher':'directions1_margin'}[name]
            require(np.array_equal(mesh['u'],data[key][i]),'Fixed/offline teacher direction differs from frozen label')
        radius=support.query(obj['logr'],mesh['u'])
        require(np.array_equal(radius,mesh['r']) and np.array_equal(mesh['vertices'],mesh['u']*radius[:,None]),'Final accepted-radius query mismatch')
        quality=named_quality(mesh['vertices'],obj['faces']);audit=audit_mesh(mesh['vertices'],obj['faces'],exact=True)
        require(quality==row['quality'] and without_timing(audit)==without_timing(row['audit']),'Quality/exact audit differs')
        fixed_r=support.query(obj['logr'],data['directions0'][i]);fixed_q=named_quality(data['directions0'][i]*fixed_r[:,None],obj['faces'])
        strict={k:fixed_q[k]+.1 for k in ANGLES}
        require(row['fixed_floor']=={k:fixed_q[k] for k in ANGLES} and row['strict_floor']==strict,'Original/strict floor differs')
        require(qualification(quality,audit,fixed_q)==row['qualified_original'] and qualification(quality,audit,strict)==row['qualified_strict'],'Qualification differs')
        for kind in ('old','margin'):
            require(endpoint_probe(mesh['u'],data['directions0'][i],data['directions1_'+kind][i],obj['faces'],p['endpoint_scale_squared'])==row['direction_errors'][kind],'Target direction error differs')
        expected_seeds=[base+p['all16_sorted_uids'].index(uid) for base in p['evaluation']['seed_bases']]
        require(row['metric_seeds']==expected_seeds and [m['seed'] for m in row['metrics']]==expected_seeds,'MC seed differs')
        cache_id=array_hash(mesh['vertices'],obj['faces'],rv,rf)
        for seed,metric in zip(expected_seeds,row['metrics']):
            cached=read(run/'metric_cache'/uid/f'{cache_id}_{seed}_8192.json')
            require(cached['mesh_reference_hash']==cache_id and cached['metrics']==metric,'Stored MC evidence differs')
        require(row['reference_hash']==array_hash(rv,rf) and row['mainfield_hash']==array_hash(obj['logr'])
            and row['data_file_sha256']==manifest[f'data/{uid}.npz'],'Field/reference identity changed')
        endpoints.append({'uid':uid,'endpoint':row['endpoint'],'actual_faces':quality['face_count'],
            'exact_valid':audit['valid'],'original_qualified':row['qualified_original']['qualified'],
            'strict_qualified':row['qualified_strict']['qualified'],'accepted_radius_exact':True})
    require(all(file_hash(run/name)==digest for name,digest in before.items()),'Existing ledger modified by read-back')
    require(all(file_hash(run/name)==digest for name,digest in manifest.items()),'Prepared input changed')
    require(not torch.cuda.is_initialized(),'Read-back initialized CUDA')
    result={'status':'PASS','meaning':'accounting, provenance and reproducibility; not scientific success',
        'readback_source_sha256':file_hash(__file__),'ledger_sha256':before,'bound_parent_deployment_dependencies':len(parent['source_sha256']),
        'prepared_files':len(manifest),'diagnostic_planned':24,'diagnostic_errors':len(diagnostic['errors']),
        'diagnostic_replays':probes,'endpoint_planned':20,'endpoint_errors':len(final['errors']),'endpoints':endpoints,
        'new_MC_evaluations':0,'old_outputs_changed':False,'cuda_initialized':False,
        'reference_is_same_frozen_radial_field_surface':True,'reference_is_original_raw_asset':False}
    write_json(run/'READBACK_REVIEW.json',result);return result


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--run',required=True);args=a.parse_args()
    value=review(args.run);print(json.dumps({'status':value['status'],'diagnostic_errors':value['diagnostic_errors'],'endpoint_errors':value['endpoint_errors']}))


if __name__=='__main__':main()
