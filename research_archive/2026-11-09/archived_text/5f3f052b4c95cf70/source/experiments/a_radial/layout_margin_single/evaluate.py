"""CPU-only fixed observations and all 20 raw endpoints against the same full field.

The reference is RadialSupport.mesh(logr), not the original raw asset. Independent
Monte Carlo samples are evaluation only. No quality fallback or model selection.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from .prepare import verify,require,read,UIDS
from .train import validate_state,step_sequence_hash
from ..io_utils import file_hash,array_hash,save_npz,write_json
from ..legacy.networks import DomainVelocity
from ..legacy.deform import deform
from ..legacy.chart import dets
from ..exporter import RadialSupport
from ..layout_student.train import model_sha256
from ..layout_student.deployment import predict_raw,named_quality,qualification
from ..layout_student.evaluate import metric_cache
from ..layout_student.integrity import mesh_identity
from ..layout_value.geometry import audit_mesh
from ..layout_fitdiag.diagnose import velocity_probe,endpoint_probe

UPDATES=(0,5000,20000)
ENDPOINTS=('fixed','old_teacher','margin_teacher','old_final','margin_final')
ANGLES=('min_triangle_angle_deg','triangle_min_angle_p05_deg')
REFERENCE='same frozen radial mainfield complete2050node4096face triangulated surface'


def configure():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU evaluation only')
    torch.set_num_threads(4)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def slots(p,kind):
    if kind=='diagnostic':return {(j,u) for j in p['jobs'] for u in UPDATES}
    require(kind=='final','Unknown ledger kind')
    return {(uid,name) for uid in UIDS for name in ENDPOINTS}


def checkpoint_path(run,job,update):
    require(update in UPDATES,'Unregistered observation')
    return Path(run)/'training'/job/('initial.pt' if update==0 else 'final.pt' if update==20000 else 'step_05000.pt')


def load_model(run,p,job_name,update):
    job=p['jobs'][job_name];path=checkpoint_path(run,job_name,update);sha=file_hash(path)
    state=torch.load(path,map_location='cpu',weights_only=False);c=state['config']
    validate_state(state,c)
    require(state['update']==update and state['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json'),'Wrong model observation/protocol')
    require(c['job']==c['job_id']==job_name and c['uid']==job['uid'] and c['seed']==10
        and c['arm']=='c0' and c['target_kind']==job['target_kind'] and c['target_key']==job['target_key']
        and c['active_training_uids']==[job['uid']] and c['total_updates']==20000,'Model job/target/budget changed')
    require(c['initial_model_sha256']==job['initial_model_sha256'] and c['initial_checkpoint_sha256']==job['initial_checkpoint_sha256']
        and c['config_sha256']==p['hashes']['config_sha256'] and c['training_labels_sha256']==p['hashes']['training_labels_sha256'],
        'Model data/initial/config changed')
    root=Path(__file__).resolve().parents[1]
    for name,digest in c['source_sha256'].items():require(file_hash(root/name)==digest,'Frozen trainer dependency changed')
    if update==0:require(state['model_sha256']==job['initial_model_sha256'],'Initial model not shared source')
    require(file_hash(path)==sha,'Checkpoint changed during load')
    model=DomainVelocity().cpu().float();model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
    require(sum(t.numel() for t in model.parameters())==29907,'Architecture changed')
    identity={'checkpoint_file':str(path.relative_to(run)),'checkpoint_file_sha256':sha,
        'model_sha256':state['model_sha256'],'job':job_name,'uid':job['uid'],'target_kind':job['target_kind'],
        'update':update,'seed':10,'step_sequence_sha256':state['step_sequence_sha256'],
        'source_import_sha256':file_hash(path.parent/'SOURCE_IMPORT.json'),'CPU_training_threads':1,'CPU_deployment_threads':4}
    return model,identity


def load_labels(run,p):
    path=run/p['paths']['training_labels'];require(file_hash(path)==p['hashes']['training_labels_sha256'],'Labels changed')
    with np.load(path,allow_pickle=False) as z:data={key:np.array(z[key]) for key in z.files}
    require(set(data)=={'train_uids','directions0','directions1_old','directions1_margin','proxy_points','point_features','faces'},
        'Training-only label archive changed')
    require(data['train_uids'].tolist()==list(UIDS),'Label order changed')
    return data


def summarize_slots(p,attempts,kind):
    expected=slots(p,kind)
    key=(lambda a:(a['job'],a['update'])) if kind=='diagnostic' else (lambda a:(a['uid'],a['endpoint']))
    keys=[key(a) for a in attempts]
    require(len(keys)==len(set(keys)) and set(keys)<=expected,'Duplicate or unregistered attempt')
    success=[a for a in attempts if a['status']=='success'];failed=[a for a in attempts if a['status']!='success']
    return {'planned_slots':len(expected),'recorded_slots':len(attempts),'success_count':len(success),
        'rows':success,'errors':failed,'unattempted':sorted(expected-set(keys)),
        'missing_success_slots':sorted(expected-{key(a) for a in success}),
        'all_attempts':attempts,'no_failure_filtering':True}


def aggregate(run,p,kind):
    sub='diagnostics' if kind=='diagnostic' else 'endpoints'
    attempts=[read(path) for path in sorted((run/sub).glob('*/*/RESULT.json'))]
    protocol_sha=file_hash(run/'RUN_PROTOCOL.json');source_sha=file_hash(__file__)
    require(all(a['protocol_sha256']==protocol_sha and a['evaluate_source_sha256']==source_sha for a in attempts),'Attempt source identity changed')
    value=summarize_slots(p,attempts,kind)
    value.update(protocol_sha256=protocol_sha,evaluate_source_sha256=source_sha,training_seeds=[10],
        scope='four previously seen parents; paired single-parent capability diagnosis only',
        reference_surface=REFERENCE,checkpoint_selection=False,fallback=False,cuda_initialized=torch.cuda.is_initialized())
    path=run/('TRAIN_DIAGNOSTICS.json' if kind=='diagnostic' else 'results.json')
    write_json(path,value,replace=path.exists());return value


def begin_attempt(folder,header):
    result=folder/'RESULT.json'
    if result.exists():
        old=read(result);require(all(old[k]==v for k,v in header.items()),'Existing attempt changed')
        return False
    if folder.exists():
        write_json(result,{**header,'status':'error','error':'Partial previous attempt retained; no automatic retry'})
        return False
    folder.mkdir(parents=True);write_json(folder/'START.json',header);return True


def diagnostics(run,p,selected_job=None,selected_update=None):
    data=load_labels(run,p);source_sha=file_hash(__file__);protocol_sha=file_hash(run/'RUN_PROTOCOL.json')
    require(selected_job is None or selected_job in p['jobs'],'Unknown diagnostic job')
    require(selected_update is None or selected_update in UPDATES,'Unknown diagnostic update')
    for job_name,job in p['jobs'].items():
        if selected_job is not None and job_name!=selected_job:continue
        i=job['index'];base=data['directions0'][i];target=data[job['target_key']][i]
        for update in UPDATES:
            if selected_update is not None and update!=selected_update:continue
            folder=run/'diagnostics'/job_name/str(update)
            header={'job':job_name,'uid':job['uid'],'target_kind':job['target_kind'],'update':update,
                'protocol_sha256':protocol_sha,'evaluate_source_sha256':source_sha}
            if not begin_attempt(folder,header):continue
            before=time.perf_counter()
            try:
                model,identity=load_model(run,p,job_name,update)
                velocity=velocity_probe(model,base,target,data['point_features'][i])
                u,path,alpha,seconds=deform(model,data['proxy_points'][i],base,data['faces'],True,32)
                save_npz(folder/'directions.npz',u=u,alpha=alpha)
                row={**header,'status':'success','identity':identity,'velocity':velocity,
                    'endpoint':endpoint_probe(u,base,target,data['faces'],p['endpoint_scale_squared']),
                    'directions_file':str((folder/'directions.npz').relative_to(run)),
                    'directions_file_sha256':file_hash(folder/'directions.npz'),'direction_alpha_hash':array_hash(u,alpha),
                    'target_hash':array_hash(target),'proxy_points_hash':array_hash(data['proxy_points'][i]),
                    'training_labels_sha256':p['hashes']['training_labels_sha256'],'rollout_seconds':seconds,
                    'elapsed_seconds':time.perf_counter()-before,'full_field_or_reference_read':False,
                    'teacher_forced_velocity_probe_is_offline':True,'free_rollout_receives_proxy_only':True}
            except Exception as exc:row={**header,'status':'error','error':repr(exc),'traceback':traceback.format_exc()}
            write_json(folder/'RESULT.json',row);aggregate(run,p,'diagnostic')
            print(json.dumps({k:row[k] for k in ('job','update','status')}),flush=True)
    return aggregate(run,p,'diagnostic')


def geometry(support,data,u):
    require(u.shape==(258,3) and u.dtype==np.float64 and np.isfinite(u).all()
        and np.allclose(np.linalg.norm(u,axis=1),1,rtol=0,atol=1e-9),'Invalid endpoint direction')
    radius=support.query(data['logr'],u)
    return {'u':u,'r':radius,'vertices':u*radius[:,None],'faces':data['faces'].astype(np.int64)}


def check_raw_readback(path,mesh,support,data):
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)==set(mesh) and all(z[k].dtype==v.dtype and np.array_equal(z[k],v) for k,v in mesh.items()),'Saved mesh changed')
        require(np.array_equal(z['r'],support.query(data['logr'],z['u']))
            and np.array_equal(z['vertices'],z['u']*z['r'][:,None]),'Saved radii stale')
    return mesh_identity(path,data['faces'])


def final_endpoints(run,p):
    labels=load_labels(run,p);support=RadialSupport();source_sha=file_hash(__file__);protocol_sha=file_hash(run/'RUN_PROTOCOL.json')
    for i,uid in enumerate(UIDS):
        data_path=run/'data'/f'{uid}.npz'
        with np.load(data_path,allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
        rv,rf=support.mesh(data['logr'])
        require(np.array_equal(rv,data['ref_vertices']) and np.array_equal(rf,data['ref_faces']),'Reference geometry changed')
        base=labels['directions0'][i];floor=named_quality(geometry(support,data,base)['vertices'],data['faces'])
        strict={k:floor[k]+.1 for k in ANGLES}
        for name in ENDPOINTS:
            folder=run/'endpoints'/uid/name;header={'uid':uid,'endpoint':name,'protocol_sha256':protocol_sha,'evaluate_source_sha256':source_sha}
            if not begin_attempt(folder,header):continue
            try:
                identity=None;timing=None;diagnostic_parity=None
                if name.endswith('_final'):
                    kind=name.split('_')[0];job_name=f'{kind}_{uid[:8]}'
                    model,identity=load_model(run,p,job_name,20000)
                    mesh,timing=predict_raw(model,data['logr'],support,base,data['faces'])
                    require(timing['proxy_points_hash']==array_hash(labels['proxy_points'][i]),'Deployment proxy changed')
                    diagnostic=read(run/'diagnostics'/job_name/'20000/RESULT.json')
                    require(diagnostic['status']=='success' and diagnostic['identity']==identity,'Final diagnostic missing or model changed')
                    with np.load(run/diagnostic['directions_file'],allow_pickle=False) as z:
                        require(np.array_equal(z['u'],mesh['u']) and np.array_equal(z['alpha'],mesh['alpha']),'Final/diagnostic raw rollout differs')
                    diagnostic_parity=True
                else:
                    key={'fixed':'directions0','old_teacher':'directions1_old','margin_teacher':'directions1_margin'}[name]
                    mesh=geometry(support,data,labels[key][i])
                mesh_path=folder/'mesh.npz';save_npz(mesh_path,**mesh)
                hashes=check_raw_readback(mesh_path,mesh,support,data)
                quality=named_quality(mesh['vertices'],data['faces']);audit=audit_mesh(mesh['vertices'],data['faces'],exact=True)
                seeds=[b+p['all16_sorted_uids'].index(uid) for b in p['evaluation']['seed_bases']]
                cached=metric_cache(run,uid,mesh['vertices'],data['faces'],data['ref_vertices'],data['ref_faces'],seeds,8192)
                row={**header,'status':'success','identity':identity,'mesh_file':str(mesh_path.relative_to(run)),**hashes,
                    'quality':quality,'audit':audit,'fixed_floor':{k:floor[k] for k in ANGLES},'strict_floor':strict,
                    'qualified_original':qualification(quality,audit,floor),'qualified_strict':qualification(quality,audit,strict),
                    'actual_margin_vs_fixed_deg':{k:quality[k]-floor[k] for k in ANGLES},
                    'direction_determinants_positive':bool(dets(mesh['u'],data['faces']).min()>0),
                    'metrics':[item['metrics'] for item in cached],'metric_seeds':seeds,'points_each_direction':8192,
                    'reference_surface':REFERENCE,'reference_hash':array_hash(rv,rf),
                    'reference_geometry_same_as_optimizer_field':True,'reference_MC_independent_of_optimizer_samples':True,
                    'mainfield_hash':array_hash(data['logr']),'data_file_sha256':file_hash(data_path),
                    'direction_errors':{kind:endpoint_probe(mesh['u'],base,labels['directions1_'+kind][i],data['faces'],p['endpoint_scale_squared']) for kind in ('old','margin')},
                    'deployment_timing':timing,'final_equals_corresponding_diagnostic':diagnostic_parity,
                    'fallback_to_fixed':False,'final_radius_requery_exact':True,'no_output_selection':True}
            except Exception as exc:row={**header,'status':'error','error':repr(exc),'traceback':traceback.format_exc()}
            write_json(folder/'RESULT.json',row);aggregate(run,p,'final')
            print(json.dumps({k:row[k] for k in ('uid','endpoint','status')}),flush=True)
    return aggregate(run,p,'final')


def pairing_audit(run,p):
    rows={}
    for name,job in p['jobs'].items():
        path=checkpoint_path(run,name,20000);s=torch.load(path,map_location='cpu',weights_only=False)
        validate_state(s,s['config']);require(s['protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json') and s['update']==20000,'Pairing checkpoint changed')
        require(s['config']['job']==name and s['config']['target_kind']==job['target_kind'],'Pairing job differs')
        rows[name]={'checkpoint_sha256':file_hash(path),'model_sha256':model_sha256(s['model']),
            'step_sequence_sha256':step_sequence_hash(s['trace']),'actual_updates':len(s['trace']),
            'actual_parent_exposures':s['sampler']['exposures'][0]}
    for uid in UIDS:
        require(rows['old_'+uid[:8]]['step_sequence_sha256']==rows['margin_'+uid[:8]]['step_sequence_sha256'],'Paired bridge time sequences differ')
    report={'status':'PASS','protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),'training_seed':10,
        'all_pairs_time_sequences_exactly_equal':True,'rows':rows,'cuda_initialized':torch.cuda.is_initialized()}
    target=run/'PAIRING_AUDIT.json'
    if target.exists():require(read(target)==report,'Pairing audit changed')
    else:write_json(target,report)
    return report


def evaluate(run,mode='all',job=None,update=None):
    configure();run=Path(run).resolve();p=verify(run)
    require(mode in ('all','diagnostics','final'),'Unknown evaluation mode')
    require((job is None and update is None) or mode=='diagnostics','Subset options only apply to diagnostic observations')
    if mode in ('all','diagnostics'):diagnostics(run,p,job,update)
    if mode in ('all','final'):pairing_audit(run,p);final_endpoints(run,p)
    require(not torch.cuda.is_initialized(),'Evaluation initialized CUDA')


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--run',required=True)
    a.add_argument('--mode',choices=['all','diagnostics','final'],default='all');a.add_argument('--job');a.add_argument('--update',type=int,choices=UPDATES)
    args=a.parse_args();evaluate(args.run,args.mode,args.job,args.update)


if __name__=='__main__':main()
