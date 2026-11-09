"""Train-only fixed-final Euler resolution diagnostic; primary deployment stays32."""
from pathlib import Path
import argparse,json,traceback
import numpy as np
import torch
from ..io_utils import new_dir,write_json,file_hash,array_hash,save_npz
from ..layout_longfit.diagnose import load_network,load_train_archive
from ..layout_fitdiag.diagnose import endpoint_probe
from ..layout_student.deployment import named_quality,qualification,predict_raw
from ..layout_value.geometry import audit_mesh
from ..exporter import RadialSupport
from ..legacy.deform import deform
from ..legacy.common import sample_surface

STEPS=(32,64,128,256)
def run(source,out):
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    source=Path(source).resolve();p=json.loads((source/'RUN_PROTOCOL.json').read_text())
    data=load_train_archive(source,p);flow,identity=load_network(source,'joint_10','240000',p)
    out=new_dir(out);support=RadialSupport();faces=data['faces'];scale=p['endpoint_scale_squared']
    source_files=['system_probe/a_integration.py','legacy/deform.py','legacy/chart.py','legacy/networks.py',
        'exporter.py','layout_student/deployment.py','layout_value/geometry.py','layout_fitdiag/diagnose.py',
        'layout_longfit/diagnose.py']
    root=Path(__file__).resolve().parents[1]
    protocol={'scope':'TRAIN12_FIXED_LONGFIT240K_SEED10_EULER_RESOLUTION_DIAGNOSIS_ONLY',
        'torch_cpu_threads':4,'torch_interop_threads':1,'source_run':str(source),'source_protocol_sha256':file_hash(source/'RUN_PROTOCOL.json'),
        'training_labels_sha256':p['hashes']['training_labels_sha256'],'checkpoint':identity,
        'uids':data['train_uids'].tolist(),'steps':list(STEPS),'primary_steps_unchanged':32,
        'model_selection':False,'generated_surface_rms_evaluated':False,'internal_check_objects_opened':0,
        'network_device':'cpu','network_dtype':'float32','geometry_dtype':'float64',
        'endpoint256_is_exact_ODE_solution':False,'quality_tolerance_deg':1e-6,
        'source_hashes':{r:file_hash(root/r) for r in source_files}}
    write_json(out/'RUN_PROTOCOL.json',protocol);rows=[];errors=[];arrays={}
    for i,uid in enumerate(protocol['uids']):
        y,base,teacher=data['logr'][i],data['directions0'][i],data['directions1'][i]
        v,f=support.mesh(y);proxy=sample_surface(v,f,256,np.random.default_rng(4501))[0]
        if not np.array_equal(proxy,data['proxy_points'][i]):raise ValueError('Original proxy256 changed')
        fixed_v=base*support.query(y,base)[:,None];floor=named_quality(fixed_v,faces)
        reference=source/'diagnostics/mechanism/joint_10/240000'/uid/'raw.npz'
        reference_hash=file_hash(reference)
        for n in STEPS:
            try:
                u,path,alpha,seconds=deform(flow,proxy,base,faces,True,n)
                radii=support.query(y,u);verts=u*radii[:,None]
                if n==32:
                    production,_=predict_raw(flow,y,support,base,faces)
                    with np.load(reference,allow_pickle=False) as stored:
                        for key,value in [('u',u),('r',radii),('vertices',verts),('alpha',alpha)]:
                            if not np.array_equal(value,production[key]) or not np.array_equal(value,stored[key]):
                                raise ValueError('Original32 replay mismatch: '+key)
                q=named_quality(verts,faces);audit=audit_mesh(verts,faces,exact=True)
                qualification_record=qualification(q,audit,floor)
                mesh=out/'meshes'/uid/f'steps{n}.npz'
                save_npz(mesh,u=u,r=radii,vertices=verts,faces=faces,alpha=alpha)
                row={'uid':uid,'steps':n,'status':'ok','mesh':str(mesh.relative_to(out)),'mesh_sha256':file_hash(mesh),
                    'proxy_sha256':array_hash(proxy),'primary32_reference_sha256':reference_hash,
                    'primary32_exact_replay':True if n==32 else None,'rollout_seconds_not_isolated':seconds,
                    'min_alpha':float(alpha.min()),'shrunk_steps':int((alpha<1).sum()),'quality':q,'audit':audit,
                    'quality_floor':floor,**qualification_record,'endpoint':endpoint_probe(u,base,teacher,faces,scale)}
                arrays[(uid,n)]=(u,verts);rows.append(row)
            except Exception as exc:errors.append({'uid':uid,'steps':n,'error':repr(exc),'traceback':traceback.format_exc()})
        if file_hash(reference)!=reference_hash:raise ValueError('Source32 output changed')
        write_json(out/'RESULTS.json',{'rows':rows,'errors':errors,'planned':48},replace=(out/'RESULTS.json').exists())
        print(json.dumps({'uid':uid,'rows':len(rows),'errors':len(errors)}),flush=True)
    for row in rows:
        uid,n=row['uid'],row['steps']
        if (uid,256) not in arrays:continue
        u,v=arrays[(uid,n)];u256,v256=arrays[(uid,256)]
        row['mse_direction_to256']=float(np.mean((u-u256)**2))
        row['mse_vertices_to256']=float(np.mean((v-v256)**2))
        e=row['endpoint']['endpoint_direction_mse']
        row['direction_to256_mse_over_teacher_mse']=float(np.mean((u-u256)**2)/e) if e>0 else None
    groups=[]
    for n in STEPS:
        rr=[r for r in rows if r['steps']==n];ep=[r['endpoint'] for r in rr]
        ratios=[e['endpoint_mse_over_zero_motion'] for e in ep if e['endpoint_mse_over_zero_motion'] is not None]
        groups.append({'steps':n,'planned':12,'success':len(rr),'qualified':sum(r['qualified'] for r in rr),
            'endpoint_global_ratio':sum(e['endpoint_direction_mse'] for e in ep)/sum(e['zero_motion_direction_mse'] for e in ep) if ep and sum(e['zero_motion_direction_mse'] for e in ep)>0 else None,
            'endpoint_macro_ratio_median':float(np.median(ratios)) if ratios else None,
            'mean_direction_mse_to256':float(np.mean([r['mse_direction_to256'] for r in rr])) if rr else None,
            'all_primary32_exact_replay':len(rr)==12 and all(r['primary32_exact_replay'] for r in rr) if n==32 else None})
    for rel,digest in protocol['source_hashes'].items():
        if file_hash(root/rel)!=digest:raise ValueError('Diagnostic dependency changed')
    if file_hash(source/'RUN_PROTOCOL.json')!=protocol['source_protocol_sha256']:raise ValueError('Source protocol changed')
    write_json(out/'RESULTS.json',{'rows':rows,'errors':errors,'planned':48},replace=True)
    write_json(out/'SUMMARY.json',{'groups':groups,'rows':len(rows),'errors':len(errors),'primary_deployment_stays32':True,
        'not_surface_RMS_or_exact_ODE_error':True,'all_train_only':True})
    print(json.dumps(groups,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.source,a.out)