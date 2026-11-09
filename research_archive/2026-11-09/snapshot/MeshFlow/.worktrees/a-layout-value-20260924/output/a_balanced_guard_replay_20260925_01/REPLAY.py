"""One predeclared original CPU4 deployment; inspection never changes the guard."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib,json,sys,traceback
from decimal import Decimal,localcontext
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.a_radial.layout_balanced.diagnose import load_network
from experiments.a_radial.layout_student.deployment import deploy
from experiments.a_radial.exporter import RadialSupport
from experiments.a_radial.io_utils import file_hash,array_hash
chart=importlib.import_module('experiments.a_radial.legacy.chart')
deformation=importlib.import_module('experiments.a_radial.legacy.deform')
run=ROOT/'output/a_balanced_20260925_01';out=Path(__file__).parent
uid='394302e20816d0fa07828ee23920679';job='joint_11'
torch.set_num_threads(4);torch.set_num_interop_threads(1)
p=json.loads((run/'RUN_PROTOCOL.json').read_text())
assert uid in p['internal_check_uids']
ledger_before={n:file_hash(run/n) for n in ['RUN_PROTOCOL.json','results.json','TRAIN_DIAGNOSTICS.json']}
model,identity=load_network(run,job,'240000',p)
data_path=run/'data'/f'{uid}.npz'
with np.load(data_path,allow_pickle=False) as z:
    y=z['logr'].copy();base=z['base_u'].copy();faces=z['faces'].copy()
support=RadialSupport();records=[];arrays={'faces':faces,'base_u':base};original=deformation.retract

def determinant_summary(u):
    values=chart.dets(u,faces)
    return {'minimum':float(values.min()),'minimum_face':int(values.argmin()),
            'nonpositive_count':int((values<=0).sum()),'abs_below_1e14_count':int((abs(values)<1e-14).sum())}

def decimal_summary(u):
    with localcontext() as context:
        context.prec=100
        points=[[Decimal.from_float(float(v)) for v in row] for row in u]
        values=[]
        for face in faces:
            a,b,c=[points[int(i)] for i in face]
            values.append(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))
        least=min(values);index=values.index(least)
        return {'minimum_decimal_exact_float_inputs':str(least),'minimum_float':float(least),
                'minimum_face':index,'nonpositive_count':sum(v<=0 for v in values)}

def wrapped(u,delta,f,margin=.05):
    number=len(records)+1;prefix=f'step_{number:02d}'
    record={'step_one_based':number,'input':determinant_summary(u),'margin':margin,
            'delta_l2':float(np.linalg.norm(delta)),'input_unit_error_max':float(abs(np.linalg.norm(u,axis=1)-1).max())}
    arrays[prefix+'_u']=u.copy();arrays[prefix+'_delta']=delta.copy()
    try:
        v,alpha=original(u,delta,f,margin)
        record.update(status='success',alpha=float(alpha),output=determinant_summary(v))
        arrays[prefix+'_v']=v.copy();records.append(record)
        return v,alpha
    except Exception as error:
        frame=None;tb=error.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code is original.__code__:frame=tb.tb_frame
            tb=tb.tb_next
        record.update(status='error',error_type=type(error).__name__,error=str(error))
        if frame is not None:
            locals_=frame.f_locals
            for key in ['d','c0','c1','c2','c3','b1','b2','b3','v','p1','p2','p3']:
                if key in locals_:arrays[prefix+'_'+key]=np.array(locals_[key],copy=True)
            alpha=float(locals_.get('alpha',float('nan')))
            record['alpha_after_original_loop']=alpha
            record['last_loop_index']=int(locals_.get('_',-1))
            record['input_decimal']=decimal_summary(u)
            if all(k in locals_ for k in ('c0','b1','b2','b3')):
                c0=locals_['c0'];bound=np.minimum(np.minimum(locals_['b1'],locals_['b2']),locals_['b3'])
                record['bernstein_last_iteration']={'c0_min':float(c0.min()),'bound_min':float(bound.min()),
                    'bound_minus_margin_c0_min':float((bound-margin*c0).min()),
                    'passes_original_float64_bound':bool(np.all(bound>margin*c0))}
            if 'v' in locals_:
                record['failed_candidate']=determinant_summary(locals_['v'])
                record['failed_candidate_decimal']=decimal_summary(locals_['v'])
            if 'd' in locals_:
                pre=u+alpha*locals_['d'];arrays[prefix+'_prenormalized']=pre.copy()
                record['prenormalized']=determinant_summary(pre)
                record['prenormalized_decimal']=decimal_summary(pre)
                if 'v' in locals_:
                    record['normalization_max_change']=float(abs(locals_['v']-pre).max())
        records.append(record)
        raise

outcome='success';failure=None
try:
    deformation.retract=wrapped
    result=deploy(model,y,support,base,faces)
    outcome='success_unexpected';failure={'raw_mesh_hash':array_hash(result['raw_mesh']['vertices'],result['raw_mesh']['faces'])}
except Exception as error:
    outcome='error';failure={'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}
finally:
    deformation.retract=original
np.savez(out/'GUARD_ARRAYS.npz',**arrays)
ledger_after={n:file_hash(run/n) for n in ledger_before};assert ledger_after==ledger_before
report={'scope':'ONE_PREDECLARED_CPU4_FAILURE_REPLAY_NOT_MODEL_SELECTION','uid':uid,'model':job,
    'checkpoint':identity,'torch_threads':4,'interop_threads':1,'cuda_context_initialized':torch.cuda.is_initialized(),
    'original_deploy_calls':1,'other_thread_counts_tried':[],'guard_or_weights_modified':False,
    'formal_ledger_unchanged':True,'formal_file_sha256':ledger_before,'data_file_sha256':file_hash(data_path),
    'outcome':outcome,'failure':failure,'guard_calls':len(records),'records':records,
    'expected_failure_reproduced':outcome=='error' and failure['type']=='AssertionError',
    'decimal_diagnostics':'100 decimal digits on the exact binary64 coordinate values, read only after failed original assert',
    'guard_arrays_sha256':file_hash(out/'GUARD_ARRAYS.npz'),
    'source_sha256':{n:file_hash(ROOT/'experiments/a_radial'/n) for n in ['legacy/chart.py','legacy/deform.py','layout_student/deployment.py','layout_balanced/evaluate.py']},
    'replay_source_sha256':file_hash(__file__)}
with (out/'REPLAY.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'outcome':outcome,'expected_failure_reproduced':report['expected_failure_reproduced'],
    'guard_calls':len(records),'last_guard':records[-1],'replay_sha256':file_hash(out/'REPLAY.json')},indent=2))
