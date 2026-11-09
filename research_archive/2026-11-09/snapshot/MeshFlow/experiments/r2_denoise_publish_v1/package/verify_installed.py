"""Isolated wheel validation; invoked from a temporary cwd, PYTHONPATH=target."""
from pathlib import Path
import contextlib
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile
import numpy as np

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_denoise_publish_v1/package'
TARGET=OUT/'target'
REF=ROOT/'experiments/r2_source_winding_export_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def arrays(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

assert os.environ['PYTHONPATH']==str(TARGET)
assert not Path.cwd().resolve().is_relative_to(ROOT)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (OUT/'started.json').exists(),'No implicit replay of package validation'
write('started.json',{'time_unix':time.time(),'cwd':str(Path.cwd()),'sys_path':sys.path,'PYTHONPATH':os.environ['PYTHONPATH'],
    'budget':{'orientation_only':1,'cached_D0_chain':1,'orientation_solves':2,'GEO_calls':1,'model_forwards':0,'RBF_solves':0}})
tick=time.perf_counter()
wheel=next(OUT.glob('*.whl'))
required=['LICENSE_MeshFlow','PROVENANCE.md','source_manifest.json']
with zipfile.ZipFile(wheel) as z:
    assert z.testzip() is None
    for name in required:assert 'meshflow_control/denoiser_models/'+name in z.namelist(),name
    wheel_members=z.namelist()
    (OUT/'wheel_members.txt').write_text('\n'.join(wheel_members)+'\n')
    for name in wheel_members:
        if name.startswith('meshflow_control/') and not name.endswith('/'):
            assert hashlib.sha256(z.read(name)).hexdigest()==sha(TARGET/name),name

from meshflow_control import r2_cli,denoise,source_winding,rbf_coordination
assert Path(denoise.__file__).resolve().is_relative_to(TARGET)
row=next(r for r in read(REF/'registration.json')['records'] if r['method']=='D0_BASE')
reffolder=REF/'outputs'/row['method']/row['request_id']
inputs=[row['raw_path'],row['post_path'],row['condition_source'],row['D1_denoised_path'],row['D1_receipt_path'],str(reffolder/'oriented.npz'),str(reffolder/'restored_C.npz')]
before={p:sha(p) for p in inputs}
source=arrays(row['condition_source']);P=source['C_source_corner_permutations'];inv=np.argsort(P,axis=1)
original=source['C'][np.arange(len(P))[:,None],inv]
original_ids=source['C_source_vertex_ids'][np.arange(len(P))[:,None],inv]
capture=OUT/'capture_input.npz'
np.savez_compressed(capture,C_original=original,source_vertex_ids=original_ids,corner_permutation=P,
                    source_face_ids=source['original_source_face_ids'])
prepared=OUT/'condition_prepared.npz'
counts={'orientation_solves':0,'GEO_calls':0,'RBF_solves':0,'model_forwards':0}
def track(fn,key):
    def wrapped(*a,**kw):counts[key]+=1;return fn(*a,**kw)
    return wrapped
def no_model(*args,**kwargs):raise AssertionError('Model invocation forbidden in wheel CPU validation')
denoise.actual_proposal=no_model
denoise.geo_c.process=track(denoise.geo_c.process,'GEO_calls')
source_winding.orient_fixed_C=track(source_winding.orient_fixed_C,'orientation_solves')
rbf_coordination.coordinate_restore=track(rbf_coordination.coordinate_restore,'RBF_solves')
commands=[]
def cli(name,args):
    output=io.StringIO();start=time.perf_counter()
    with contextlib.redirect_stdout(output):
        try:code=r2_cli.main(args)
        except SystemExit as e:code=e.code
    assert code==0,(args,code)
    assert 'torch' not in sys.modules,'CPU CLI unexpectedly imported torch'
    text=output.getvalue();(OUT/(name+'.stdout.txt')).write_text(text)
    commands.append(dict(name=name,argv=['meshflow-r2',*map(str,args)],returncode=code,seconds=time.perf_counter()-start,torch_imported=False))
cli('help',['--help'])
cli('recipe',['recipe'])
cli('prepare_condition',['prepare-condition','--input',str(capture),'--input-key','C_original','--vertex-ids-key','source_vertex_ids',
    '--permutation-key','corner_permutation','--face-ids-key','source_face_ids','--out',str(prepared)])
captured=arrays(prepared)
assert bits(captured['C'],source['C']) and bits(captured['C_source_vertex_ids'],source['C_source_vertex_ids'])
cli('orient',['orient','--input',row['post_path'],'--condition',str(prepared),'--out',str(OUT/'orient')])
cli('cached_d0',['denoise','--input',row['raw_path'],'--known-faces',str(row['K']),
    '--condition',str(prepared),'--proposal',row['D1_denoised_path'],'--proposal-receipt',row['D1_receipt_path'],
    '--coordination','none','--winding','source','--out',str(OUT/'cached_d0')])
assert counts==dict(orientation_solves=2,GEO_calls=1,RBF_solves=0,model_forwards=0)
diff=[]
for action in ['orient','cached_d0']:
    for filename,reffile in [('post.npz','oriented.npz'),('post_restored_C.npz','restored_C.npz')]:
        actual=arrays(OUT/action/filename);expected=arrays(reffolder/reffile)
        for key,b in expected.items():
            newkey='C_source_corner_permutations' if key=='C_source_corner_permutation' else key
            ok=newkey in actual and bits(actual[newkey],b)
            diff.append(dict(command=action,stage=filename,reference_key=key,installed_key=newkey,status='PASS' if ok else 'FAIL'))
        assert all(r['status']=='PASS' for r in diff)
actual=arrays(OUT/'cached_d0/denoised.npz');old=arrays(row['D1_denoised_path'])
for key in ['output','C']:assert bits(actual[key],old[key])
with (OUT/'array_validation.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(diff[0]));w.writeheader();w.writerows(diff)
after={p:sha(p) for p in inputs};assert before==after

# This read-only source identity check imports the installed torch module, but
# never constructs/loads a model and does not initialize CUDA.
from meshflow_control import denoise_model
identity=denoise_model._source_identity()
import torch
assert not torch.cuda.is_initialized()
loaded={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.copy().items()
        if getattr(module,'__file__',None) and name.startswith('meshflow_control')}
assert all(Path(p).is_relative_to(TARGET.resolve()) for p in loaded.values())
for name,module in sys.modules.copy().items():
    p=getattr(module,'__file__',None)
    if not p:continue
    p=Path(p).resolve()
    assert not p.is_relative_to(ROOT) or p.is_relative_to(TARGET.resolve()),(name,str(p))
assert not any(n.startswith(('native_t1','r2.tools')) for n in sys.modules)
manifest=read(TARGET/'meshflow_control/denoiser_models/source_manifest.json')
assert identity['vendored_file_sha256']==manifest['vendored_file_sha256']
write('completion.json',dict(status='PASS',wheel=str(wheel),wheel_sha256=sha(wheel),wheel_bytes=wheel.stat().st_size,
    commands=commands,counts=counts,model_loads=0,CUDA_initialized=False,new_scores=0,new_noise=0,
    array_comparisons=len(diff),pre_GEO_output_C_equal=True,all_old_arrays_bitwise_equal=True,
    capture_canonical_C_exact=True,source_identity=identity,required_package_data=required,
    imported_package_modules=loaded,source_imports_outside_target=False,original_before=before,original_after=after,
    seconds=time.perf_counter()-tick,default_venv_unchanged=True,
    note='Installed target only. CPU CLI imported no torch; final source-identity checker imports torch without model construction or CUDA initialization. Exact extra CPU budget: 2 orientations and 1 GEO, using existing cached D0; no new research scoring.'))
print(json.dumps({'status':'PASS','counts':counts,'array_comparisons':len(diff),'wheel_sha256':sha(wheel)}))
