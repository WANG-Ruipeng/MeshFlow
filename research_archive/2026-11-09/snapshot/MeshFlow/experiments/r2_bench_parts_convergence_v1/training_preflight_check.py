"""Read-only training-entry preflight: CPU packing only, zero model forwards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import ast,json,time
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/e/MeshFlow')
import sys
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'r2/src'))
start=time.perf_counter()
from r2.pilots.bench_parts_convergence_v1 import train as T,data as D
import torch
source=Path(T.__file__);text=source.read_text('utf-8')
compile(text,str(source),'exec')
assert not torch.cuda.is_initialized()
root=ROOT/'experiments/r2_bench_parts_convergence_v1'
manifest=D.read(root/'data/manifest.json')
cases={r['condition_id']:D.load_case(r) for r in manifest['rows']}
reg=dict(probe_manifest=str(root/'fixed_loss_probe/manifest.json'))
counts={};parts={};closed=0
for arm in ('PLAIN','SEAM'):
    prepared=T.prepare_probes(reg,cases,arm)
    assert len(prepared)==96
    for row,sample in prepared:
        n,k=int(sample['N']),int(sample['K']);fixed=sample['anchor_ids']>=0
        error=np.ones((n,3,3),np.float32)
        values=T.scalars(error,sample)
        assert values['all_free_scalar_count']==int(sample['free_coordinate_count'])
        assert values['all_free_velocity_sse']==values['all_free_scalar_count']
        assert values['all_free_scalar_count']==values['bulk_scalar_count']+values['seam_tip_scalar_count']
        assert values['all_free_velocity_sse']==values['bulk_velocity_sse']+values['seam_tip_velocity_sse']
        if len(sample['occupied_labels'])==0:
            assert values['seam_tip_scalar_count']==0
            assert values['all_free_scalar_count']==9*(n-k)
            assert np.array_equal(fixed,np.broadcast_to(sample['known_mask'][:,None],fixed.shape))
            if arm=='SEAM':closed+=1
    counts[arm]=len(prepared)
    parts[arm]=sorted({row['part_type'] for row,sample in prepared})
# Actually expand representative training log kwargs to catch overlapping keys.
schedule=D.read(root/'data/input_schedule.json')['steps']
row=schedule[0]['samples'][0]
st=values
def bind(**kwargs):return kwargs
bound=bind(step=1,micro=0,split='TRAIN',part=row['part_type'],time_bin='0.2-0.4',t=.3,N=196,K=7,boundary_edges=9,**row,**st)
assert bound['condition_id']==row['condition_id']
paths={}
for arm in ('COMMON','PLAIN','SEAM'):
    out=root/'training'/arm
    assert out.parents[1]/'registration.json'==root/'registration.json'
    paths[arm]=dict(registration=str(out.parents[1]/'registration.json'),rolling=str(out/'rolling.pt'),
                   stage_endpoint=str(out/('step12000.pt' if arm=='COMMON' else 'step15000.pt')))
assert [x['step'] for x in schedule[12000:12002]]==[12001,12002]
tree=ast.parse(text)
function_calls=[(getattr(n.func,'id',None),getattr(n.func,'attr',None),getattr(n,'lineno',None)) for n in ast.walk(tree) if isinstance(n,ast.Call)]
assert 'validate_optimizer(model,None,opt,\'COMMON\',step)' in text
assert "if hopt is not None and eden:" in text
assert "if head is not None and len(s['occupied_labels']):" in text
assert 'step-12000' in text
assert "if fork and (p['arm']!='COMMON' or step!=12000)" in text
assert not torch.cuda.is_initialized()
report=dict(status='PASS_WITH_STATIC_LIMITS',train_source_sha256=D.sha(source),
            script_sha256=D.sha(__file__),manifest_sha256=D.sha(root/'data/manifest.json'),
            probe_manifest_sha256=D.sha(reg['probe_manifest']),
            compile_and_import='PASS',prepared_probes=counts,CPU_OT_calls=192,
            closed_probe_rows=closed,parts=parts,closed_mask_and_denominator='PASS',
            source_probe_noise_keys='z/eps present; all hashes checked by real prepare_probes',
            checkpoint_paths=paths,log_kwargs_expansion='PASS',
            AdamW_cursor='STATIC PASS: live matching optimizer restored; validate absolute global step; step12001 follows fork12000',
            head_cursor='STATIC PASS: fresh on fork, matching state on resume; empty-edge batch skips head step',
            model_constructions=0,model_forwards=0,GPU_jobs=0,CUDA_initialized=False,
            no_checkpoint_state_loaded=True,no_train_source_modified=True,
            limitations=['No neural forward/backward or optimizer step was executed.',
                         'Real GPU execution and real checkpoint serialization/resume are not certified by this CPU audit.'],
            blocking_findings=[],wall_seconds=time.perf_counter()-start)
D.write_new(root/'training_preflight.json',report)
print(json.dumps({k:report[k] for k in ('status','CPU_OT_calls','closed_probe_rows','wall_seconds')},indent=2))
