"""Read-only port audit of existing assets. No model forward or OT."""
from pathlib import Path
import hashlib,json,sys,time
root=Path('/mnt/e/MeshFlow')
sys.path.insert(0,str(root/'r2/src'))
from meshflow_control import runtime
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
import meshflow_control
from meshflow_control.r2_checkpoints import inspect_checkpoint
from meshflow_control.r2_spec import RETAINED
from meshflow_control.r2_runs import check_assets
from meshflow_control.data.recipe_factorial import RecipeFactorialStream,PAIR_FIELDS
from meshflow_control.data.io import read_json,digest
audit=root/'native_t1/maintenance/r2_port_20261009'
run=Path('/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1')
result=dict(status='RUNNING',package=str(Path(meshflow_control.__file__).resolve()),model_forwards=0,training_updates=0,new_conditions=0,new_OT_calls=0,checkpoints={})
def save():
    (audit/'real_asset_validation.json').write_text(json.dumps(result,indent=2)+'\n')
assert Path(meshflow_control.__file__).resolve().is_relative_to(root/'r2/src')
manifest=read_json(root/'r2/docs/source_manifest.json')
for row in manifest['source_files_copied_byte_exact']:
    assert digest(root/'r2'/row['path'])==row['sha256']==digest(row['source']),row['path']
result['byte_exact_sources_checked']=len(manifest['source_files_copied_byte_exact'])
for stage in ('main','confirmation'):
    folder=run if stage=='main' else run/'confirmation'
    for kind,rel in [('generator','inference/R2_MIX_H_ALL_generator_step6000.pt'),
                     ('training','checkpoints/R2_MIX_H_ALL_step6000.pt')]:
        tick=time.monotonic()
        print('CPU inspect:',stage,kind,flush=True)
        value=inspect_checkpoint(folder/rel,RETAINED[stage][kind+'_sha256'])
        value['seconds']=time.monotonic()-tick
        result['checkpoints'][stage+'_'+kind]=value
        save()
print('Read frozen R2 assets',flush=True)
data=Path('/mnt/e/MeshFlow-Control-artifacts/data')
stream_root=run/'streams'
before={str(p):digest(p) for p in [data/'train_manifest.json',stream_root/'recipe_plan.json',stream_root/'paired_inputs/R2_MIX_H_ALL.json']}
result['asset_declarations']=check_assets(data/'train_manifest.json',stream_root)
stream=RecipeFactorialStream(data,stream_root,recipe='R2',hybrid_schedule='all',require_cache=True)
expected={r['sample_index']:r for r in read_json(stream.paired_path)['records']}
rows=[]
for record in stream.records[:8]:
    sample=stream.sample(record)
    target=expected[sample['sample_index']]
    for key in PAIR_FIELDS:
        assert sample[key]==target[key],(record['sample_index'],key)
    rows.append(dict(sample_index=sample['sample_index'],ratio=record['ratio'],alpha=record['alpha'],
                     **{k:sample[k] for k in PAIR_FIELDS if k!='sample_index'}))
result['frozen_sample_checks']=rows
result['cache_counts']=dict(stream.counts,base40=stream._base40.counts)
assert stream.counts['actual_OT_attempts']==stream._base40.counts['actual_OT_attempts']==0
assert not torch.cuda.is_initialized()
assert all(digest(p)==sha for p,sha in before.items())
result.update(status='PASS',cuda_initialized=False,manifests_unchanged=True)
save()
print(json.dumps({k:v for k,v in result.items() if k not in ('checkpoints','frozen_sample_checks','asset_declarations')},indent=2),flush=True)
