"""One-time source assembly; only writes this independent branch's new files."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[4]
DEST=Path(__file__).parent

def function(path,name):
    text=(ROOT/path).read_text(encoding='utf-8')
    node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    return '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'

def save(name,text):
    p=DEST/name
    if p.exists():raise RuntimeError('Refuse replacing assembled source: '+str(p))
    p.write_text(text,encoding='utf-8')

runtime='''"""Original owner-only SEAM, independent natural-N128..799 compatibility.
No new parameters, loss, source-shared noise, head training, or state projection.
Only explicit N bounds differ from the old SEAM adapter. Closed C is Native.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'r2/src'))
import numpy as np
import torch
from r2.pilots.bench_natural_n_recipe_v1.runtime import (
    NativeModel, build_model, native_execution, edge_features, N_MIN, N_MAX)
from meshflow_control.models.native import assert_native_runtime, _face_select, _corners
from meshflow_control.models.geometry import inject_context
from meshflow_control.seam.model_adapter import _corner_select, _validate_corner_masks, corner_masked_fm_loss
from meshflow_control.training.coupling import nested_ot_map, to_noise_slots
VERSION='bench_four_routes_original_seam128_799_v1'

'''
runtime+=function('r2/src/meshflow_control/seam/slot_data.py','inference_layout').replace("    n, k = int(total_N), len(original)","    n, k = int(total_N), len(original)\n    if len(edges)>n-k:raise ValueError('Original boundary slot budget exceeded')")+'\n'
runtime+=function('r2/pilots/seam_abc_v1/slot_data_extension.py','supervised_sample').replace(
    'def supervised_sample(case, raw_noise, epsilon, t, permutation_seed):',
    "def supervised_sample(case, raw_noise, epsilon, t, permutation_seed, arm='SEAM'):\n    if arm!='SEAM':raise ValueError('Only original SEAM is registered')")+'\n'
runtime+=function('r2/pilots/bench_natural_n_recipe_v1/runtime.py','sample_for_training').replace('arm="PLAIN"','arm="SEAM"')+'\n'
runtime+=function('r2/pilots/bench_full_seat_curriculum_v1/training_state.py','verify_sample').replace('n <= 256','n <= 799').replace('N128..256','N128..799')+'\n'
runtime+=function('r2/src/meshflow_control/seam/model_adapter.py','seam_forward').replace('(y > 256)','(y > 799)').replace('N128..256','N128..799')+'\n'
save('runtime.py',runtime)

text=(ROOT/'r2/pilots/bench_natural_n_recipe_v1/training.py').read_text(encoding='utf-8')
text=text.replace('Paired Native natural-N continuations from retained exposure B global21000.',
    'Original owner-only natural-N SEAM continuation from narrow B global22500.')
text=text.replace('New data stream, inherited live generator/AdamW/RNG; no head, probe or SEAM\ntraining.',
    'New registered stream, inherited live generator/AdamW/RNG; historical head frozen.\nNo head/probe updates; source owner training, predicted C-only inference.')
text=text.replace('parents[3]','parents[4]')
text=text.replace('sample_for_training, verify_sample, VERSION','sample_for_training, verify_sample, seam_forward, VERSION')
text=text.replace("STAGE = 'NATIVE'","STAGE = 'SEAM'")
text=text.replace('bench_natural_n_recipe_training_v1','bench_four_routes_seam_training_v1')
text=text.replace('bench_natural_n_recipe_generator_v1','bench_four_routes_seam_generator_v1')
text=text.replace('(1500,BASE_STEP,22500)','(1500,BASE_STEP,24000)')
text=text.replace('Exactly 1500 new updates from exposure B21000 to22500 required','Exactly1500 new SEAM updates from narrow B22500 to24000 required')
text=text.replace('curriculum_step=3000+step','curriculum_step=4500+step')
text=text.replace('native_total_step=9000+step,seam_total_step=0','native_total_step=10500,seam_total_step=step')
text=text.replace("'bench_natural_n_'","'bench_four_routes_seam_'")
text=text.replace("model_arm='PLAIN',representation='NATIVE',head_frozen=True,head_used=False", "model_arm='SEAM',representation='SEAM',head_frozen=True,head_used=True")
text=text.replace("sample_for_training(cases[r['condition_id']],r,'PLAIN')", "sample_for_training(cases[r['condition_id']],r,'SEAM')")
text=text.replace("S.verify_sample(s,'NATIVE')", "S.verify_sample(s,'SEAM')")
text=text.replace("call=lambda:model(b['x'],b['t'],b['y'],valid_mask=b['valid'],known_mask=b['known']).float()",
    "call=lambda:seam_forward(model,b['x'],b['t'],b['y'],b['valid'],b['known'],b['fixed']).float()")
start=text.index('def checked(arm):');end=text.index('\ndef _build_restore(',start)
replacement='''def checked(arm):
    reg=T.read(OUT/'training_registration.json')
    if (reg['status'],reg['experiment'],reg['batch'],reg['microbatch'],reg['hybrid_lambda'],reg['model_config']) != ('FROZEN',EXPERIMENT,8,1,.25,MODEL_CONFIG):
        raise ValueError('Frozen SEAM contract differs')
    S.stage_spec(reg)
    if arm not in Q.ARMS or reg['arms']!=list(Q.ARMS):raise ValueError('One SEAM1500 arm only')
    if reg['runtime_version']!=RUNTIME_VERSION or reg['N_contract']!=[128,799]:raise ValueError('Natural N contract changed')
    if reg['loss_normalization']!='whole_effective_batch_free_coordinates':raise ValueError('Original FM normalization changed')
    if any(reg[k]!=v for k,v in SOURCE_IDENTITIES.items()) or resolve(reg['source_checkpoint']).resolve()!=SOURCE.resolve():raise ValueError('Only narrow B22500 source')
    if any(reg[k]!=0 for k in ('head_forward_budget','head_optimizer_updates','probe_forward_budget')):raise ValueError('No new head/probe training')
    for p,h in reg['training_code_hashes'].items():
        if digest(resolve(p))!=h:raise ValueError('Frozen training code changed: '+p)
    for key in ('manifest','eval_manifest','source_checkpoint'):
        if digest(resolve(reg[key]))!=reg[key+'_sha256']:raise ValueError('Frozen input changed: '+key)
    for p,h in reg['gate_hashes'].items():
        if digest(resolve(p))!=h:raise ValueError('Frozen gate changed: '+p)
    m=T.read(resolve(reg['manifest']))
    spec=reg['schedules'][arm]
    if digest(resolve(spec['path']))!=spec['sha256']:raise ValueError('Stream changed')
    schedule=T.read(resolve(spec['path']));Q.validate(schedule,m['train'],1500,arm)
    if m['train']!=T.read(SOURCE_ROOT/'data/manifest.json')['train']:raise ValueError('Must retain all B26 conditions unchanged')
    for r in m['train']:
        if digest(resolve(r['case_path']))!=r['case_sha256']:raise ValueError('Case changed')
    return reg,m,schedule['steps']


def _source_payload(reg):
    p=torch.load(resolve(reg['source_checkpoint']),map_location='cpu',weights_only=True,mmap=True)
    if (p['schema'],p['arm'],p['stage'],p['stage_step'],p['global_step'],p['EMA'],p['update_in_progress']) != ('bench_natural_n_recipe_training_v1','NATURAL_HIGH','NATIVE',1500,22500,False,False):
        raise ValueError('Narrow B endpoint differs')
    for key,hkey,fn in (('generator_state','source_generator_sha256',state_hash),('optimizer','source_optimizer_sha256',T.base.tree_hash),('rng','source_rng_sha256',T.base.tree_hash)):
        if fn(p[key])!=reg[hkey]:raise ValueError('Source tensor identity differs: '+key)
    if p['model_config']!=MODEL_CONFIG or p['native_total_step']!=10500 or p['seam_total_step']!=0:raise ValueError('Source history differs')
    if p['head_calls']!=0 or p['head_optimizer'] is not None:raise ValueError('Old head must remain frozen')
    S.validate_head_state(p['head_state']);_check_generator(p['generator_state'])
    return p


'''
text=text[:start]+replacement+text[end:]
save('training.py',text)
print('Created runtime.py and training.py; no models imported or called.')
