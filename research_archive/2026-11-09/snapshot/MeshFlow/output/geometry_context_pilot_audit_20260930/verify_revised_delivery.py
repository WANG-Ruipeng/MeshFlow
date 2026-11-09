from pathlib import Path
import hashlib,json,subprocess,sys,numpy as np
root=Path('/mnt/e/MeshFlow');run=root/'native_t1/runs/geometry_context_revised_20260930_153000'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
reg=read(run/'registration.json');gen=read(run/'generation.json')
source_ok={name:sha(root/name)==s for name,s in reg['execution_sources'].items()}
assets={name:sha(Path(p))==s for name,p,s in [
 ('initialization',reg['initialization_path'],reg['initialization_file_sha256']),
 ('inputs',reg['inputs_path'],reg['inputs_sha256']),
 ('evaluation',str(run/'registration/evaluation_inputs.npz'),reg['evaluation_inputs_sha256'])]}
history=read(run/'diagnostic_history.json')
history_ok={e['label']:sha(Path(e['source']['path']))==e['source']['sha256'] for e in history['entries']}
noise={};raw=[]
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
with np.load(run/'registration/evaluation_inputs.npz',allow_pickle=False) as bank:
 for row in gen['rows']:
  p=Path(row['raw_path']);C=bank[f"parent{row['parent']}_patch{row['patch_id']}_constraints"].reshape(12,9)
  with np.load(p,allow_pickle=False) as f:
   output=f['output'];path=f['path'];z=f['initial_gaussian']
   if row['seed'] not in noise:noise[row['seed']]=z.copy()
   checks=dict(sha=sha(p)==row['raw_sha256'],shape=output.shape==(112,9) and path.shape==(51,112,9),
    finite=bool(np.isfinite(path).all()),endpoint=bits(path[-1],output),
    C_all51=bits(path[:,:12],np.broadcast_to(C,(51,12,9))),paired_gaussian=bits(z,noise[row['seed']]))
  raw.append(dict(output_id=row['output_id'],checks=checks))
manifest=read(run/'code_manifest.json')
manifest_ok={r['path']:sha(root/r['path'])==r['sha256'] for r in manifest['files']}
protected=subprocess.check_output(['git','-c','core.autocrlf=true','diff','--name-only','2502d5f89c54706cc4700d3cdea4a12fd6792529','--','models','configs','native_t1/data.py','native_t1/training.py','native_t1/sampling.py','native_t1/runtime.py','native_t1/baseline.py','native_t1/portable_checkpoint.py','native_t1/train_cli.py'],text=True).splitlines()
ok=all(source_ok.values()) and all(assets.values()) and all(history_ok.values()) and all(manifest_ok.values()) and all(all(r['checks'].values()) for r in raw) and not protected
result=dict(status='PASS' if ok else 'FAIL',execution_sources_unchanged=source_ok,assets_unchanged=assets,diagnostic_sources_unchanged=history_ok,code_manifest_matches=manifest_ok,protected_source_changes=protected,raw_outputs=raw,raw_count=len(raw),noise_seeds=sorted(noise),model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,geometry_rescoring=0,scope='Independent file bytes and array identity checks only; no model load or CUDA.')
(run/'delivery_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],raw_count=len(raw),source_files=len(source_ok),noise_seeds=sorted(noise),protected_changes=protected)))
assert ok
