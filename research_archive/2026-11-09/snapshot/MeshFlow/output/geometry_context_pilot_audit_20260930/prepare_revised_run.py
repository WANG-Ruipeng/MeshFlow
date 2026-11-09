from pathlib import Path
import shutil,hashlib,json
root=Path('/mnt/e/MeshFlow')
run=root/'native_t1/runs/geometry_context_revised_20260930_153000'
run.mkdir(exist_ok=False)
(run/'registration').mkdir()
old=root/'native_t1/runs/geometry_context_20260930_142000/registration/evaluation_inputs.npz'
shutil.copyfile(old,run/'registration/evaluation_inputs.npz')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
entries=[]
for label,rel,n,partial in [
 ('original_stopped_gate','native_t1/runs/geometry_context_20260930_142000/gates.json',3,{}),
 ('base_commit_code_regression','native_t1/runs/geometry_gate_code_audit_20260930_150400/results.json',9,{}),
 ('FP32_and_first_block_localization','native_t1/runs/geometry_precision_audit_20260930_152000/results.json',4,{'BF16_first_block_replays':2,'condition_embedder_calls':3})]:
 p=root/rel
 entries.append(dict(label=label,source=dict(path=str(p),sha256=sha(p)),
  counts=dict(forward_attempts=n,forward_returns=n,backward_attempts=0,backward_returns=0,optimizer_updates=0,complete_rollouts=0,OT_calls=0),
  partial_operator_calls=partial))
(run/'diagnostic_history.json').write_text(json.dumps(dict(status='RECORDED',entries=entries),indent=2)+'\n')
print(str(run))
print(sha(run/'registration/evaluation_inputs.npz'))
