"""Fix only final-delivery hash ordering; no scientific data/ledger changes."""
from pathlib import Path
import json,hashlib,shutil
O=Path(__file__).resolve().parents[1]
p=O/"audit/finalize_delivery.py"
old=p.read_text(encoding="utf8")
bak=O/"audit/finalize_delivery.pre_seal_fix.py"
if not bak.exists():bak.write_text(old,encoding="utf8")
new=old.replace('event=dict(event="DELIVERY_COMPLETE",time_unix=time.time(),completion_sha256=digest(OUT/"completion.json"),','event=dict(event="DELIVERY_COMPLETE",time_unix=time.time(),completion_path="completion.json",')
if new==old:raise RuntimeError("Expected event statement not found")
block='''write(checks_path,dict(status="PASS",checks=checks,model_calls=0,geometry_calls=0,weight_loads=0,
 small_text_and_image_files_hashed=len(rows),binary_hash_evidence="Prior successful execution audit and endpoint/raw/input receipts; no duplicate weight read"))
'''
assert block in new
new=new.replace(block,"")
anchor='gpu_stdout=resource.get("gpu_processes",{}).get("stdout","")\n'
assert anchor in new
new=new.replace(anchor,anchor+block)
p.write_text(new,encoding="utf8")
log=O/"controller_events.jsonl"
lines=log.read_text(encoding="utf8").splitlines()
last=json.loads(lines[-1])
assert last["event"]=="DELIVERY_COMPLETE" and "completion_sha256" in last
receipt=dict(schema="delivery_hash_ordering_fix_v1",
 scope="Only delivery-helper record; no model/training/sampling/score/input event altered",
 original_event=last,
 reason="The completion document is resealed after final controller-log inventory refresh; its pre-refresh hash must not be labeled final",
 model_calls=0,geometry_calls=0,weight_loads=0,
 old_helper_sha256=hashlib.sha256(old.encode()).hexdigest(),
 new_helper_sha256=hashlib.sha256(new.encode()).hexdigest())
last["provisional_completion_sha256"]=last.pop("completion_sha256")
last["digest_scope"]="Before final artifact inventory refresh; not final completion hash"
lines[-1]=json.dumps(last)
log.write_text("\n".join(lines)+"\n",encoding="utf8")
(O/"audit/delivery_hash_ordering_fix.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
print(json.dumps(dict(status="PASS",edited_last_delivery_event_only=True)))
