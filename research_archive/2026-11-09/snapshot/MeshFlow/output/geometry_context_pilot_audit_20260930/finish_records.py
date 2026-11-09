from pathlib import Path
import json,hashlib
R=Path("/mnt/e/MeshFlow/native_t1/runs/geometry_context_20260930_142000")
p=R/"code_manifest.json";m=json.loads(p.read_text())
m["patch_validation"]=dict(status="PASS",command="git apply --check --reverse native_t1/runs/geometry_context_20260930_142000/code.diff",
                           exit_code=0,applied=False,working_files_changed=False)
p.write_text(json.dumps(m,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
notes=dict(status="RESOLVED_DELIVERY_ONLY",additional_model_forwards=0,additional_model_backwards=0,
 optimizer_updates=0,complete_rollouts=0,items=[
 dict(kind="integrity_probe",first_status="FAIL",reason="WSL Git global autocrlf differs from Windows Git; 28 protected files contained only unchanged CRLF line endings",resolution="Canonical bytes independently matched base; per-command core.autocrlf=true returned no protected changes. No working source or Git config was changed.",evidence="delivery_integrity_attempt1.json and delivery_integrity.json"),
 dict(kind="report",first_status="UnicodeDecodeError",reason="Earlier PowerShell unittest transcript is UTF-16 BOM",resolution="Reporter detects UTF-16/UTF-8 BOM. Original transcript preserved; report execution subsequently succeeded."),
 dict(kind="diff_validation",failed_checks=3,reason="Two newly added files have no final newline; initial handcrafted unified diff omitted standard no-newline markers",resolution="Preserved file contents and added proper unified-diff markers. Final git apply --check --reverse passes; no patch was applied.")])
(R/"delivery_tooling_notes.json").write_text(json.dumps(notes,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print("Delivery metadata finalized; no model calls.")
