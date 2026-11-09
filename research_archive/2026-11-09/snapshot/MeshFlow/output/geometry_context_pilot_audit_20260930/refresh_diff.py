from pathlib import Path
import json,difflib,hashlib,subprocess
ROOT=Path("/mnt/e/MeshFlow"); RUN=ROOT/"native_t1/runs/geometry_context_20260930_142000"
manifest=json.loads((RUN/"code_manifest.json").read_text())
parts=[]
for row in manifest["files"]:
    name=row["path"];p=ROOT/name
    old=subprocess.run(["git","show",manifest["baseline_commit"]+":"+name],cwd=ROOT,capture_output=True)
    oldtext=old.stdout.decode("utf-8-sig") if old.returncode==0 else ""
    newtext=p.read_bytes().decode("utf-8-sig")
    header="diff --git a/"+name+" b/"+name+"\n"
    if old.returncode:header+="new file mode 100644\n"
    delta="".join(line if line.endswith("\n") else line+"\n\\ No newline at end of file\n" for line in difflib.unified_diff(oldtext.splitlines(keepends=True),newtext.splitlines(keepends=True),
        fromfile="a/"+name if old.returncode==0 else "/dev/null",tofile="b/"+name))
    if delta: parts.append(header+delta)
    row.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
diff="".join(parts).encode("utf-8")
(RUN/"code.diff").write_bytes(diff)
manifest["diff_sha256"]=hashlib.sha256(diff).hexdigest()
manifest["new_file_line_endings"]="Preserved working-copy new-file line endings; no source changed to fix patch."
(RUN/"code_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps(dict(diff_bytes=len(diff),files=len(manifest["files"]),model_calls=0)))
