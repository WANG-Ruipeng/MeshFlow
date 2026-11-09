from pathlib import Path
import shutil,difflib,json
from control import OUT,CODE,sha,write
dest=OUT/"code_manifest.json"
if dest.exists():raise RuntimeError("Implementation already frozen")
sources={str(p):sha(p) for p in sorted(CODE.glob("*.py"))}
for p,h in sources.items():
 compile(Path(p).read_text(),p,"exec")
 q=OUT/"source_snapshot"/Path(p).name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
shutil.copyfile(CODE/"README.md",OUT/"source_snapshot/README.md")
write(dest,dict(schema="r2_user_condition_finetune_code_manifest_v1",source_hashes=sources))
diff="".join("".join(difflib.unified_diff([],Path(p).read_text().splitlines(True),fromfile="/dev/null",tofile="r2/pilots/user_condition_finetune_v1/"+Path(p).name)) for p in sources)
(OUT/"code.diff").write_text(diff)
print(json.dumps(dict(status="FROZEN",files=len(sources),code_manifest_sha256=sha(dest))))
