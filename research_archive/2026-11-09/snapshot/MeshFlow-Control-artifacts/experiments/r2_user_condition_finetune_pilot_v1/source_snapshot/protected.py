"""Read-only before/after audit of original sources and condition assets."""
from pathlib import Path
import argparse,json,time
from control import OUT,DATA,REPO,BASE,sha,write,event
def before():
 p=OUT/"protected_before.json"
 if p.exists(): print("REUSE",p);return
 paths=set(q for q in DATA.rglob("*") if q.is_file())
 paths.update(q for q in (REPO/"r2/src").rglob("*.py"))
 old=Path("/mnt/e/MeshFlow-Control-artifacts/data")
 paths.add(old/"train_manifest.json")
 paths.update((old/"parents").glob("*.npz"))
 paths.update(q for q in (Path("/mnt/e/MeshFlow-Control/src/meshflow_control/evaluation")).glob("*.py"))
 rows={str(q):dict(sha256=sha(q),bytes=q.stat().st_size) for q in sorted(paths)}
 rows[str(BASE)]=dict(sha256=json.loads((OUT/"registration.json").read_text())["base_checkpoint_sha256"],bytes=BASE.stat().st_size)
 write(p,dict(created_unix=time.time(),files=rows))
 print(json.dumps(dict(status="PROTECTED_BASELINE_SAVED",files=len(rows))))
def after():
 rows=json.loads((OUT/"protected_before.json").read_text())["files"]
 results=[dict(path=p,exists=Path(p).is_file(),expected_sha256=r["sha256"],actual_sha256=sha(p) if Path(p).is_file() else None) for p,r in rows.items()]
 bad=[r for r in results if r["actual_sha256"]!=r["expected_sha256"]]
 write(OUT/"protected_after.json",dict(status="PASS" if not bad else "FAIL",files=len(rows),mismatches=bad,results=results))
 print(json.dumps(dict(status="PASS" if not bad else "FAIL",files=len(rows),mismatches=len(bad))))
 if bad:raise RuntimeError("Protected inputs changed")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("stage",choices=["before","after"]);a=p.parse_args();globals()[a.stage]()
