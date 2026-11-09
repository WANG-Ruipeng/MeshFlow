from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json,csv,os
D=Path("/mnt/e/MeshFlow-Control-artifacts/reviews/storage_inventory_20261008")
rows=json.loads((D/"weights_windows_metadata.json").read_text(encoding="utf-8-sig"))
def enrich(r):
 p=Path("/mnt/e/"+r["path"][3:].replace("\\","/"));s=p.stat()
 assert s.st_size==r["bytes"],r
 parts=p.parts
 if ".worktrees" in parts:
  j=parts.index(".worktrees");group="/".join(parts[j:j+2])
  tail=parts[j+2:]
  detail=group+"/"+"/".join(tail[:2]) if len(tail)>1 else group+"/"+"/".join(tail)
 elif "native_t1" in parts:
  j=parts.index("native_t1");group="/".join(parts[j:j+2]);detail="/".join(parts[j:j+3])
 elif "MeshFlow-Control-artifacts" in parts:
  group="current_contextual_alignment";detail="current_contextual_alignment/"+p.parent.name
 elif "experiments" in parts:
  group="legacy_experiments";detail="/".join(parts[parts.index("experiments")+1:-1])
 else:group="other";detail=str(p.parent)
 return dict(r,linux_path=str(p),device=s.st_dev,inode=s.st_ino,links=s.st_nlink,group=group,detail=detail)
with ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(enrich,rows))
rows.sort(key=lambda r:r["path"])
(D/"weights_inventory.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf8")
with (D/"weights_inventory.csv").open("w",encoding="utf-8-sig",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups=defaultdict(list);details=defaultdict(list);inodes=defaultdict(list)
for r in rows:groups[r["group"]].append(r);details[r["detail"]].append(r);inodes[(r["device"],r["inode"])].append(r)
def summary(k,v):
 seen={}
 for r in v:seen[(r["device"],r["inode"])]=r["bytes"]
 return dict(group=k,paths=len(v),logical_bytes=sum(r["bytes"] for r in v),unique_file_ids=len(seen),unique_file_id_bytes=sum(seen.values()))
result=dict(status="INVENTORY_ONLY_NO_DELETION",weight_like_paths=len(rows),logical_bytes=sum(r["bytes"] for r in rows),unique_file_id_bytes=sum(v[0]["bytes"] for v in inodes.values()),groups=sorted([summary(k,v) for k,v in groups.items()],key=lambda x:-x["logical_bytes"]),worktree_details=sorted([summary(k,v) for k,v in details.items() if k.startswith(".worktrees")],key=lambda x:-x["logical_bytes"]),multi_path_file_ids=[dict(paths=[r["path"] for r in v],links=v[0]["links"],bytes=v[0]["bytes"]) for v in inodes.values() if len(v)>1],note="NTFS metadata identity via WSL stat; unique logical size, not physical allocated bytes; no content duplicate claim")
(D/"inventory_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
print(json.dumps(result,ensure_ascii=False,indent=2))
