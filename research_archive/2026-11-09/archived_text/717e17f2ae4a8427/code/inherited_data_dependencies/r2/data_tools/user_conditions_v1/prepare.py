"""R2 user-condition data preparation. CPU only; source meshes are immutable."""
from pathlib import Path
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="2"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse, collections, hashlib, json, platform, subprocess, sys, time
import numpy as np

DEFAULT_OUT=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_r2_user_conditions_v1")
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
    if path.exists() and path.read_text(encoding="utf-8-sig")!=text:
        raise RuntimeError("Refuse differing immutable JSON: "+str(path))
    if not path.exists():
        tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(text,encoding="utf-8");tmp.replace(path)
def event(out,name,**fields):
    path=Path(out)/"ledger.jsonl"
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(dict(time_unix=time.time(),event=name,**fields),ensure_ascii=False,allow_nan=False)+"\n")
        f.flush();os.fsync(f.fileno())
def initialize(out):
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(schema="chair_r2_user_conditions_v1",stage="DATA_PREPARATION_ONLY",
       maintained_future_training_recipe="R2: Native/T1+role+Geo; FM; full-course HYBRID=.25",
       future_initial_checkpoint="main R2 full training step6000 (not loaded this stage)",
       max_training_parents=26,max_training_regions=52,max_unique_training_conditions=156,
       max_dev_parents=8,max_new_dev_conditions=8,old_dev_conditions=8,
       target_dev_cells={"seat_small":1,"seat_medium":1,"seat_full":2,"back_small":1,"back_medium":1,"back_full":2},
       region_definition={"seat":"coherent sitting upper surface, excluding underside and side walls",
                          "back":"coherent seat-facing supporting surface; curved allowed; no isolated rail called complete back"},
       extent_reference="region area 25/50/100 percent; exact source-face subsets, nested and deduplicated",
       no_forced_counts=True,manual_review_kind="AI_PROXY; geometry-informed visual review, not human ground truth",
       preservation="Existing FP32 repair_v2 M, source face and corner maps unchanged; Y exact source-ID complement",
       forbidden=["training","model_loading","model_forward","teacher","OT","Gaussian_generation","full_mesh_generation",
                  "retessellation","mesh_repair","old_data_mutation","automatic_git_commit_push"],
       cpu_dispatch={"max_concurrent_heavy_processes":2,"threads_per_process":2,
                     "stages":"one quality process + one render process; source/metadata work may overlap"},
       stop="Deliver actual accepted/rejected/uncertain regions, C/Y arrays, images and audits; no training until data review")
    write(out/"PROTOCOL.json",protocol)
    if not (out/"resources_before.json").exists():
        facts=dict(time_unix=time.time(),cpu_cores=os.cpu_count(),platform=platform.platform(),cpu_threads_per_process=2,
                   GPU_workers=0,model_calls=0)
        for key,cmd in [
          ("gpu",["nvidia-smi","--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu","--format=csv"]),
          ("gpu_processes",["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv"]),
          ("RAM",["free","-b"]),("cpu_load",["ps","-eo","pid,pcpu,pmem,comm","--sort=-pcpu"])]:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
            facts[key]=dict(exit_code=p.returncode,stdout="\n".join(p.stdout.splitlines()[:35]),stderr=p.stderr)
        write(out/"resources_before.json",facts)
        event(out,"PREPARATION_STARTED",protocol_sha256=sha(out/"PROTOCOL.json"),model_calls=0)

def geometry(parent):
    assert sha(parent["npz"])==parent["sha256"], "Source mesh hash changed"
    with np.load(parent["npz"],allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    t=a["full_target"].reshape(-1,3,3)
    assert t.dtype==np.float32 and len(t)==parent["N"] and np.isfinite(t).all()
    assert a["model_vertices"][a["source_face_vertex_ids"]].tobytes()==t.tobytes()
    return a
def components(nodes,adj):
    left=set(map(int,nodes));groups=[]
    while left:
        todo=[min(left)];left.remove(todo[0]);group=[]
        while todo:
            i=todo.pop();group.append(i)
            for j in sorted(adj.get(i,set())):
                if j in left:left.remove(j);todo.append(j)
        groups.append(sorted(group))
    return groups
def propose(parent):
    a=geometry(parent); t=a["full_target"].astype(np.float64).reshape(-1,3,3); f=a["source_faces"]
    cent=t.mean(1); cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);length=np.linalg.norm(cross,axis=1)
    area=length/2;norm=cross/np.maximum(length[:,None],1e-300)
    edges=collections.defaultdict(list)
    for i,face in enumerate(f):
        for j in range(3):edges[tuple(sorted((int(face[j]),int(face[(j+1)%3]))))].append(i)
    adj={i:set() for i in range(len(t))}
    full_adj={i:set() for i in range(len(t))}
    for faces in edges.values():
        for i in faces:full_adj[i].update(set(faces)-{i})
        if len(faces)==2:
            i,j=faces
            # Candidate patches only. These numerical proposals are NOT semantic labels.
            if abs(float(norm[i]@norm[j]))>=np.cos(np.deg2rad(25)):
                adj[i].add(j);adj[j].add(i)
    groups=components(range(len(t)),adj)
    groups.sort(key=lambda ids:(-float(area[ids].sum()),min(ids)))
    candidates=[]
    for k,ids in enumerate(groups):
        w=area[ids]; pts=t[ids].reshape(-1,3); center=(cent[ids]*w[:,None]).sum(0)/max(w.sum(),1e-300)
        # unsigned dominant axis; canonical corner permutations need not preserve winding.
        normal_abs=(np.abs(norm[ids])*w[:,None]).sum(0)/max(w.sum(),1e-300)
        candidates.append(dict(candidate_id=f"p{k:02d}",source_face_ids=ids,K=len(ids),
            area=float(w.sum()),parent_area_fraction=float(w.sum()/area.sum()),
            center=center.tolist(),bounds=[pts.min(0).tolist(),pts.max(0).tolist()],
            normal_abs=normal_abs.tolist(),proposal_only=True))
    return dict(uid=parent["uid"],split=parent["split"],npz=parent["npz"],sha256=parent["sha256"],N=parent["N"],
                area=float(area.sum()),bounds=[t.reshape(-1,3).min(0).tolist(),t.reshape(-1,3).max(0).tolist()],
                candidates=candidates,adjacency={str(k):sorted(v) for k,v in full_adj.items()},
                face_centers=cent.tolist(),face_areas=area.tolist())

def candidates(out):
    inventory=read(out/"parent_inventory.json");rows=[]
    for p in inventory["rows"]:
        path=out/"candidates"/(p["uid"]+".json")
        value=propose(p);write(path,value);rows.append(dict(uid=p["uid"],split=p["split"],path=str(path),sha256=sha(path),count=len(value["candidates"])))
    write(out/"candidate_inventory.json",dict(schema="geometric_proposals_v1",semantic_status="UNREVIEWED",rows=rows))
    event(out,"GEOMETRIC_PROPOSALS_READY",parents=len(rows),model_calls=0,semantic_labels_accepted=0)
    print(json.dumps({"status":"CANDIDATES_READY","parents":len(rows)}),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["init","candidates"]);ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()
    if args.stage=="init":initialize(args.out)
    else:candidates(args.out)
if __name__=="__main__":main()
