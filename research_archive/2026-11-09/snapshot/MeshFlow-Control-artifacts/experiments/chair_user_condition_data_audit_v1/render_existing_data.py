#!/usr/bin/env python3
"""CPU-only gallery of existing training parents and frozen C face subsets."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[key]="2"
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys, time, json, hashlib, html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

ART=Path("/mnt/e/MeshFlow-Control-artifacts")
DATA=ART/"data"
RUN=ART/"experiments/chair_user_condition_data_audit_v1"
OUT=RUN/"gallery"; OUT.mkdir(parents=True,exist_ok=True)
MAN=DATA/"train_manifest.json"
PLAN=ART/"experiments/chair_condition_recipe_factorial_v1/streams/recipe_plan.json"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
def shaded(tri,color):
 n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(n,axis=1)
 n=np.divide(n,norm[:,None],out=np.zeros_like(n),where=norm[:,None]>0)
 light=np.abs(n @ (np.array([.3,.7,.6])/np.linalg.norm([.3,.7,.6])))
 return np.clip(np.array(to_rgb(color))[None,:]*(.65+.35*light[:,None]),0,1)
def draw(ax,tri,colors,center,span,elev,azim,wire=False):
 t=np.asarray(tri)[:,:,[0,2,1]]
 if wire:
  edge=t[:,((0,1),(1,2),(2,0)),:].reshape(-1,2,3)
  obj=Line3DCollection(edge,colors="#56616b",linewidths=.55)
 else:
  obj=Poly3DCollection(t,facecolors=colors,edgecolors="#253543",linewidths=.35,alpha=1)
 ax.add_collection3d(obj)
 cc=center[[0,2,1]]
 ax.set(xlim=(cc[0]-span/2,cc[0]+span/2),ylim=(cc[1]-span/2,cc[1]+span/2),zlim=(cc[2]-span/2,cc[2]+span/2))
 ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim);ax.set_axis_off()
def main():
 start=time.perf_counter();m=read(MAN);plan=read(PLAN);uids=sorted(plan["eligible_parents"])
 assert len(uids)==26
 parents={p["uid"]:p for p in m["parents"]};tasks=[t for t in m["tasks"] if t["uid"] in uids]
 assert len(tasks)==411
 inputs={str(MAN):sha(MAN),str(PLAN):sha(PLAN)};pages=[]
 for pi,uid in enumerate(uids):
  pr=parents[uid];npz=DATA/pr["npz"];inputs[str(npz)]=sha(npz);assert inputs[str(npz)]==pr["sha256"]
  with np.load(npz,allow_pickle=False) as z:full=z["full_target"].reshape(-1,3,3).astype(np.float64)
  assert np.isfinite(full).all()
  lo=full.reshape(-1,3).min(0);hi=full.reshape(-1,3).max(0);center=(lo+hi)/2;span=float((hi-lo).max())*1.1
  areas=np.linalg.norm(np.cross(full[:,1]-full[:,0],full[:,2]-full[:,0]),axis=1)/2
  for ratio in (.2,.4):
   tt=sorted([t for t in tasks if t["uid"]==uid and float(t["ratio"])==ratio],key=lambda t:t["task_id"])
   assert tt
   name=f"{uid}__C{int(ratio*100)}.png";path=OUT/name
   # Resume only pages whose receipt matches both image and exact source identities.
   sidecar=OUT/(name+".json")
   if path.exists() and sidecar.exists():
    old=read(sidecar)
    if old["sha256"]==sha(path) and old["parent_sha256"]==inputs[str(npz)] and old["manifest_sha256"]==inputs[str(MAN)]:
     for t in tt:
      pp=DATA/t["condition_npz"];inputs[str(pp)]=sha(pp);assert inputs[str(pp)]==t["condition_sha256"]
     pages.append(old);print(json.dumps({"parent":pi+1,"ratio":ratio,"status":"REUSED_PAGE"}),flush=True);continue
    raise RuntimeError("Existing page identity differs")
   fig=plt.figure(figsize=(15.3,3.25*(len(tt)+1)),dpi=100,facecolor="white")
   for j,(e,a,w) in enumerate(((22,-60,False),(22,120,False),(22,-60,True))):
    ax=fig.add_subplot(len(tt)+1,3,j+1,projection="3d")
    draw(ax,full,shaded(full,"#9aa4ac"),center,span,e,a,w)
    ax.set_title("Actual repaired training parent"+(" / wire" if w else (" / front" if j==0 else " / back")),fontsize=10)
   listed=[]
   for i,t in enumerate(tt,1):
    tp=DATA/t["condition_npz"];inputs[str(tp)]=sha(tp);assert inputs[str(tp)]==t["condition_sha256"]
    with np.load(tp,allow_pickle=False) as z:
     ids=z["source_face_ids"].astype(int);C=z["C"].reshape(-1,3,3)
     assert np.array_equal(C,full[ids].astype(np.float32))
    selected=np.zeros(len(full),bool);selected[ids]=True
    cs=shaded(C,"#348dca");mixed=shaded(full,"#c2c8ce");mixed[selected]=shaded(full[selected],"#348dca")
    frac=float(areas[ids].sum()/areas.sum())
    tag=t["task_id"].split("_train",1)[1]
    for j,(tri,col,e,a,label) in enumerate(((C,cs,22,-60,"C front"),(C,cs,22,120,"C back"),(full,mixed,22,-60,"C in source"))):
     ax=fig.add_subplot(len(tt)+1,3,i*3+j+1,projection="3d")
     draw(ax,tri,col,center,span,e,a)
     ax.set_title(f"{tag} | {label} | K={len(ids)} | area={100*frac:.1f}%",fontsize=9)
    listed.append({"task_id":t["task_id"],"K":len(ids),"N":len(full),"area_fraction":frac,"condition_npz":str(tp),"condition_sha256":inputs[str(tp)],"face_ids":ids.tolist()})
   fig.suptitle(f"TRAIN DATA ONLY | {uid} | C{int(ratio*100)} | BLUE=C / GREY=source remainder | NO GENERATED MESH",fontsize=11,y=.999)
   fig.subplots_adjust(left=0,right=1,top=.985,bottom=.002,wspace=0,hspace=.02)
   fig.savefig(path,facecolor="white");plt.close(fig)
   page={"uid":uid,"ratio":ratio,"image":str(path),"sha256":sha(path),"parent_npz":str(npz),"parent_sha256":inputs[str(npz)],"manifest_sha256":inputs[str(MAN)],"tasks":listed,"views":["C_front","C_back","C_in_source_front"],"whole_parent_views":["front","back","wire"],"camera":{"center":center.tolist(),"span":span,"scope":"full actual parent, unchanged for all conditions of this parent"},"mesh_modified":False,"AI_VISUAL_REVIEW":"PENDING"}
   write(sidecar,page);pages.append(page)
   write(OUT/"progress.json",{"status":"RUNNING","pages":len(pages),"parents_target":26,"pages_target":52,"wall_seconds":time.perf_counter()-start})
   print(json.dumps({"parent":pi+1,"ratio":ratio,"page":name,"tasks":len(tt),"seconds":time.perf_counter()-start}),flush=True)
 for p,h in inputs.items():assert sha(Path(p))==h,p
 assert len(pages)==52 and sum(len(p["tasks"]) for p in pages)==411
 result={"status":"COMPLETE","pages":pages,"parents":26,"conditions":411,"source_before_after_same":True,"input_hashes":inputs,"wall_seconds":time.perf_counter()-start,"cpu_workers":1,"threads":2,"model_calls":0,"training":0,"new_generated_mesh":0,"AI_VISUAL_REVIEW":"PENDING","torch_imported":"torch" in sys.modules}
 assert not result["torch_imported"]
 write(OUT/"gallery_manifest.json",result)
 rows=["<!doctype html><html lang='zh'><meta charset='utf-8'><title>训练条件数据检查</title><style>body{font:16px system-ui;max-width:1200px;margin:auto;background:#f4f6f8}img{max-width:100%;background:white}section{background:white;padding:12px;margin:16px 0}button,input{font:inherit;padding:8px}a{color:#176293}</style><h1>26把训练椅子 · 411份真实条件面片</h1><p>全部来自原有训练数据：蓝色C，灰色为同一源网格其余面。不是模型生成结果。每页顶部为实际训练父网格，下方逐行是不同C。相机与尺度按整个父网格固定，不把小碎片放大成大面片。几何统计与AI逐图复核另附，生成图片本身不代表通过检查。</p><input id='q' placeholder='筛选UID或C20/C40' oninput='filter()'><p><a href='../DATA_AUDIT_REPORT.md'>最终检查报告（完成后可用）</a></p>"]
 for p in pages:
  label=p["uid"]+" C"+str(int(p["ratio"]*100))
  rows.append(f"<section data-label='{label}'><h2>{label} · {len(p['tasks'])}条件</h2><a href='{Path(p['image']).name}' target='_blank'>原尺寸查看</a><details><summary>展开该父条件图</summary><img loading='lazy' src='{Path(p['image']).name}'></details></section>")
 rows.append("<script>function filter(){let q=document.querySelector('#q').value.toLowerCase();document.querySelectorAll('section').forEach(x=>x.hidden=!x.dataset.label.toLowerCase().includes(q))}</script></html>")
 (OUT/"INDEX.html").write_text("\n".join(rows),encoding="utf-8")
 print(json.dumps({k:v for k,v in result.items() if k not in ("pages","input_hashes")}),flush=True)
if __name__=="__main__":main()
