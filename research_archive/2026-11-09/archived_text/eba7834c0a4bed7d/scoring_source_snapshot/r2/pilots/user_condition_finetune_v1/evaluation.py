"""R2 user-condition pilot: frozen inputs, bounded Euler50, uniform CPU scoring.
No model is imported by prepare/score/render. Only sample is a GPU entrypoint.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, importlib, importlib.util, json, os, shutil, sys, time, traceback, types
from pathlib import Path
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="2"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np

VERSION="r2_user_condition_finetune_evaluation_v1"
NAMESPACE="R2_USER_CONDITION_FINETUNE_PILOT_V1"
LABELS=("R2_FROZEN","R2_CONT_OLD","R2_CONT_USER")
ROOT_NAME="r2_user_condition_finetune_pilot_v1"
ART=Path("/mnt/e/MeshFlow-Control-artifacts")
DATA=ART/"experiments/chair_r2_user_conditions_v1"
R2=Path(__file__).resolve().parents[2]
LEGACY=Path("/mnt/e/MeshFlow-Control/src/meshflow_control/evaluation")
L=float(np.sqrt(3.)*2/.3762)
META=("request_id","parent_id","condition_id","panel","candidate","region_type","tier","N","K","ratio")

def resolve(p):
    raw=str(p)
    if os.name!="nt" and len(raw)>2 and raw[1]==":":
        return Path("/mnt/"+raw[0].lower()+"/"+raw[3:].replace("\\","/"))
    return Path(raw)

def root(p):
    p=resolve(p).resolve()
    if p.name!=ROOT_NAME or p.parent.name!="experiments":raise ValueError("Wrong experiment root")
    return p

def digest(p):
    h=hashlib.sha256()
    with resolve(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def array_sha(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode("ascii")+str(a.shape).encode("ascii")+a.tobytes()).hexdigest()

def read(p):return json.loads(resolve(p).read_text(encoding="utf-8-sig"))
def arrays(p):
    with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def write(p,value,freeze=True):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if p.exists() and freeze:
        if read(p)!=value:raise ValueError("Refuse change frozen JSON: "+str(p))
        return
    temp=p.with_suffix(p.suffix+".tmp");temp.write_text(payload,encoding="utf8");temp.replace(p)

def save_npz(p,a):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        old=arrays(p)
        if set(old)!=set(a) or any(not byte_equal(old[k],a[k]) for k in a):raise ValueError("Frozen NPZ differs")
        return
    temp=p.with_suffix(p.suffix+".tmp")
    with temp.open("wb") as f:np.savez_compressed(f,**a)
    temp.replace(p)

def write_csv(p,rows):
    p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def byte_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def source_module(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,resolve(path))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def cpu_math():
    # Isolated alias prevents swapping r2 model/sampling package dependencies.
    alias="_r2_user_pilot_original_evaluation"
    if alias not in sys.modules:
        pkg=types.ModuleType(alias);pkg.__path__=[str(LEGACY)];sys.modules[alias]=pkg
    return {name:importlib.import_module(alias+"."+name) for name in
            ("control","geometry","shape","surface","quality","seams")}

def scorer_sources():
    names=("__init__.py","control.py","geometry.py","shape.py","surface.py","quality.py","seams.py","pair_audit.py","intersections.py","hashing.py")
    result={str(LEGACY/n):digest(LEGACY/n) for n in names}
    historical=Path("/mnt/e/MeshFlow-Control/tools/score_recipe_factorial_conflict.py")
    result[str(historical)]=digest(historical)
    return result

def definition():
    m=cpu_math()
    return {"version":VERSION,"source_metric_version":m["control"].VERSION,
       "original_definition":m["control"].frozen_definition(),"L":L,"scale_source":"Fixed canonical training cube diagonal sqrt(3)*2/.3762",
       "C_queries":2048,"queries_frozen_before_first_new_rollout":True,"denominator":"Full registered C query weight; CF/FF free-face denominator N-K",
       "aggregation":"NEWDEV and OLDREG separate; mean two noises within condition, then equal parents; type/tier descriptive",
       "S":"NA_NO_REGISTERED_REQUIRED_INTERFACE","E":"NA_NO_EXPOSURE_TASK","GT":"NA_NOT_UNIQUE_GENERATIVE_TARGET",
       "topology":"Original exact uint32-coordinate diagnostic, signed-zero bit patterns distinct; no tolerance welding; not original vertex-ID certification",
       "no_POST":True,"no_GT_ranking":True,"raw_geometry_scores_recomputed_uniformly":84,
       "Q_F_rule":"2048 area samples, original quality.seed(Q_F, condition_id, candidate, 2048); same seed rule across models",
       "software_versions":{name:importlib.import_module("importlib.metadata").version(name) for name in ("numpy","scipy","trimesh","matplotlib")}}

def read_input(q):
    if digest(q["input_path"])!=q["input_sha256"]:raise ValueError("Frozen inference input changed")
    b=arrays(q["input_path"]);n,k=q["N"],q["K"]
    if not {"C","z","N","K"}<=set(b) or set(b)-{"C","z","N","K","known_mask","valid_mask"}:raise ValueError("Unauthorized inference fields")
    if (int(b["N"]),int(b["K"]))!=(n,k) or not 128<=n<=256 or not 0<k<n:raise ValueError("N/K mismatch")
    if b["C"].shape!=(k,3,3) or b["z"].shape!=(1,n,9) or b["C"].dtype!=np.float32 or b["z"].dtype!=np.float32:raise ValueError("Inference dtype/shape mismatch")
    if not np.isfinite(b["C"]).all() or not np.isfinite(b["z"]).all() or np.any(b["z"][:,:k]):raise ValueError("Invalid Gaussian/condition")
    for field,key in (("C_sha256","C"),("Gaussian_sha256","z")):
        if q[field]!=array_sha(b[key]):raise ValueError(field+" mismatch")
    return b

def verify_output(q,r,checkpoint_sha=None):
    if checkpoint_sha and r["checkpoint_sha256"]!=checkpoint_sha:raise ValueError("Wrong checkpoint binding")
    if r.get("status") not in ("COMPLETE","VERIFIED_CACHE"):raise ValueError("Incomplete receipt")
    for field in ("raw","ply","result"):
        if digest(r[field+"_path"])!=r[field+"_sha256"]:raise ValueError("Output artifact hash changed")
    b=read_input(q);n,k=q["N"],q["K"];a=arrays(r["raw_path"])
    raw=np.asarray(a.get("raw",a.get("output"))).reshape(n,3,3)
    traj=np.asarray(a.get("trajectory",a.get("path"))).reshape(51,n,3,3)
    C=np.asarray(a["C"]).reshape(k,3,3);z=np.asarray(a.get("z",a.get("initial_gaussian"))).reshape(1,n,9)
    mesh=source_module("_r2_user_pilot_mesh_io",R2/"src/meshflow_control/mesh_io.py")
    ply,_=mesh.read_ply(resolve(r["ply_path"]))
    facts={"C_output_bitwise":byte_equal(raw[:k],b["C"]),"C_all51_bitwise":all(byte_equal(t[:k],b["C"]) for t in traj),
       "actual_Gaussian_bitwise":byte_equal(z,b["z"]),"NPZ_C_bitwise":byte_equal(C,b["C"]),
       "PLY_C_bitwise":byte_equal(ply[:k],b["C"]),"PLY_full_bitwise":byte_equal(ply,raw),
       "trajectory_final_is_RAW":byte_equal(traj[-1],raw),
       "initial_free_is_Gaussian":byte_equal(traj[0,k:].reshape(1,n-k,9),b["z"][:,k:]),
       "N_preserved":len(raw)==n,"finite":bool(np.isfinite(traj).all()),
       "FP32":raw.dtype==traj.dtype==z.dtype==np.float32}
    runtime=read(r["result_path"]);runtime=runtime.get("sampling",runtime)
    if runtime.get("nfe")!=50 or runtime.get("readout_mode")!="none" or any(s not in str(runtime.get("dtype","")).upper() for s in ("BF16","FP32")):raise ValueError("Missing Euler50 BF16/FP32 runtime proof")
    if not all(facts.values()):raise ValueError("Output contract failed: "+str(facts))
    return {"raw":raw,"trajectory":traj,"C":C,"z":z},facts

def query_bank(C,cid):
    m=cpu_math();ctl=m["control"];q,face,w,bary,seed=ctl.sample_C(C,cid,2048)
    normal,good,_=ctl.normals(C,L);edges,_=ctl.open_edges(C)
    border=np.sqrt(m["seams"].nearest_segments(q,edges)[0]) if len(edges) else np.full(2048,np.inf)
    return {"C_points":q,"C_point_faces":face,"C_weights":w,"C_barycentric":bary,"C_point_normals":normal[face],
            "C_point_normal_valid":good[face],"C_open_edges":edges,"C_border_distance":border},seed

def load_registration(out):
    out=root(out);r=read(out/"evaluation/registration.json")
    if r["version"]!=VERSION or r["adapter_sha256"]!=digest(__file__) or r["scorer_sources"]!=scorer_sources():raise ValueError("Frozen evaluation source identity changed")
    if r["definition"]!=definition() or len(r["requests"])!=28 or len(r["queries"])!=14:raise ValueError("Evaluation registration differs")
    if r["render_helper_sha256"]!=digest(R2/"data_tools/user_conditions_v1/render.py") or r["sampling_sha256"]!=digest(R2/"src/meshflow_control/sampling.py"):raise ValueError("Sampling/render helper changed")
    for q in r["requests"]:read_input(q)
    return r

def prepare(out,data_root=DATA):
    out=root(out);data_root=resolve(data_root)
    dest=out/"evaluation/registration.json"
    if dest.exists():return load_registration(out)
    if any((out/"evaluation/outputs").glob("*/*/calls.jsonl")):raise ValueError("Cannot prepare after first rollout")
    files={name:data_root/name for name in ("active_conditions.json","dataset_handoff_active.json","old_regression_selection.json")}
    documents={name:read(path) for name,path in files.items()}
    active=documents["active_conditions.json"];dev=[r for r in active["rows"] if r["split"]=="dev"]
    if len(dev)!=6 or active["train_condition_count"]!=117:raise ValueError("Wrong active pool")
    handoff=documents["dataset_handoff_active.json"];frozen=handoff["checkpoint_handoff"]["frozen_generator"]
    if digest(frozen["path"])!=frozen["sha256"]:raise ValueError("Frozen main R2 generator hash mismatch")
    old=documents["old_regression_selection.json"];requests=[];new_draws=0
    for row in sorted(dev,key=lambda r:r["condition_id"]):
        if digest(row["npz"])!=row["sha256"]:raise ValueError("DEV source changed")
        c=arrays(row["npz"])["C"].reshape(row["K"],3,3)
        for candidate in (0,1):
            rid="NEWDEV__"+row["condition_id"]+"__n"+str(candidate)
            seed=int.from_bytes(hashlib.sha256((NAMESPACE+"|EVAL_NEWDEV|"+row["condition_id"]+"|"+str(candidate)).encode()).digest()[:4],"little")
            p=out/"evaluation/inputs"/(rid+".npz")
            if p.exists():
                b=arrays(p)
                if not byte_equal(b["C"],c) or int(b["N"])!=row["N"] or int(b["K"])!=row["K"]:raise ValueError("Existing input differs")
            else:
                z=np.zeros((1,row["N"],9),np.float32);z[:,row["K"]:]=np.random.RandomState(seed).randn(row["N"]-row["K"],3,3).astype(np.float32).reshape(1,-1,9)
                b={"C":c,"z":z,"N":np.array(row["N"]),"K":np.array(row["K"])};save_npz(p,b);new_draws+=1
            requests.append({"request_id":rid,"parent_id":row["uid"],"condition_id":row["condition_id"],"panel":"NEWDEV",
                 "candidate":candidate,"region_type":row["region_type"],"tier":row["tier"],"N":row["N"],"K":row["K"],"ratio":row["K"]/row["N"],
                 "noise_seed":seed,"input_path":str(p),"input_sha256":digest(p),"C_sha256":array_sha(c),"Gaussian_sha256":array_sha(b["z"]),
                 "condition_source":row["npz"],"condition_source_sha256":row["sha256"],"noise_provenance":"ONCE_REGISTERED_MT19937_NEWDEV; three labels share actual array"})
    for row in old["rows"]:
        original=row["original_request"];b=arrays(row["input_path"])
        if digest(row["input_path"])!=row["input_sha256"]:raise ValueError("Old input hash mismatch")
        requests.append({"request_id":"OLDREG__"+row["request_id"],"parent_id":row["uid"],"condition_id":row["condition_id"],
            "panel":"OLDREG","candidate":row["candidate"],"region_type":"NA_OLD_GEOMETRIC","tier":"C"+str(int(row["ratio"]*100)),
            "N":int(b["N"]),"K":int(b["K"]),"ratio":row["ratio"],"noise_seed":original.get("noise_seed"),
            "input_path":row["input_path"],"input_sha256":row["input_sha256"],"C_sha256":array_sha(b["C"]),"Gaussian_sha256":array_sha(b["z"]),
            "historical_receipt":row["existing_R2_RAW"]["original_completed_row"],"noise_provenance":"READ_ONLY_ORIGINAL_ACTUAL_C_AND_GAUSSIAN"})
    if len(requests)!=28 or len({q["request_id"] for q in requests})!=28:raise ValueError("Request count")
    queries=[]
    for cid in sorted({q["condition_id"] for q in requests}):
        qs=[q for q in requests if q["condition_id"]==cid];q=qs[0]
        if len(qs)!=2 or len({x["C_sha256"] for x in qs})!=1:raise ValueError("Unpaired condition")
        path=out/"evaluation/queries"/(cid+".npz")
        C=read_input(q)["C"]
        if path.exists():a=arrays(path);seed=int.from_bytes(hashlib.sha256(("CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_V1|20261007|C_AREA|"+cid+"|2048").encode()).digest()[:4],"little")
        else:a,seed=query_bank(C,cid);save_npz(path,a)
        queries.append({"condition_id":cid,"parent_id":q["parent_id"],"panel":q["panel"],"path":str(path),"sha256":digest(path),
            "array_sha256":{k:array_sha(v) for k,v in a.items()},"identity":{"condition_id":cid,"C_seed":seed,"C_sha256":q["C_sha256"],"L0":L},
            "query_count":2048,"weight_denominator":float(a["C_weights"].sum())})
    source_hashes=scorer_sources()
    for path,h in source_hashes.items():
        target=out/"source_external_snapshot"/Path(path).name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and digest(target)!=h:raise ValueError("Source snapshot conflict")
        if not target.exists():shutil.copyfile(path,target)
    r={"version":VERSION,"status":"FROZEN","adapter_sha256":digest(__file__),"source_inputs":{str(p):digest(p) for p in files.values()},
       "scorer_sources":source_hashes,"render_helper_sha256":digest(R2/"data_tools/user_conditions_v1/render.py"),"sampling_sha256":digest(R2/"src/meshflow_control/sampling.py"),"definition":definition(),"labels":list(LABELS),"frozen_checkpoint":frozen,"requests":requests,"queries":queries,
       "budget":{"logical_outputs":84,"new_rollouts":68,"sampling_forwards":3400,"old_reuses":16,"per_label_new":{"R2_FROZEN":12,"R2_CONT_OLD":28,"R2_CONT_USER":28}},
       "new_Gaussian_arrays_materialized_this_call":new_draws,"query_count_total":14*2048,"model_calls":0,"no_new_source_mesh":True}
    # Actual existing RAW contracts audited before registration permits any GPU sampling.
    reuse=[]
    for q in requests:
        if q["panel"]!="OLDREG":continue
        historic=q["historical_receipt"];_,facts=verify_output(q,historic,frozen["sha256"])
        record={k:v for k,v in q.items() if k!="historical_receipt"}
        record.update(label="R2_FROZEN",status="COMPLETE",checkpoint_sha256=frozen["sha256"],generation_status="VERIFIED_HISTORICAL_RAW_REUSE",new_rollouts=0,contracts=facts)
        record.update({k:historic[k] for k in ("raw_path","raw_sha256","ply_path","ply_sha256","result_path","result_sha256")})
        write(out/"evaluation/outputs/R2_FROZEN"/q["request_id"]/"receipt.json",record);reuse.append(record)
    if len(reuse)!=16:raise ValueError("Expected exactly16 reuse proofs")
    write(out/"evaluation/reuse_receipt.json",{"status":"COMPLETE","rows":reuse,"reused":16,"new_rollouts":0,"model_calls":0})
    write(dest,r);bind_checkpoint(out,"R2_FROZEN",frozen["path"])
    return r

def bind_checkpoint(out,label,path):
    out=root(out)
    if label not in LABELS:raise ValueError("Unregistered label")
    r=load_registration(out);path=resolve(path);h=digest(path)
    if label=="R2_FROZEN" and h!=r["frozen_checkpoint"]["sha256"]:raise ValueError("Not main frozen R2")
    if label!="R2_FROZEN" and path.resolve()!=(out/"inference"/(label+"_generator_step7000.pt")).resolve():raise ValueError("Unexpected finetuned generator path")
    value={"label":label,"path":str(path),"sha256":h,"evaluation_registration_sha256":digest(out/"evaluation/registration.json")}
    write(out/"evaluation/checkpoints"/(label+".json"),value);return value


class CellBudget:
    """Durable reserve-before-call ledger; an attempted cell is never replayed."""
    def __init__(self,path,context):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.context=context;self.counts={}
        with self.path.open("x",encoding="utf8") as f:
            f.write(json.dumps({"event":"CELL_REGISTERED","context":context})+"\n");f.flush();os.fsync(f.fileno())
    def event(self,event,**kwargs):
        with self.path.open("a",encoding="utf8") as f:
            f.write(json.dumps({"time_unix":time.time(),"event":event,**kwargs},allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
    def reserve(self,kind,context=None):
        count=self.counts.get(kind,0)+1
        if kind not in ("rollout_attempt","sampling_forward") or count>{"rollout_attempt":1,"sampling_forward":50}[kind]:raise RuntimeError("Cell budget exceeded")
        self.event("RESERVE",kind=kind,ordinal=count,context=context or self.context);self.counts[kind]=count
    def call(self,kind,fn,context=None):
        self.reserve(kind,context);t=time.perf_counter()
        try:value=fn()
        except BaseException as exc:self.event("ERROR",kind=kind,error=repr(exc));raise
        self.event("RETURN",kind=kind,seconds=time.perf_counter()-t);return value

def sample_job(out,label,attempt="initial",controller_owned=False):
    if not controller_owned:raise ValueError("Only explicit resource-aware root dispatcher may own GPU")
    if attempt not in ("initial","serial_retry","oom_serial_retry"):raise ValueError("Attempt contract")
    out=root(out);r=load_registration(out)
    if label not in LABELS:raise ValueError("Unknown label")
    binding=read(out/"evaluation/checkpoints"/(label+".json"))
    if binding["evaluation_registration_sha256"]!=digest(out/"evaluation/registration.json") or digest(binding["path"])!=binding["sha256"]:raise ValueError("Model binding changed")
    folder=out/"evaluation/sampling"/label;folder.mkdir(parents=True,exist_ok=True)
    dispatch="initial" if attempt=="initial" else "serial_retry"
    if dispatch=="serial_retry" and not (folder/"initial_failure.json").exists():raise ValueError("No initial failure for serial continuation")
    with (folder/(dispatch+"_dispatch.json")).open("x",encoding="utf8") as f:json.dump({"attempt":attempt,"binding":binding,"pid":os.getpid(),"started":time.time()},f)
    done=[];blocked=[];pending=[];t=time.perf_counter()
    try:
        for q in r["requests"]:
            dest=out/"evaluation/outputs"/label/q["request_id"]
            if (dest/"receipt.json").exists():
                receipt=read(dest/"receipt.json");verify_output(q,receipt,binding["sha256"]);done.append(receipt)
            elif (dest/"calls.jsonl").exists():blocked.append(q["request_id"])
            else:
                if label=="R2_FROZEN" and q["panel"]=="OLDREG":raise ValueError("No automatic extra rollout replacing cache")
                pending.append(q)
        if pending:
            sys.path.insert(0,str(R2/"src"))
            from meshflow_control import runtime
            import torch
            torch.set_num_threads(2);runtime.configure_stable_runtime()
            from meshflow_control.sampling import sample
            if label=="R2_FROZEN":
                from meshflow_control.training.recipe_factorial import load_generator
                model,identity=load_generator(resolve(binding["path"]),device="cuda")
            else:
                trainer=source_module("_r2_user_pilot_trainer",Path(__file__).with_name("trainer.py"))
                model,identity=trainer.load_generator(resolve(binding["path"]),device="cuda",expected_sha256=binding["sha256"])
                if identity["arm"]!=label or identity["global_step"]!=7000 or identity["finetune_step"]!=1000:raise ValueError("Wrong finetuned endpoint")
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
            write(folder/(dispatch+"_generator_identity.json"),identity)
            for q in pending:
                b=read_input(q);dest=out/"evaluation/outputs"/label/q["request_id"]
                context={"label":label,"request_id":q["request_id"],"attempt":attempt,"phase":"pilot","input_sha256":q["input_sha256"],"checkpoint_sha256":binding["sha256"]}
                budget=CellBudget(dest/"calls.jsonl",context)
                _,_,audit=sample(model,b["C"],q["N"],seed=q.get("noise_seed"),z=b["z"],budget=budget,out=dest,context=context)
                receipt={k:v for k,v in q.items() if k!="historical_receipt"}
                receipt.update(label=label,status="COMPLETE",checkpoint_sha256=binding["sha256"],generation_status="NEW_EULER50",new_rollouts=1,attempt=attempt)
                for name,filename in (("raw","raw.npz"),("ply","raw.ply"),("result","run.json")):
                    receipt[name+"_path"]=str(dest/filename);receipt[name+"_sha256"]=digest(dest/filename)
                _,facts=verify_output(q,receipt,binding["sha256"]);receipt["contracts"]=facts
                write(dest/"receipt.json",receipt);done.append(receipt)
                write(folder/"progress.json",{"status":"RUNNING","label":label,"attempt":attempt,"completed":len(done),"registered":28,
                    "blocked":blocked,"seconds":time.perf_counter()-t,"cuda_peak_allocated_bytes":audit["cuda_peak_allocated_bytes"],
                    "cuda_peak_reserved_bytes":audit["cuda_peak_reserved_bytes"],"peak_scope":audit["peak_scope"]},freeze=False)
        result={"status":"COMPLETE" if len(done)==28 else "PARTIAL_BLOCKED","label":label,"rows":done,"blocked":blocked,"registered":28,"seconds":time.perf_counter()-t}
        write(folder/"completed.json",result,freeze=False);return result
    except BaseException as exc:
        failure={"status":"OOM" if "out of memory" in str(exc).lower() else "FAILED","label":label,"attempt":attempt,
                 "completed":len(done),"blocked":blocked,"error":repr(exc),"traceback":traceback.format_exc(),"seconds":time.perf_counter()-t}
        write(folder/(dispatch+"_failure.json"),failure);raise

def collect_outputs(out,require_complete=True):
    out=root(out);reg=load_registration(out);rows=[];missing=[]
    for label in LABELS:
        for q in reg["requests"]:
            path=out/"evaluation/outputs"/label/q["request_id"]/"receipt.json"
            if not path.exists():missing.append({"label":label,"request_id":q["request_id"]});continue
            r=read(path)
            if r["request_id"]!=q["request_id"] or r["label"]!=label or r["input_sha256"]!=q["input_sha256"]:raise ValueError("Receipt input/label identity mismatch")
            verify_output(q,r);rows.append(r)
    result={"status":"COMPLETE" if not missing else "PENDING","registered":84,"completed":len(rows),"missing":missing,"rows":rows}
    write(out/"evaluation/raw_index.json",result,freeze=False)
    write_csv(out/"evaluation/raw_index.csv",[{k:r.get(k) for k in (*META,"label","generation_status","new_rollouts","input_path","input_sha256","raw_path","raw_sha256","checkpoint_sha256")} for r in rows])
    if missing and require_complete:raise ValueError("All84 RAW required before scoring/render: "+str(len(missing))+" missing")
    return rows

def load_queries(query):
    if digest(query["path"])!=query["sha256"]:raise ValueError("C query bank changed")
    a=arrays(query["path"])
    if {k:array_sha(v) for k,v in a.items()}!=query["array_sha256"] or len(a["C_points"])!=2048:raise ValueError("Query identity mismatch")
    return {"arrays":a,"identity":query["identity"]}

def score_one(out,q,receipt,query,registration_sha):
    out=root(out);dest=out/"evaluation/scores"/receipt["label"]/(q["request_id"]+".json")
    identity={"version":VERSION,"registration_sha256":registration_sha,"label":receipt["label"],"request_id":q["request_id"],
        "raw_sha256":receipt["raw_sha256"],"input_sha256":q["input_sha256"],"query_sha256":query["sha256"],"scorer_sources":scorer_sources()}
    if dest.exists():
        old=read(dest)
        if old["identity"]!=identity:raise ValueError("Scoring identity changed")
        for path,h in old.get("files",{}).items():
            if digest(path)!=h:raise ValueError("Score vectors changed")
        return old
    t=time.perf_counter();row={k:q[k] for k in META};row.update(label=receipt["label"],identity=identity,status="BLOCKED",metrics={},files={})
    try:
        raw,facts=verify_output(q,receipt);b=read_input(q);ref=load_queries(query);C=b["C"];F=raw["raw"][q["K"]:]
        mod=cpu_math();m,cf,ff=mod["geometry"].scan_geometry(C,F,L)
        trusted=mod["control"].trusted_cf(C,F,L,cf_scan=cf)
        im,iv=mod["control"].conflict_metrics(F,{"C":C,"L0":L},ref,trusted)
        if abs(im["ConflictCov_C"]-im["ConflictCov_C_cross"]-im["ConflictCov_C_dup"]+im["ConflictCov_C_cross_and_dup"])>1e-12:raise ValueError("I union identity")
        m.update(im);m.update(mod["geometry"].topology(raw["raw"],q["K"]))
        illegal={"proper_transverse_penetration","coplanar_positive_area_overlap"}
        cf_free={r["pair"][1] for r in cf["contacts"] if r["classification"] in illegal}
        m["CF_illegal_pair_count"]=sum(r["classification"] in illegal for r in cf["contacts"])
        m["CF_illegal_affected_free_fraction"]=len(cf_free)/len(F)
        cq=mod["shape"].shape_quality(C,L);whole=mod["shape"].shape_quality(raw["raw"],L)
        m.update(C_total_surface_area=cq["total_area"],C_near_degenerate_count=cq["near_degenerate_area_count"],
            C_exact_zero_area_count=cq["exact_zero_area_count"],whole_total_surface_area=whole["total_area"],
            whole_near_degenerate_count=whole["near_degenerate_area_count"],whole_minimum_angle_lt5_fraction=whole["min_angle_lt5_fraction"])
        qp=mod["surface"].sample_mesh_surface(F,2048,np.random.RandomState(mod["quality"].seed("Q_F",q["condition_id"],q["candidate"],2048)))
        if qp is None:raise ValueError("No finite positive free surface")
        folder=out/"evaluation/score_details"/receipt["label"]/q["request_id"]
        pairpath=folder/"pairs.json";vectorpath=folder/"vectors.npz"
        write(pairpath,{"CF":cf,"FF":ff});save_npz(vectorpath,dict(iv,Q_F2048=qp))
        row.update(status="PASS",metrics=m,contracts=facts,files={str(pairpath):digest(pairpath),str(vectorpath):digest(vectorpath)},
             vectors_path=str(vectorpath),pair_audit_path=str(pairpath),metric_status={"ConflictCov_C":im["ConflictCov_C_status"],
             "SeamCov":"NA_NO_REGISTERED_REQUIRED_INTERFACE","ExposedRetain":"NA_NO_EXPOSURE_TASK","GT_reconstruction":"NA_NOT_UNIQUE_GENERATIVE_TARGET",
             "watertight_certification":"NOT_IMPLEMENTED","human_usability":"NOT_RUN"},C_query_denominator=float(ref["arrays"]["C_weights"].sum()),
             C_query_count=2048,free_face_denominator=len(F),full_CF_scans=1,full_FF_scans=1,I_reuses_CF=True)
    except Exception as exc:row.update(reason=str(exc),traceback=traceback.format_exc())
    row["seconds"]=time.perf_counter()-t;write(dest,row);return row

def score_all(out):
    out=root(out);reg=load_registration(out);raws=collect_outputs(out);rb={(r["label"],r["request_id"]):r for r in raws};queries={q["condition_id"]:q for q in reg["queries"]}
    rows=[];started=time.perf_counter()
    for label in LABELS:
        for q in reg["requests"]:
            row=score_one(out,q,rb[label,q["request_id"]],queries[q["condition_id"]],digest(out/"evaluation/registration.json"));rows.append(row)
            write(out/"evaluation/scoring_progress.json",{"completed":len(rows),"registered":84,"seconds":time.perf_counter()-started},freeze=False)
            print(json.dumps({"stage":"score","label":label,"request_id":q["request_id"],"status":row["status"],"completed":len(rows)}),flush=True)
    write(out/"evaluation/scores_all.json",{"status":"COMPLETE" if all(r["status"]=="PASS" for r in rows) else "PARTIAL_BLOCKED","rows":rows,"registered":84,"seconds":time.perf_counter()-started})
    flat=[{**{k:r.get(k) for k in (*META,"label","status")},**r["metrics"]} for r in rows];write_csv(out/"evaluation/per_output.csv",flat)
    numeric=sorted({k for r in rows for k,v in r["metrics"].items() if isinstance(v,(int,float)) and not isinstance(v,bool)})
    parents=[];paired_noise=[];paired_conditions=[]
    by={(r["label"],r["request_id"]):r for r in rows}
    for panel in ("NEWDEV","OLDREG"):
        conditions=sorted({q["condition_id"] for q in reg["requests"] if q["panel"]==panel})
        for cid in conditions:
            qs=[q for q in reg["requests"] if q["condition_id"]==cid]
            for label in LABELS:
                rr=[by[label,q["request_id"]] for q in qs]
                values={k:float(np.mean([r["metrics"][k] for r in rr])) if all(r["status"]=="PASS" and isinstance(r["metrics"].get(k),(int,float)) and np.isfinite(r["metrics"][k]) for r in rr) else None for k in numeric}
                parents.append({"panel":panel,"condition_id":cid,"parent_id":qs[0]["parent_id"],"label":label,"registered_noises":2,
                     "region_type":qs[0]["region_type"],"tier":qs[0]["tier"],**values})
    condition_map={(r["label"],r["condition_id"]):r for r in parents}
    for first,second in (("R2_CONT_USER","R2_CONT_OLD"),("R2_CONT_USER","R2_FROZEN"),("R2_CONT_OLD","R2_FROZEN")):
        for q in reg["requests"]:
            a,b=by[first,q["request_id"]],by[second,q["request_id"]]
            paired_noise.append({**{k:q[k] for k in META},"comparison":first+" minus "+second,
                **{k:a["metrics"][k]-b["metrics"][k] if isinstance(a["metrics"].get(k),(int,float)) and isinstance(b["metrics"].get(k),(int,float)) else None for k in numeric}})
        for cid in sorted({q["condition_id"] for q in reg["requests"]}):
            a,b=condition_map[first,cid],condition_map[second,cid]
            paired_conditions.append({**{k:a[k] for k in ("panel","condition_id","parent_id","region_type","tier")},"comparison":first+" minus "+second,
                **{k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in numeric}})
    write_csv(out/"evaluation/per_condition.csv",parents);write_csv(out/"evaluation/paired_noise.csv",paired_noise);write_csv(out/"evaluation/paired_conditions.csv",paired_conditions)
    summary=[]
    for panel in ("NEWDEV","OLDREG"):
        for label in LABELS:
            pp=[r for r in parents if r["panel"]==panel and r["label"]==label];pids=sorted({r["parent_id"] for r in pp})
            record={"panel":panel,"label":label,"registered_conditions":len(pp),"registered_parents":len(pids),"registered_outputs":2*len(pp)}
            for k in numeric:
                values=[np.mean([r[k] for r in pp if r["parent_id"]==pid]) if all(r[k] is not None for r in pp if r["parent_id"]==pid) else None for pid in pids]
                record[k]=float(np.mean(values)) if all(v is not None for v in values) else None
            summary.append(record)
    write_csv(out/"evaluation/summary_equal_parent.csv",summary)
    # With two noises there is one symmetric free-surface pair per condition/label.
    diversity=[]
    for p in parents:
        qs=[q for q in reg["requests"] if q["condition_id"]==p["condition_id"]];a,b=[by[p["label"],q["request_id"]] for q in qs]
        row={k:p[k] for k in ("label","panel","condition_id","parent_id")}
        if a["status"]==b["status"]=="PASS":
            av,bv=arrays(a["vectors_path"])["Q_F2048"],arrays(b["vectors_path"])["Q_F2048"]
            ar,_=verify_output(qs[0],rb[p["label"],qs[0]["request_id"]]);br,_=verify_output(qs[1],rb[p["label"],qs[1]["request_id"]])
            surface=cpu_math()["surface"];d1=surface.surface_distance_squared(av,br["raw"][qs[1]["K"]:]);d2=surface.surface_distance_squared(bv,ar["raw"][qs[0]["K"]:])
            row.update(status="PASS",D_free=float((d1.mean()+d2.mean())/(2*L**2)),noise_pairs=1)
        else:row.update(status="BLOCKED",D_free=None,noise_pairs=1)
        diversity.append(row)
    write_csv(out/"evaluation/diversity.csv",diversity)
    return {"registered":84,"scored":sum(r["status"]=="PASS" for r in rows),"seconds":time.perf_counter()-started}


def render_all(out):
    out=root(out);reg=load_registration(out);raws=collect_outputs(out);rb={(r["label"],r["request_id"]):r for r in raws}
    helper=source_module("_r2_user_pilot_render",R2/"data_tools/user_conditions_v1/render.py")
    plt=helper.plt;pages=[];started=time.perf_counter();bounds={}
    for cid in sorted({q["condition_id"] for q in reg["requests"]}):
        qs=[q for q in reg["requests"] if q["condition_id"]==cid]
        all_tri=[]
        for q in qs:
            for label in LABELS:
                a,_=verify_output(q,rb[label,q["request_id"]]);all_tri.append(a["raw"].reshape(-1,3))
        all_points=np.concatenate(all_tri);lo=all_points.min(0);hi=all_points.max(0)
        bounds[cid]={"center_model_xyz":((lo+hi)/2).tolist(),"span":float(np.max(hi-lo))*1.14,
           "scope":"Union of C+RAW all3models/all2noises for this condition; no crop, display-only",
           "projection":"orthographic","axis_transform":"model[x,y,z]->plot[x,z,y]"}
    for q in reg["requests"]:
        for mode in ("neutral","colored"):
            path=out/"gallery"/mode/(q["request_id"]+".png");sidecar=path.with_suffix(".png.json")
            identity={"version":VERSION,"evaluation_registration_sha256":digest(out/"evaluation/registration.json"),
                "request_id":q["request_id"],"condition_id":q["condition_id"],"panel":q["panel"],"mode":mode,
                "columns":list(LABELS),"cameras":[[22,-60],[22,120]],"bounds":bounds[q["condition_id"]],
                "raw_sha256":{label:rb[label,q["request_id"]]["raw_sha256"] for label in LABELS},
                "source_sha256":digest(__file__),"renderer_helper_sha256":digest(R2/"data_tools/user_conditions_v1/render.py")}
            if path.exists() and sidecar.exists():
                page=read(sidecar)
                if page["identity"]!=identity or page["image_sha256"]!=digest(path):raise ValueError("Gallery identity changed")
                pages.append(page);continue
            if path.exists() or sidecar.exists():raise ValueError("Incomplete gallery artifact")
            fig=plt.figure(figsize=(14,9),facecolor="white")
            for col,label in enumerate(LABELS):
                a,_=verify_output(q,rb[label,q["request_id"]]);tri=a["raw"]
                colors=helper._shaded(tri,"#acb3bb" if mode=="neutral" else "#de9758")
                if mode=="colored":colors[:q["K"]]=helper._shaded(tri[:q["K"]],"#268cca")
                for rr,view in enumerate(((22,-60),(22,120))):
                    ax=fig.add_subplot(2,3,rr*3+col+1,projection="3d")
                    helper._draw(ax,tri,colors,bounds[q["condition_id"]],view)
                    ax.set_title(label+" | "+("front oblique" if rr==0 else "opposite oblique"),fontsize=10)
            fig.suptitle(q["panel"]+" | "+q["parent_id"]+" | "+q["region_type"]+"/"+q["tier"]+
                         " | n"+str(q["candidate"])+" | N="+str(q["N"])+" K="+str(q["K"]),fontsize=10,y=.985)
            fig.text(.5,.018,"RAW ONLY; no postprocessing | "+("BLUE=C / ORANGE=generated free" if mode=="colored" else "Neutral geometry")+
                     " | Identical condition-scale bounds across models and noises",ha="center",fontsize=9)
            fig.subplots_adjust(left=0,right=1,top=.92,bottom=.05,wspace=0,hspace=.03)
            path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(".png.tmp")
            fig.savefig(temp,format="png",dpi=150,facecolor="white");plt.close(fig);temp.replace(path)
            page={"status":"COMPLETE","identity":identity,"image":str(path),"image_sha256":digest(path),"AI_PROXY_review":"PENDING"}
            write(sidecar,page);pages.append(page)
        write(out/"gallery/progress.json",{"pages":len(pages),"target":56,"seconds":time.perf_counter()-started},freeze=False)
    write(out/"gallery/manifest.json",{"status":"COMPLETE","pages":pages,"requests":28,"model_outputs":84,"PNG_count":56,
          "seconds":time.perf_counter()-started,"model_calls":0,"GPU":0})
    template=out/"gallery/AI_PROXY_review_template.csv"
    if not template.exists():
        notes=[]
        for q in reg["requests"]:
            for label in LABELS:
                notes.append({**{k:q[k] for k in META},"label":label,"reviewer":"AI_PROXY","actually_viewed":False,
                    "neutral_image":str(out/"gallery/neutral"/(q["request_id"]+".png")),
                    "colored_image":str(out/"gallery/colored"/(q["request_id"]+".png")),
                    "overall_observation":"","C_role_compatibility":"","visible_fragments_or_misuse":"",
                    "comparison_improved_similar_degraded_uncertain":"","uncertainty":"","human_rating":False})
        write_csv(template,notes)
    return {"status":"COMPLETE","PNG_count":56,"model_outputs":84,"seconds":time.perf_counter()-started}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",required=True,choices=("prepare","bind","sample","collect","score","render"))
    ap.add_argument("--out",default=str(ART/"experiments"/ROOT_NAME))
    ap.add_argument("--data-root",default=str(DATA));ap.add_argument("--label",choices=LABELS)
    ap.add_argument("--checkpoint");ap.add_argument("--attempt",default="initial",choices=("initial","serial_retry","oom_serial_retry"))
    ap.add_argument("--controller-owned",action="store_true")
    args=ap.parse_args()
    if args.stage=="prepare":
        result=prepare(args.out,args.data_root)
        result={"status":result["status"],"requests":len(result["requests"]),"conditions":len(result["queries"]),"budget":result["budget"],"model_calls":0}
    elif args.stage=="bind":
        if not args.label or not args.checkpoint:ap.error("bind needs label/checkpoint")
        result=bind_checkpoint(args.out,args.label,args.checkpoint)
    elif args.stage=="sample":
        if not args.label:ap.error("sample needs label")
        result=sample_job(args.out,args.label,args.attempt,args.controller_owned)
        result={k:v for k,v in result.items() if k!="rows"}
    elif args.stage=="collect":
        rows=collect_outputs(args.out,False);result={"completed":len(rows),"registered":84,"status":"COMPLETE" if len(rows)==84 else "PENDING"}
    elif args.stage=="score":result=score_all(args.out)
    else:result=render_all(args.out)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=="__main__":main()
