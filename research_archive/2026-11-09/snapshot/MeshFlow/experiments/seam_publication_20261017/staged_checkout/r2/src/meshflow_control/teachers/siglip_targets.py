"""Frozen, training-only SigLIP targets. Imports torch/transformers only in encode."""
from __future__ import annotations
from collections import Counter
import importlib.metadata,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .siglip_assets import MODEL_ID,REVISION,WEIGHT_SHA256,digest,save
from .siglip_render import RENDER_CONFIG,array_hash,json_hash,camera_from_training,render

IMAGE_LIMIT=4096
def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def freeze(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError("Frozen identity differs: "+str(path))
    else:save(path,value)
def npz_save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".partial")
    with tmp.open("wb") as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)
def source_identity():
    return {p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name("siglip_render.py"),Path(__file__).with_name("siglip_assets.py"))}
def collect_targets(data_root,train_plan_path):
    from ..data.dataset import TrainingDataset
    data=TrainingDataset(data_root);plan=read(train_plan_path)
    if plan["data_manifest_sha256"]!=data.manifest_sha256:raise ValueError("Stream/data identity differs")
    records=plan["records"]
    if len(records)!=8000 or any(sum(r["alpha"]==1. for r in records[i:i+8])!=4 for i in range(0,8000,8)):
        raise ValueError("Expected original 1000x8 with four exact alpha1")
    selected=[r for r in records if r["alpha"]==1.]
    tasks={r["task_id"] for r in selected};groups={};mapping={}
    train=set(data.manifest["splits"]["train"])
    for task_id in sorted(tasks):
        task=data.tasks[task_id]
        if task["role"]!="train" or task["split"]!="train" or task["status"]!="READY" or task["uid"] not in train:
            raise ValueError("Teacher target outside actual TRAINING READY split")
        p=data.task(task_id,1.0);full=np.ascontiguousarray(p["full_target"].reshape(-1,3,3))
        ids=np.r_[p["source_face_ids"],p["free_source_ids"]]
        combined=np.concatenate((full[p["source_face_ids"]],full[p["free_source_ids"]]))
        restored=np.empty_like(full);restored[ids]=combined
        if restored.tobytes()!=full.tobytes():raise ValueError("C union clean free mismatch")
        gh=array_hash(full)
        if gh not in groups:groups[gh]=dict(mesh=full,task_ids=[],uids=[])
        groups[gh]["task_ids"].append(task_id)
        if task["uid"] not in groups[gh]["uids"]:groups[gh]["uids"].append(task["uid"])
        mapping[task_id]=dict(geometry_sha256=gh,uid=task["uid"],N=int(task["N"]),K=int(task["K"]))
    if len(groups)*4>IMAGE_LIMIT:raise ValueError("Teacher unique-image cap exceeded")
    frequency=Counter(mapping[r["task_id"]]["geometry_sha256"] for r in selected)
    identity=dict(data_root=str(Path(data_root).resolve()),data_manifest_path=str(data.manifest_path),
        data_manifest_sha256=data.manifest_sha256,train_plan_path=str(Path(train_plan_path).resolve()),
        train_plan_sha256=digest(train_plan_path),alignment_samples=len(selected),
        unique_meshes=len(groups),unique_tasks=len(mapping),unique_parents=len({r["uid"] for r in selected}),
        target_geometry="FP32 full_target from actual alpha1 training targets; C plus clean GT-free verified by source-face mapping",
        inference_inputs_unchanged=True,frequency=dict(frequency),tasks=mapping,
        alpha1_samples={str(r["sample_index"]):dict(task_id=r["task_id"],geometry_sha256=mapping[r["task_id"]]["geometry_sha256"]) for r in selected})
    return groups,identity

def prepare_targets(data_root,train_plan_path,out):
    out=Path(out);folder=out/"teacher";folder.mkdir(parents=True,exist_ok=True)
    assets=read(folder/"assets/assets_verified.json")
    if assets["status"]!="PASS" or assets["revision"]!=REVISION:raise ValueError("Teacher asset identity unavailable")
    # Every cache binds physical asset hashes, not model ID alone.
    for row in assets["files"]:
        if digest(row["path"])!=row["sha256"]:raise ValueError("Teacher asset changed")
    groups,inputs=collect_targets(data_root,train_plan_path)
    keys=sorted(groups,key=lambda k:(min(groups[k]["uids"]),k))
    camera=camera_from_training([groups[k]["mesh"] for k in keys])
    config=dict(renderer=RENDER_CONFIG,camera=camera,renderer_source_sha256=digest(Path(__file__).with_name("siglip_render.py")),
        pillow_version=importlib.metadata.version("Pillow"),numpy_version=np.__version__)
    config_sha=json_hash(config)
    identity=dict(inputs=inputs,model_id=MODEL_ID,revision=REVISION,assets_sha256=digest(folder/"assets/assets_verified.json"),
        weight_sha256=WEIGHT_SHA256,processor_sha256=digest(folder/"assets/preprocessor_config.json"),
        render_config_sha256=config_sha,render_config=config)
    freeze(folder/"target_registration.json",identity)
    rows=[];started=time.perf_counter()
    for gh in keys:
        group=groups[gh];geopath=folder/"geometry"/(gh+".npz")
        if geopath.exists():
            with np.load(geopath,allow_pickle=False) as z:
                if array_hash(z["full_target"])!=gh:raise ValueError("Cached target differs")
        else:npz_save(geopath,full_target=group["mesh"])
        views=[]
        for v,az in enumerate(RENDER_CONFIG["azimuth_degrees"]):
            path=folder/"renders"/gh/(str(v)+".png");meta=path.with_suffix(".json")
            view_identity=dict(geometry_sha256=gh,render_config_sha256=config_sha,view_index=v)
            if meta.exists():
                receipt=read(meta)
                if receipt["identity"]!=view_identity or digest(path)!=receipt["png_sha256"]:raise ValueError("Render cache differs")
            else:
                if path.exists():raise ValueError("Unreceipted image requires engineering recovery")
                image,audit=render(group["mesh"],camera,v);path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
                receipt=dict(identity=view_identity,path=str(path),png_sha256=digest(path),audit=audit,
                    view_azimuth=az,teacher_image_no_labels=True)
                save(meta,receipt)
            views.append(receipt)
        rows.append(dict(geometry_sha256=gh,geometry_path=str(geopath),geometry_file_sha256=digest(geopath),
                         uids=sorted(group["uids"]),task_ids=group["task_ids"],training_frequency=inputs["frequency"][gh],views=views))
    manifest=dict(status="COMPLETE",identity=identity,rows=rows,unique_meshes=len(rows),images=4*len(rows),
        model_forwards=0,teacher_forwards=0,sources=source_identity())
    if (folder/"render_manifest.json").exists():
        previous=read(folder/"render_manifest.json")
        if previous["identity"]!=manifest["identity"] or previous["rows"]!=manifest["rows"]:
            raise ValueError("Frozen training render geometry/config/receipt differs")
        manifest=previous
    else:freeze(folder/"render_manifest.json",manifest)
    # Predetermined first eight TRAINING geometries; labels exist only outside encoded images.
    selected=rows[:8];w,h=4*160, len(selected)*180+32
    contact=Image.new("RGB",(w,h),"white");draw=ImageDraw.Draw(contact)
    draw.text((8,6),"TRAINING targets - first 8 by UID; all fixed views; not teacher input",fill="black")
    for j,row in enumerate(selected):
        for v,view in enumerate(row["views"]):
            with Image.open(view["path"]) as im:contact.paste(im.resize((160,160)),(v*160,32+j*180))
        draw.text((8,32+j*180+160),row["uids"][0]+"  "+row["geometry_sha256"][:12],fill="black")
    contact_path=out/"teacher_render_contact_sheet.png";contact.save(contact_path)
    save(folder/"render_receipt.json",dict(status="COMPLETE",images=manifest["images"],unique_meshes=len(rows),
        contact_sheet=str(contact_path),contact_sheet_sha256=digest(contact_path),selected_geometry=[r["geometry_sha256"] for r in selected],
        render_manifest_sha256=digest(folder/"render_manifest.json"),seconds=time.perf_counter()-started,
        visual_review="PENDING",teacher_forwards=0))
    return dict(status="COMPLETE",unique_meshes=len(rows),images=manifest["images"],contact_sheet=str(contact_path))

def normalize_multiview(poolers):
    values=np.asarray(poolers)
    if values.dtype!=np.float32 or values.shape!=(4,768) or not np.isfinite(values).all():
        raise ValueError("Four finite official FP32 pooler768 vectors required")
    norms=np.linalg.norm(values,axis=1,keepdims=True)
    if np.any(norms==0):raise ValueError("Undefined zero teacher vector")
    views=values/norms;mean=views.mean(axis=0,dtype=np.float32);length=np.linalg.norm(mean)
    if not length>0:raise ValueError("Undefined normalized mean target")
    return views.astype(np.float32),np.asarray(mean/length,np.float32)

def load_teacher(asset_folder,device):
    import torch
    from safetensors import safe_open
    from transformers import SiglipVisionConfig,SiglipVisionModel,SiglipImageProcessor
    folder=Path(asset_folder)
    if digest(folder/"model.safetensors")!=WEIGHT_SHA256:raise ValueError("Exact public teacher weight hash differs")
    config=SiglipVisionConfig.from_pretrained(str(folder),local_files_only=True)
    if config.hidden_size!=768 or config.image_size!=256 or config.patch_size!=16:raise ValueError("Unexpected official vision config")
    config._attn_implementation="eager"
    with torch.random.fork_rng(devices=[]):
        model=SiglipVisionModel(config)
    with safe_open(str(folder/"model.safetensors"),framework="pt",device="cpu") as source:
        state={key:source.get_tensor(key) for key in source.keys() if key.startswith("vision_model.")}
    model.load_state_dict(state,strict=True);del state
    model.eval();model.requires_grad_(False);model.to(device=device,dtype=torch.float32)
    processor=SiglipImageProcessor.from_pretrained(str(folder),local_files_only=True)
    info=dict(class_name=type(model).__name__,output_field="pooler_output",feature_dimension=768,
        parameters=sum(p.numel() for p in model.parameters()),state_tensors=len(model.state_dict()),
        loading="Official vision-only architecture; exact vision_model.* safetensors subset loaded strict=True",
        missing_keys=[],unexpected_keys=[],eval=True,requires_grad=False,dtype="torch.float32",
        attention_implementation="eager",processor_config=processor.to_dict(),
        versions={name:importlib.metadata.version(name) for name in ("torch","transformers","huggingface-hub","safetensors","Pillow","numpy")})
    return model,processor,info

def encode_targets(out,budget,device="cuda",batch_size=16):
    import torch
    from ..accounting import resource_snapshot
    out=Path(out);folder=out/"teacher";manifest=read(folder/"render_manifest.json")
    if str(device)!="cuda":raise ValueError("Actual teacher forwards require root-owned GPU worker")
    review=read(folder/"render_review.json")
    if review.get("status")!="PASS" or review.get("render_manifest_sha256")!=digest(folder/"render_manifest.json"):
        raise ValueError("Actual fixed TRAINING contact review must precede teacher encoding")
    if not 1<=batch_size<=32:raise ValueError("Bounded teacher batch required")
    plan=[dict(geometry_sha256=r["geometry_sha256"],view_index=i,path=v["path"],sha256=v["png_sha256"])
          for r in manifest["rows"] for i,v in enumerate(r["views"])]
    if len(plan)>IMAGE_LIMIT:raise ValueError("Teacher image cap exceeded")
    identity=dict(render_manifest_sha256=digest(folder/"render_manifest.json"),
        assets_sha256=digest(folder/"assets/assets_verified.json"),sources=source_identity(),batch_size=batch_size,
        precision="FP32 without autocast",pooler_output="official SiglipVisionModel.pooler_output",images=plan)
    freeze(folder/"encoding_plan.json",identity)
    if (out/"teacher_manifest.json").exists():
        completed=read(out/"teacher_manifest.json")
        if completed["status"]!="COMPLETE" or completed["encoding_plan_sha256"]!=digest(folder/"encoding_plan.json"):
            raise ValueError("Completed teacher identity differs")
        TargetStore(out)
        return completed
    for row in plan:
        if digest(row["path"])!=row["sha256"]:raise ValueError("Rendered image changed")
    chunks=[plan[i:i+batch_size] for i in range(0,len(plan),batch_size)]
    # Completed batches can be reused; partial actual-call attempts are never silently repeated.
    loaded=[];pending=[]
    for i,chunk in enumerate(chunks):
        stem=folder/"encoding_batches"/str(i).zfill(4);meta=stem.with_suffix(".json");data=stem.with_suffix(".npz");attempt=stem.with_suffix(".attempt.json")
        if meta.exists():
            receipt=read(meta)
            if receipt["plan_sha256"]!=digest(folder/"encoding_plan.json") or receipt["npz_sha256"]!=digest(data):
                raise ValueError("Teacher batch receipt changed")
            with np.load(data,allow_pickle=False) as z:loaded.append((i,z["pooler"].copy()))
        else:
            if attempt.exists() or data.exists():raise RuntimeError("Incomplete teacher batch requires explicit engineering recovery")
            pending.append((i,chunk,stem))
    model=processor=None;info=None;started=time.perf_counter()
    if pending:
        model,processor,info=load_teacher(folder/"assets",device)
        save(folder/"teacher_loading.json",info)
        torch.cuda.synchronize()
        save(folder/"resources_after_teacher_load.json",resource_snapshot())
        versions_before={name:int(p._version) for name,p in model.named_parameters()}
        for i,chunk,stem in pending:
            images=[]
            for row in chunk:
                with Image.open(row["path"]) as image:images.append(image.convert("RGB").copy())
            pixels=processor(images=images,return_tensors="pt")["pixel_values"].to(device=device,dtype=torch.float32)
            if pixels.shape!=(len(chunk),3,256,256):raise ValueError("Unexpected official processor output")
            contexts=[dict(teacher=MODEL_ID,revision=REVISION,geometry_sha256=r["geometry_sha256"],view_index=r["view_index"],batch=i) for r in chunk]
            freeze(stem.with_suffix(".attempt.json"),dict(plan_sha256=digest(folder/"encoding_plan.json"),images=chunk))
            for ctx in contexts:budget.reserve("teacher_image",context=ctx)
            measurement={}
            def actual_teacher_forward():
                # Synchronization stays inside the actual budgeted call. No probe forward.
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                tick=time.perf_counter()
                value=model(pixel_values=pixels,return_dict=True)
                torch.cuda.synchronize()
                measurement.update(seconds=time.perf_counter()-tick,
                    allocated_bytes=torch.cuda.max_memory_allocated(),
                    reserved_bytes=torch.cuda.max_memory_reserved(),
                    scope="One real teacher forward, synchronized; resident weights and input tensor included; no backward")
                return value
            try:
                with torch.inference_mode(),torch.autocast(device_type="cuda",enabled=False):
                    result=budget.call("teacher_forward",actual_teacher_forward,
                        context=dict(batch=i,image_count=len(chunk),teacher=MODEL_ID,revision=REVISION))
                pooler=result.pooler_output
                if pooler.dtype!=torch.float32 or pooler.shape!=(len(chunk),768) or not torch.isfinite(pooler).all():
                    raise ValueError("Official pooler output is nonfinite or wrong dtype/shape")
                value=pooler.cpu().numpy().copy()
                for ctx in contexts:budget.event("RETURN",kind="teacher_image",context=ctx)
            except BaseException as error:
                for ctx in contexts:budget.event("ERROR",kind="teacher_image",context=ctx,error=repr(error))
                raise
            npz_save(stem.with_suffix(".npz"),pooler=value)
            save(stem.with_suffix(".json"),dict(status="PASS",plan_sha256=digest(folder/"encoding_plan.json"),
                npz_sha256=digest(stem.with_suffix(".npz")),images=len(chunk),pooler_shape=list(value.shape),pooler_dtype=str(value.dtype),measurement=measurement))
            loaded.append((i,value));del result,pooler,pixels
        if any(int(p._version)!=versions_before[name] for name,p in model.named_parameters()):raise ValueError("Frozen teacher parameters changed")
        torch.cuda.synchronize()
        measurements=[read((folder/"encoding_batches"/str(i).zfill(4)).with_suffix(".json"))["measurement"] for i in range(len(chunks))]
        save(folder/"encoding_resource_peak.json",dict(
            allocated_bytes=max(m["allocated_bytes"] for m in measurements),
            reserved_bytes=max(m["reserved_bytes"] for m in measurements),
            synchronized_forward_seconds=sum(m["seconds"] for m in measurements),
            actual_batches=len(measurements),
            scope="Maximum across individually reset real forward batches; resident teacher/input included. PyTorch process allocator only, not total process VRAM"))
        save(folder/"resources_after_encoding_before_unload.json",resource_snapshot())
        del model,processor;torch.cuda.empty_cache();torch.cuda.synchronize()
        save(folder/"resources_after_teacher_unload.json",resource_snapshot())
    else:info=read(folder/"teacher_loading.json")
    all_poolers=np.concatenate([v for _,v in sorted(loaded)],axis=0)
    features=[];rows=[]
    for j,row in enumerate(manifest["rows"]):
        poolers=all_poolers[4*j:4*j+4];views,target=normalize_multiview(poolers)
        path=folder/"features"/(row["geometry_sha256"]+".npz")
        npz_save(path,pooler=poolers,normalized_views=views,target=target)
        features.append(target)
        rows.append(dict(geometry_sha256=row["geometry_sha256"],uids=row["uids"],task_ids=row["task_ids"],
            training_frequency=row["training_frequency"],path=str(path),sha256=digest(path),
            target_array_sha256=array_hash(target),pooler_norms=np.linalg.norm(poolers,axis=1).tolist()))
    features=np.stack(features);diagnostics=feature_statistics(features,np.asarray([r["training_frequency"] for r in rows]))
    if diagnostics["all_targets_identical"]:raise ValueError("Teacher targets are all identical: implementation/asset blocked")
    save(folder/"feature_statistics.json",diagnostics)
    npz_save(folder/"feature_pairwise_cosine.npz",cosine=features.astype(np.float64)@features.astype(np.float64).T,targets=features)
    final=dict(status="COMPLETE",model_id=MODEL_ID,revision=REVISION,weight_sha256=WEIGHT_SHA256,
        processor_sha256=manifest["identity"]["processor_sha256"],render_manifest_sha256=digest(folder/"render_manifest.json"),
        encoding_plan_sha256=digest(folder/"encoding_plan.json"),inputs=manifest["identity"]["inputs"],
        rows=rows,images=len(plan),unique_meshes=len(rows),teacher_batches=len(chunks),loading=info,
        feature_statistics=diagnostics,teacher_seconds=time.perf_counter()-started,
        model_not_loaded_for_training=True,supervision="alpha1 only; normalize each official pooler, equal 4-view mean, normalize again",
        ledger_snapshot=budget.snapshot())
    save(out/"teacher_manifest.json",final)
    return final

def feature_statistics(features,frequency):
    features=np.asarray(features,np.float32);frequency=np.asarray(frequency,np.float64)
    if features.ndim!=2 or features.shape[1]!=768 or not np.isfinite(features).all():raise ValueError("Invalid frozen targets")
    q=features.astype(np.float64);cos=q@q.T;lower=cos[np.tril_indices(len(q),-1)]
    mean=q.mean(0);mean/=np.linalg.norm(mean)
    weighted=(q*frequency[:,None]).sum(0);weighted/=np.linalg.norm(weighted)
    return dict(unique_targets=len(q),all_targets_identical=bool(np.all(features==features[0])),
        pairwise_cosine=dict(min=float(lower.min()) if len(lower) else None,max=float(lower.max()) if len(lower) else None,
            mean=float(lower.mean()) if len(lower) else None,median=float(np.median(lower)) if len(lower) else None),
        constant_unique_teacher_mean=dict(mean_alignment_loss=float((1-q@mean).mean()),normalization="L2(normalized target mean), equal unique geometry"),
        constant_training_frequency_mean=dict(mean_alignment_loss=float(np.average(1-q@weighted,weights=frequency)),
            normalization="L2(target mean weighted by actual alpha1 occurrence count)"),
        target_norm_min=float(np.linalg.norm(q,axis=1).min()),target_norm_max=float(np.linalg.norm(q,axis=1).max()),
        interpretation="Global semantic distinguishability only; no proof of local seam/control quality")

class TargetStore:
    def __init__(self,out):
        self.out=Path(out);self.path=self.out/"teacher_manifest.json";self.manifest=read(self.path)
        if self.manifest["status"]!="COMPLETE":raise ValueError("Teacher cache not complete")
        self.sha256=digest(self.path);self.tasks=self.manifest["inputs"]["tasks"]
        self.samples=self.manifest["inputs"]["alpha1_samples"];self.targets={}
        for row in self.manifest["rows"]:
            if digest(row["path"])!=row["sha256"]:raise ValueError("Teacher cache file changed")
            with np.load(row["path"],allow_pickle=False) as z:target=z["target"].copy()
            if target.dtype!=np.float32 or target.shape!=(768,) or not np.isfinite(target).all() or array_hash(target)!=row["target_array_sha256"]:
                raise ValueError("Teacher target identity differs")
            self.targets[row["geometry_sha256"]]=target
    def for_sample(self,sample):
        if float(sample["alpha"])!=1.:raise ValueError("Width-augmented sample cannot receive clean teacher")
        tid=sample["task_id"];row=self.tasks[tid]
        if str(sample["sample_index"]) not in self.samples or self.samples[str(sample["sample_index"])]["task_id"]!=tid:
            raise ValueError("Sample is not in registered alpha1 supervision")
        if sample["object_id"]!=row["uid"]:raise ValueError("Target parent identity differs")
        n=int(sample["y"]);known=np.asarray(sample["known_mask"],bool);valid=np.asarray(sample["valid_mask"],bool)
        if n!=row["N"] or known.sum()!=row["K"] or not valid.all():raise ValueError("Unpadded training target mapping differs")
        coordinates=np.where(known[:,None],sample["context"],sample["x1"]).reshape(n,3,3)
        ids=np.asarray(sample["source_face_ids"],np.int64);corners=np.asarray(sample["corner_permutations"],np.int64)
        if not np.array_equal(np.sort(ids),np.arange(n)) or corners.shape!=(n,3) or not np.all(np.sort(corners,axis=1)==np.arange(3)):
            raise ValueError("Target source/corner mapping is not bijective")
        full=np.empty_like(coordinates);full[ids[:,None],corners]=coordinates
        if array_hash(full)!=row["geometry_sha256"]:raise ValueError("Actual alpha1 C plus clean target is not teacher geometry")
        return self.targets[row["geometry_sha256"]].copy()

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--stage",choices=("prepare","encode"),default="prepare")
    parser.add_argument("--data",type=Path);parser.add_argument("--plan",type=Path)
    parser.add_argument("--ledger",type=Path);parser.add_argument("--batch-size",type=int,default=16)
    args=parser.parse_args()
    if args.stage=="prepare":
        if args.data is None or args.plan is None:parser.error("CPU preparation requires --data and --plan")
        from meshflow_control.accounting import resource_snapshot
        save(args.out/"teacher/resources_before_render.json",resource_snapshot())
        result=prepare_targets(args.data,args.plan,args.out)
        save(args.out/"teacher/resources_after_render.json",resource_snapshot())
    else:
        if args.ledger is None:parser.error("Actual encoding requires the shared experiment ledger")
        from meshflow_control.runtime import gpu_worker,configure_stable_runtime
        from meshflow_control.training.context_budget import Budget
        with gpu_worker(args.ledger,"context_teacher"):
            configure_stable_runtime()
            from torch.nn.attention import sdpa_kernel,SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                result=encode_targets(args.out,Budget(args.ledger),batch_size=args.batch_size)
    print(json.dumps({k:result[k] for k in ("status","unique_meshes","images")}))
