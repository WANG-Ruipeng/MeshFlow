#!/usr/bin/env python3
"""Document ten evaluated generator instances from frozen small checkpoint bindings."""
import argparse,json,hashlib
from pathlib import Path
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=data:raise ValueError("Refuse replacing different model index")
 if not p.exists():p.write_bytes(data)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);a=ap.parse_args();run=Path(a.run).resolve()
 if run.name!="chair_condition_recipe_factorial_v1":raise ValueError("Wrong run")
 notes={
  ("main","START5000"):"历史条件训练累计5000；保留的默认参照",
  ("main","HIST_FM6000"):"历史START继续FM至6000；实用参照",
  ("main","R1_40_H_ALL"):"官方起点，全程40%/H_ALL至6000；复用既有匹配FM_ONLY",
  ("main","R2_MIX_H_ALL"):"官方起点，全程混合/H_ALL至6000",
  ("main","R3_40_H_LATE"):"官方起点，全程40%/H_LATE至6000",
  ("main","R4_MIX_H_LATE"):"官方起点，全程混合/H_LATE至6000",
  ("course","C20_40_MIX"):"官方起点，20→40→混合/H_ALL至6000",
  ("course","C40_20_MIX"):"官方起点，40→20→混合/H_ALL至6000",
  ("confirm","C20_40_MIX"):"官方起点，新独立训练输入流，20→40→混合至6000",
  ("confirm","R2_MIX_H_ALL"):"官方起点，新独立训练输入流，全程混合至6000"}
 rows=[]
 for phase,count in (("main",384),("course",128),("confirm",128)):
  receipt=read(run/"phases"/phase/"evaluation/scoring_receipt.json")
  if receipt.get("status")!="COMPLETE" or receipt.get("scored")!=count:raise ValueError("Evaluation incomplete")
  p=run/"phases"/phase/"registration.json";reg=read(p)
  for label in reg["labels"]:
   binding=run/"phases"/phase/"checkpoints"/(label+".json");b=read(binding)
   if b["label"]!=label or b["registration_sha256"]!=digest(p):raise ValueError("Checkpoint binding differs")
   file=Path(b["checkpoint_path"])
   if not file.is_file():raise ValueError("Evaluated retained checkpoint missing")
   rows.append(dict(instance=phase+"/"+label,phase=phase,label=label,description=notes[(phase,label)],
     checkpoint_path=str(file),checkpoint_sha256=b["checkpoint_sha256"],binding_path=str(binding.relative_to(run)),
     binding_sha256=digest(binding),registered_outputs=64,weight_hash_source="Frozen binding verified by actual sampler; index does not reload weights"))
 if len(rows)!=10 or len({r["instance"] for r in rows})!=10:raise ValueError("Expected ten distinct generator instances")
 lines=["# 本轮模型与检查点索引","","默认仍为 START。下列是10个被评价的生成器实例；同名配方出现在 confirm 时是新的独立训练终点。","","新矩阵、课程和确认都从同一官方权重起步，各自训练到6000步，使用新AdamW。课程并非从R2续训，确认也并非从旧课程续训。历史START/HIST的训练历史不同。","","|实例|训练关系|实际绑定与完整SHA|","|---|---|---|"]
 for r in rows:
  lines.append("|"+r["instance"]+"|"+r["description"]+"|[绑定]("+r["binding_path"]+")；"+r["checkpoint_sha256"]+"|")
 lines+=["","完整路径和绑定来源见 [model_index.json](analysis/model_index.json)。权重留在本地，不进入审阅ZIP。","本文件只整理已完成评价的身份，不选择新默认，不加载或转换模型。"]
 freeze(run/"MODEL_INDEX.md",("\n".join(lines)+"\n").encode())
 freeze(run/"analysis/model_index.json",(json.dumps(dict(status="COMPLETE",instances=rows,model_calls=0,default_changed=False,implementation_sha256=digest(Path(__file__))),ensure_ascii=False,indent=2)+"\n").encode())
 print(json.dumps({"status":"COMPLETE","instances":len(rows),"model_calls":0}))
if __name__=="__main__":main()
