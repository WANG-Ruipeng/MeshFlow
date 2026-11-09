from pathlib import Path
import csv,json,subprocess,time
r=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_contextual_alignment_sie_v1")
def read(p):return json.loads(p.read_text()) if p.exists() else {}
out={"time_unix":time.time(),"controller":read(r/"orchestration/status.json").get("phase"),"controller_status":read(r/"orchestration/status.json").get("status"),"training":{}}
for arm in "ABC":
 p=r/"training"/arm/"training_log.csv"
 if p.exists():
  with p.open() as f:rows=list(csv.DictReader(f))
  if rows:
   last=rows[-1]
   out["training"][arm]={k:last[k] for k in ("step","FM","raw_align","generator_gradient_norm","aligner_gradient_norm","update_seconds","cuda_peak_reserved_bytes")}
for label in ("START","A_FM_1000","B_FREE_ALIGN_1000","C_KNOWN_ALIGN_1000"):
 p=r/"outputs"/(label+"_completed.json")
 if p.exists():out.setdefault("outputs",{})[label]=len(read(p)["rows"])
ev=read(r/"analysis/results.json")
out["scored"]=ev.get("scored");out["diversity_complete"]=ev.get("diversity",{}).get("complete_pairs")
q=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True)
out["gpu_used_free_MiB_util_percent"]=q.stdout.strip()
print(json.dumps(out))
