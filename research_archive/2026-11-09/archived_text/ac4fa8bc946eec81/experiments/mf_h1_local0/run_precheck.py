"""Strict real EMA precheck; disposable probes, never formal training."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse,json,resource,time,traceback
from pathlib import Path
import torch
from .official_wrapper import load_official,state_sha256
from .prompt_dataset import file_sha256,RealChairStream,collate_samples
from .tests.official_checks import run_official_checks

def run(checkpoint,output):
    output=Path(output)
    path=output/"precheck_execution.json"
    if path.exists(): raise FileExistsError(path)
    report=dict(status="RUNNING",official_backbone_forwards=0,official_examples=0,
                complete_rollouts=0,probe_updates=0,phase2_updates=0,phase3="NOT_AUTHORIZED")
    start=time.perf_counter()
    try:
        torch.set_num_threads(8)
        checkpoint=Path(checkpoint)
        digest=file_sha256(checkpoint)
        if checkpoint.stat().st_size!=2084287208 or digest!="bda891a0f046ca6015f70f745fa24a8036c64dda12175b246d853c957f63b92d":
            raise ValueError("Official chair checkpoint identity mismatch")
        report["checkpoint"]=dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,sha256=digest)
        manifest=json.loads((output/"split_manifest.json").read_text())
        base,audit=load_official(checkpoint,Path(__file__).resolve().parents[2]/"configs/snet/base-120m-ot-v-chair.yaml","cuda")
        report["strict_load"]=audit
        report["base_state_sha256"]=state_sha256(base)
        def count(module,args):
            report["official_backbone_forwards"]+=1
            report["official_examples"]+=len(args[0])
        base.register_forward_pre_hook(count)
        path.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(dict(stage="strict_load",status="PASS",parameters=audit["parameters"])),flush=True)
        samples=RealChairStream(manifest["train"],seed=10).next_effective_batch(8)
        units=run_official_checks(base,"cuda",output/"official_tests",collate_samples(samples))
        report["official_units_status"]=units["status"]
        report["probe_updates"]=units["probe_optimizer_updates"]
        report["status"]=units["status"]
    except BaseException as error:
        report.update(status="FAIL",error=repr(error),traceback=traceback.format_exc())
        print(traceback.format_exc(),flush=True)
    finally:
        report["seconds"]=time.perf_counter()-start
        report["cpu_peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(dict(stage="official_units",status=report["status"],forwards=report["official_backbone_forwards"],updates=report["probe_updates"])),flush=True)
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();r=run(a.checkpoint,a.output)
    raise SystemExit(0 if r["status"]=="PASS" else 1)

