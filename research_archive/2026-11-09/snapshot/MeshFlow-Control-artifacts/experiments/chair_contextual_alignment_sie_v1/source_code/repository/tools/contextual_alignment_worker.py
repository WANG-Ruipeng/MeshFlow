"""Single locked GPU phase and CPU-only inference export for the fixed study."""
from meshflow_control import runtime
from meshflow_control.training.context_budget import Budget
from meshflow_control.artifacts import atomic_json,file_sha256
from pathlib import Path
import argparse,json
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["train","export","check-export"],required=True)
    p.add_argument("--out",required=True);p.add_argument("--arm",choices=list("ABC"),required=True)
    p.add_argument("--start");p.add_argument("--data");p.add_argument("--resume")
    a=p.parse_args();root=Path(a.out).resolve();repo=Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    cfg={"A":"coal_fm.yaml","B":"coal_free.yaml","C":"coal_known.yaml"}[a.arm]
    training=root/"checkpoints"/f"{a.arm}_step1000.pt";export=root/"inference"/f"{a.arm}_generator_step1000.pt"
    if a.stage=="train":
        import yaml
        from meshflow_control.training.context_trainer import train
        config=yaml.safe_load((repo/"configs"/cfg).read_text())
        with runtime.gpu_worker(root/"audit/budget","train_"+a.arm):
            state=train(config,Path(a.data)/"train_manifest.json",root/"training_stream",a.start,root,
                Budget(root/"audit/budget"),root/"teacher_manifest.json",stop_step=1000,resume=a.resume,device="cuda")
        if state["status"]!="COMPLETE":raise RuntimeError("Training did not reach primary endpoint")
        print(json.dumps(state),flush=True)
    elif a.stage=="export":
        from meshflow_control.training.context_checkpoint import export_generator
        receipt=root/"inference"/f"{a.arm}_export_receipt.json"
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if file_sha256(export)!=saved["sha256"]:raise ValueError("Prior export changed")
            print(json.dumps(saved));return
        result=export_generator(training,export);atomic_json(receipt,result);print(json.dumps(result))
    else:
        from meshflow_control.training.context_checkpoint import load_generator
        from meshflow_control.training.context_alignment import AttentionCapture
        from meshflow_control.data.contextual_stream import HybridStream
        from meshflow_control.data.dataset import collate,model_inputs
        from meshflow_control.models.native import native_execution
        receipt=root/"audit"/f"{a.arm}_export_capture_check.json"
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved["generator_file_sha256"]!=file_sha256(export):raise ValueError("Export binding changed")
            print(json.dumps(saved));return
        with runtime.gpu_worker(root/"audit/budget","export_check_"+a.arm):
            model,audit=load_generator(export,device="cuda",expected_sha256=file_sha256(export))
            stream=HybridStream(a.data,root/"training_stream",a.arm);s=stream.sample(stream.records[0])
            batch=collate([s],"cuda");budget=Budget(root/"audit/budget")
            with torch.no_grad(),native_execution(model,coordinates=(batch["xt"],batch["t"]),autocast=True):
                plain=budget.call("preflight_forward",lambda:model(*model_inputs(batch)),context=dict(arm=a.arm,check="export_capture_off"))
                with AttentionCapture(model) as capture:
                    hooked=budget.call("preflight_forward",lambda:model(*model_inputs(batch)),context=dict(arm=a.arm,check="export_capture_on"))
                    shape=list(capture.get().shape)
            if not torch.equal(plain,hooked):raise AssertionError("Read-only capture changed exported generator")
            if any(m._forward_hooks for m in model.modules()):raise AssertionError("Capture left a live hook")
            result=dict(status="PASS",generator_file_sha256=file_sha256(export),
                generator_state_sha256=audit["generator_state_sha256"],arm=a.arm,capture_on_off_bitwise=True,
                capture_shape=shape,no_remaining_hooks=True,strict_export_load=True,
                generator_forwards=2,backwards=0,updates=0,aligner_instantiated=False,teacher_loaded=False)
            atomic_json(receipt,result);print(json.dumps(result))
if __name__=="__main__":main()
