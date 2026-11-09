"""Audit the external START without copying its large file or running a forward."""
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--out",required=True)
    a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    from meshflow_control.checkpoints import load_start
    model,optimizer,audit=load_start(a.checkpoint,device="cpu",trainable=True)
    if torch.cuda.is_initialized(): raise RuntimeError("CPU import initialized CUDA")
    audit["external_checkpoint"]=str(Path(a.checkpoint).resolve())
    audit["large_checkpoint_copied"]=False
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ("old_tensor_count","old_parameter_count","old_tensors_equal","optimizer_restored")}))
if __name__=="__main__":main()
