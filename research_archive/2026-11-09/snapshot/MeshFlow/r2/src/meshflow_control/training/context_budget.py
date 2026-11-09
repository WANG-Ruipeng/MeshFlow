"""Fixed context-alignment V1 budget, separate from the completed readout study."""
from pathlib import Path
import json,time
from ..accounting import Budget as BaseBudget
LIMITS=dict(train_forward=24000,train_backward=24000,optimizer_update=3000,
    aligner_forward=8000,aligner_backward=8000,preflight_forward=24,preflight_backward=16,
    gradient_observation_backward=12,teacher_forward=4096,teacher_image=4096,
    rollout_attempt=200,sampling_forward=10000)
class Budget(BaseBudget):
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/"calls.jsonl";self.counts={k:0 for k in LIMITS};self.by_arm={}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row=json.loads(line)
                if row["event"]=="RESERVE":
                    k=row["kind"]
                    if k not in LIMITS:raise ValueError("Unknown ledger category")
                    self.counts[k]+=1
                    if k=="optimizer_update":
                        arm=row.get("context",{}).get("arm","unknown");self.by_arm[arm]=self.by_arm.get(arm,0)+1
    def reserve(self,kind,context=None):
        from ..runtime import check_gpu_deadline
        check_gpu_deadline()
        if kind not in LIMITS:raise ValueError("Unknown budget category: "+kind)
        if self.counts[kind]>=LIMITS[kind]:raise RuntimeError("Budget exhausted: "+kind)
        ctx=context or {}
        if kind=="optimizer_update" and self.by_arm.get(ctx.get("arm","unknown"),0)>=1000:
            raise RuntimeError("Per-arm update budget exhausted")
        self.event("RESERVE",kind=kind,context=ctx,ordinal=self.counts[kind]+1);self.counts[kind]+=1
        if kind=="optimizer_update":
            arm=ctx.get("arm","unknown");self.by_arm[arm]=self.by_arm.get(arm,0)+1
    def snapshot(self):
        return dict(counts=dict(self.counts),limits=dict(LIMITS),updates_by_arm=dict(self.by_arm))
