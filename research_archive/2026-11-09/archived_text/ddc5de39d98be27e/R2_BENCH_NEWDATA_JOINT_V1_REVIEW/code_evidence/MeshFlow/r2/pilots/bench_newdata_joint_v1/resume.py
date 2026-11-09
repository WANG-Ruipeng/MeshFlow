"""Read-only cursor inspection or exact safe-cursor resume; no silent replay."""
from __future__ import annotations
import argparse
import json
from .common import OUT, ARM, read, resolve, sha


def inspect():
    path = OUT / "training" / ARM / "status.json"
    if not path.exists():
        return dict(status="NOT_STARTED", training_updates=0, exact_resume_available=False)
    status = read(path)
    receipt = status.get("checkpoint")
    valid = bool(receipt and resolve(receipt["path"]).is_file()
                 and sha(receipt["path"]) == receipt["sha256"])
    return dict(status=status["status"], local_step=status.get("local_step"),
                checkpoint_valid=valid,
                exact_resume_available=status["status"] == "INTERRUPTED_SAFE" and valid,
                note="FAILED/OOM/unresolved calls require separately recorded recovery; never rollback automatically")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    state = inspect()
    if args.execute:
        if not state["exact_resume_available"]:
            raise RuntimeError("No verified safe-cursor resume is available")
        from .training import train
        state = train(ARM, resume=True)
    print(json.dumps(state, ensure_ascii=False, indent=2))
