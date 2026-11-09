"""Resource ledger and incremental release, not a scientific job launcher."""
import argparse
import math
import time
from .common import OUT, ARMS, sha, resources, read, write_new, atomic_write


def release(arm, observed_peak=4500., note="initial prior-comparable conservative process estimate"):
    snap = resources()
    vals = snap["gpu"]["stdout"].strip().splitlines()[0].split(",")
    total, used, free, compute = [float(v.strip().split()[0]) for v in vals[1:]]
    headroom = max(2048., .1*total)
    slots = math.floor(max(0., free-headroom)/(1.2*observed_peak))
    if slots < 1:
        raise RuntimeError("RESOURCE_WAIT no incremental slot")
    path = OUT/"training_dispatch_go.json"
    old = read(path) if path.exists() else dict(authorized=True, arms=[],
        training_registration_sha256=sha(OUT/"training_registration.json"))
    if arm in old["arms"]:
        raise ValueError("Already released; use exact-cursor recovery without a second release")
    old["arms"].append(arm)
    event = dict(arm=arm, observed_or_estimated_process_peak_mib=observed_peak,
                 estimate_note=note, headroom_mib=headroom, incremental_slots=slots,
                 selected_additional_workers=1, current_gpu_compute_percent=compute,
                 dispatch="incremental real authorized task", resource=snap)
    write_new(OUT/"dispatch"/f"{time.time_ns()}_{arm}.json", event)
    atomic_write(path, old)
    return event


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--observed-peak-mib", type=float, default=4500.)
    p.add_argument("--note", default="initial comparable estimate; real first-update peak pending")
    a = p.parse_args()
    event = release(a.arm, a.observed_peak_mib, a.note)
    print({k:v for k,v in event.items() if k != "resource"})
