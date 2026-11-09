"""Bind successful CPU initialization and fixed evaluation checks to dispatch."""
from .common import OUT, read, sha, write_new, resources


def run():
    dest = OUT / "training_gate.json"
    if dest.exists():
        r=read(dest)
        assert r["registration_sha256"]==sha(OUT/"training_registration.json")
        return r
    pre=OUT/"training_audit/initialization_preflight.json"
    ev=OUT/"evaluation_cpu_checks.json"
    a,b=read(pre),read(ev)
    assert a["status"]==b["status"]=="PASS"
    assert a["training_registration_sha256"]==sha(OUT/"training_registration.json")
    assert a["CUDA_initialized"] is False and a["model_forwards"]==0
    assert sum(b["sizes"].values())==224 and b["generator_forwards"]==b["D0_forwards"]==0
    r=dict(status="PASS",registration_sha256=sha(OUT/"training_registration.json"),
           initialization_audit_sha256=sha(pre),evaluation_audit_sha256=sha(ev),
           scope_sha256=sha(OUT/"registration.json"),resources=resources(),
           additional_model_forwards=0,retired_weights_not_restored=True)
    write_new(dest,r)
    return r


if __name__=="__main__":
    print(run()["status"],flush=True)
