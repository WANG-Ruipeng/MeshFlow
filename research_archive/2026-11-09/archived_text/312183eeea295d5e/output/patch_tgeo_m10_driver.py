
from pathlib import Path
p=Path("experiments/mf_h1_local0/tgeo_m10_run.py")
s=p.read_text()
s=s.replace('''def make_stream(r):''','''def verify_assets(r):
    for name, expected in r["assets"].items():
        if not isinstance(expected, dict) or "path" not in expected or "sha256" not in expected:
            continue
        actual = record(expected["path"])
        if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
            raise RuntimeError("Registered asset changed: " + name)
    return dict(status="PASS", assets=len(r["assets"]))


def make_stream(r):''')
s=s.replace('''def get_relations(r):
    return load_source_relations''','''def get_relations(r):
    for name in ("relation_inputs", "relation_manifest"):
        if file_sha256(r["assets"][name]["path"]) != r["assets"][name]["sha256"]:
            raise RuntimeError("Registered relation asset changed: " + name)
    return load_source_relations''')
s=s.replace('''    velocities=[];gradient_sets=[];losses=[]''','''    velocities=[];gradient_sets=[];losses=[];all_gradient_checks=[]''')
s=s.replace('''        velocities.append(velocity.detach().cpu())
        gradient_sets.append''','''        family = {}
        for n, parameter in model.named_parameters():
            group = "encoder" if n.startswith("context_encoder.") else "role" if n=="role_embedding" else "backbone"
            value = family.setdefault(group, dict(parameters_with_gradient=0, gradient_squared=0.))
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("Nonfinite full-model preflight gradient: " + n)
                value["parameters_with_gradient"] += parameter.numel()
                value["gradient_squared"] += float(parameter.grad.detach().float().square().sum())
        all_gradient_checks.append({k:dict(parameters_with_gradient=v["parameters_with_gradient"],
                                          gradient_l2=v["gradient_squared"]**.5) for k,v in family.items()})
        velocities.append(velocity.detach().cpu())
        gradient_sets.append''')
s=s.replace('''        runtime=runtime,load=audit,gate_sample_index=index,''','''        runtime=runtime,load=audit,gate_sample_index=index,all_parameter_gradient_checks=all_gradient_checks,''')
s=s.replace('''    with np.load(r['assets']['evaluation_noise']['path'],allow_pickle=False) as value:
        noises={s:value[f'noise_seed{s}'].copy() for s in SEEDS}
    tick=''','''    if file_sha256(r["assets"]["evaluation_noise"]["path"]) != r["assets"]["evaluation_noise"]["sha256"]:
        raise RuntimeError("Registered evaluation noises changed")
    with np.load(r['assets']['evaluation_noise']['path'],allow_pickle=False) as value:
        noises={s:value[f'noise_seed{s}'].copy() for s in SEEDS}
    if any(n.dtype != np.float32 or n.shape != (1,112,9) or not np.isfinite(n).all() for n in noises.values()):
        raise RuntimeError("Invalid fixed evaluation Gaussian")
    tick=''')
s=s.replace('''            verify_sources(r)
            if args.phase=='train':''','''            verify_sources(r)
            verify_assets(r)
            if args.phase=='train':''')
p.write_text(s)
