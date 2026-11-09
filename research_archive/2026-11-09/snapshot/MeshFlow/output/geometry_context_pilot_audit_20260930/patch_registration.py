from pathlib import Path
p=Path("/mnt/e/MeshFlow/native_t1/geometry_pilot.py")
t=p.read_text(encoding="utf-8")
t=t.replace('    from .geometry_evaluation import prepare_evaluation\n','    from .geometry_evaluation import prepare_evaluation, load_evaluation\n    from .context_geometry import face_geometry, ENCODER_METADATA\n',1)
old='    # The bank may already have been frozen by the offline registration step.\n    contract["evaluation_registration"]=str((args.out/"registration").resolve())\n'
new='''    evaluation_path=args.out/"registration/evaluation_inputs.npz"
    if evaluation_path.exists():
        bank,evaluation=load_evaluation(evaluation_path)
    else:
        bank,evaluation=prepare_evaluation(args.inputs,args.out/"registration")
    if evaluation["inputs"]["sha256"]!=archive.sha256:
        raise RuntimeError("Evaluation and training source archives differ")
    if evaluation["status"]!="PASS":
        raise RuntimeError("Registered evaluation patches insufficient; no replacement or training")
    contract["evaluation_registration"]=str((args.out/"registration").resolve())
    contract["evaluation_inputs_sha256"]=file_sha256(evaluation_path)
    contract["evaluation_case_registration"]=evaluation
    contract["encoder_definition"]=dict(ENCODER_METADATA)
    contract["generation_seed_history_check"]=dict(status="REGISTERED",
        seeds=list(GENERATION_SEEDS),
        scope="Local native_t1 and experiments source/JSON/Markdown searched with rg; no prior use as generation/model-selection seeds found. Numeric 9702 occurrences were counters.",
        external_history_claim=False)
    condition_checks=[]
    for scope,cases in (("training_archive",archive.cases),("evaluation",bank["cases"])):
        for key,case in sorted(cases.items()):
            phi,adjacency,facts=face_geometry(torch.from_numpy(case.constraints.copy()))
            condition_checks.append(dict(scope=scope,key=list(key),K=case.K,
                minimum_area=float(facts["area"].min()),nonmanifold_edges=int(facts["nonmanifold_edge_count"]),
                adjacency_directed_edges=int(adjacency.sum()),phi_finite=bool(torch.isfinite(phi).all())))
    contract["condition_checks"]=condition_checks
    contract["registration_analytic_geometry_calls"]=len(condition_checks)
    root=Path(__file__).resolve().parents[1]
    source_names=("native_t1/model.py","native_t1/context_geometry.py","native_t1/geometry_checkpoint.py",
        "native_t1/geometry_pilot.py","native_t1/geometry_gates.py","native_t1/geometry_generation.py",
        "native_t1/geometry_evaluation.py","native_t1/data.py","native_t1/training.py",
        "native_t1/sampling.py","native_t1/runtime.py","native_t1/baseline.py",
        "native_t1/portable_checkpoint.py","native_t1/train_cli.py",
        "models/equidit.py","models/attention.py","models/utils.py","utils/ot_utils.py",
        "configs/snet/base-120m-ot-v-chair.yaml")
    contract["execution_sources"]={name:file_sha256(root/name) for name in source_names}
'''
assert old in t
t=t.replace(old,new)
old='    archive=load_archive(args.inputs)\n    reference_inputs=None\n'
new='''    root=Path(__file__).resolve().parents[1]
    if any(file_sha256(root/name)!=digest for name,digest in contract["execution_sources"].items()):
        raise RuntimeError("Execution code changed after registration")
    archive=load_archive(args.inputs)
    reference_inputs=None
'''
assert old in t
t=t.replace(old,new)
p.write_text(t,encoding="utf-8")
