from pathlib import Path
p=Path('native_t1/geometry_pilot.py')
s=p.read_text()
s=s.replace('from .training import train_step, make_optimizer','from .training import train_step, make_optimizer\nfrom .geometry_accounting import count_ot_calls')
s=s.replace('    archive=load_archive(args.inputs)\n    # Construct', '''    from .geometry_gates import GATE_POLICY, BASELINE_EVIDENCE_RESULTS, BASELINE_EVIDENCE_VELOCITIES
    evidence_path=getattr(args,"baseline_regression_evidence",None)
    if evidence_path is None or evidence_path.resolve()!=BASELINE_EVIDENCE_RESULTS.resolve():
        raise ValueError("Register the completed baseline code audit explicitly with --baseline-regression-evidence")
    evidence=dict(results_path=str(evidence_path.resolve()),results_sha256=file_sha256(evidence_path),
        velocities_path=str(BASELINE_EVIDENCE_VELOCITIES.resolve()),
        velocities_sha256=file_sha256(BASELINE_EVIDENCE_VELOCITIES))
    archive=load_archive(args.inputs)
    # Construct''')
s=s.replace('        baseline_commit="2502d5f89c54706cc4700d3cdea4a12fd6792529",','''        baseline_commit="2502d5f89c54706cc4700d3cdea4a12fd6792529",
        gate_policy=GATE_POLICY,baseline_regression_evidence=evidence,
        gate_policy_reason="Original T0 and current wrapper are bitwise identical; FP32/first-block diagnosis localizes inherited BF16 order sensitivity. Historical numerical FAIL is retained separately from hard wiring gates.",''')
s=s.replace('"native_t1/geometry_pilot.py","native_t1/geometry_gates.py","native_t1/geometry_generation.py",','"native_t1/geometry_pilot.py","native_t1/geometry_gates.py","native_t1/geometry_generation.py",\n        "native_t1/geometry_accounting.py",')
s=s.replace('optimizer_step_attempts=0,optimizer_updates=0,OT_calls=0)','optimizer_step_attempts=0,optimizer_updates=0,OT_calls=0,OT_returns=0)')
s=s.replace('                    samples=stream.next_effective_batch()\n                    counts["OT_calls"]+=8','                    with count_ot_calls(counts):\n                        samples=stream.next_effective_batch()')
s=s.replace('    parser.add_argument("--audit-results",type=Path)','    parser.add_argument("--audit-results",type=Path)\n    parser.add_argument("--baseline-regression-evidence",type=Path)')
p.write_text(s,encoding='utf-8')
