from pathlib import Path
p=Path('native_t1/geometry_pilot.py');s=p.read_text()
start=s.index('    from .geometry_gates import GATE_POLICY, BASELINE_EVIDENCE_RESULTS')
end=s.index('    archive=load_archive(args.inputs)',start)
s=s[:start]+'''    from .geometry_gates import GATE_POLICY, STRICT_GATE_POLICY, BASELINE_EVIDENCE_RESULTS, BASELINE_EVIDENCE_VELOCITIES
    evidence_path=getattr(args,"baseline_regression_evidence",None)
    evidence=None
    if evidence_path is not None:
        if evidence_path.resolve()!=BASELINE_EVIDENCE_RESULTS.resolve():
            raise ValueError("Revised policy requires the authorized baseline code audit")
        evidence=dict(results_path=str(evidence_path.resolve()),results_sha256=file_sha256(evidence_path),
            velocities_path=str(BASELINE_EVIDENCE_VELOCITIES.resolve()),
            velocities_sha256=file_sha256(BASELINE_EVIDENCE_VELOCITIES))
'''+s[end:]
s=s.replace('gate_policy=GATE_POLICY,baseline_regression_evidence=evidence,','gate_policy=GATE_POLICY if evidence else STRICT_GATE_POLICY,baseline_regression_evidence=evidence,')
s=s.replace('gate_policy_reason="Original T0 and current wrapper are bitwise identical; FP32/first-block diagnosis localizes inherited BF16 order sensitivity. Historical numerical FAIL is retained separately from hard wiring gates.",','''gate_policy_reason=("Original T0 and current wrapper are bitwise identical; FP32/first-block diagnosis localizes inherited BF16 order sensitivity. Historical numerical FAIL is retained separately from hard wiring gates."
            if evidence else "No baseline exemption: historical BF16 permutation tolerance remains a hard gate."),''')
p.write_text(s)
run=Path('native_t1/runs/geometry_context_revised_20260930_153000')
for ext in ('log','json'):
 (run/('all_cpu_tests.'+ext)).rename(run/('pre_portable_policy_cpu_tests.'+ext))
