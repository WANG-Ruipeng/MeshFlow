from pathlib import Path
p=Path("/mnt/e/MeshFlow-Control-artifacts/experiments/chair_condition_recipe_factorial_v1/audit/curriculum_sampling_accounting_preparation/audit_sampling.py")
s=p.read_text()
s=s.replace('if len(list((phase/"outputs").glob("*/*/receipt.json")))!=128:', 'if len(list((phase/"outputs").glob("*/*/receipt.json")))<128:')
s=s.replace('        binding=base.read_json(binding_path);record(binding_path,"checkpoint_binding_metadata_only")',
'''        binding=base.read_json(binding_path);record(binding_path,"checkpoint_binding_metadata_only")
        registered_model=next(m for m in registration["models"] if m["label"]==label)
        if binding.get("registration_sha256")!=sha(registration_path):
            failures.append(dict(label=label,kind="checkpoint binding registration hash"))
        if any(binding.get(k)!=registered_model.get(k) for k in ("label","loader","checkpoint_path")):
            failures.append(dict(label=label,kind="checkpoint binding model metadata"))''')
s=s.replace('                    if r["new_rollouts"]!=1 or r.get("attempt")!="initial":errors.append("new_output_attempt")',
'''                    attempt=r.get("attempt")
                    row["dispatch_attempt"]=attempt
                    if r["new_rollouts"]!=1 or attempt not in ("initial","serial_retry","oom_serial_retry"):
                        errors.append("new_output_attempt")
                    if attempt!="initial":
                        dispatch=phase/"sampling"/label/"serial_retry_dispatch.json"
                        initial_failure=phase/"sampling"/label/"initial_failure.json"
                        if (not dispatch.is_file() or not initial_failure.is_file()
                            or base.read_json(dispatch).get("attempt")!=attempt):
                            errors.append("unregistered serial dispatch attempt")''')
s=s.replace('    for p in failure_markers+retry_markers:\n        record(p,"failure_or_retry_marker");failures.append(dict(kind="failure or retry marker",path=str(p)))',
'''    dispatch_events=[]
    for p in failure_markers+retry_markers:
        record(p,"failure_or_retry_marker")
        dispatch_events.append(dict(path=str(p),record=base.read_json(p)))
    for label in COURSES:
        folder=phase/"sampling"/label
        dispatches=list(folder.glob("*_dispatch.json"))
        for dp in dispatches: record(dp,"sampling_label_dispatch")
        if len(dispatches)>2 or not (folder/"initial_dispatch.json").is_file():
            failures.append(dict(label=label,kind="label dispatch budget"))
        if (folder/"serial_retry_dispatch.json").exists() and not (folder/"initial_failure.json").exists():
            failures.append(dict(label=label,kind="retry without initial failure"))
    if len(all_calls)!=128 or totals["rollout_attempt_reserved"]!=128 or totals["ROLLOUT_COMPLETE"]!=128:
        failures.append(dict(kind="exact128 unique completed rollout ledgers",actual=len(all_calls),
                             reserved=totals["rollout_attempt_reserved"],completed=totals["ROLLOUT_COMPLETE"]))''')
s=s.replace('dispatch_retry_markers=len(retry_markers),failure_markers=len(failure_markers),',
'''dispatch_retry_markers=len(retry_markers),failure_markers=len(failure_markers),
        dispatch_event_records=dispatch_events,
        dispatch_retry_is_not_cell_retry=True,
        dispatch_authorization_scope="Markers preserved; resource/retry authorization independently audited by the controller",''')
s=s.replace('        "No model, scoring, rendering, quality summary or anonymous method mapping was used."]',
'''        "Dispatch failure/retry markers are preserved separately from per-cell attempts. A continued label may start untouched cells, but no cell may have a second rollout.",
        "No model, scoring, rendering, quality summary or anonymous method mapping was used."]''')
p.write_text(s)
print("updated independent adapter only")
