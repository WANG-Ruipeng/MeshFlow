# Report and completion semantics revision 003

Only reporting.py and aggregation.py changed. Original sources, evaluation-source manifest, derived summary, main report and indices are preserved under originals/ with exact hashes. No training, sampling, control metric, old source, raw, or scored metric was altered.

A versus START now selects the seven START reference cost checks. It does not reuse A versus A interval comparisons, which could invent nonzero uncertainty for an identical quantity. Registered B/C/D quality gates are unchanged.

A fixed task with missing outputs/panels cannot issue a completed overall pilot verdict or promotion recommendation. Completed P1 screens remain available as provisional evidence. Once the task is complete, B then C are still preferred over D if they independently pass; descriptive A and retained START behavior remain.

Twenty-two CPU semantic checks pass, including reproducing the original premature recommendation and proving its removal. AST equality verifies that interval construction, per-parent units, contrasts, interaction arithmetic and the complete scientific gate are unchanged. This revision does not rerender partial tables or inspect model tensors. Root owns active manifest refresh and the separate execute.py final-audit failure status guard.
