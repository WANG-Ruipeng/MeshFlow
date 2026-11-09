# Legacy collector superseded

The original frozen `evaluation.py` collector is retained unchanged. Its
`per_output.csv` overwrites the original whole-mesh `boundary_edges` score with
the input C boundary-edge count; its aggregate/pair metric list also excludes
that score. The original per-cell `row.json` files retain the correct score.
No model output or per-cell scoring is affected.

The independent `collect_v2.py` reads all 672 original rows. It preserves the
whole-mesh `boundary_edges` value and adds the input count as `C_boundary_edges`.
The original whole-mesh count participates in V2 aggregate and paired tables.

The authoritative corrected tables are `../results_v2/` once that directory's
`completion.json` reports `COMPLETE`. Until then V2 is incomplete; this notice
does not certify completed collection. Legacy files here remain unchanged.
See `../diagnostics/collector_v2_revision.json` for the engineering record.
