"""Resource-only dispatch receipt; no model/scoring call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from r2.pilots.bench_backend_relative_deformation_v1.common import resource_snapshot, write, OUT
write(OUT/'audits/resource_wrapper_note.json', dict(
    event='PRE_DISPATCH_SHELL_QUOTING_ERROR',
    reason='Nested PowerShell/WSL -c quoting rejected by bash before Python ran; use this script entry instead.',
    model_calls=0, scoring_calls=0, geometry_calls=0, resource_helper_executed_on_failed_attempt=False))
resource_snapshot('cpu_backend_dispatch')
