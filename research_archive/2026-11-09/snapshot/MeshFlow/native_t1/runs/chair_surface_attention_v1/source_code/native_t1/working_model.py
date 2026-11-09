"""Named local Geo profiles; FM is the default, Edge5 stays explicitly available."""
from .checkpoint import REPO, DEFAULT_CONFIG

FM_GEO_CHECKPOINT = REPO / "native_t1/checkpoints/ablations/fm_cumulative3000.pt"
EDGE5_CHECKPOINT = REPO / "native_t1/checkpoints/jedge5_cumulative3000.pt"
DEFAULT_WORKING_CHECKPOINT = FM_GEO_CHECKPOINT
# Backward-compatible name from the former Edge5-default workflow.
# New default routing must use DEFAULT_WORKING_CHECKPOINT explicitly.
WORKING_CHECKPOINT = EDGE5_CHECKPOINT


def load_fm_geo(path=DEFAULT_WORKING_CHECKPOINT, config_path=DEFAULT_CONFIG, *, device="cuda"):
    from .geometry_checkpoint import load_geometry_checkpoint
    return load_geometry_checkpoint(path, config_path, device=device, expected_recipe="fm")


def load_edge5(path=EDGE5_CHECKPOINT, config_path=DEFAULT_CONFIG, *, device="cuda"):
    from .geometry_checkpoint import load_geometry_checkpoint
    return load_geometry_checkpoint(path, config_path, device=device, expected_recipe="edge5")
