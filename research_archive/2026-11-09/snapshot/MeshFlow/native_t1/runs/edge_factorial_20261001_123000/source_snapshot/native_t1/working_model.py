"""Single working JEdge5 profile, without importing training losses."""
from .checkpoint import REPO, DEFAULT_CONFIG

WORKING_CHECKPOINT = REPO / "native_t1/checkpoints/jedge5_cumulative3000.pt"


def load_edge5(path=WORKING_CHECKPOINT, config_path=DEFAULT_CONFIG, *, device="cuda"):
    from .geometry_checkpoint import load_geometry_checkpoint
    return load_geometry_checkpoint(path, config_path, device=device, expected_recipe="edge5")
