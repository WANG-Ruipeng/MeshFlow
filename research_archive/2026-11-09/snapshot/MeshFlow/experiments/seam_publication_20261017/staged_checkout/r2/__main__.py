"""Run the bundled package without depending on a sibling checkout."""
from pathlib import Path
import sys


def main(argv=None):
    source = Path(__file__).resolve().parent / "src"
    loaded = sys.modules.get("meshflow_control")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
        raise RuntimeError("A different meshflow_control package is already imported")
    sys.path.insert(0, str(source))
    from meshflow_control.r2_cli import main as run
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
