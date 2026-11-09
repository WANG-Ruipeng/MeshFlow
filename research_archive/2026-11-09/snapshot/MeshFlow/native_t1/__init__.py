"""Native in-context T1 with lazy model imports and optional NumPy-only cleanup."""
from ._runtime_environment import ensure_cublas_environment

ensure_cublas_environment()
__version__ = "0.1.0"


def __getattr__(name):
    # Preserve ``native_t1.runtime`` access without loading Torch for geometry IO.
    if name == "runtime":
        from importlib import import_module
        module = import_module(".runtime", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
