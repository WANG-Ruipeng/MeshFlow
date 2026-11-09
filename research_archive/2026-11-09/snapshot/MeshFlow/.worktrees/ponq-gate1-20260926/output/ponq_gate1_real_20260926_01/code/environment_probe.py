"""Print installed dependency versions without modifying the environment."""
import importlib
import json
import platform
import sys

result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
for name in ("numpy", "scipy", "torch", "trimesh", "networkx", "shapely", "pandas", "matplotlib"):
    try:
        module = importlib.import_module(name)
        result[name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        result[name] = {"error": repr(exc)}
print(json.dumps(result, indent=2))
