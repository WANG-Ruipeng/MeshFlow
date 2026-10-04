"""CPU-only import boundary checks for the standalone Native T1 package.

Run from the repository root:
    python -m unittest discover -s native_t1/tests -p test_isolation.py -v

No checkpoint, model forward, CUDA initialization, sampling, or training occurs.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parent
LEGACY_PREFIXES = (
    "experiments", "adapter", "adapters", "control_adapter", "s32_adapter", "s32_",
    "guidance", "guided_native", "gpu_guidance", "soft_matching",
    "native_patch", "native_inpainting", "nxc112", "official_wrapper",
    "stable_runtime", "prompt_dataset",
)


def _forbidden(name: str) -> bool:
    return any(part == prefix or part.startswith(prefix + "_") or
               (prefix.endswith("_") and part.startswith(prefix))
               for part in name.split(".") for prefix in LEGACY_PREFIXES)


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def inspect_source(text: str, filename: str) -> list[str]:
    """Inspect import statements and common dynamic-import escape routes."""
    tree = ast.parse(text, filename=filename)
    problems = []
    aliases = {}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            for alias in node.names:
                qualified = ".".join(filter(None, (node.module, alias.name)))
                names.append(qualified)
                aliases[alias.asname or alias.name] = qualified
            if node.level > len(Path(filename).with_suffix("").parts):
                problems.append(f"{filename}:{node.lineno}: relative import escapes package")
        for name in names:
            if _forbidden(name):
                problems.append(f"{filename}:{node.lineno}: legacy import {name}")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = _dotted(node.func)
        if not called:
            continue
        head, *tail = called.split(".")
        called = ".".join([aliases.get(head, head), *tail])
        if called in ("importlib.import_module", "__import__", "builtins.__import__"):
            if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                problems.append(f"{filename}:{node.lineno}: nonliteral dynamic import is not auditable")
            elif _forbidden(node.args[0].value):
                problems.append(f"{filename}:{node.lineno}: legacy dynamic import {node.args[0].value}")
        if called in ("importlib.util.spec_from_file_location", "importlib.machinery.SourceFileLoader",
                      "runpy.run_path", "runpy.run_module", "exec", "eval"):
            problems.append(f"{filename}:{node.lineno}: dynamic code/file loader {called}")
        if called.startswith("sys.path."):
            problems.append(f"{filename}:{node.lineno}: import search path mutation {called}")
    return problems


def _sources():
    return sorted(path for path in PACKAGE.rglob("*.py")
                  if not ({"tests", "runs", "validation", "maintenance", "reviews", "analysis", "checkpoints", "data", "__pycache__"}
                          & set(path.relative_to(PACKAGE).parts)))


class IsolationTests(unittest.TestCase):
    def test_static_import_boundary(self):
        sources = _sources()
        self.assertTrue(sources, "No production native_t1 source files found")
        problems = []
        for path in sources:
            problems.extend(inspect_source(path.read_text(encoding="utf-8-sig"), str(path.relative_to(PACKAGE))))
        self.assertEqual(problems, [], "\n".join(problems))

    def test_detector_catches_direct_relative_and_dynamic_legacy_imports(self):
        cases = (
            "import experiments.mf_h1_local0.native_patch_run",
            "from experiments import mf_h1_local0",
            "from . import control_adapter",
            "import importlib as imp\nimp.import_module('soft_matching_energy')",
            "from importlib import import_module as get\nget('gpu_guidance_energies')",
            "__import__('experiments')",
            "import importlib\nimportlib.import_module(module_name)",
            "import importlib.util\nimportlib.util.spec_from_file_location('x', 'legacy.py')",
            "import sys as system\nsystem.path.insert(0, 'legacy')",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assertTrue(inspect_source(source, "probe.py"))

    def test_detector_allows_package_official_and_asset_paths(self):
        source = ("from .model import NativeInpaintingModel\n"
                  "from native_t1.runtime import configure_runtime\n"
                  "from models.equidit import DiT\n"
                  "import numpy as np\n"
                  "checkpoint = 'experiments/historical/endpoint_step500.pt'\n")
        self.assertEqual(inspect_source(source, "probe.py"), [])

    def test_fresh_process_imports_do_not_load_experiments_or_initialize_cuda(self):
        sources = _sources()
        self.assertTrue((PACKAGE / "__init__.py").is_file(), "native_t1 must be a concrete package")
        names = []
        for path in sources:
            relative = path.relative_to(REPOSITORY).with_suffix("")
            parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
            names.append(".".join(parts))
        names = sorted(set(names), key=lambda name: (name.count("."), name))
        worker = r'''
import importlib, importlib.abc, json, sys
import torch
forbidden = json.loads(sys.argv[2])
def legacy(name):
    return any(part == prefix or part.startswith(prefix + '_') or
               (prefix.endswith('_') and part.startswith(prefix))
               for part in name.split('.') for prefix in forbidden)
assert not torch.cuda.is_initialized(), 'CUDA initialized while importing torch'
assert not any(legacy(name) for name in sys.modules), 'Legacy module present before package import'
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if legacy(fullname):
            raise AssertionError('Forbidden legacy module import: ' + fullname)
        return None
sys.meta_path.insert(0, Boundary())
def forbidden_operation(*args, **kwargs):
    raise AssertionError('Import attempted checkpoint loading or CUDA initialization')
torch.load = forbidden_operation
torch.cuda._lazy_init = forbidden_operation
for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
bad = sorted(name for name in sys.modules if legacy(name))
assert not bad, bad
assert not torch.cuda.is_initialized(), 'Package import initialized CUDA'
print(json.dumps({'status':'PASS', 'imported_modules':json.loads(sys.argv[1]),
                  'legacy_modules':bad, 'CUDA_initialized':False,
                  'checkpoint_loads':0, 'model_forwards':0, 'optimizer_updates':0}))
'''
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        completed = subprocess.run([sys.executable, "-B", "-c", worker,
                                    json.dumps(names), json.dumps(LEGACY_PREFIXES)],
                                   cwd=REPOSITORY, env=env, capture_output=True, text=True,
                                   timeout=60, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        record = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["legacy_modules"], [])
        self.assertFalse(record["CUDA_initialized"])


if __name__ == "__main__":
    unittest.main()
