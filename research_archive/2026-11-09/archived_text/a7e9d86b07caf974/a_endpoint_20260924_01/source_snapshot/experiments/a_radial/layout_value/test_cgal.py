"""Native guard and analytic mesh tests, independent of the sixteen objects."""
from pathlib import Path
import subprocess
import numpy as np
import trimesh
from .cgal_runner import build_cgal, run_cgal

def test_exact_guard_cases():
    binary, _ = build_cgal()
    p = subprocess.run([str(binary), "--self-test"], capture_output=True, text=True, check=True)
    assert '"intersection_selftest":"PASS"' in p.stdout

def test_analytic_ellipsoid_reduction(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    v = np.asarray(mesh.vertices) * [1.0, .8, .6]
    vo, fo, meta = run_cgal(v, mesh.faces, 80, 0., 0., tmp_path / "ordinary")
    result = trimesh.Trimesh(vo, fo, process=False)
    assert len(fo) == 80
    assert result.is_watertight and result.euler_number == 2
    assert not meta["final_self_intersection"]
    assert meta["filter_pass"] == meta["accepted_collapses"]
    assert meta["accepted_collapses"] == (len(mesh.faces) - len(fo)) // 2
    assert meta["collision_exact_pairs"] > 0
    assert meta["global_q05_enforced_per_step"] is False

def test_unreachable_quality_retains_mesh(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=1)
    vo, fo, meta = run_cgal(mesh.vertices, mesh.faces, 20, 59.99, 59.99, tmp_path / "unreachable")
    assert len(fo) > 20
    assert meta["budget_pass"] is False
    assert meta["reject_min_angle"] > 0
    assert len(vo) > 0 and meta["postrepair"] is False
