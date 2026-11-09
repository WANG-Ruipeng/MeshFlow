# Guarded CGAL baseline

Implementation identity: **CGAL 6.1.1
Surface_mesh_simplification::GarlandHeckbert_plane_policies** with
discontinuity multiplier 100 (the default), plus the filters in
guarded_qem.cpp. CGAL 6.1.1 exposes classic-plane, triangle and probabilistic
quadric policies, but no public "plane_and_line" policy. This is deliberately
labelled classic-plane QEM, not a different policy or generic stock CGAL.

The closed input is the same F4096 mainfield mesh as VTK QEM. Native CGAL
edge-collapse link conditions preserve manifold topology. Positions and
quadrics use EPICK/double coordinates. Every evaluated candidate additionally:

1. Rejects nonfinite placement or exact collinearity of any surviving changed
   face.
2. Requires strictly positive old/new face-normal dot product.
3. Requires every changed face minimum angle >= the supplied Fixed minimum
   angle (1e-10 degree comparison tolerance).
4. Tests all prospective changed-vs-rest and changed-vs-changed triangle pairs,
   using bounding boxes and EPECK exact triangle intersection constructions.
   The intersection must be wholly contained in the topologically shared
   vertex/edge; nonadjacent contact and adjacent overlap are rejected.

The input and final output are also checked by CGAL PMP::does_self_intersect.
Those final checks are backend diagnostics; the experiment's common independent
endpoint audit remains required. There is no envelope filter, radial
reprojection, repair, retriangulation, restart or target-face coercion.

**Quality qualification is not automatic.** The minimum-angle filter is local:
pre-existing small triangles can persist unchanged. Global minimum and q05 of
per-face minimum angles are measured at the endpoint. The q05 threshold is
NOT enforced globally at every collapse. A target-unreachable run returns its
actual mesh and marks budget_pass false. A quality-failing run also keeps its
mesh. A 300-second native wall budget retains the best available *current*
mesh (not a retrospectively selected output); wrapper timeout is 360 seconds.

## Build and tests

Ubuntu packages installed for this experiment:
build-essential, cmake, libcgal-dev, libeigen3-dev (GMP/MPFR transitively).
PyTorch/CUDA/NVIDIA packages are unchanged.

From the independent worktree:

    cmake -S experiments/a_radial/layout_value/cgal \
      -B experiments/a_radial/layout_value/cgal/build -DCMAKE_BUILD_TYPE=Release
    cmake --build experiments/a_radial/layout_value/cgal/build --parallel 2
    python -m pytest -q -p no:cacheprovider experiments/a_radial/layout_value/test_cgal.py

Test coverage:
- Six exact contact/crossing cases: disjoint, valid shared edge/vertex,
  invalid coplanar overlap with shared edge/vertex, nonadjacent crossing.
- Analytic ellipsoid F320 -> F80: closed manifold, no final intersection,
  correct accepted-collapse counts.
- A deliberately unreachable 59.99 degree quality floor preserves actual
  geometry and face count, reporting failure to attain F20.

## Python interface and timing

    v_out, f_out, meta = run_cgal(
        v, f, target=512, min_angle_deg=fixed_min,
        angle_q05_deg=fixed_q05, work_dir=new_backend_dir)

The directory retains input.off, output.off, native_report.json, native.log,
runner_report.json. The wrapper refuses to overwrite a used run directory.
Metadata records source/binary SHA256, policy, precise guard coverage,
selected/accepted/rejected candidate counts, stop reason, actual face count,
local-filter time, native simplify time, both native audits, and end-to-end
wrapper time (process startup, OFF I/O and native audits included). Build time
is reported separately and excluded from wrapper export time. Experiment-level
outer timing should likewise warm/compile the executable before measurement.

Exact constructions certify tested intersections of stored finite coordinates;
they do not establish a universal numerical theorem for all inputs.
