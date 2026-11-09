"""Numerically guarded PoNQ adapter; the vendored extractor is unchanged.

Only roundoff-sized arccos domain excess is clipped. Quadric terms and graph
capacities are never clipped. A finite nonnegative *aggregated* DiGraph is
required before calling the original min_cut implementation.
"""
from pathlib import Path
import copy
import sys
import numpy as np
import torch
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor'))
from PoNQ_to_mesh import MeshFromPoNQ

BACKEND_VERSION = 'PONQ_CLAY_SAFE_EXTRACT_V1'
COS_TOLERANCE = 64 * np.finfo(np.float64).eps


class SafeExtractionError(RuntimeError):
    def __init__(self, message, audit=None, trace_arrays=None):
        super().__init__(message)
        self.audit = copy.deepcopy(audit or {})
        self.trace_arrays = {k: np.array(v, copy=True) for k, v in (trace_arrays or {}).items()}


def guarded_arccos(dots):
    """Return (angles, audit); refuse invalid or materially out-of-domain input."""
    dots = np.asarray(dots, dtype=np.float64)
    record = dict(tolerance=COS_TOLERANCE, scalar_count=int(dots.size),
                  nonfinite_count=int(np.sum(~np.isfinite(dots))),
                  clipped_count=0, max_domain_excess=None)
    if not np.isfinite(dots).all():
        raise SafeExtractionError('Nonfinite normalized normal dot product', record)
    excess = np.maximum(np.abs(dots) - 1., 0.)
    record.update(max_domain_excess=float(excess.max()) if excess.size else 0.,
                  minimum=float(dots.min()) if dots.size else None,
                  maximum=float(dots.max()) if dots.size else None,
                  clipped_count=int(np.count_nonzero(excess)))
    if np.any(excess > COS_TOLERANCE):
        raise SafeExtractionError('Normal dot product exceeds the 64eps arccos boundary', record)
    return np.arccos(np.clip(dots, -1., 1.)), record


def _score_stats(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return dict(count=int(values.size), nonfinite_count=int((~np.isfinite(values)).sum()),
                negative_count=int((values < 0).sum()),
                minimum_finite=float(finite.min()) if finite.size else None,
                maximum_finite=float(finite.max()) if finite.size else None)


class SafeMeshFromPoNQ(MeshFromPoNQ):
    def __init__(self, *args, **kwargs):
        self.trace_arrays = {}
        self.audit = dict(backend_version=BACKEND_VERSION, arccos_tolerance=COS_TOLERANCE,
                          arccos_calls=[], quadric_repair_performed=False,
                          capacity_repair_performed=False, original_mincut_called=False)
        try:
            super().__init__(*args, **kwargs)
        except Exception as exc:
            details = self.audit.copy()
            details.update(status='FAILED', error=str(exc), error_type=type(exc).__name__)
            if isinstance(exc, SafeExtractionError):
                details['failure_detail'] = exc.audit
            raise SafeExtractionError(str(exc), details, self.trace_arrays) from exc

    def get_init_tet_color(self):
        result = super().get_init_tet_color()
        self.trace_arrays.update(
            delaunay_points=self.points.copy(), delaunay_simplices=self.simplices.copy(),
            delaunay_neighbors=self.neighbors.copy(), delaunay_coplanar=self.coplanar.copy(),
            circumcenters=self.circum_centers.copy(), initial_labels=result.copy())
        return result

    def correct_tet_color(self):
        before = self.tet_colors.copy()
        super().correct_tet_color()
        self.trace_arrays['corrected_labels'] = self.tet_colors.copy()
        self.trace_arrays['correction_changed_tetrahedra'] = np.flatnonzero(before != self.tet_colors)

    def add_void_vertices(self):
        if 'corrected_labels' not in self.trace_arrays:
            self.trace_arrays['corrected_labels'] = self.tet_colors.copy()
        before = self.tet_colors.copy()
        super().add_void_vertices()
        self.trace_arrays['void_labels'] = self.tet_colors.copy()
        self.trace_arrays['void_changed_tetrahedra'] = np.flatnonzero(before != self.tet_colors)

    def get_face_normal_align(self):
        # Keep the arithmetic/order of the original method through the dot product.
        vertex_normals = self.normals[self.triangle_faces].cpu().detach().numpy()
        if not np.isfinite(vertex_normals).all():
            raise SafeExtractionError('Nonfinite input normal')
        lengths = np.sqrt((vertex_normals ** 2).sum(-1))
        if not np.isfinite(lengths).all() or np.any(lengths == 0):
            raise SafeExtractionError('Zero or nonfinite input-normal length')
        vertex_normals /= lengths[..., None]
        face_normals = np.cross(
            self.points[self.triangle_faces[:, 1]] - self.points[self.triangle_faces[:, 0]],
            self.points[self.triangle_faces[:, 2]] - self.points[self.triangle_faces[:, 0]])
        face_lengths = np.sqrt((face_normals ** 2).sum(-1, keepdims=True))
        if not np.isfinite(face_lengths).all() or np.any(face_lengths == 0):
            raise SafeExtractionError('Zero or nonfinite candidate-face normal length')
        face_normals /= face_lengths
        dots = (face_normals[:, None, :] * vertex_normals).sum(-1)
        self.trace_arrays['normal_dot_products_unclipped'] = dots.copy()
        angles, audit = guarded_arccos(dots)
        audit['faces_with_clipped_dot'] = int(np.any(np.abs(dots) > 1., axis=1).sum())
        self.audit['arccos_calls'].append(audit)
        return angles.sum(-1)

    def min_cut(self):
        # Identical assembly to original min_cut lines 159-178; no objective repair.
        scores = self.final_scores
        colors = self.tet_colors.astype(int)
        tet_edges = self.triangle_faces_neighbors.copy()
        non_inf = (tet_edges != -1).all(-1)
        inside = np.flatnonzero(colors == 4)
        outside = np.flatnonzero(colors == 0)
        if not len(inside) or not len(outside):
            raise SafeExtractionError('Missing inside or outside terminal seed')
        min_in, min_out = int(inside.min()), int(outside.min())
        tet_edges[colors[tet_edges] == 4] = min_in
        tet_edges[colors[tet_edges] == 0] = min_out
        relevant = (tet_edges[:, 0] != tet_edges[:, 1]) * non_inf
        edges = np.fliplr(tet_edges[relevant])
        r_scores = scores[relevant]
        network = nx.DiGraph()
        network.add_nodes_from(np.unique(edges))
        for e, score in zip(edges, r_scores):
            if network.has_edge(e[0], e[1]):
                network[e[0]][e[1]]['capacity'] += score
            else:
                network.add_edge(e[0], e[1], capacity=score)
        graph_edges = np.array(list(network.edges()), dtype=np.int64).reshape(-1, 2)
        capacities = np.array([network[u][v]['capacity'] for u, v in graph_edges], dtype=np.float64)
        self.trace_arrays.update(
            precut_labels=colors.copy(), inside_seed_ids=inside, outside_seed_ids=outside,
            free_tetrahedron_ids=np.flatnonzero((colors != 0) & (colors != 4)),
            candidate_faces=self.triangle_faces.copy(),
            candidate_face_neighbors=self.triangle_faces_neighbors.copy(),
            candidate_scores=np.array(scores, copy=True), relevant_mask=relevant.copy(),
            relevant_directed_edges=edges.copy(), relevant_directed_scores=np.array(r_scores, copy=True),
            graph_nodes=np.array(list(network.nodes()), dtype=np.int64),
            graph_edges=graph_edges, graph_capacities=capacities,
            graph_terminals=np.array([min_in, min_out], dtype=np.int64))
        self.audit.update(candidate_score_stats=_score_stats(scores),
                          relevant_score_stats=_score_stats(r_scores),
                          graph_capacity_stats=_score_stats(capacities),
                          free_tetrahedron_count=int(np.sum((colors != 0) & (colors != 4))),
                          source=min_in, sink=min_out)
        if not np.isfinite(capacities).all() or np.any(capacities < 0):
            raise SafeExtractionError('Actual aggregated graph has nonfinite or negative capacity')
        self.audit['original_mincut_called'] = True
        super().min_cut()
        self.trace_arrays['postcut_labels'] = self.tet_colors.copy()
        source_side = self.tet_colors[graph_edges[:, 0]] == 4
        sink_side = self.tet_colors[graph_edges[:, 1]] == 0
        self.trace_arrays['cut_graph_edge_mask'] = source_side & sink_side
        self.audit.update(
            cut_energy_from_final_partition=float(capacities[source_side & sink_side].sum()),
            inside_seeds_preserved=bool(np.all(self.tet_colors[inside] == 4)),
            outside_seeds_preserved=bool(np.all(self.tet_colors[outside] == 0)))


def extract(params, *, grid_scale=32):
    """Return (vertices, faces, input_vertex_ids, trace_arrays, audit).

    Input params require v/n: (N,3), Q: (N,4,4), eigs: (N,3). Input sites and
    normals are copied without repair; extra fitting/encoding fields are ignored.
    The returned IDs index Delaunay inputs (original elements then eight corners).
    Failures raise SafeExtractionError carrying available audit and trace_arrays.
    """
    arrays = {}
    try:
        count = len(params['v'])
        for key, shape in dict(v=(count, 3), n=(count, 3), Q=(count, 4, 4), eigs=(count, 3)).items():
            value = np.array(params[key], dtype=np.float64, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise SafeExtractionError('Invalid shape or nonfinite input: '+key)
            arrays[key] = value
        if count < 4 or np.any(arrays['eigs'][:, 0] <= 0):
            raise SafeExtractionError('At least four sites and positive normalization eigenvalues required')
        if np.any(np.linalg.norm(arrays['n'], axis=1) == 0):
            raise SafeExtractionError('Zero input normal')
        backend = SafeMeshFromPoNQ(
            torch.from_numpy(arrays['v']), torch.from_numpy(arrays['eigs']),
            torch.from_numpy(arrays['Q']), torch.from_numpy(arrays['n']),
            grid_scale=grid_scale, add_corners=True, correct_tet_color=True, compute_mincut=True)
        v, f, ids = backend.get_surface(return_indices=True)
        backend.trace_arrays['surface_input_vertex_ids'] = ids.copy()
        backend.trace_arrays['surface_faces_input_ids'] = ids[f].copy()
        backend.audit.update(status='EXTRACTED', input_elements=count,
                             output_vertices=int(len(v)), output_faces=int(len(f)),
                             anchor_vertices_in_output=int(np.sum(ids >= count)),
                             arccos_clipped_count=sum(a['clipped_count'] for a in backend.audit['arccos_calls']))
        return (v.copy(), f.copy(), ids.copy(),
                {k: np.array(a, copy=True) for k, a in backend.trace_arrays.items()},
                copy.deepcopy(backend.audit))
    except SafeExtractionError:
        raise
    except Exception as exc:
        if 'backend' in locals():
            raise SafeExtractionError(str(exc), backend.audit, backend.trace_arrays) from exc
        raise SafeExtractionError(str(exc), dict(backend_version=BACKEND_VERSION, status='FAILED')) from exc
