"""CPU mathematical/adapter checks, without instantiating an official backbone.

Run from the repository root:
    python -m experiments.mf_h1_local0.tests.run_units --output OUTPUT.json

Reports are created exclusively (never overwritten). A failed check remains in
its report; reruns must use a new output path. No Phase 2 updates occur here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from torch import nn

from ..adapters import ARMS, ControlAdapter, point_features
from ..geometry import (closest_point_segment, closest_point_triangle,
                        closest_points_primitives, primitive_centroids, relation_features)
from .geometry_oracle import segment_closest, triangle_closest


ATOL = 1e-5
RTOL = 1e-5
GEOMETRY_ATOL = 1e-10
MODELS = []
COUNTERS = {"cpu_probe_optimizer_updates": 0, "official_forward_calls": 0,
            "official_complete_samples": 0, "phase2_optimizer_updates": 0}


class Recorder:
    def __init__(self):
        self.results = []

    def compare(self, name, actual, expected, inputs, atol=ATOL, rtol=RTOL, **extra):
        a, b = torch.as_tensor(actual).detach().double(), torch.as_tensor(expected).detach().double()
        if a.shape != b.shape:
            self.results.append(dict(name=name, inputs=inputs, actual_error="shape_mismatch",
                                     actual_shape=list(a.shape), expected_shape=list(b.shape),
                                     threshold=dict(atol=atol, rtol=rtol), status="FAIL", **extra))
            return
        difference = (a - b).abs()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        absolute = float(difference.max()) if difference.numel() else 0.0
        relative = float((difference / b.abs().clamp_min(1e-30)).max()) if difference.numel() else 0.0
        passed = finite and bool((difference <= atol + rtol * b.abs()).all())
        self.results.append(dict(name=name, inputs=inputs,
                                 actual_error=dict(max_abs=absolute, max_rel=relative, finite=finite),
                                 threshold=dict(atol=atol, rtol=rtol),
                                 status="PASS" if passed else "FAIL", **extra))

    def predicate(self, name, passed, observed, inputs, threshold, **extra):
        self.results.append(dict(name=name, inputs=inputs, actual_error=observed,
                                 threshold=threshold, status="PASS" if passed else "FAIL", **extra))

    def run(self, label, function):
        started = time.perf_counter()
        try:
            function(self)
        except Exception as exc:
            self.results.append(dict(name=label, status="FAIL", inputs={"test_group": label},
                                     actual_error={"exception": repr(exc), "traceback": traceback.format_exc()},
                                     threshold={"exceptions": 0}))
        print(json.dumps({"finished_group": label, "seconds": time.perf_counter()-started,
                          "checks": len(self.results),
                          "failures": sum(item["status"] == "FAIL" for item in self.results)}), flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def adapter(arm, seed=10, nonzero=False, query_chunk=4):
    torch.manual_seed(seed)
    model = ControlAdapter(arm, width=768, layers=4, query_chunk=query_chunk).cpu()
    if nonzero:
        generator = torch.Generator().manual_seed(921)
        with torch.no_grad():
            for layer in model.layers:
                layer.output_projection.weight.copy_(torch.randn(layer.output_projection.weight.shape,
                                                                 generator=generator) * 0.03)
                layer.output_projection.bias.copy_(torch.randn(layer.output_projection.bias.shape,
                                                               generator=generator) * 0.01)
    MODELS.append(model)
    return model


def fixture(kind="mixed", batch=1, dtype=torch.float32):
    vertices = torch.tensor([[[[-.7, -.2, .1], [.6, -.1, .2], [0, 0, 0]],
                              [[-.5, .4, -.2], [.4, .6, .3], [0, 0, 0]],
                              [[-.3, -.1, -.4], [.8, .1, -.2], [.1, .9, .3]]]], dtype=dtype)
    arity = torch.tensor([[2, 2, 3]])
    if kind == "edge":
        vertices, arity, counts = vertices[:, :1], arity[:, :1], [32]
    elif kind == "face":
        vertices, arity, counts = vertices[:, 2:], arity[:, 2:], [32]
    else:
        counts = [8, 8, 16]
    points, groups = [], []
    generator = torch.Generator().manual_seed(228)
    for index, count in enumerate(counts):
        v = vertices[0, index]
        if int(arity[0, index]) == 2:
            parameters = torch.linspace(0, 1, count, dtype=dtype)
            values = (1-parameters[:, None])*v[0] + parameters[:, None]*v[1]
        else:
            coordinates = torch.rand((count-3, 2), generator=generator, dtype=dtype)
            root = coordinates[:, :1].sqrt()
            barycentric = torch.cat((1-root, root*(1-coordinates[:, 1:]), root*coordinates[:, 1:]), -1)
            values = torch.cat((v, barycentric @ v), 0)
        points.append(values)
        groups += [index]*count
    return {"vertices": vertices.repeat(batch, 1, 1, 1), "arity": arity.repeat(batch, 1),
            "primitive_mask": torch.ones((batch, len(counts)), dtype=torch.bool),
            "points": torch.cat(points).unsqueeze(0).repeat(batch, 1, 1),
            "point_primitive": torch.tensor([groups]).repeat(batch, 1),
            "point_mask": torch.ones((batch, 32), dtype=torch.bool)}


def query(batch=1, corners=9):
    generator = torch.Generator().manual_seed(190)
    return (torch.randn((batch, corners, 768), generator=generator),
            torch.randn((batch, corners, 3), generator=generator), torch.full((batch,), .37))


def clone_prompt(prompt):
    return {key: value.clone() for key, value in prompt.items()}


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.detach().cpu().contiguous()).tobytes())
    return digest.hexdigest()


def json_safe(value):
    """A numerical failure must still produce a valid, permanent JSON report."""
    if isinstance(value, dict):
        return {key: json_safe(member) for key, member in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(member) for member in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def geometry_cases(rec):
    segments = [
        ("interior", [.3, .4, 0], [0, 0, 0], [1, 0, 0], [.3, 0, 0]),
        ("before_endpoint", [-2, .4, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]),
        ("after_endpoint", [3, .4, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]),
        ("repeated_point", [2, 3, 4], [.1, .2, .3], [.1, .2, .3], [.1, .2, .3]),
        ("far_query", [10000, 20000, -30000], [0, 0, 0], [1, 0, 0], [1, 0, 0]),
    ]
    for name, point, a, b, expected in segments:
        actual = closest_point_segment(*(torch.tensor(value, dtype=torch.float64) for value in (point, a, b)))
        rec.compare("geometry_segment_"+name, actual, torch.tensor(expected, dtype=torch.float64),
                    {"point": point, "a": a, "b": b, "dtype": "float64"}, GEOMETRY_ATOL, 0)
    triangles = [
        ("interior_projection", [.2, .3, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [.2, .3, 0]),
        ("edge_boundary", [.8, .8, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [.5, .5, 0]),
        ("nearest_vertex", [2, -.3, .2], [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]),
        ("repeated_vertex", [.3, .2, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0], [.3, 0, 0]),
        ("all_repeated", [2, 3, 4], [.1, .2, .3], [.1, .2, .3], [.1, .2, .3], [.1, .2, .3]),
        ("collinear", [1.3, .2, .1], [0, 0, 0], [1, 0, 0], [2, 0, 0], [1.3, 0, 0]),
        ("near_degenerate", [.7, .1, 1], [0, 0, 0], [1, 0, 0], [.5, 1e-12, 0], None),
        ("subresolution_degenerate", [.7, .1, 1], [0, 0, 0], [1, 0, 0], [.5, 1e-17, 0], None),
        ("far_query", [10000, 20000, -30000], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]),
    ]
    for name, point, a, b, c, expected in triangles:
        inputs = {"point": point, "vertices": [a, b, c], "dtype": "float64"}
        tensors = [torch.tensor(value, dtype=torch.float64) for value in (point, a, b, c)]
        actual = closest_point_triangle(*tensors)
        if expected is None:
            expected = triangle_closest(point, a, b, c)
        rec.compare("geometry_triangle_"+name, actual, torch.as_tensor(expected, dtype=torch.float64), inputs,
                    GEOMETRY_ATOL, 0)
        permutations = torch.stack([closest_point_triangle(tensors[0], *(tensors[i+1] for i in permutation))
                                    for permutation in itertools.permutations(range(3))])
        rec.compare("geometry_triangle_six_permutations_"+name, permutations,
                    actual.expand_as(permutations), inputs, GEOMETRY_ATOL, 0)


def geometry_random(rec):
    generator = np.random.default_rng(20260928)
    count = 1000
    points = generator.normal(size=(count, 3))*3
    vertices = generator.normal(size=(count, 3, 3))
    expected_segment = np.stack([segment_closest(p, v[0], v[1]) for p, v in zip(points, vertices)])
    expected_triangle = np.stack([triangle_closest(p, *v) for p, v in zip(points, vertices)])
    p, v = torch.tensor(points), torch.tensor(vertices)
    actual_segment = closest_point_segment(p, v[:, 0], v[:, 1])
    actual_triangle = closest_point_triangle(p, v[:, 0], v[:, 1], v[:, 2])
    inputs = {"seed": 20260928, "cases": count, "dtype": "float64", "device": "cpu",
              "point_distribution": "N(0,3^2)", "vertex_distribution": "N(0,1)",
              "oracle": "independent NumPy scalar active-set enumeration + lstsq"}
    rec.compare("geometry_random_segments_independent_oracle", actual_segment, expected_segment,
                inputs, GEOMETRY_ATOL, 0)
    rec.compare("geometry_random_triangles_independent_oracle", actual_triangle, expected_triangle,
                inputs, GEOMETRY_ATOL, 0)
    rec.compare("geometry_random_segment_reversal", closest_point_segment(p, v[:, 1], v[:, 0]),
                actual_segment, inputs, GEOMETRY_ATOL, 0)
    for permutation in itertools.permutations(range(3)):
        permuted = closest_point_triangle(p, *(v[:, i] for i in permutation))
        rec.compare("geometry_random_triangle_permutation_"+"".join(map(str, permutation)), permuted,
                    actual_triangle, inputs, GEOMETRY_ATOL, 0)
    prompts = fixture(dtype=torch.float64, batch=2)
    x = torch.tensor(generator.normal(size=(2, 13, 3)))
    actual = closest_points_primitives(x, prompts["vertices"], prompts["arity"])
    expected = np.empty((2, 13, 3, 3))
    for batch in range(2):
        for q in range(13):
            for primitive in range(3):
                primitive_vertices = prompts["vertices"][batch, primitive].numpy()
                expected[batch, q, primitive] = (segment_closest(x[batch, q], *primitive_vertices[:2])
                    if prompts["arity"][batch, primitive] == 2 else
                    triangle_closest(x[batch, q], *primitive_vertices))
    rec.compare("geometry_batched_mixed_broadcast_oracle", actual, expected,
                {"B": 2, "Q": 13, "K": 3, "arity": [2, 2, 3], "seed": 20260928}, GEOMETRY_ATOL, 0)
    padded = clone_prompt(prompts)
    padded["vertices"][:, :2, 2] = float("nan")
    rec.compare("geometry_edge_third_vertex_ignored", closest_points_primitives(x, padded["vertices"], padded["arity"]),
                actual, {"edge_third_slot": "NaN", "B": 2, "Q": 13, "K": 3}, GEOMETRY_ATOL, 0)
    centers = primitive_centroids(prompts["vertices"], prompts["arity"])
    rec.compare("geometry_centroid_ignores_edge_third_slot", primitive_centroids(padded["vertices"], padded["arity"]),
                centers, {"edge_third_slot": "NaN", "arity": [2, 2, 3]}, GEOMETRY_ATOL, 0)


def architecture_and_zero(rec):
    models = [adapter(arm) for arm in ARMS]
    reference = models[1].state_dict()
    counts = {model.arm: sum(p.numel() for p in model.parameters()) for model in models}
    for model in models[2:]:
        state = model.state_dict()
        same_shapes = list(state) == list(reference) and all(state[k].shape == reference[k].shape for k in reference)
        max_difference = max(float((state[k]-reference[k]).abs().max()) for k in reference) if same_shapes else float("inf")
        rec.predicate("identical_qgp_initial_state_"+model.arm, same_shapes and max_difference == 0,
                      {"max_tensor_abs": max_difference, "shape_match": same_shapes, "parameter_counts": counts},
                      {"seed": 10, "width": 768, "layers": 4, "initialization": "same RNG seed per arm"},
                      {"state_keys_shapes_values": "identical", "max_abs": 0})
    rec.predicate("static_has_no_dummy_dynamic_parameters", counts["S_static"] < counts["Q_center"] and
                  not any("dynamic_mlp" in name for name, _ in models[0].named_parameters()),
                  counts, {"arms": list(ARMS)}, {"static_count": "strictly smaller", "unused_dummy_parameters": 0})
    h, x, t = query()
    prompts = fixture()
    for model in models:
        encoded = model.encode(prompts)
        for index, layer in enumerate(model.layers):
            residual = model.forward_layer(index, h, x, t, encoded)
            rec.compare(f"zero_output_{model.arm}_layer{index}", residual, torch.zeros_like(residual),
                        {"seed": 10, "B": 1, "Q": 9, "K": 3, "width": 768}, 0, 0)
            magnitude = float(layer.aggregate(h, x, t, encoded).abs().max())
            rec.predicate(f"only_exit_zero_{model.arm}_layer{index}", magnitude > 1e-6,
                          {"preprojection_max_abs": magnitude}, {"same_fixture_as": f"zero_output_{model.arm}_layer{index}"},
                          {"preprojection_max_abs_gt": 1e-6})


def single_token(rec):
    h, x, t = query()
    prompts = fixture("edge")
    for arm in ("S_static", "Q_center", "G_closest"):
        model = adapter(arm)
        encoded = model.encode(prompts)
        layer = model.layers[0]
        message = layer.aggregate(h, x, t, encoded)
        magnitude = float(message.abs().max())
        spread = float((message-message[:, :1]).abs().max())
        rec.predicate("single_token_message_nonzero_"+arm, magnitude > 1e-6,
                      {"message_max_abs": magnitude}, {"K": 1, "Q": 9, "seed": 10}, {"message_max_abs_gt": 1e-6})
        if arm == "S_static":
            rec.compare("single_token_static_query_independence", message, message[:, :1].expand_as(message),
                        {"K": 1, "distinct_hidden_and_xt": 9, "location": "before zero output projection"}, 0, 0)
            q = layer.query_projection(layer.query_norm(h)).reshape(1, 9, 4, 32)
            k = layer.key_projection(encoded.tokens).reshape(1, 1, 4, 32)
            logits = torch.einsum("bqhd,bahd->bhqa", q, k)/np.sqrt(32)
            distance_bias = -((x.unsqueeze(2)-encoded.centers.unsqueeze(1))**2).sum(-1)[:, None]
            alpha = (logits+distance_bias).softmax(-1)
            value = layer.value_projection(encoded.tokens).reshape(1, 1, 4, 32)
            biased = torch.einsum("bhqa,bahd->bqhd", alpha, value).flatten(-2)
            rec.compare("single_token_logits_only_distance_bias_no_effect", biased, message,
                        {"K": 1, "bias": "-squared_distance", "no_gate_or_null": True}, 0, 0)
        else:
            rec.predicate("single_token_dynamic_query_allowed_"+arm, spread > 1e-6,
                          {"max_query_message_difference": spread},
                          {"K": 1, "Q": 9, "distinct_h_and_xt": True, "not_a_quality_claim": True},
                          {"max_abs_difference_gt": 1e-6})
            constant_h = h[:, :1].expand_as(h)
            by_xt = layer.aggregate(constant_h, x, t, encoded)
            xt_difference = float((by_xt-by_xt[:, :1]).abs().max())
            rec.predicate("single_token_current_xt_dependence_"+arm, xt_difference > 1e-6,
                          {"max_message_difference": xt_difference}, {"h": "constant", "xt": "distinct", "t": .37},
                          {"max_abs_difference_gt": 1e-6})
            by_h = layer.aggregate(h, x[:, :1].expand_as(x), t, encoded)
            h_difference = float((by_h-by_h[:, :1]).abs().max())
            rec.predicate("single_token_hidden_dependence_"+arm, h_difference > 1e-6,
                          {"max_message_difference": h_difference}, {"h": "distinct", "xt": "constant", "t": .37},
                          {"max_abs_difference_gt": 1e-6})
            changed_t = layer.aggregate(h, x, torch.full_like(t, .71), encoded)
            t_difference = float((changed_t-message).abs().max())
            rec.predicate("single_token_time_dependence_"+arm, t_difference > 1e-6,
                          {"max_message_difference": t_difference}, {"h_xt": "fixed", "t": [.37, .71]},
                          {"max_abs_difference_gt": 1e-6})


def geometry_message_contract(rec):
    h, x, t = query()
    prompt = fixture()
    captured = {}
    for arm in ("Q_center", "G_closest"):
        model = adapter(arm)
        values = []
        handle = model.layers[0].dynamic_mlp.register_forward_pre_hook(
            lambda _module, args: values.append(args[0].detach().clone()))
        model.layers[0].aggregate(h, x, t, model.encode(prompt))
        handle.remove()
        captured[arm] = torch.cat(values, 1)
    q_value, g_value = captured["Q_center"], captured["G_closest"]
    rec.compare("qg_only_relation_input_changes_prefix", q_value[..., :256], g_value[..., :256],
                {"input_parts": "q(128), e(128)", "shared_seed": 10}, 0, 0)
    rec.compare("qg_only_relation_input_changes_time", q_value[..., 284:], g_value[..., 284:],
                {"input_part": "time(32)", "t": .37}, 0, 0)
    for arm in ("Q_center", "G_closest"):
        expected = np.empty((1, 9, 3, 3))
        for corner in range(9):
            for primitive in range(3):
                vertices = prompt["vertices"][0, primitive].numpy().astype(np.float64)
                arity = int(prompt["arity"][0, primitive])
                location = x[0, corner].numpy().astype(np.float64)
                if arm == "Q_center":
                    anchor = vertices[:arity].mean(0)
                else:
                    anchor = (segment_closest(location, *vertices[:2]) if arity == 2 else
                              triangle_closest(location, *vertices))
                expected[0, corner, primitive] = location-anchor
        phase = expected[..., None]*np.pi*(2.0**np.arange(4))
        relations = np.concatenate((expected, (expected**2).sum(-1, keepdims=True),
                                    np.sin(phase).reshape(1, 9, 3, 12),
                                    np.cos(phase).reshape(1, 9, 3, 12)), -1)
        rec.compare("dynamic_input_independent_relation_"+arm, captured[arm][..., 256:284], relations,
                    {"B": 1, "Q": 9, "K": 3, "source": "current Xt", "GT_argument": False,
                     "oracle": "NumPy independently assembled relation encoding"})
    delta = torch.tensor([[-.2, .3, .7]], dtype=torch.float64)
    phase = delta.numpy()[..., None]*np.pi*(2.0**np.arange(4))
    expected = np.concatenate((delta.numpy(), (delta.numpy()**2).sum(-1, keepdims=True),
                               np.sin(phase).reshape(1, 12), np.cos(phase).reshape(1, 12)), -1)
    rec.compare("relation_encoding_exact_28_components", relation_features(delta), expected,
                {"delta": delta.tolist(), "fourier_bands": 4}, GEOMETRY_ATOL, 0)


def point_group_contract(rec):
    model = adapter("P_point", nonzero=True)
    prompt = fixture()
    encoded = model.encode(prompt)
    raw = model.condition_encoder(point_features(prompt["points"],
                                                   prompt["arity"].gather(1, prompt["point_primitive"])))
    expected = raw.clone()
    expected_weight = torch.zeros((1, 32))
    for primitive in range(3):
        indices = prompt["point_primitive"][0] == primitive
        expected[0, indices] += raw[0, indices].mean(0)
        expected_weight[0, indices] = -np.log(int(indices.sum()))
    rec.compare("point_tokens_keep_group_mean", encoded.tokens, expected,
                {"counts": [8, 8, 16], "formula": "e_point + mean(e_same_primitive)", "trainable_extra_parameters": 0})
    rec.compare("point_logits_equal_primitive_mass", encoded.log_weights, expected_weight,
                {"counts": [8, 8, 16], "formula": "-log(m_j)"})
    for primitive in range(3):
        indices = prompt["point_primitive"][0] == primitive
        rec.compare(f"point_log_weights_primitive_{primitive}_mass", encoded.log_weights[0, indices].exp().sum(),
                    torch.tensor(1.0), {"primitive": primitive, "points": int(indices.sum()), "logits_before_content": True})


def pooled_group_contract(rec):
    prompt = fixture()
    for arm in ("S_static", "Q_center", "G_closest"):
        model = adapter(arm)
        pooled = model.encode(prompt)
        point_values = model.condition_encoder(point_features(prompt["points"],
                                                prompt["arity"].gather(1, prompt["point_primitive"])))
        expected_means = torch.stack([point_values[0, prompt["point_primitive"][0] == primitive].mean(0)
                                      for primitive in range(3)]).unsqueeze(0)
        rec.compare("primitive_token_is_point_feature_mean_"+arm, pooled.tokens, expected_means,
                    {"counts": [8, 8, 16], "pooled_tokens": 3, "source": "same globally positioned point samples"})
        rec.compare("pooled_primitives_have_equal_log_weight_"+arm, pooled.log_weights,
                    torch.zeros_like(pooled.log_weights), {"primitive_weights": [1, 1, 1]}, 0, 0)


def permutation_checks(rec):
    h, x, t = query()
    prompt = fixture()
    generator = torch.Generator().manual_seed(22)
    for arm in ARMS:
        model = adapter(arm, nonzero=True)
        encoded = model.encode(prompt)
        baseline = model.forward_layer(0, h, x, t, encoded)
        magnitude = float(baseline.abs().max())
        rec.predicate("symmetry_residual_nonzero_"+arm, magnitude > 1e-6, {"max_abs_residual": magnitude},
                      {"output_projection": "independent deterministic nonzero fixture", "arm": arm}, {"max_abs_gt": 1e-6})
        for name, order in (("face_order", torch.tensor([6, 7, 8, 0, 1, 2, 3, 4, 5])),
                            ("face_corner_order", torch.tensor([2, 0, 1, 4, 5, 3, 7, 6, 8]))):
            actual = model.forward_layer(0, h[:, order], x[:, order], t, encoded)
            rec.compare(name+"_"+arm, actual, baseline[:, order],
                        {"order": order.tolist(), "B": 1, "faces": 3, "residual_nonzero_max": magnitude})
        point_order = torch.randperm(32, generator=generator)
        changed = clone_prompt(prompt)
        for key in ("points", "point_primitive", "point_mask"):
            changed[key] = changed[key][:, point_order]
        rec.compare("condition_point_order_"+arm, model.forward_layer(0, h, x, t, model.encode(changed)), baseline,
                    {"permutation": point_order.tolist(), "seed": 22, "residual_nonzero_max": magnitude})
        primitive_order = torch.tensor([2, 0, 1])
        inverse = torch.argsort(primitive_order)
        changed = clone_prompt(prompt)
        for key in ("vertices", "arity", "primitive_mask"):
            changed[key] = changed[key][:, primitive_order]
        changed["point_primitive"] = inverse[changed["point_primitive"]]
        rec.compare("primitive_order_"+arm, model.forward_layer(0, h, x, t, model.encode(changed)), baseline,
                    {"permutation": [2, 0, 1], "point_groups_remapped": True, "residual_nonzero_max": magnitude})
        changed = clone_prompt(prompt)
        changed["vertices"][:, :2, :2] = changed["vertices"][:, :2, [1, 0]]
        rec.compare("edge_endpoint_exchange_"+arm, model.forward_layer(0, h, x, t, model.encode(changed)), baseline,
                    {"edge_permutation": [1, 0], "residual_nonzero_max": magnitude})
        for permutation in itertools.permutations(range(3)):
            changed = clone_prompt(prompt)
            changed["vertices"][:, 2] = changed["vertices"][:, 2, list(permutation)]
            actual = model.forward_layer(0, h, x, t, model.encode(changed))
            rec.compare("triangle_vertex_order_"+arm+"_"+"".join(map(str, permutation)), actual, baseline,
                        {"permutation": permutation, "residual_nonzero_max": magnitude})


def padding_checks(rec):
    h, x, t = query()
    prompt = fixture()
    for arm in ARMS:
        model = adapter(arm, nonzero=True)
        baseline = model.forward_layer(0, h, x, t, model.encode(prompt))
        magnitude = float(baseline.abs().max())
        changed = clone_prompt(prompt)
        changed["vertices"] = torch.cat((changed["vertices"], torch.full((1, 2, 3, 3), float("nan"))), 1)
        changed["arity"] = torch.cat((changed["arity"], torch.tensor([[0, 99]])), 1)
        changed["primitive_mask"] = torch.cat((changed["primitive_mask"], torch.zeros((1, 2), dtype=torch.bool)), 1)
        rec.compare("primitive_padding_"+arm, model.forward_layer(0, h, x, t, model.encode(changed)), baseline,
                    {"valid_primitives": 3, "padded_primitives": 2, "padding": "NaN vertices + invalid arity masked",
                     "nonzero_valid_residual_max": magnitude})
        changed = clone_prompt(prompt)
        changed["points"] = torch.cat((changed["points"], torch.full((1, 7, 3), float("nan"))), 1)
        changed["point_primitive"] = torch.cat((changed["point_primitive"], torch.full((1, 7), -99)), 1)
        changed["point_mask"] = torch.cat((changed["point_mask"], torch.zeros((1, 7), dtype=torch.bool)), 1)
        rec.compare("point_padding_"+arm, model.forward_layer(0, h, x, t, model.encode(changed)), baseline,
                    {"valid_points": 32, "padded_points": 7, "padding": "NaN points + invalid group masked",
                     "nonzero_valid_residual_max": magnitude})
        extended_h = torch.cat((h, torch.full((1, 6, 768), float("nan"))), 1)
        extended_x = torch.cat((x, torch.full((1, 6, 3), float("nan"))), 1)
        query_mask = torch.tensor([[True]*9+[False]*6])
        output = model.forward_layer(0, extended_h, extended_x, t, model.encode(prompt), query_mask)
        rec.compare("face_padding_valid_"+arm, output[:, :9], baseline,
                    {"valid_faces": 3, "padded_faces": 2, "mask_semantics": "True means valid", "padding": "NaN"})
        rec.compare("face_padding_masked_zero_"+arm, output[:, 9:], torch.zeros_like(output[:, 9:]),
                    {"padding_faces": 2, "output_projection_bias": "nonzero"}, 0, 0)
        empty = clone_prompt(prompt)
        empty["primitive_mask"].zero_()
        empty["point_mask"].zero_()
        output = model.forward_layer(0, h, x, t, model.encode(empty))
        rec.compare("empty_all_masked_zero_"+arm, output, torch.zeros_like(output),
                    {"K": 3, "M": 32, "all_valid_flags": False, "output_projection_bias": "nonzero"}, 0, 0)
        mixed_empty = {key: torch.cat((empty[key], prompt[key]), 0) for key in prompt}
        batch_h, batch_x, batch_t = query(batch=2)
        mixed_output = model.forward_layer(0, batch_h, batch_x, batch_t, model.encode(mixed_empty))
        expected_valid = model.forward_layer(0, batch_h[1:], batch_x[1:], batch_t[1:], model.encode(prompt))
        rec.compare("mixed_empty_and_nonempty_batch_"+arm, mixed_output,
                    torch.cat((torch.zeros_like(expected_valid), expected_valid), 0),
                    {"per_row_valid_primitives": [0, 3], "nonzero_projection": True,
                     "all_masked_softmax_must_be_finite": True})
        empty = {key: value[:, :0] for key, value in prompt.items()}
        output = model.forward_layer(0, h, x, t, model.encode(empty))
        rec.compare("empty_zero_length_zero_"+arm, output, torch.zeros_like(output),
                    {"K": 0, "M": 0, "output_projection_bias": "nonzero"}, 0, 0)
        single = fixture("edge")
        single_padded = clone_prompt(single)
        single_padded["vertices"] = torch.cat((single["vertices"], torch.full((1, 2, 3, 3), float("nan"))), 1)
        single_padded["arity"] = torch.cat((single["arity"], torch.zeros((1, 2), dtype=torch.long)), 1)
        single_padded["primitive_mask"] = torch.tensor([[True, False, False]])
        mixed = {key: torch.cat((single_padded[key], prompt[key]), 0) for key in prompt}
        batch_h, batch_x, batch_t = query(batch=2)
        actual = model.forward_layer(0, batch_h, batch_x, batch_t, model.encode(mixed))
        expected = torch.cat((model.forward_layer(0, batch_h[:1], batch_x[:1], batch_t[:1], model.encode(single)),
                              model.forward_layer(0, batch_h[1:], batch_x[1:], batch_t[1:], model.encode(prompt))), 0)
        rec.compare("m1_m3_batch_equals_separate_"+arm, actual, expected,
                    {"batch_primitive_counts": [1, 3], "points_per_object": 32, "Q": 9,
                     "residual_nonzero_max": float(expected.abs().max())})
        other = adapter(arm, nonzero=True, query_chunk=100)
        other.load_state_dict(model.state_dict(), strict=True)
        rec.compare("query_chunk_equivalence_"+arm,
                    other.forward_layer(0, h, x, t, other.encode(prompt)), baseline,
                    {"query_chunks": [4, 100], "Q": 9, "same_state": True})


def gradient_probes(rec):
    prompt = fixture()
    h, x, t = query(corners=6)
    target = torch.randn(h.shape, generator=torch.Generator().manual_seed(398))*.2
    for arm in ("Q_center", "G_closest"):
        model = adapter(arm)
        # This is only an autograd unit fixture, never a replacement backbone.
        frozen = nn.ModuleList([nn.Linear(768, 768) for _ in range(4)]).eval()
        frozen.requires_grad_(False)
        before = state_hash(frozen.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=0)
        first = None
        for step in range(10):
            optimizer.zero_grad(set_to_none=True)
            encoded = model.encode(prompt)
            current = h
            for index, block in enumerate(frozen):
                current = block(current)
                current = current + model.forward_layer(index, current, x, t, encoded)
            # Frozen transformation after the final adapter also must pass gradients.
            loss = (frozen[-1](current)-target).square().mean()
            loss.backward()
            gradients = {name: float(parameter.grad.abs().max()) if parameter.grad is not None else None
                         for name, parameter in model.named_parameters()}
            if step == 0:
                first = gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            COUNTERS["cpu_probe_optimizer_updates"] += 1
        outlet = [v for k, v in first.items() if "output_projection" in k]
        internal = [v for k, v in first.items() if "output_projection" not in k]
        rec.predicate("gradient_first_exit_"+arm, all(v is not None and v > 0 for v in outlet),
                      {k: v for k, v in first.items() if "output_projection" in k},
                      {"step": 0, "zero_initial_output": True, "backbone": "CPU frozen affine unit fixture"},
                      {"every_output_weight_and_bias_grad_gt": 0})
        rec.predicate("gradient_first_internal_zero_expected_"+arm, all(v is not None and v == 0 for v in internal),
                      {"max_internal_grad": max(v for v in internal if v is not None)},
                      {"step": 0, "zero_initial_output": True}, {"max_abs": 0})
        key_names = ["condition_encoder.0.weight", "condition_encoder.2.weight"]
        for index in range(4):
            key_names += [f"layers.{index}.{part}.weight" for part in
                          ("query_norm", "query_projection", "key_projection", "value_projection", "dynamic_mlp.0", "dynamic_mlp.2")]
        selected = {key: gradients[key] for key in key_names}
        rec.predicate("gradient_after_10_probe_updates_"+arm,
                      all(value is not None and np.isfinite(value) and value > 0 for value in selected.values()), selected,
                      {"updates": 10, "K": 3, "frozen_layers": 4,
                       "early_adapter_index": 0, "probe_not_phase2_training": True}, {"every_key_gradient_gt": 0})
        after = state_hash(frozen.state_dict())
        rec.predicate("autograd_fixture_frozen_hash_no_grad_"+arm,
                      before == after and all(p.grad is None for p in frozen.parameters()) and not frozen.training,
                      {"before_sha256": before, "after_sha256": after,
                       "any_frozen_grad": any(p.grad is not None for p in frozen.parameters())},
                      {"fixture": "frozen CPU affine transforms; official hash checked separately", "updates": 10},
                      {"hash_equal": True, "frozen_grad": None, "training": False})


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def flattened_state(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, member in value.items():
            result.update(flattened_state(member, prefix+"/"+str(key)))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, member in enumerate(value):
            result.update(flattened_state(member, prefix+"/"+str(index)))
        return result
    if isinstance(value, (torch.Tensor, np.ndarray, int, float, bool)):
        return {prefix: torch.as_tensor(value).double()}
    return {prefix: value}


def compare_nested(rec, name, actual, expected, inputs):
    aa, bb = flattened_state(actual), flattened_state(expected)
    mismatches, maximum = [], 0.0
    for key in set(aa)|set(bb):
        if key not in aa or key not in bb:
            mismatches.append(key)
        elif isinstance(aa[key], torch.Tensor) and isinstance(bb[key], torch.Tensor):
            if aa[key].shape != bb[key].shape:
                mismatches.append(key)
            else:
                difference = float((aa[key]-bb[key]).abs().max()) if aa[key].numel() else 0
                maximum = max(maximum, difference)
                if not torch.equal(aa[key], bb[key]):
                    mismatches.append(key)
        elif aa[key] != bb[key]:
            mismatches.append(key)
    rec.predicate(name, not mismatches, {"max_abs": maximum, "mismatched_fields": mismatches},
                  inputs, {"all_nested_values": "bitwise_equal", "max_abs": 0})


def restore_check(rec):
    prompt = fixture()
    for arm in ("Q_center", "G_closest"):
        def fresh():
            model = adapter(arm)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=0)
            return model, optimizer

        def update(model, optimizer):
            # Every random stream affects this step, so restoration is observable.
            shift = random.random()+float(np.random.rand())
            h = torch.randn(1, 3, 768)+shift*.05
            x = torch.randn(1, 3, 3)
            t = torch.rand(1)
            target = torch.randn_like(h)*.1
            optimizer.zero_grad(set_to_none=True)
            encoded = model.encode(prompt)
            output = h
            for index in range(4):
                output = output+model.forward_layer(index, output, x, t, encoded)
            loss = (output-target).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            COUNTERS["cpu_probe_optimizer_updates"] += 1
            return float(loss)

        seed_all(777)
        continuous, continuous_optimizer = fresh()
        initial_rng = rng_state()
        continuous_losses = [update(continuous, continuous_optimizer) for _ in range(10)]
        final_rng = rng_state()
        split, split_optimizer = fresh()
        set_rng_state(initial_rng)
        losses = [update(split, split_optimizer) for _ in range(5)]
        # Save exactly the adapter/optimizer/step/RNG state to a real serializer.
        stream = io.BytesIO()
        state = {"adapter": split.state_dict(), "optimizer": split_optimizer.state_dict(),
                 "step": 5, "rng": rng_state(), "arm": arm}
        torch.save(state, stream)
        # This trusted self-created fixture includes known NumPy RNG types;
        # minimally allowlist them without enabling unrestricted pickle loading.
        stream.seek(0)
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray,
                                               np.dtype, type(np.dtype(np.uint32))]):
            loaded = torch.load(stream, map_location="cpu", weights_only=True)
        restored, restored_optimizer = fresh()
        restored.load_state_dict(loaded["adapter"], strict=True)
        restored_optimizer.load_state_dict(loaded["optimizer"])
        set_rng_state(loaded["rng"])
        losses += [update(restored, restored_optimizer) for _ in range(5)]
        resumed_rng = rng_state()
        inputs = {"arm": arm, "continuous_steps": 10, "resume_at": 5,
                  "serializer": "torch.save + torch.load(weights_only=True)",
                  "rng_streams": ["Python", "NumPy", "Torch CPU"], "device": "cpu",
                  "probe_not_phase2_training": True}
        rec.compare("restore_loss_sequence_"+arm, torch.tensor(losses), torch.tensor(continuous_losses), inputs, 0, 0)
        compare_nested(rec, "restore_adapter_state_"+arm, restored.state_dict(), continuous.state_dict(), inputs)
        compare_nested(rec, "restore_optimizer_state_"+arm, restored_optimizer.state_dict(), continuous_optimizer.state_dict(), inputs)
        compare_nested(rec, "restore_rng_state_"+arm, resumed_rng, final_rng, inputs)
        h, x, t = query()
        with torch.no_grad():
            expected = continuous.forward_layer(0, h, x, t, continuous.encode(prompt))
            actual = restored.forward_layer(0, h, x, t, restored.encode(prompt))
        rec.compare("restore_output_"+arm, actual, expected, inputs, 0, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite previous test report: "+str(args.output))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rec = Recorder()
    started = time.perf_counter()
    groups = [geometry_cases, geometry_random, architecture_and_zero, single_token,
              geometry_message_contract, point_group_contract, pooled_group_contract, permutation_checks,
              padding_checks, gradient_probes, restore_check]
    for group in groups:
        rec.run(group.__name__, group)
    passed = all(item["status"] == "PASS" for item in rec.results)
    COUNTERS["adapter_layer_forward_calls"] = sum(model.call_count for model in MODELS)
    COUNTERS["adapter_encoder_calls"] = sum(model.encoder_call_count for model in MODELS)
    report = {"schema_version": 1, "suite": "MF-H1-LOCAL0 independent CPU mathematical + adapter tests",
              "status": "PASS" if passed else "FAIL", "device": "cpu", "torch_version": torch.__version__,
              "python": sys.version, "elapsed_seconds": time.perf_counter()-started,
              "official_status": "NOT_RUN_THIS_SUITE", "phase2_status": "NOT_RUN_THIS_SUITE",
              "precision": {"geometry": "FP64", "adapter": "FP32"},
              "scope": "Structural checks only; no official model execution or generated-chair quality inference.",
              "counts": COUNTERS, "passed": sum(item["status"] == "PASS" for item in rec.results),
              "failed": sum(item["status"] == "FAIL" for item in rec.results), "tests": rec.results}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(json_safe(report), output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"report": str(args.output), "status": report["status"], "passed": report["passed"],
                      "failed": report["failed"], "counts": COUNTERS}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
