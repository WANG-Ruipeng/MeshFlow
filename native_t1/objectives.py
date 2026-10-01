"""Frozen training recipes; they never enter the generator or sampler."""
from copy import deepcopy

LAMBDA_EDGE = 4.071385484299878
LAMBDA_L5 = 0.3266104383520167
NAMES = ("fm", "surface", "edge5")


def objective_spec(name):
    if name not in NAMES:
        raise ValueError("Unknown maintained training objective")
    return dict(name=name, version=1, FM="masked_free_coordinate_MSE",
        time_gate="original_FP32_t>=0.5", ramp="min(additional_step/50,1)",
        lambda_edge=LAMBDA_EDGE if name == "edge5" else 0.,
        lambda_surface=LAMBDA_L5 if name != "fm" else 0.,
        auxiliary_denominator="all_8_samples_including_inactive",
        geometry_dtype="float32", source_ids_are_model_inputs=False,
        geometry="finite_corresponding_interface_edges_and_area_unsigned_normal_MMD")


def validate_objective(spec):
    if not isinstance(spec, dict) or spec != objective_spec(spec.get("name")):
        raise ValueError("Training objective differs from its registered recipe")
    return deepcopy(spec)


class GeometryObjective:
    """Per-sample auxiliary term, already divided by the effective batch8.

    Fixed GT targets may be cached within one batch. Predicted coordinates,
    areas, normals and model condition encodings always retain their gradients.
    """
    def __init__(self, name, archive):
        if name not in ("surface", "edge5"):
            raise ValueError("GeometryObjective requires surface or edge5")
        from .losses import TrainingGeometryTargets
        self.spec = objective_spec(name)
        self.bank = TrainingGeometryTargets(archive)

    def prepare(self, samples):
        from .losses import prepare_targets
        return [prepare_targets(self.bank.sample(sample)) for sample in samples]

    def auxiliary(self, velocity, batch, target, *, additional_step):
        import torch
        from .losses import evaluate
        if type(additional_step) is not int or additional_step < 1:
            raise ValueError("An explicit positive additional update index is required")
        if batch["t"].dtype != torch.float32 or batch["t"].numel() != 1:
            raise ValueError("Original single-sample FP32 time required")
        active = bool(batch["t"][0] >= .5)
        ramp = min(additional_step / 50., 1.)
        diagnostic = dict(active=active, ramp=ramp, raw_losses={}, counts={})
        total = velocity.float().sum() * 0.
        if active:
            with torch.autocast(device_type=velocity.device.type, enabled=False):
                free = batch["valid_mask"] & ~batch["known_mask"]
                prediction = (batch["xt"].float() + (1-batch["t"][:, None, None])
                              * velocity.float())[free].reshape(-1, 3, 3)
                terms = [("D_edge", self.spec["lambda_edge"])] if self.spec["name"] == "edge5" else []
                terms.append(("L5", self.spec["lambda_surface"]))
                for name, coefficient in terms:
                    result = evaluate(name, prediction, target[name])
                    if not bool(torch.isfinite(result["loss"])):
                        raise FloatingPointError("Nonfinite geometry loss: " + name)
                    total = total + coefficient * result["loss"]
                    diagnostic["raw_losses"][name] = float(result["loss"].detach())
                    diagnostic["counts"][name] = {
                        key: int(value.detach()) if torch.is_tensor(value) else value
                        for key, value in result.get("counts", {}).items()}
        total = ramp * total / 8
        diagnostic["weighted_auxiliary"] = float(total.detach())
        return total, diagnostic
