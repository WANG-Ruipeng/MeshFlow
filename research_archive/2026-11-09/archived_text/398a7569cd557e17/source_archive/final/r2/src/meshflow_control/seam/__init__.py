"""Packaged computational core of the bounded SEAM_SLOT_C200 prototype.

Importing this namespace does not construct a model, read weights, or import
Torch. Import the explicit submodule needed by the caller. This is an optional
research API; it does not change the maintained sampling profile.
"""

__all__ = ["corner_contract", "inference", "model_adapter", "occupancy", "slot_data"]
