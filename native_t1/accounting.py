"""Attempt/return accounting for the original CPU OT preprocessing.

The portable trainer consumes its training stream on one thread. This temporary wrapper
does not change OT arguments, results, exceptions, or random state. Other
threads and nested accounting scopes are intentionally unsupported.
"""
from contextlib import contextmanager
from functools import wraps


_ACTIVE = False


@contextmanager
def count_ot_calls(counts):
    """Count actual calls, including a partially completed eight-sample batch.

``OT_calls`` counts attempts before entering the original function;
``OT_returns`` counts successful returns. The original callable is restored
even when either OT or the surrounding batch preparation raises. The caller
must remove any separate fixed ``OT_calls += 8`` bookkeeping.
    """
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("OT accounting scopes cannot be nested")
    from utils import ot_utils

    original = ot_utils.optimal_sum_numpy
    counts.setdefault("OT_calls", 0)
    counts.setdefault("OT_returns", 0)

    @wraps(original)
    def recorded(*args, **kwargs):
        counts["OT_calls"] += 1
        result = original(*args, **kwargs)
        counts["OT_returns"] += 1
        return result

    _ACTIVE = True
    ot_utils.optimal_sum_numpy = recorded
    try:
        yield
    finally:
        ot_utils.optimal_sum_numpy = original
        _ACTIVE = False
