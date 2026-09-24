"""ulpwise: numerical conformance testing for ML code.

The heavy lifting (ordered float views, exact rounding oracles, knife-edge scans) is in the Rust
extension ``ulpwise._core``. This module adds the array plumbing and the assertion helpers.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

from ._core import (  # noqa: F401
    UNARY_OPS,
    all_floats,
    binade_edges,
    knife_edges_raw,
    midpoint,
    neighbours,
    next_down,
    next_up,
    ordered,
    spacing,
    special,
    sqrt_cr,
    ulp_distance,
    ulp_distances,
)

__version__ = "0.1.2"

__all__ = [
    "UNARY_OPS",
    "all_floats",
    "assert_max_ulp",
    "binade_edges",
    "flatten",
    "knife_edges",
    "knife_edges_raw",
    "max_ulp",
    "midpoint",
    "neighbours",
    "next_down",
    "next_up",
    "ordered",
    "spacing",
    "special",
    "sqrt_cr",
    "ulp_distance",
    "ulp_distances",
]

_DTYPE_NAMES = {
    "float32": "f32",
    "float64": "f64",
    "f32": "f32",
    "f64": "f64",
    "float": "f64",
    "double": "f64",
    "torch.float32": "f32",
    "torch.float64": "f64",
}


def knife_edges(
    op: str,
    lo: float,
    hi: float,
    tol_ulp: float = 1e-3,
    dtype: str = "f32",
    limit: int = 100,
    stride: int = 1,
    max_evals: int = 100_000_000,
) -> list:
    """Inputs in ``[lo, hi]`` whose exact ``op`` result lies within ``tol_ulp`` of a rounding
    midpoint, as ``(x, rounded, distance_ulp, exact_above)`` tuples.

    Two implementations that differ by one ulp return different floats exactly at these inputs, so
    they are the inputs to pin in a test that must agree across platforms. ``sqrt``, ``recip`` and
    ``div`` use exact integer arithmetic, the other ops use the f64 libm as reference (trust
    ``tol_ulp`` down to about 1e-8). A warning is raised when ``max_evals`` stopped the scan before
    ``hi`` and fewer than ``limit`` hits were found.
    """
    hits, evaluated, exhausted = knife_edges_raw(op, lo, hi, tol_ulp, dtype, limit, stride, max_evals)
    if not exhausted and len(hits) < limit:
        import warnings

        warnings.warn(
            f"knife_edges stopped after max_evals={evaluated} inputs before reaching hi={hi!r}; "
            "narrow the range, raise max_evals or use a stride",
            RuntimeWarning,
            stacklevel=2,
        )
    return hits


def flatten(x: Any, dtype: Optional[str] = None) -> Tuple[list, str]:
    """Turn a float, a nested sequence, a numpy array or a torch tensor into a flat list of Python
    floats plus the ulpwise dtype name ("f32" or "f64").

    The dtype is taken from the array when it has one; Python floats default to "f64". Pass
    ``dtype`` to override. Integer arrays are rejected: ulps only make sense for floats.
    """
    inferred = None
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch tensor
        x = x.detach().cpu()
        inferred = str(x.dtype)
        if inferred not in _DTYPE_NAMES:
            raise TypeError(f"ulpwise handles float32 and float64 tensors, got {inferred}")
        values = x.reshape(-1).tolist()
    elif hasattr(x, "dtype") and hasattr(x, "tolist"):  # numpy array or scalar
        inferred = str(x.dtype)
        if inferred not in _DTYPE_NAMES:
            raise TypeError(f"ulpwise handles float32 and float64 arrays, got {inferred}")
        values = x.reshape(-1).tolist() if hasattr(x, "reshape") else [x.tolist()]
    elif isinstance(x, (int, float)):
        values = [float(x)]
    else:
        values = []
        stack = list(x)
        stack.reverse()
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                stack.extend(reversed(item))
            elif hasattr(item, "tolist"):
                sub, sub_dtype = flatten(item)
                inferred = inferred or sub_dtype
                values.extend(sub)
            else:
                values.append(float(item))
    if dtype is not None:
        if dtype not in _DTYPE_NAMES:
            raise ValueError(f"unknown dtype {dtype!r}")
        return values, _DTYPE_NAMES[dtype]
    return values, _DTYPE_NAMES.get(inferred or "f64", "f64")


def max_ulp(actual: Any, expected: Any, dtype: Optional[str] = None) -> Tuple[Optional[int], int]:
    """Largest elementwise ulp distance between ``actual`` and ``expected`` and the flat index
    where it occurs. The distance is ``None`` when some element pair involves a NaN on one side
    only (NaN against NaN counts as 0)."""
    a, da = flatten(actual, dtype)
    b, db = flatten(expected, dtype)
    if len(a) != len(b):
        raise ValueError(f"shape mismatch: {len(a)} vs {len(b)} elements")
    if not a:
        return 0, 0
    dt = dtype and _DTYPE_NAMES[dtype] or ("f32" if "f32" in (da, db) else "f64")
    worst, worst_i = -1, 0
    for i, (x, y, d) in enumerate(zip(a, b, ulp_distances(a, b, dt))):
        if d is None:
            if x != x and y != y:
                continue
            return None, i
        if d > worst:
            worst, worst_i = d, i
    return max(worst, 0), worst_i


def assert_max_ulp(actual: Any, expected: Any, max_ulp_: int = 1, dtype: Optional[str] = None, msg: str = "") -> int:
    """Assert that every element of ``actual`` is within ``max_ulp_`` ulps of ``expected``.

    Works on floats, nested lists, numpy arrays and torch tensors. The dtype is inferred from the
    inputs (a float32 tensor is measured in float32 ulps) unless ``dtype`` is given. NaN matches
    only NaN. Returns the largest distance seen.
    """
    worst, i = max_ulp(actual, expected, dtype)
    a, _ = flatten(actual, dtype)
    b, _ = flatten(expected, dtype)
    if worst is None:
        raise AssertionError(f"{msg}NaN mismatch at flat index {i}: actual={a[i]!r} expected={b[i]!r}")
    if worst > max_ulp_:
        raise AssertionError(
            f"{msg}max ulp distance {worst} > {max_ulp_} at flat index {i}: actual={a[i]!r} expected={b[i]!r}"
        )
    return worst


def edge_values(dtype: str = "f32", names: Optional[Iterable[str]] = None) -> Sequence[float]:
    """The named edge values of a dtype, optionally restricted to ``names``."""
    pairs = special(dtype)
    if names is None:
        return [v for _, v in pairs]
    wanted = set(names)
    return [v for n, v in pairs if n in wanted]
