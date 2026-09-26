"""ulpwise: numerical conformance testing for ML code.

The heavy lifting (ordered float views, exact rounding oracles, knife-edge scans) is in the Rust
extension ``ulpwise._core``. This module adds the array plumbing, the assertion helpers and the
float16 / bfloat16 ulp distances, which are pure Python on top of the 16 bit patterns.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction
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
    spacing,
    sqrt_cr,
)
from ._core import ordered as _ordered_core
from ._core import special as _special_core
from ._core import ulp_distance as _ulp_distance_core
from ._core import ulp_distances as _ulp_distances_core

__version__ = "0.3.0"

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
    "float16": "f16",
    "f16": "f16",
    "half": "f16",
    "torch.float16": "f16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "torch.bfloat16": "bf16",
}
_HALF_DTYPES = ("f16", "bf16")
_PRECISION_ORDER = ("bf16", "f16", "f32", "f64")  # fewest significand bits first


def _check_dtype(dtype: str) -> None:
    if dtype not in _PRECISION_ORDER:
        raise ValueError(f"unsupported dtype {dtype!r}: use 'f64', 'f32', 'f16' or 'bf16'")


def _bits16(x: float, dtype: str) -> int:
    """Bit pattern of ``x`` rounded to nearest even in float16 or bfloat16.

    ``struct`` rounds to float16 straight from the double. bfloat16 is the top half of the float32
    pattern rounded at bit 16; going through float32 first is harmless because 24 bits are more
    than the 2 * 8 + 2 that make double rounding innocuous. Overflow becomes the signed infinity.
    """
    if dtype == "f16":
        if x != x:
            return 0xFE00 if math.copysign(1.0, x) < 0 else 0x7E00
        try:
            return struct.unpack("<H", struct.pack("<e", x))[0]
        except OverflowError:
            return 0xFC00 if x < 0 else 0x7C00
    try:
        bits = struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0xFF80 if x < 0 else 0x7F80
    if bits & 0x7F800000 == 0x7F800000:  # infinity or NaN: keep the top bits, no rounding
        return (bits >> 16) | (0x40 if bits & 0x007FFFFF else 0)
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def _ordered16(x: float, dtype: str) -> int:
    bits = _bits16(x, dtype)
    return -(bits & 0x7FFF) if bits & 0x8000 else bits


def _from_bits16(bits: int, dtype: str) -> float:
    if dtype == "f16":
        return struct.unpack("<e", struct.pack("<H", bits))[0]
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _round16(x: float, dtype: str) -> float:
    return _from_bits16(_bits16(x, dtype), dtype)


def _step16(x: float, dtype: str, n: int) -> float:
    """``x`` moved ``n`` representable values up (``n > 0``) or down in the 16 bit dtype."""
    o = _ordered16(x, dtype) + n
    return _from_bits16(-o | 0x8000 if o < 0 else o, dtype)


def _special16(dtype: str) -> list:
    """The 29 named edge values of float16 (p = 11, emin = -14, emax = 15) or bfloat16 (p = 8,
    emin = -126, emax = 127), the same names and meanings as the Rust ``special`` for f32 and f64."""
    p, emin, emax = (11, -14, 15) if dtype == "f16" else (8, -126, 127)
    fmax = (2 - 2.0 ** (1 - p)) * 2.0**emax
    min_subnormal = 2.0 ** (emin - p + 1)
    below = Fraction(2) ** (emin - p)  # half the smallest subnormal: a square this small rounds to zero
    sq_zero = _round16(math.sqrt(2.0 ** (emin - p)), dtype)
    while Fraction(sq_zero) ** 2 > below:
        sq_zero = _step16(sq_zero, dtype, -1)
    while Fraction(_step16(sq_zero, dtype, 1)) ** 2 <= below:
        sq_zero = _step16(sq_zero, dtype, 1)
    overflow = (Fraction(2) - Fraction(2) ** -p) * Fraction(2) ** emax  # max plus half an ulp: rounds to infinity
    sq_finite = _round16(math.sqrt(fmax), dtype)
    while Fraction(sq_finite) ** 2 >= overflow:
        sq_finite = _step16(sq_finite, dtype, -1)
    while Fraction(_step16(sq_finite, dtype, 1)) ** 2 < overflow:
        sq_finite = _step16(sq_finite, dtype, 1)
    return [
        ("zero", 0.0),
        ("neg_zero", -0.0),
        ("min_subnormal", min_subnormal),
        ("neg_min_subnormal", -min_subnormal),
        ("max_subnormal", 2.0**emin - min_subnormal),
        ("min_normal", 2.0**emin),
        ("neg_min_normal", -(2.0**emin)),
        ("reciprocal_overflows", 2.0 ** -(emax + 1)),
        ("square_underflows_to_zero", sq_zero),
        ("square_is_subnormal", _step16(2.0 ** (emin // 2), dtype, -1)),
        ("tenth", _round16(0.1, dtype)),
        ("third", _round16(1 / 3, dtype)),
        ("half", 0.5),
        ("one_minus_ulp", _step16(1.0, dtype, -1)),
        ("one", 1.0),
        ("one_plus_ulp", _step16(1.0, dtype, 1)),
        ("neg_one", -1.0),
        ("two", 2.0),
        ("e", _round16(math.e, dtype)),
        ("pi", _round16(math.pi, dtype)),
        ("integer_limit", 2.0**p),
        ("sqrt_max", _round16(math.sqrt(fmax), dtype)),
        ("square_just_finite", sq_finite),
        ("square_overflows", _step16(sq_finite, dtype, 1)),
        ("max", fmax),
        ("neg_max", -fmax),
        ("inf", math.inf),
        ("neg_inf", -math.inf),
        ("nan", math.nan),
    ]


def special(dtype: str = "f32") -> list:
    """Named edge values of the dtype as (name, value) pairs; ``f64``, ``f32``, ``f16`` or ``bf16``."""
    _check_dtype(dtype)
    if dtype in _HALF_DTYPES:
        return _special16(dtype)
    return _special_core(dtype)


def ordered(x: float, dtype: str = "f64") -> int:
    """Monotone integer view of ``x`` in the dtype (-0.0 and +0.0 both map to 0)."""
    _check_dtype(dtype)
    if dtype in _HALF_DTYPES:
        return _ordered16(x, dtype)
    return _ordered_core(x, dtype)


def ulp_distance(a: float, b: float, dtype: str = "f64") -> Optional[int]:
    """Number of representable floats of the dtype between ``a`` and ``b``, None if either is NaN.

    ``dtype`` is ``f64``, ``f32``, ``f16`` or ``bf16``; values that are not representable in it
    are rounded to nearest even first.
    """
    _check_dtype(dtype)
    if dtype in _HALF_DTYPES:
        if a != a or b != b:
            return None
        return abs(_ordered16(a, dtype) - _ordered16(b, dtype))
    return _ulp_distance_core(a, b, dtype)


def ulp_distances(a: Sequence[float], b: Sequence[float], dtype: str = "f64") -> list:
    """Elementwise :func:`ulp_distance` over two equally long sequences."""
    _check_dtype(dtype)
    if dtype in _HALF_DTYPES:
        a, b = list(a), list(b)
        if len(a) != len(b):
            raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
        return [ulp_distance(x, y, dtype) for x, y in zip(a, b)]
    return _ulp_distances_core(a, b, dtype)


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
    floats plus the ulpwise dtype name ("f64", "f32", "f16" or "bf16").

    The dtype is taken from the array when it has one; Python floats default to "f64". Pass
    ``dtype`` to override. Integer arrays are rejected: ulps only make sense for floats.
    """
    inferred = None
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch tensor
        x = x.detach().cpu()
        inferred = str(x.dtype)
        if inferred not in _DTYPE_NAMES:
            raise TypeError(f"ulpwise handles float64, float32, float16 and bfloat16 tensors, got {inferred}")
        values = x.reshape(-1).tolist()
    elif hasattr(x, "dtype") and hasattr(x, "tolist"):  # numpy array or scalar
        inferred = str(x.dtype)
        if inferred not in _DTYPE_NAMES:
            raise TypeError(f"ulpwise handles float64, float32 and float16 arrays, got {inferred}")
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
    only (NaN against NaN counts as 0). Without ``dtype`` the distance is measured in ulps of the
    less precise of the two inputs, so a bfloat16 tensor against a float64 reference is measured in
    bfloat16 ulps."""
    a, da = flatten(actual, dtype)
    b, db = flatten(expected, dtype)
    if len(a) != len(b):
        raise ValueError(f"shape mismatch: {len(a)} vs {len(b)} elements")
    if not a:
        return 0, 0
    dt = _DTYPE_NAMES[dtype] if dtype else next(d for d in _PRECISION_ORDER if d in (da, db))
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
