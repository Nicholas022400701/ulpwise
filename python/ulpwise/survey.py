"""Accuracy survey of elementary and special functions against a high precision reference.

``ulpwise survey`` evaluates every function in :data:`REGISTRY` on a fixed grid of inputs (log spaced
points over the function's domain plus the named edge values of the dtype), computes the exact result
with ``mpmath`` at the rounded inputs, and reports the error of each backend in ulps of the dtype.
For torch it also reports whether the errors would be caught by the tolerances of torch's own
``OpInfo`` reference tests, both the per dtype defaults and the per op overrides, and whether the
vectorized kernel agrees with the scalar tail.

The reference is computed at the input the backend actually saw (the float of the dtype), so the
numbers are the error of the implementation, not of the input rounding.
"""

from __future__ import annotations

import csv
import importlib
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import ulpwise

DTYPES = {"f32": np.float32, "f64": np.float64}
DEFAULT_TOL = {"f32": (1.3e-6, 1e-5), "f64": (1e-7, 1e-7)}  # (rtol, atol) of torch.testing for the dtype


@dataclass(frozen=True)
class Entry:
    name: str
    ref: Callable[[Any], Any]  # mpf -> mpf, or None for a domain error
    domain: Tuple  # see grid()
    torch: Optional[str] = None  # attribute path below `torch.`; a callable is allowed too
    numpy: Optional[str] = None
    scipy: Optional[str] = None  # attribute path below `scipy.special.`
    jax: Optional[str] = None  # attribute path below `jax.`
    opinfo: Optional[str] = None  # torch OpInfo name for the tolerance lookup
    note: str = ""


def _mp():
    import mpmath as mp

    return mp


def grid(domain: Tuple, dtype: str, points: int) -> np.ndarray:
    """Input grid for a domain spec, in the dtype, unique and sorted."""
    kind = domain[0]
    npdt = DTYPES[dtype]
    half = max(points // 2, 8)
    if kind == "log":  # (lo, hi) positive
        xs = np.logspace(math.log10(domain[1]), math.log10(domain[2]), points)
    elif kind == "symlog":  # (lo, hi) both signs
        pos = np.logspace(math.log10(domain[1]), math.log10(domain[2]), half)
        xs = np.concatenate([-pos[::-1], pos])
    elif kind == "lin":
        xs = np.linspace(domain[1], domain[2], points)
    elif kind == "unit":  # open interval (0, 1), dense at both ends
        lo = -38 if dtype == "f32" else -300
        eps = -7 if dtype == "f32" else -16
        xs = np.concatenate(
            [np.logspace(lo, -1, half), np.linspace(0.1, 0.9, half // 2), 1 - np.logspace(eps, -1, half)]
        )
    elif kind == "unitsym":  # open interval (-1, 1)
        eps = -7 if dtype == "f32" else -16
        pos = np.concatenate([np.logspace(-8, math.log10(0.9), half), 1 - np.logspace(eps, -1, half // 2)])
        xs = np.concatenate([-pos[::-1], pos])
    elif kind == "ge1":  # [1, hi) as 1 + logspace
        eps = -7 if dtype == "f32" else -16
        xs = 1 + np.logspace(eps, math.log10(domain[1]), points)
    else:
        raise ValueError(f"unknown domain kind {kind!r}")
    xs = xs.astype(npdt)
    lo, hi = float(xs.min()), float(xs.max())
    edges = [v for _, v in ulpwise.special(dtype) if lo <= v <= hi and math.isfinite(v)]
    xs = np.unique(np.concatenate([xs, np.asarray(edges, dtype=npdt)]).astype(npdt))
    xs = xs[np.isfinite(xs)]
    return xs


def _mpf_ref(fn, x: float):
    mp = _mp()
    try:
        v = fn(mp.mpf(x))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if isinstance(v, mp.mpc):
        return None
    return v


# ----------------------------------------------------------------------------------------------
# reference functions (mpf -> mpf)


def _ref_log_ndtr(x):
    mp = _mp()
    return mp.log1p(-mp.ncdf(-x)) if x > 0 else mp.log(mp.ncdf(x))


def _ref_ndtri(p):
    mp = _mp()
    if p <= 0 or p >= 1:
        return mp.mpf("-inf") if p == 0 else (mp.mpf("inf") if p == 1 else None)
    # Newton on ncdf(x) = p starting from the double precision estimate; converges in a few steps.
    from scipy.special import ndtri as _ndtri  # noqa: PLC0415

    x = mp.mpf(float(_ndtri(float(p))))
    for _ in range(6):
        x = x - (mp.ncdf(x) - p) / mp.npdf(x)
    return x


def _ref_erfinv(y):
    mp = _mp()
    if y <= -1 or y >= 1:
        return mp.mpf("-inf") if y == -1 else (mp.mpf("inf") if y == 1 else None)
    return mp.erfinv(y)


def _ref_entr(x):
    mp = _mp()
    if x < 0:
        return mp.mpf("-inf")
    if x == 0:
        return mp.mpf(0)
    return -x * mp.log(x)


def _ref_sinc(x):
    # sincpi is exact at the integers, sin(pi * x) / (pi * x) with a rounded pi is not
    return _mp().sincpi(x)


def _ref_lgamma(x):
    mp = _mp()
    if x <= 0 and x == mp.floor(x):
        return mp.mpf("inf")
    return mp.log(abs(mp.gamma(x)))


def _ref_digamma(x):
    mp = _mp()
    if x <= 0 and x == mp.floor(x):
        return None
    return mp.digamma(x)


def _ref_gelu(x):
    mp = _mp()
    return x * mp.ncdf(x)


def _ref_gelu_tanh(x):
    mp = _mp()
    t = mp.sqrt(2 / mp.pi) * (x + mp.mpf("0.044715") * x**3)
    # 0.5 * x * (1 + tanh(t)) written as x / (1 + exp(-2 t)) so the negative tail does not cancel
    return x / (1 + mp.exp(-2 * t))


def _ref_mish(x):
    mp = _mp()
    return x * mp.tanh(mp.log1p(mp.exp(x)))


def _ref_selu(x):
    mp = _mp()
    scale = mp.mpf("1.0507009873554804934193349852946")
    alpha = mp.mpf("1.6732632423543772848170429916717")
    return scale * (x if x > 0 else alpha * mp.expm1(x))


def _ref_spherical_j0(x):
    mp = _mp()
    return mp.mpf(1) if x == 0 else mp.sin(x) / x


def _ref_zeta(x):
    mp = _mp()
    if x == 1:
        return mp.mpf("inf")
    if x < 1:
        return None  # torch.special.zeta returns nan for x < 1
    return mp.zeta(x)


def _ref_polygamma(n):
    def f(x):
        mp = _mp()
        if x <= 0 and x == mp.floor(x):
            return None
        return mp.polygamma(n, x)

    return f


def _torch_polygamma(n):
    def f(t):
        import torch  # noqa: PLC0415

        return torch.special.polygamma(n, t)

    return f


def _scipy_polygamma(n):
    def f(x):
        import scipy.special as sp  # noqa: PLC0415

        return sp.polygamma(n, x)

    return f


def _jax_polygamma(n):
    def f(x):
        import jax.numpy as jnp  # noqa: PLC0415
        from jax.scipy.special import polygamma  # noqa: PLC0415

        return polygamma(jnp.asarray(n, dtype=x.dtype), x)

    return f


def _M(fn_name):
    """mpmath function by name, as a reference."""

    def f(x):
        return getattr(_mp(), fn_name)(x)

    return f


def _torch_zeta(t):
    import torch  # noqa: PLC0415

    return torch.special.zeta(t, torch.ones_like(t))


def _scipy_zeta(x):
    import scipy.special as sp  # noqa: PLC0415

    return sp.zeta(x, 1.0)


def _jax_zeta(x):
    from jax.scipy.special import zeta  # noqa: PLC0415

    return zeta(x, 1.0)


def _scipy_airy(x):
    import scipy.special as sp  # noqa: PLC0415

    return sp.airy(x)[0]


def _torch_gelu_tanh(t):
    import torch  # noqa: PLC0415

    return torch.nn.functional.gelu(t, approximate="tanh")


def _jax_gelu_exact(x):
    import jax  # noqa: PLC0415

    return jax.nn.gelu(x, approximate=False)


def _jax_gelu_tanh(x):
    import jax  # noqa: PLC0415

    return jax.nn.gelu(x, approximate=True)


def _scipy_spherical_j0(x):
    import scipy.special as sp  # noqa: PLC0415

    return sp.spherical_jn(0, x)


REGISTRY: List[Entry] = [
    # elementary
    Entry("exp", _M("exp"), ("symlog", 1e-20, 700), "exp", "exp", None, "numpy.exp", "exp"),
    Entry("exp2", lambda x: _mp().power(2, x), ("symlog", 1e-20, 1000), "exp2", "exp2", None, "numpy.exp2", "exp2"),
    Entry("expm1", _M("expm1"), ("symlog", 1e-20, 700), "expm1", "expm1", None, "numpy.expm1", "expm1"),
    Entry("log", _M("log"), ("log", 1e-300, 1e300), "log", "log", None, "numpy.log", "log"),
    Entry("log2", lambda x: _mp().log(x, 2), ("log", 1e-300, 1e300), "log2", "log2", None, "numpy.log2", "log2"),
    Entry("log10", _M("log10"), ("log", 1e-300, 1e300), "log10", "log10", None, "numpy.log10", "log10"),
    Entry("log1p", _M("log1p"), ("log", 1e-300, 1e300), "log1p", "log1p", None, "numpy.log1p", "log1p"),
    Entry("sqrt", _M("sqrt"), ("log", 1e-300, 1e300), "sqrt", "sqrt", None, "numpy.sqrt", "sqrt"),
    Entry("rsqrt", lambda x: 1 / _mp().sqrt(x), ("log", 1e-300, 1e300), "rsqrt", None, None, "lax.rsqrt", "rsqrt"),
    Entry("reciprocal", lambda x: 1 / x, ("symlog", 1e-300, 1e300), "reciprocal", "reciprocal", None, "numpy.reciprocal", "reciprocal"),
    Entry("sin", _M("sin"), ("symlog", 1e-8, 1e6), "sin", "sin", None, "numpy.sin", "sin"),
    Entry("cos", _M("cos"), ("symlog", 1e-8, 1e6), "cos", "cos", None, "numpy.cos", "cos"),
    Entry("tan", _M("tan"), ("symlog", 1e-8, 1e6), "tan", "tan", None, "numpy.tan", "tan"),
    Entry("asin", _M("asin"), ("unitsym",), "asin", "arcsin", None, "numpy.arcsin", "asin"),
    Entry("acos", _M("acos"), ("unitsym",), "acos", "arccos", None, "numpy.arccos", "acos"),
    Entry("atan", _M("atan"), ("symlog", 1e-8, 1e6), "atan", "arctan", None, "numpy.arctan", "atan"),
    Entry("sinh", _M("sinh"), ("symlog", 1e-20, 700), "sinh", "sinh", None, "numpy.sinh", "sinh"),
    Entry("cosh", _M("cosh"), ("symlog", 1e-8, 700), "cosh", "cosh", None, "numpy.cosh", "cosh"),
    Entry("tanh", _M("tanh"), ("symlog", 1e-20, 30), "tanh", "tanh", None, "numpy.tanh", "tanh"),
    Entry("asinh", _M("asinh"), ("symlog", 1e-20, 1e300), "asinh", "arcsinh", None, "numpy.arcsinh", "asinh"),
    Entry("acosh", _M("acosh"), ("ge1", 1e300), "acosh", "arccosh", None, "numpy.arccosh", "acosh"),
    Entry("atanh", _M("atanh"), ("unitsym",), "atanh", "arctanh", None, "numpy.arctanh", "atanh"),
    Entry("erf", _M("erf"), ("symlog", 1e-8, 6), "erf", None, "erf", "scipy.special.erf", "erf"),
    Entry("erfc", _M("erfc"), ("symlog", 1e-8, 27), "erfc", None, "erfc", "scipy.special.erfc", "erfc"),
    Entry("erfinv", _ref_erfinv, ("unitsym",), "erfinv", None, "erfinv", "scipy.special.erfinv", "erfinv"),
    Entry("lgamma", _ref_lgamma, ("symlog", 1e-8, 1e5), "lgamma", None, "gammaln", "scipy.special.gammaln", "lgamma"),
    Entry("digamma", _ref_digamma, ("symlog", 1e-8, 1e6), "digamma", None, "psi", "scipy.special.digamma", "digamma"),
    Entry("sigmoid", lambda x: 1 / (1 + _mp().exp(-x)), ("symlog", 1e-8, 100), "sigmoid", None, "expit", "nn.sigmoid", "sigmoid"),
    # torch.special
    Entry("erfcx", lambda x: _mp().exp(x * x) * _mp().erfc(x), ("symlog", 1e-8, 25), "special.erfcx", None, "erfcx", None, "special.erfcx"),
    Entry("i0", lambda x: _mp().besseli(0, x), ("symlog", 1e-8, 80), "special.i0", None, "i0", "scipy.special.i0", "special.i0"),
    Entry("i0e", lambda x: _mp().exp(-abs(x)) * _mp().besseli(0, x), ("symlog", 1e-8, 1e6), "special.i0e", None, "i0e", "scipy.special.i0e", "special.i0e"),
    Entry("i1", lambda x: _mp().besseli(1, x), ("symlog", 1e-8, 80), "special.i1", None, "i1", "scipy.special.i1", "special.i1"),
    Entry("i1e", lambda x: _mp().exp(-abs(x)) * _mp().besseli(1, x), ("symlog", 1e-8, 1e6), "special.i1e", None, "i1e", "scipy.special.i1e", "special.i1e"),
    Entry("ndtri", _ref_ndtri, ("unit",), "special.ndtri", None, "ndtri", "scipy.special.ndtri", "special.ndtri"),
    Entry("log_ndtr", _ref_log_ndtr, ("symlog", 1e-8, 40), "special.log_ndtr", None, "log_ndtr", "scipy.special.log_ndtr", "special.log_ndtr"),
    Entry("logit", lambda p: _mp().log(p / (1 - p)), ("unit",), "special.logit", None, "logit", "scipy.special.logit", "logit"),
    Entry("sinc", _ref_sinc, ("symlog", 1e-8, 1e5), "special.sinc", "sinc", None, "numpy.sinc", "special.sinc"),
    Entry("entr", _ref_entr, ("log", 1e-300, 1e5), "special.entr", None, "entr", "scipy.special.entr", "special.entr"),
    Entry("bessel_j0", lambda x: _mp().besselj(0, x), ("symlog", 1e-8, 1e4), "special.bessel_j0", None, "j0", None, "special.bessel_j0"),
    Entry("bessel_j1", lambda x: _mp().besselj(1, x), ("symlog", 1e-8, 1e4), "special.bessel_j1", None, "j1", None, "special.bessel_j1"),
    Entry("bessel_y0", lambda x: _mp().bessely(0, x), ("log", 1e-8, 1e4), "special.bessel_y0", None, "y0", None, "special.bessel_y0"),
    Entry("bessel_y1", lambda x: _mp().bessely(1, x), ("log", 1e-8, 1e4), "special.bessel_y1", None, "y1", None, "special.bessel_y1"),
    Entry("modified_bessel_i0", lambda x: _mp().besseli(0, x), ("symlog", 1e-8, 80), "special.modified_bessel_i0", None, "i0", None, "special.modified_bessel_i0"),
    Entry("modified_bessel_i1", lambda x: _mp().besseli(1, x), ("symlog", 1e-8, 80), "special.modified_bessel_i1", None, "i1", None, "special.modified_bessel_i1"),
    Entry("modified_bessel_k0", lambda x: _mp().besselk(0, x), ("log", 1e-8, 80), "special.modified_bessel_k0", None, "k0", None, "special.modified_bessel_k0"),
    Entry("modified_bessel_k1", lambda x: _mp().besselk(1, x), ("log", 1e-8, 80), "special.modified_bessel_k1", None, "k1", None, "special.modified_bessel_k1"),
    Entry("scaled_modified_bessel_k0", lambda x: _mp().exp(x) * _mp().besselk(0, x), ("log", 1e-8, 1e6), "special.scaled_modified_bessel_k0", None, "k0e", None, "special.scaled_modified_bessel_k0"),
    Entry("scaled_modified_bessel_k1", lambda x: _mp().exp(x) * _mp().besselk(1, x), ("log", 1e-8, 1e6), "special.scaled_modified_bessel_k1", None, "k1e", None, "special.scaled_modified_bessel_k1"),
    Entry("spherical_bessel_j0", _ref_spherical_j0, ("symlog", 1e-8, 1e5), "special.spherical_bessel_j0", None, _scipy_spherical_j0, None, "special.spherical_bessel_j0"),
    Entry("airy_ai", _M("airyai"), ("symlog", 1e-8, 50), "special.airy_ai", None, _scipy_airy, None, "special.airy_ai"),
    Entry("polygamma_1", _ref_polygamma(1), ("symlog", 1e-8, 1e5), _torch_polygamma(1), None, _scipy_polygamma(1), _jax_polygamma(1), "polygamma", "trigamma"),
    Entry("polygamma_2", _ref_polygamma(2), ("symlog", 1e-8, 1e5), _torch_polygamma(2), None, _scipy_polygamma(2), _jax_polygamma(2), "polygamma"),
    Entry("zeta", _ref_zeta, ("ge1", 1e3), _torch_zeta, None, _scipy_zeta, _jax_zeta, "special.zeta", "Riemann zeta, q = 1"),
    # activations
    Entry("gelu", _ref_gelu, ("symlog", 1e-8, 40), "nn.functional.gelu", None, None, _jax_gelu_exact, "nn.functional.gelu"),
    Entry("gelu_tanh", _ref_gelu_tanh, ("symlog", 1e-8, 40), _torch_gelu_tanh, None, None, _jax_gelu_tanh, None),
    Entry("silu", lambda x: x / (1 + _mp().exp(-x)), ("symlog", 1e-8, 100), "nn.functional.silu", None, None, "nn.silu", "nn.functional.silu"),
    Entry("mish", _ref_mish, ("symlog", 1e-8, 100), "nn.functional.mish", None, None, "nn.mish", "nn.functional.mish"),
    Entry("softplus", lambda x: _mp().log1p(_mp().exp(x)), ("symlog", 1e-8, 800), "nn.functional.softplus", None, None, "nn.softplus", "nn.functional.softplus"),
    Entry("logsigmoid", lambda x: -_mp().log1p(_mp().exp(-x)), ("symlog", 1e-8, 800), "nn.functional.logsigmoid", None, None, "nn.log_sigmoid", "nn.functional.logsigmoid"),
    Entry("elu", lambda x: x if x > 0 else _mp().expm1(x), ("symlog", 1e-8, 100), "nn.functional.elu", None, None, "nn.elu", "nn.functional.elu"),
    Entry("selu", _ref_selu, ("symlog", 1e-8, 100), "nn.functional.selu", None, None, "nn.selu", "nn.functional.selu"),
]


# ----------------------------------------------------------------------------------------------
# backends


def _resolve(path: Any, root_name: str):
    if path is None:
        return None
    if callable(path):
        return path
    root = importlib.import_module(root_name)
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def backend_callable(backend: str, entry: Entry, dtype: str):
    """Return a numpy array -> numpy array callable for the backend, or None if unavailable."""
    try:
        if backend == "torch":
            import torch  # noqa: PLC0415

            fn = _resolve(entry.torch, "torch")
            if fn is None:
                return None
            tdt = {"f32": torch.float32, "f64": torch.float64}[dtype]

            def run(x):
                return fn(torch.from_numpy(np.ascontiguousarray(x)).to(tdt)).detach().cpu().numpy()

            return run
        if backend == "numpy":
            fn = _resolve(entry.numpy, "numpy")
            return (lambda x: np.asarray(fn(x))) if fn is not None else None
        if backend == "scipy":
            fn = _resolve(entry.scipy, "scipy.special")
            return (lambda x: np.asarray(fn(x))) if fn is not None else None
        if backend == "jax":
            import jax  # noqa: PLC0415
            import jax.numpy as jnp  # noqa: PLC0415

            jax.config.update("jax_enable_x64", True)
            fn = _resolve(entry.jax, "jax")
            if fn is None:
                return None

            def run(x):
                return np.asarray(fn(jnp.asarray(x, dtype=DTYPES[dtype])))

            return run
    except Exception:  # noqa: BLE001
        return None
    raise ValueError(f"unknown backend {backend!r}")


def torch_opinfo_tolerance(opinfo_name: Optional[str], dtype: str) -> Tuple[float, float]:
    """Effective (rtol, atol) of torch's TestUnaryUfuncs.test_reference_numerics_* for the op on CPU."""
    rtol, atol = DEFAULT_TOL[dtype]
    if opinfo_name is None:
        return rtol, atol
    try:
        import torch  # noqa: PLC0415
        from torch.testing._internal.common_device_type import precisionOverride, toleranceOverride  # noqa: PLC0415
        from torch.testing._internal.common_methods_invocations import op_db  # noqa: PLC0415
        from torch.testing._internal.opinfo.core import DecorateInfo  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return rtol, atol
    tdt = {"f32": torch.float32, "f64": torch.float64}[dtype]
    for op in op_db:
        if op.name != opinfo_name:
            continue
        for d in op.decorators:
            if isinstance(d, precisionOverride) and tdt in d.d:
                atol = max(atol, d.d[tdt])
            elif isinstance(d, DecorateInfo):
                if d.cls_name not in (None, "TestUnaryUfuncs"):
                    continue
                if d.test_name is not None and not d.test_name.startswith("test_reference_numerics"):
                    continue
                if d.device_type not in (None, "cpu"):
                    continue
                if d.dtypes is not None and tdt not in d.dtypes:
                    continue
                for dec in d.decorators:
                    if isinstance(dec, toleranceOverride) and tdt in dec.d:
                        t = dec.d[tdt]
                        atol, rtol = max(atol, t.atol), max(rtol, t.rtol)
                    elif isinstance(dec, precisionOverride) and tdt in dec.d:
                        atol = max(atol, dec.d[tdt])
        break
    return rtol, atol


# ----------------------------------------------------------------------------------------------
# measurement


def _ulp_error(got: float, ref, dtype: str):
    """Error of ``got`` in ulps of the dtype at ``ref`` (an mpf), or None when both are the same non finite."""
    mp = _mp()
    npdt = DTYPES[dtype]
    fmax = float(np.finfo(npdt).max)
    if ref is None:  # domain error: the backend should return nan
        return 0.0 if math.isnan(got) else math.inf
    if math.isnan(got):
        return math.inf
    if mp.isinf(ref):
        return 0.0 if math.isinf(got) and (got > 0) == (ref > 0) else math.inf
    if math.isinf(got):
        # an overflowing exact result rounds to inf in the dtype, that is not an error
        return 0.0 if abs(ref) >= mp.mpf(fmax) * (1 + mp.mpf(float(np.finfo(npdt).eps)) / 2) and (got > 0) == (ref > 0) else math.inf
    r = float(npdt(float(ref))) if abs(ref) < mp.mpf(fmax) else float(np.copysign(fmax, float(ref)))
    spacing = float(np.spacing(npdt(abs(r))))
    return float(abs(mp.mpf(float(got)) - ref) / mp.mpf(float(spacing)))


@dataclass
class Result:
    function: str
    backend: str
    dtype: str
    n: int
    max_ulp: float
    p99_ulp: float
    median_ulp: float
    frac_gt1: float
    frac_gt10: float
    max_abs_err: float
    nonfinite_mismatch: int
    worst_x: float
    worst_got: float
    worst_ref: float
    vec_scalar_mismatch: Optional[int]
    default_tol_fail: Optional[int]
    op_tol_fail: Optional[int]
    op_rtol: Optional[float]
    op_atol: Optional[float]
    seconds: float


def survey(
    backends: Sequence[str] = ("torch", "numpy", "scipy", "jax"),
    dtypes: Sequence[str] = ("f32", "f64"),
    points: int = 600,
    functions: Optional[Sequence[str]] = None,
    prec: int = 200,
    log=None,
) -> List[Result]:
    """Run the survey and return one Result per (function, backend, dtype) that could be evaluated."""
    mp = _mp()
    mp.mp.prec = prec
    out: List[Result] = []
    for entry in REGISTRY:
        if functions and entry.name not in functions:
            continue
        for dtype in dtypes:
            npdt = DTYPES[dtype]
            xs = grid(entry.domain, dtype, points)
            t0 = time.time()
            refs = [_mpf_ref(entry.ref, float(x)) for x in xs]
            tref = time.time() - t0
            for backend in backends:
                fn = backend_callable(backend, entry, dtype)
                if fn is None:
                    continue
                t1 = time.time()
                try:
                    got = np.asarray(fn(xs))
                except Exception as ex:  # noqa: BLE001
                    if log:
                        print(f"{entry.name} {backend} {dtype}: error {ex}", file=log)
                    continue
                if got.shape != xs.shape:
                    continue
                if got.dtype != npdt:
                    got = got.astype(np.float64)  # scipy may return float64 for float32 input
                errs = np.array([_ulp_error(float(g), r, dtype) for g, r in zip(got.tolist(), refs)])
                finite_ref = np.array([r is not None and not mp.isinf(r) for r in refs])
                nonfinite_mismatch = int(np.isinf(errs).sum())
                e = errs[np.isfinite(errs)]
                worst = int(np.nanargmax(np.where(np.isfinite(errs), errs, -1.0))) if len(errs) else 0
                absdiff = np.array(
                    [abs(float(g) - float(r)) if (r is not None and not mp.isinf(r) and math.isfinite(g)) else 0.0 for g, r in zip(got.tolist(), refs)]
                )
                vec_mismatch = default_fail = op_fail = None
                op_rtol = op_atol = None
                if backend == "torch":
                    scalar = np.array([float(fn(xs[i : i + 1])[0]) for i in range(len(xs))], dtype=np.float64)
                    g64 = got.astype(np.float64)
                    vec_mismatch = int(((scalar != g64) & ~(np.isnan(scalar) & np.isnan(g64))).sum())
                    refd = np.array([float(r) if (r is not None and not mp.isinf(r)) else np.nan for r in refs])
                    rtol, atol = DEFAULT_TOL[dtype]
                    default_fail = int((absdiff[finite_ref] > atol + rtol * np.abs(refd[finite_ref])).sum())
                    op_rtol, op_atol = torch_opinfo_tolerance(entry.opinfo, dtype)
                    op_fail = int((absdiff[finite_ref] > op_atol + op_rtol * np.abs(refd[finite_ref])).sum())
                res = Result(
                    entry.name,
                    backend,
                    dtype,
                    int(len(xs)),
                    float(e.max()) if len(e) else math.nan,
                    float(np.percentile(e, 99)) if len(e) else math.nan,
                    float(np.median(e)) if len(e) else math.nan,
                    float((e > 1).mean()) if len(e) else math.nan,
                    float((e > 10).mean()) if len(e) else math.nan,
                    float(absdiff.max()) if len(absdiff) else math.nan,
                    nonfinite_mismatch,
                    float(xs[worst]),
                    float(got[worst]),
                    float(refs[worst]) if refs[worst] is not None else math.nan,
                    vec_mismatch,
                    default_fail,
                    op_fail,
                    op_rtol,
                    op_atol,
                    round(time.time() - t1 + tref, 2),
                )
                out.append(res)
                if log:
                    print(
                        f"{entry.name:28s} {backend:6s} {dtype} max {res.max_ulp:10.3g} ulp  p99 {res.p99_ulp:8.3g}  "
                        f"median {res.median_ulp:6.3g}  nonfinite {nonfinite_mismatch}"
                        + (f"  vec!=scalar {vec_mismatch}  default_fail {default_fail}  op_fail {op_fail}" if backend == "torch" else ""),
                        file=log,
                        flush=True,
                    )
    return out


def write_csv(results: List[Result], path: str) -> None:
    fields = list(Result.__dataclass_fields__)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in fields})


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if v == 0:
        return "0"
    if v < 10:
        return f"{v:.2g}"
    return f"{v:,.0f}" if v < 1e6 else f"{v:.1e}"


def write_markdown(results: List[Result], path: str, versions: Dict[str, str]) -> None:
    """Pivot table: one row per function and dtype, one column per backend, plus the torch tolerance columns."""
    backends = [b for b in ("torch", "numpy", "scipy", "jax") if any(r.backend == b for r in results)]
    by = {(r.function, r.backend, r.dtype): r for r in results}
    lines = ["# Accuracy survey", ""]
    lines.append("Max error in ulps of the dtype against mpmath at 200 bits, about 60 decimal digits (p99 in brackets), on log spaced inputs over each function's domain plus the named edge values of the dtype. The reference is evaluated at the rounded input, so the numbers measure the implementation and not the conditioning of the function. An error is |got - exact| divided by the spacing of the dtype at the exact value, which is why an exact result of zero or a result in the subnormal range gives enormous numbers: read those rows together with the notes in README.md next to this file.")
    lines.append("")
    lines.append("Versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    lines.append("")
    lines.append("torch columns: `vec!=scalar` is the number of inputs where the vectorized kernel and the scalar tail disagree; `default fail` and `op fail` count inputs that fail torch's reference test tolerance for the dtype, with the default tolerance and with the op's own override respectively. A row with `default fail > 0` and `op fail = 0` is an error that torch's test suite tolerates because of the override.")
    lines.append("")
    head = ["function", "dtype"] + [f"{b} max (p99)" for b in backends] + ["torch vec!=scalar", "torch default fail", "torch op fail", "op (rtol, atol)"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    functions = []
    for r in results:
        if r.function not in functions:
            functions.append(r.function)
    for fn in functions:
        for dtype in ("f32", "f64"):
            row = [fn, dtype]
            any_row = False
            for b in backends:
                r = by.get((fn, b, dtype))
                if r is None:
                    row.append("")
                else:
                    any_row = True
                    cell = f"{_fmt(r.max_ulp)} ({_fmt(r.p99_ulp)})"
                    if r.nonfinite_mismatch:
                        cell += f" +{r.nonfinite_mismatch} nonfinite"
                    row.append(cell)
            t = by.get((fn, "torch", dtype))
            if t is not None:
                row += [str(t.vec_scalar_mismatch), str(t.default_tol_fail), str(t.op_tol_fail), f"({t.op_rtol:g}, {t.op_atol:g})"]
            else:
                row += ["", "", "", ""]
            if any_row:
                lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    masked = [r for r in results if r.backend == "torch" and r.default_tol_fail and not r.op_tol_fail]
    if masked:
        lines.append("## Errors tolerated by a torch override")
        lines.append("")
        for r in masked:
            lines.append(
                f"- `{r.function}` {r.dtype}: {r.default_tol_fail} of {r.n} inputs fail the default tolerance, none fail the op's (rtol {r.op_rtol:g}, atol {r.op_atol:g}); worst x = {r.worst_x!r}, got {r.worst_got!r}, exact {r.worst_ref!r} ({_fmt(r.max_ulp)} ulp)."
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def versions() -> Dict[str, str]:
    out = {"ulpwise": getattr(ulpwise, "__version__", "?"), "numpy": np.__version__}
    for name in ("torch", "scipy", "jax", "mpmath"):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001
            pass
    try:
        import torch  # noqa: PLC0415

        cfg = torch.__config__.show()
        for line in cfg.splitlines():
            if "CPU capability" in line:
                out["torch cpu capability"] = line.split(":")[-1].strip()
    except Exception:  # noqa: BLE001
        pass
    return out


def main(args) -> int:
    functions = [f.strip() for f in args.functions.split(",")] if args.functions else None
    results = survey(
        backends=[b.strip() for b in args.backends.split(",")],
        dtypes=[d.strip() for d in args.dtypes.split(",")],
        points=args.points,
        functions=functions,
        log=sys.stderr,
    )
    import os  # noqa: PLC0415

    os.makedirs(args.out, exist_ok=True)
    write_csv(results, os.path.join(args.out, "results.csv"))
    write_markdown(results, os.path.join(args.out, "results.md"), versions())
    print(f"{len(results)} rows written to {args.out}/results.csv and results.md", file=sys.stderr)
    return 0
