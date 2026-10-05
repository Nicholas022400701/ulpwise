"""Unit tests for the survey machinery that do not need any backend."""

import importlib.util
import math
import sys

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from ulpwise import survey  # noqa: E402


def test_grid_is_sorted_unique_finite_and_in_dtype():
    xs = survey.grid(("symlog", 1e-8, 100), "f32", 200)
    assert xs.dtype == np.float32
    assert np.all(np.isfinite(xs))
    assert np.all(np.diff(xs) > 0)
    # the named edge values of the dtype inside the range are part of the grid
    tiny = np.float32(np.finfo(np.float32).tiny)
    assert tiny in xs


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("kind", ["log", "symlog", "lin", "unit", "unitsym", "ge1"])
def test_grid_half_dtypes_hold_representable_values_and_their_edges(dtype, kind):
    import ulpwise

    domain = {
        "log": ("log", 1e-300, 1e300),
        "symlog": ("symlog", 1e-300, 1e300),
        "lin": ("lin", -4, 4),
        "unit": ("unit",),
        "unitsym": ("unitsym",),
        "ge1": ("ge1", 1e300),
    }[kind]
    xs = survey.grid(domain, dtype, 64)
    assert xs.dtype == survey.DTYPES[dtype]  # bfloat16 values live in a float32 array
    assert np.all(np.isfinite(xs))
    assert np.all(np.diff(xs) > 0)
    for x in xs.tolist():  # every point is a float of the dtype, not just of the array's dtype
        assert ulpwise.next_up(ulpwise.next_down(x, dtype), dtype) == x, x
    edges = [v for _, v in ulpwise.special(dtype) if math.isfinite(v) and xs[0] <= v <= xs[-1]]
    assert edges and all(v in xs for v in edges)
    below = ulpwise.next_down(1.0, dtype)
    if kind == "unit":  # open interval: the grid used to reach 1.0 in the half dtypes, 1 - 1e-4 rounds to 1 in
        assert 0 < xs[0] < 1e-3 and xs[-1] == below  # float16 and 1 - 1e-3 in bfloat16
    if kind == "unitsym":
        assert xs[0] == -below and xs[-1] == below
    if kind == "ge1":  # clipped to the largest float too: 1 + logspace up to 1e300 kept a handful of points
        assert xs[0] == 1.0 and xs[-1] == survey._fmax(dtype) and len(xs) >= 60
    if kind in ("log", "symlog"):  # clipped to the finite range of the dtype, so the points are not wasted
        assert xs[-1] == survey._fmax(dtype) and len(xs) >= 60
        assert xs[0] == (ulpwise.next_up(0.0, dtype) if kind == "log" else -survey._fmax(dtype))


def test_grid_clips_a_wide_log_domain_to_the_range_of_the_dtype():
    import ulpwise

    for dtype, lo, hi in (("f16", 2**-24, 65504.0), ("bf16", 2**-133, 2**128 - 2**120), ("f32", 2**-149, 2**128 - 2**104)):
        xs = survey.grid(("log", 1e-300, 1e300), dtype, 600)
        assert float(xs[0]) == lo == ulpwise.next_up(0.0, dtype), dtype  # the smallest subnormal
        assert float(xs[-1]) == hi == max(v for _, v in ulpwise.special(dtype) if math.isfinite(v)), dtype
        assert len(xs) >= 570, (dtype, len(xs))  # the float16 grid had 32 points when the 1e-300 .. 1e300 points were spaced first
    xs = survey.grid(("log", 1e-300, 1e300), "f64", 600)  # inside the float64 range, nothing to clip
    assert float(xs[0]) == 1e-300 and float(xs[-1]) == 1e300 and len(xs) >= 600


def test_fmax_matches_the_largest_edge_value():
    import ulpwise

    for dtype in survey.DTYPES:
        assert survey._fmax(dtype) == max(v for _, v in ulpwise.special(dtype) if math.isfinite(v))
    assert survey._fmax("f16") == 65504.0 and survey._fmax("bf16") == float(2**128 - 2**120)


def test_ulp_error_is_half_for_a_correctly_rounded_result():
    mp.mp.prec = 200
    exact = mp.sqrt(mp.mpf(2))
    got = float(np.float64(math.sqrt(2.0)))
    err = survey._ulp_error(got, exact, "f64")
    assert err <= 0.5
    # the neighbour on the far side of the exact value is exactly one ulp further away
    far = float(np.nextafter(got, np.inf if got > exact else -np.inf))
    assert abs(survey._ulp_error(far, exact, "f64") - (err + 1)) < 1e-9


def test_ulp_error_treats_non_finite_results():
    assert survey._ulp_error(math.nan, None, "f64") == 0.0  # domain error, nan expected
    assert survey._ulp_error(1.0, None, "f64") == math.inf
    assert survey._ulp_error(math.inf, mp.mpf("inf"), "f64") == 0.0
    assert survey._ulp_error(math.inf, mp.mpf(1), "f64") == math.inf


def test_ulp_error_in_the_half_dtypes():
    mp.mp.prec = 200
    one = mp.mpf(1)
    assert survey._ulp_error(1 + 2**-7, one, "bf16") == 1.0  # the bfloat16 next to 1
    assert survey._ulp_error(1 + 2**-10, one, "f16") == 1.0  # the float16 next to 1
    assert survey._ulp_error(1 + 2**-10, one, "bf16") == 2**-3  # a float32 step is an eighth of a bfloat16 ulp
    assert survey._ulp_error(0.0, mp.mpf(2) ** -25, "f16") == 0.5  # half the smallest subnormal, rounds to 0
    assert survey._ulp_error(65504.0, mp.mpf(65504 + 16), "f16") == 0.5  # the spacing at the largest float16 is 32
    assert survey._ulp_error(math.inf, mp.mpf(65520), "f16") == 0.0  # 65504 + 16 rounds to inf (ties to even)
    assert survey._ulp_error(math.inf, mp.mpf(65519), "f16") == math.inf
    bmax = mp.mpf(survey._fmax("bf16"))
    assert survey._ulp_error(math.inf, bmax + mp.mpf(2) ** 119, "bf16") == 0.0  # half a bfloat16 ulp above the max
    assert survey._ulp_error(math.inf, bmax + mp.mpf(2) ** 118, "bf16") == math.inf


def test_ulp_error_at_the_largest_float_and_the_rounding_threshold_to_infinity():
    # at the largest finite value numpy's spacing is inf, which made every finite result there a 0 ulp error, and
    # the overflow threshold was fmax * (1 + eps / 2), about one ulp above fmax, half an ulp too high: an exact
    # result between fmax + ulp/2 and fmax + ulp rounds to inf in the dtype, so a backend returning inf is right
    import ulpwise

    mp.mp.prec = 200
    for dtype in ("f64", "f32", "f16", "bf16"):
        fmax = mp.mpf(survey._fmax(dtype))
        ulp = mp.mpf(ulpwise.spacing(float(fmax), dtype))
        assert mp.isfinite(ulp) and ulp > 0
        got = float(fmax)
        assert abs(survey._ulp_error(got, fmax + ulp * mp.mpf("0.4"), dtype) - 0.4) < 1e-12, dtype
        assert survey._ulp_error(got, fmax, dtype) == 0.0
        assert survey._ulp_error(math.inf, fmax + ulp * mp.mpf("0.75"), dtype) == 0.0, dtype
        assert survey._ulp_error(math.inf, fmax + ulp / 2, dtype) == 0.0, dtype
        assert survey._ulp_error(math.inf, fmax + ulp * mp.mpf("0.25"), dtype) == math.inf, dtype
        assert survey._ulp_error(-math.inf, -(fmax + ulp), dtype) == 0.0, dtype
        assert survey._ulp_error(math.inf, -(fmax + ulp), dtype) == math.inf, dtype  # wrong sign of infinity


def test_reference_formulas_keep_the_tails():
    mp.mp.prec = 200
    reg = {e.name: e for e in survey.REGISTRY}
    x = mp.mpf(-149.20600729076193)
    assert reg["softplus"].ref(x) > 0  # log1p keeps exp(-149), log(1 + exp(-149)) at 60 digits does not
    assert reg["logsigmoid"].ref(-x) < 0
    assert reg["sinc"].ref(mp.mpf(100000)) == 0  # exact at the integers
    assert reg["gelu_tanh"].ref(mp.mpf(-7.3)) < 0


def test_ndtri_reference_needs_no_scipy(monkeypatch):
    # the reference used to start its Newton iteration from scipy's ndtri, so a survey without scipy installed
    # died at ndtri with a ModuleNotFoundError and lost every row before it
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.special", None)
    mp = survey._mp()
    mp.mp.prec = 200
    for p in (5e-324, 1e-300, 1e-38, 0.02425, 0.3, 0.7, 1 - 1e-16, 1 - 2**-53):
        x = survey._mpf_ref(survey._ref_ndtri, p)
        assert abs(mp.ncdf(x) - p) <= mp.mpf(10) ** -50 * p, p
    assert survey._mpf_ref(survey._ref_ndtri, 0.5) == 0  # exactly, torch and scipy return 0.0 there
    assert survey._mpf_ref(survey._ref_ndtri, 0.75) == -survey._mpf_ref(survey._ref_ndtri, 0.25)  # both exact doubles
    assert survey._mpf_ref(survey._ref_ndtri, 0.0) == mp.mpf("-inf")
    assert survey._mpf_ref(survey._ref_ndtri, 1.0) == mp.mpf("inf")
    assert survey._mpf_ref(survey._ref_ndtri, -0.1) is None and survey._mpf_ref(survey._ref_ndtri, 1.1) is None


def test_registry_names_are_unique():
    names = [e.name for e in survey.REGISTRY]
    assert len(names) == len(set(names))


# ----------------------------------------------------------------------------------------------
# the survey itself, the backend plumbing and the writers


def _has(module):
    import importlib.util

    return importlib.util.find_spec(module) is not None


def _no_op_db():
    # torch.testing._internal imports expecttest, which a plain torch install does not bring
    return not _has("torch") or survey.opinfo_unavailable() is not None


def _sqrt_entry():
    return next(e for e in survey.REGISTRY if e.name == "sqrt")


@pytest.mark.parametrize("entry", survey.REGISTRY, ids=lambda e: e.name)
def test_every_reference_agrees_with_the_libraries_it_measures(entry):
    # A wrong reference (another function, a sign, a swapped argument) is thousands of ulps from every library
    # on most of the grid, a right one is within a few even where a library has a bad point: on the 24 point
    # float64 grid the worst median is torch's polygamma_1 at 9 ulp, log2 with a natural log reference is 3e15.
    backends = [b for b in ("torch", "numpy", "scipy") if _has(b)]
    results = survey.survey(backends=backends, dtypes=("f64",), points=24, functions=[entry.name])
    if not results:
        pytest.skip(f"{entry.name} is not implemented by any of {backends}")
    for r in results:
        assert r.n >= 20
        assert r.median_ulp <= 32, (r.backend, r.median_ulp, r.worst_x, r.worst_got, r.worst_ref)


@pytest.mark.skipif(not _has("torch"), reason="needs torch as the second opinion")
@pytest.mark.parametrize(
    ("name", "x", "expected"),
    [
        # the median above does not see a slip in one branch of a piecewise reference: selu with alpha 1.77 in
        # place of 1.67 moves a third of the grid and keeps the median; one point per branch does see it
        ("selu", -1.0, "torch"),
        ("selu", 1.0, "torch"),
        ("elu", -1.0, "torch"),
        ("elu", 1.0, "torch"),
        ("entr", 0.5, "torch"),
        ("entr", 0.0, 0.0),
        ("entr", -1.0, -math.inf),
        ("gelu_tanh", -1.0, "torch"),  # at -7.3 torch's formula cancels to -0.0, that is the survey's finding
        ("gelu_tanh", 1.0, "torch"),
        ("spherical_bessel_j0", 0.0, 1.0),
        ("spherical_bessel_j0", 2.0, "torch"),
        ("sinc", 0.5, "torch"),
        ("log_ndtr", -5.0, "torch"),
        ("log_ndtr", 1.0, "torch"),  # the log1p branch; at 5 torch is 5 ulp off, that is the survey's finding
        ("lgamma", -2.0, math.inf),
        ("lgamma", -2.5, "torch"),
        ("digamma", -2.0, None),  # torch returns nan at the poles
        ("erfinv", 1.0, math.inf),
        ("erfinv", -1.0, -math.inf),
        ("ndtri", 0.0, -math.inf),
        ("ndtri", 1.0, math.inf),
        ("zeta", 1.0, math.inf),
        ("zeta", 0.5, None),  # torch.special.zeta returns nan below 1
    ],
)
def test_piecewise_references_in_each_branch(name, x, expected):
    mp.mp.prec = 200
    entry = next(e for e in survey.REGISTRY if e.name == name)
    ref = survey._mpf_ref(entry.ref, x)
    if expected == "torch":
        got = float(survey.backend_callable("torch", entry, "f64")(np.array([x], dtype=np.float64))[0])
        assert survey._ulp_error(got, ref, "f64") <= 4, (got, ref)
    elif expected is None:
        assert ref is None
    elif math.isinf(expected):
        assert mp.isinf(ref) and (ref > 0) == (expected > 0)
    else:
        assert ref == expected


def test_resolve_follows_dotted_paths_and_passes_callables_through():
    assert survey._resolve(None, "numpy") is None
    assert survey._resolve(np.exp, "numpy") is np.exp
    assert survey._resolve("linalg.norm", "numpy") is np.linalg.norm
    assert survey._resolve("linalg.no_such_function", "numpy") is None


def test_backend_callable_returns_none_for_a_missing_function_and_raises_for_an_unknown_backend():
    entry = survey.Entry("twice", lambda x: 2 * x, ("lin", 0, 1), numpy="no_such_function", torch=None)
    assert survey.backend_callable("numpy", entry, "f64") is None
    assert survey.backend_callable("torch", entry, "f64") is None  # the entry has no torch path
    with pytest.raises(ValueError, match="unknown backend"):
        survey.backend_callable("photon", entry, "f64")


def test_backend_callable_accepts_a_callable_and_keeps_the_input_dtype():
    entry = survey.Entry("twice", lambda x: 2 * x, ("lin", 0, 1), numpy=lambda x: x * 2)
    fn = survey.backend_callable("numpy", entry, "f32")
    xs = np.array([0.5, 1.5], dtype=np.float32)
    assert np.array_equal(fn(xs), np.array([1.0, 3.0], dtype=np.float32))


@pytest.mark.skipif(_has("jax"), reason="jax is installed, the unavailable path is not reachable")
def test_backend_callable_returns_none_when_jax_is_not_installed():
    assert survey.backend_callable("jax", _sqrt_entry(), "f64") is None


def test_survey_measures_sqrt_in_numpy_and_torch(monkeypatch):
    import io

    monkeypatch.setattr(survey, "REGISTRY", [_sqrt_entry()])
    log = io.StringIO()
    results = survey.survey(backends=("numpy", "torch", "scipy", "jax"), points=48, log=log)
    rows = {(r.backend, r.dtype): r for r in results}
    expected = {("numpy", "f32"), ("numpy", "f64")}
    if _has("torch"):
        expected |= {("torch", "f32"), ("torch", "f64")}
    assert set(rows) == expected  # the sqrt entry has no scipy path and jax is skipped when missing
    for (backend, dtype), r in rows.items():
        assert r.function == "sqrt"
        assert r.n == len(survey.grid(("log", 1e-300, 1e300), dtype, 48))
        if backend == "numpy":
            assert r.max_ulp <= 0.5  # IEEE sqrt, correctly rounded
        else:
            # torch's x86 sqrt is faithful but not correctly rounded (pytorch #198448, README table): under one ulp
            assert r.max_ulp < 1.0
        assert r.p99_ulp <= r.max_ulp
        assert r.median_ulp <= r.p99_ulp
        assert r.frac_gt1 == 0 and r.frac_gt10 == 0
        assert r.nonfinite_mismatch == 0
        assert math.isfinite(r.worst_x) and math.isfinite(r.worst_got) and math.isfinite(r.worst_ref)
        assert r.seconds >= 0
        if backend == "torch":
            assert r.vec_scalar_mismatch == 0
            assert r.default_tol_fail == 0 and r.op_tol_fail == 0
            assert r.op_rtol >= survey.DEFAULT_TOL[dtype][0] and r.op_atol >= survey.DEFAULT_TOL[dtype][1]
        else:
            assert r.vec_scalar_mismatch is None and r.default_tol_fail is None and r.op_tol_fail is None
            assert r.op_rtol is None and r.op_atol is None
    text = log.getvalue()
    assert text.count("sqrt") == len(results)
    assert "nonfinite 0" in text


def test_survey_evaluates_the_reference_only_when_a_backend_has_the_function(monkeypatch):
    def ref(x):
        raise AssertionError("the reference was evaluated with no backend to compare")

    entry = survey.Entry("only_jax", ref, ("log", 1e-300, 1e300), jax="numpy.sqrt")
    monkeypatch.setattr(survey, "REGISTRY", [entry])
    assert survey.survey(backends=("numpy", "torch", "scipy"), points=8) == []


def test_survey_is_silent_at_the_edge_values_of_the_grid():
    # the grid ends at 0 and at the overflow edge of the dtype, where numpy's reciprocal and log warn about a
    # division by zero and an overflow; the survey counts those points as non finite and must not pass the
    # warnings on, a 61 function run would otherwise print hundreds of them
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = survey.survey(backends=("numpy",), dtypes=("f32", "f64"), points=16, functions=["reciprocal", "log"])
    assert {(r.function, r.dtype) for r in results} == {(f, d) for f in ("reciprocal", "log") for d in ("f32", "f64")}
    assert all(r.nonfinite_mismatch == 0 for r in results if r.function == "log")  # log(0) = -inf on both sides


def test_survey_measures_an_inexact_backend_and_filters_by_function_name(monkeypatch):
    # a sqrt that is off by one ulp at every point: the survey must see max 1.5 ulp at most and never below 0.5
    def off_by_one(x):
        r = np.sqrt(x)
        return np.nextafter(r, np.inf * np.ones_like(r))

    sloppy = survey.Entry("sloppy_sqrt", survey._M("sqrt"), ("log", 1e-6, 1e6), numpy=off_by_one)
    exact = survey.Entry("exact_sqrt", survey._M("sqrt"), ("log", 1e-6, 1e6), numpy="sqrt")
    monkeypatch.setattr(survey, "REGISTRY", [sloppy, exact])
    results = survey.survey(backends=("numpy",), dtypes=("f64",), points=40, functions=["sloppy_sqrt"])
    assert [r.function for r in results] == ["sloppy_sqrt"]
    (r,) = results
    assert 0.5 <= r.median_ulp <= r.max_ulp <= 1.5
    assert r.frac_gt1 > 0  # most points are more than one ulp away after the extra step
    assert r.frac_gt10 == 0
    assert r.worst_got == float(off_by_one(np.array([r.worst_x]))[0])


def test_survey_skips_a_backend_that_raises_or_returns_the_wrong_shape(monkeypatch):
    import io

    def boom(x):
        raise RuntimeError("no kernel")

    raising = survey.Entry("raising", survey._M("sqrt"), ("log", 1e-6, 1e6), numpy=boom)
    scalar = survey.Entry("scalar", survey._M("sqrt"), ("log", 1e-6, 1e6), numpy=lambda x: np.sqrt(x[0]))
    monkeypatch.setattr(survey, "REGISTRY", [raising, scalar])
    log = io.StringIO()
    assert survey.survey(backends=("numpy",), dtypes=("f64",), points=24, log=log) == []
    assert "raising numpy f64: error no kernel" in log.getvalue()
    assert "scalar" not in log.getvalue()  # a wrong shape is dropped silently


def _exp_entry():
    return next(e for e in survey.REGISTRY if e.name == "exp")


def test_backend_callable_half_dtypes():
    entry = _exp_entry()
    assert survey.backend_callable("numpy", entry, "bf16") is None  # numpy has no bfloat16
    erf = next(e for e in survey.REGISTRY if e.name == "erf")  # exp has no scipy path, erf has scipy.special.erf
    if _has("scipy"):
        assert survey.backend_callable("scipy", erf, "f32") is not None
    assert survey.backend_callable("scipy", erf, "f16") is None  # scipy.special computes float16 in float32
    assert survey.backend_callable("scipy", erf, "bf16") is None
    xs = np.array([0.5, 1.5], dtype=np.float16)
    got = survey.backend_callable("numpy", entry, "f16")(xs)
    assert got.dtype == np.float16 and np.allclose(got, np.exp(xs))
    if _has("torch"):
        import torch

        got = survey.backend_callable("torch", entry, "f16")(xs)
        assert got.dtype == np.float16 and np.array_equal(got, torch.exp(torch.tensor([0.5, 1.5], dtype=torch.float16)).numpy())
        got = survey.backend_callable("torch", entry, "bf16")(np.array([0.5, 1.5], dtype=np.float32))
        assert got.dtype == np.float32  # bfloat16 values returned in a float32 array
        assert np.array_equal(got, torch.exp(torch.tensor([0.5, 1.5], dtype=torch.bfloat16)).float().numpy())


def test_survey_rejects_an_unknown_dtype_before_doing_anything():
    with pytest.raises(ValueError, match="unknown dtype 'f8': use f64, f32, f16 or bf16"):
        survey.survey(backends=("numpy",), dtypes=("f64", "f8"), points=8, functions=["exp"])


def test_survey_drops_a_backend_without_a_kernel_in_the_dtype_before_the_reference(monkeypatch):
    import io

    calls = []

    def ref(x):
        calls.append(x)
        return mp.sqrt(x)

    def picky(x):
        if x.dtype == np.float16:
            raise RuntimeError('"sqrt_cpu" not implemented for \'Half\'')
        return np.sqrt(x)

    monkeypatch.setattr(survey, "REGISTRY", [survey.Entry("picky", ref, ("log", 1e-3, 1e3), numpy=picky)])
    log = io.StringIO()
    results = survey.survey(backends=("numpy",), dtypes=("f16", "f32"), points=16, log=log)
    assert [(r.dtype, r.backend) for r in results] == [("f32", "numpy")]
    assert len(calls) == len(survey.grid(("log", 1e-3, 1e3), "f32", 16))  # no reference for the f16 grid
    assert "picky numpy f16: error \"sqrt_cpu\" not implemented for 'Half'" in log.getvalue()
    assert "picky numpy f32: error" not in log.getvalue()


def test_survey_measures_exp_in_the_half_dtypes(monkeypatch):
    import io

    monkeypatch.setattr(survey, "REGISTRY", [_exp_entry()])
    log = io.StringIO()
    results = survey.survey(backends=("numpy", "torch", "scipy"), dtypes=("f16", "bf16"), points=48, log=log)
    rows = {(r.backend, r.dtype): r for r in results}
    expected = {("numpy", "f16")}  # no numpy bf16 row, no scipy half rows
    if _has("torch"):
        expected |= {("torch", "f16"), ("torch", "bf16")}
    assert set(rows) == expected
    for (backend, dtype), r in rows.items():
        assert r.n == len(survey.grid(_exp_entry().domain, dtype, 48))
        assert r.max_ulp <= 1.0, (backend, dtype, r)  # exp computed in float32 and rounded once: within an ulp
        assert r.frac_gt1 == 0 and r.nonfinite_mismatch == 0  # inf at the overflow edge, 0 at the underflow edge
        assert math.isfinite(r.worst_x) and math.isfinite(r.worst_got)
        if backend == "torch":
            assert r.default_tol_fail == 0 and r.op_tol_fail == 0
            assert (r.op_rtol, r.op_atol) >= survey.DEFAULT_TOL[dtype]
    assert "exp" in log.getvalue() and "scipy" not in log.getvalue()


@pytest.mark.skipif(not _has("torch"), reason="torch is not installed")
def test_default_tolerances_are_those_of_torch_testing():
    import torch
    from torch.testing._comparison import default_tolerances

    for dtype, tdt in (("bf16", torch.bfloat16), ("f16", torch.float16), ("f32", torch.float32), ("f64", torch.float64)):
        assert default_tolerances(tdt) == survey.DEFAULT_TOL[dtype], dtype


def test_torch_opinfo_tolerance_defaults_without_an_op_and_never_tightens():
    assert survey.torch_opinfo_tolerance(None, "f32") == survey.DEFAULT_TOL["f32"]
    assert survey.torch_opinfo_tolerance("no_such_op_in_op_db", "f64") == survey.DEFAULT_TOL["f64"]
    for dtype in ("f32", "f64"):
        rtol0, atol0 = survey.DEFAULT_TOL[dtype]
        for entry in survey.REGISTRY:
            rtol, atol = survey.torch_opinfo_tolerance(entry.opinfo, dtype)
            assert rtol >= rtol0 and atol >= atol0, (entry.name, dtype)


@pytest.mark.skipif(_no_op_db(), reason="needs torch's op_db, torch plus expecttest")
def test_torch_opinfo_tolerance_reads_an_override_from_op_db():
    import torch
    from torch.testing._internal.common_device_type import precisionOverride, toleranceOverride
    from torch.testing._internal.common_methods_invocations import op_db
    from torch.testing._internal.opinfo.core import DecorateInfo

    # pick any unary op whose op_db entry loosens float32 for the CPU reference numerics tests
    for op in op_db:
        for d in op.decorators:
            if isinstance(d, precisionOverride) and torch.float32 in d.d:
                loosened, variant = (op.name, d.d[torch.float32], "atol"), op.variant_test_name
                break
            if (
                isinstance(d, DecorateInfo)
                and d.cls_name in (None, "TestUnaryUfuncs")
                and (d.test_name is None or d.test_name.startswith("test_reference_numerics"))
                and d.device_type in (None, "cpu")
                and (d.dtypes is None or torch.float32 in d.dtypes)
                and any(isinstance(dec, toleranceOverride) and torch.float32 in dec.d for dec in d.decorators)
            ):
                dec = next(dec for dec in d.decorators if isinstance(dec, toleranceOverride) and torch.float32 in dec.d)
                loosened, variant = (op.name, dec.d[torch.float32].atol, "atol"), op.variant_test_name
                break
        else:
            continue
        break
    else:
        pytest.skip("this torch has no float32 override in op_db")
    name, atol_override, _ = loosened
    rtol, atol = survey.torch_opinfo_tolerance(name, "f32", variant)
    assert atol >= atol_override
    assert atol >= survey.DEFAULT_TOL["f32"][1] and rtol >= survey.DEFAULT_TOL["f32"][0]


@pytest.mark.skipif(_no_op_db(), reason="needs torch's op_db, torch plus expecttest")
def test_torch_opinfo_tolerance_selects_the_op_db_variant():
    # polygamma has one op_db entry per order; the first, polygamma_n_0, has no float32 override and
    # polygamma_n_1 has one, so the lookup by name alone used to return the default for polygamma_1
    base = survey.torch_opinfo_tolerance("polygamma", "f32")
    n1 = survey.torch_opinfo_tolerance("polygamma", "f32", "polygamma_n_1")
    assert base == survey.DEFAULT_TOL["f32"]
    assert n1[0] > base[0] and n1[1] > base[1]
    assert survey.torch_opinfo_tolerance("polygamma", "f32", "no_such_variant") == survey.DEFAULT_TOL["f32"]
    reg = {e.name: e for e in survey.REGISTRY}
    assert reg["polygamma_1"].opinfo_variant == "polygamma_n_1"
    assert reg["polygamma_2"].opinfo_variant == "polygamma_n_2"
    # an op with a base variant and another variant resolves to the base one without a variant argument
    assert survey.torch_opinfo_tolerance("nn.functional.silu", "f32") == survey.torch_opinfo_tolerance("nn.functional.silu", "f32", "")


@pytest.mark.skipif(not _has("torch"), reason="needs torch as a backend")
def test_survey_says_once_when_op_db_cannot_be_read_and_uses_the_default(monkeypatch):
    import io

    reason = "ModuleNotFoundError: No module named 'expecttest'"
    uncached = survey._opinfo.__wrapped__
    monkeypatch.setattr(survey, "_opinfo", lambda: (None, reason))
    assert survey.opinfo_unavailable() == reason
    assert survey.torch_opinfo_tolerance("polygamma", "f32", "polygamma_n_1") == survey.DEFAULT_TOL["f32"]
    monkeypatch.setattr(survey, "REGISTRY", [_sqrt_entry()])
    log = io.StringIO()
    results = survey.survey(backends=("numpy", "torch"), points=24, log=log)
    torch_rows = [r for r in results if r.backend == "torch"]
    assert len(torch_rows) == 2
    for r in torch_rows:
        assert (r.op_rtol, r.op_atol) == survey.DEFAULT_TOL[r.dtype]
    text = log.getvalue()
    assert text.count("torch op_db unavailable") == 1
    assert reason in text and "expecttest" in text
    # nothing to say when torch is not surveyed, and no log means no crash
    log = io.StringIO()
    survey.survey(backends=("numpy",), points=24, log=log)
    assert "op_db" not in log.getvalue()
    assert len(survey.survey(backends=("torch",), dtypes=("f64",), points=24)) == 1
    # the uncached import itself: a module set to None in sys.modules raises on import
    monkeypatch.setitem(sys.modules, "torch.testing._internal.common_methods_invocations", None)
    imports, why = uncached()
    assert imports is None and why.split(":")[0] in ("ImportError", "ModuleNotFoundError")
    # and nothing to report when torch is not installed at all
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "torch" else find_spec(name, *a))
    assert survey.opinfo_unavailable() is None


def test_fmt_formats_by_magnitude():
    assert survey._fmt(None) == ""
    assert survey._fmt(math.nan) == ""
    assert survey._fmt(0) == "0"
    assert survey._fmt(0.5) == "0.5"
    assert survey._fmt(2.345) == "2.3"
    assert survey._fmt(1234.0) == "1,234"
    assert survey._fmt(2.5e6) == "2.5e+06"


def _result(function, backend, dtype, **kw):
    base = dict(
        n=10,
        max_ulp=3.0,
        p99_ulp=2.0,
        median_ulp=0.5,
        frac_gt1=0.1,
        frac_gt10=0.0,
        max_abs_err=1e-6,
        nonfinite_mismatch=0,
        worst_x=1.5,
        worst_got=1.2247,
        worst_ref=1.22474,
        vec_scalar_mismatch=None,
        default_tol_fail=None,
        op_tol_fail=None,
        op_rtol=None,
        op_atol=None,
        seconds=0.1,
    )
    base.update(kw)
    return survey.Result(function, backend, dtype, **base)


def test_write_csv_round_trips_every_result_field(tmp_path):
    import csv

    results = [
        _result("exp", "numpy", "f64"),
        _result("exp", "torch", "f64", vec_scalar_mismatch=1, default_tol_fail=3, op_tol_fail=0, op_rtol=1e-7, op_atol=1e-6),
    ]
    path = tmp_path / "results.csv"
    survey.write_csv(results, str(path))
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = list(survey.Result.__dataclass_fields__)
    assert list(rows[0]) == fields
    assert len(rows) == 2
    for row, r in zip(rows, results):
        assert row == {k: "" if getattr(r, k) is None else str(getattr(r, k)) for k in fields}  # csv writes None as empty


def test_write_markdown_pivots_backends_and_lists_tolerated_errors(tmp_path):
    results = [
        _result("exp", "numpy", "f64", nonfinite_mismatch=2),
        _result("exp", "torch", "f64", vec_scalar_mismatch=1, default_tol_fail=3, op_tol_fail=0, op_rtol=1e-7, op_atol=1e-6),
        _result("log", "numpy", "f32", max_ulp=0.5, p99_ulp=0.5),
    ]
    path = tmp_path / "results.md"
    survey.write_markdown(results, str(path), {"ulpwise": "0.3.0", "numpy": np.__version__})
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# Accuracy survey"
    assert f"Versions: ulpwise 0.3.0, numpy {np.__version__}" in lines
    head = next(line for line in lines if line.startswith("| function"))
    assert head == "| function | dtype | torch max (p99) | numpy max (p99) | torch vec!=scalar | torch default fail | torch op fail | op (rtol, atol) |"
    assert "| exp | f64 | 3 (2) | 3 (2) +2 nonfinite | 1 | 3 | 0 | (1e-07, 1e-06) |" in lines
    assert "| log | f32 |  | 0.5 (0.5) |  |  |  |  |" in lines
    assert not any(line.startswith("| exp | f32") or line.startswith("| log | f64") for line in lines)  # no result, no row
    assert "## Errors tolerated by a torch override" in lines
    assert any(line.startswith("- `exp` f64: 3 of 10 inputs fail the default tolerance, none fail the op's (rtol 1e-07, atol 1e-06); worst x = 1.5") for line in lines)


def test_write_markdown_lists_the_half_dtypes_in_dtype_order(tmp_path):
    results = [_result("exp", "numpy", d) for d in ("f64", "f16", "f32")] + [
        _result("exp", "torch", "bf16", vec_scalar_mismatch=0, default_tol_fail=0, op_tol_fail=0, op_rtol=1.6e-2, op_atol=1e-5)
    ]
    path = tmp_path / "results.md"
    survey.write_markdown(results, str(path), {})
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [line.split(" | ")[1] for line in lines if line.startswith("| exp |")]
    assert rows == ["bf16", "f16", "f32", "f64"]  # the order of DTYPES, only the dtypes with a result
    assert "| exp | bf16 | 3 (2) |  | 0 | 0 | 0 | (0.016, 1e-05) |" in lines
    assert "| exp | f16 |  | 3 (2) |  |  |  |  |" in lines


def test_write_markdown_without_tolerated_errors_has_no_override_section(tmp_path):
    path = tmp_path / "results.md"
    survey.write_markdown([_result("exp", "numpy", "f64")], str(path), {})
    text = path.read_text(encoding="utf-8")
    assert "Versions: " in text
    assert "## Errors tolerated by a torch override" not in text
    assert "| exp | f64 | 3 (2) |  |  |  |  |" in text


def test_versions_reports_the_installed_backends():
    v = survey.versions()
    assert v["numpy"] == np.__version__
    assert v["mpmath"] == mp.__version__
    assert "ulpwise" in v
    for name in ("torch", "scipy", "jax"):
        assert (name in v) == _has(name)


def test_main_writes_both_files_for_the_requested_function(tmp_path, monkeypatch, capsys):
    import csv
    from types import SimpleNamespace

    monkeypatch.setattr(survey, "REGISTRY", [_sqrt_entry()])
    out = tmp_path / "out"
    args = SimpleNamespace(functions="sqrt", backends="numpy", dtypes="f64", points=32, out=str(out))
    assert survey.main(args) == 0
    with open(out / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [(r["function"], r["backend"], r["dtype"]) for r in rows] == [("sqrt", "numpy", "f64")]
    assert float(rows[0]["max_ulp"]) <= 0.5
    assert "| sqrt | f64 |" in (out / "results.md").read_text(encoding="utf-8")
    assert "1 rows written to" in capsys.readouterr().err


def test_cli_survey_subcommand_filters_the_registry(tmp_path):
    from ulpwise.__main__ import main

    out = tmp_path / "survey"
    code = main(["survey", "--functions", "sqrt", "--backends", "numpy", "--dtypes", "f64", "--points", "32", "--out", str(out)])
    assert code == 0
    text = (out / "results.csv").read_text(encoding="utf-8")
    assert text.count("\n") == 2  # header plus the one sqrt row
    assert "sqrt,numpy,f64," in text


def test_cli_survey_accepts_the_half_dtypes(tmp_path):
    from ulpwise.__main__ import main

    out = tmp_path / "survey"
    code = main(["survey", "--functions", "exp", "--backends", "numpy", "--dtypes", "f16,bf16", "--points", "16", "--out", str(out)])
    assert code == 0
    text = (out / "results.csv").read_text(encoding="utf-8")
    assert text.count("\n") == 2  # header plus the numpy f16 row, numpy has no bfloat16
    assert "exp,numpy,f16," in text
    assert "| exp | f16 |" in (out / "results.md").read_text(encoding="utf-8")
