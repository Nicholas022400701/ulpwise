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


def test_reference_formulas_keep_the_tails():
    mp.mp.prec = 200
    reg = {e.name: e for e in survey.REGISTRY}
    x = mp.mpf(-149.20600729076193)
    assert reg["softplus"].ref(x) > 0  # log1p keeps exp(-149), log(1 + exp(-149)) at 60 digits does not
    assert reg["logsigmoid"].ref(-x) < 0
    assert reg["sinc"].ref(mp.mpf(100000)) == 0  # exact at the integers
    assert reg["gelu_tanh"].ref(mp.mpf(-7.3)) < 0


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
