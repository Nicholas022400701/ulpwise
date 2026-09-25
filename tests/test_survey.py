"""Unit tests for the survey machinery that do not need any backend."""

import math

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
