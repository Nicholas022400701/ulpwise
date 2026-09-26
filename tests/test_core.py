"""Cross-check the Rust core against independent Python oracles (Fraction, Decimal, numpy)."""

import math
import warnings
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np
import pytest

import ulpwise

getcontext().prec = 80

NP = {"f32": np.float32, "f64": np.float64}


def rounded_in(x, dtype):
    return float(NP[dtype](x))


def oracle_midpoint(exact: Fraction, rounded: float, dtype: str):
    """(distance_ulp, exact_above, exact) computed with exact rationals and numpy's nextafter."""
    r = Fraction(rounded)
    if exact == r:
        return 0.5, False, True
    above = exact > r
    t = NP[dtype]
    nxt = float(np.nextafter(t(rounded), t(np.inf) if above else t(-np.inf)))
    u = abs(Fraction(nxt) - r)
    m = r + u / 2 if above else r - u / 2
    return float(abs(exact - m) / u), above, False


@pytest.mark.parametrize("dtype", ["f32", "f64"])
@pytest.mark.parametrize(
    "x", [0.8528626561164856, 2.0, 3.0, 0.1, 12345.678, 0.5625, 1e30, 3.0e-39, 1e-40, 7.0, 1e-310, 0.75]
)
def test_sqrt_midpoint_matches_decimal_oracle(x, dtype):
    x = rounded_in(x, dtype)
    if x == 0.0:
        pytest.skip("underflowed to zero in this dtype")
    got = ulpwise.midpoint("sqrt", x, dtype)
    assert got is not None
    rounded, dist, above, exact = got
    # numpy's sqrt is the hardware sqrt, which is correctly rounded.
    assert rounded == float(np.sqrt(NP[dtype](x)))
    want_dist, want_above, want_exact = oracle_midpoint(Fraction(Decimal(x).sqrt()), rounded, dtype)
    assert exact == want_exact
    if not exact:
        assert above == want_above
        assert dist == pytest.approx(want_dist, rel=1e-9)


def test_the_kornia_knife_edge():
    rounded, dist, above, exact = ulpwise.midpoint("sqrt", 0.8528626561164856, "f32")
    assert rounded == 0.9235056042671204 and above and not exact
    assert 3e-4 < dist < 5e-4
    assert ulpwise.ulp_distance(0.9235056042671204, 0.9235056638717651, "f32") == 1
    hits = ulpwise.knife_edges("sqrt", 0.85, 0.86, 1e-3, "f32", limit=10_000)
    assert any(x == 0.8528626561164856 for x, *_ in hits)
    assert all(d < 1e-3 for _, _, d, _ in hits)


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_div_midpoint_matches_fraction_oracle(dtype):
    rng = np.random.default_rng(0)
    pairs = [(1.0, 3.0), (2.0, 3.0), (1.0, 10.0), (7.0, 7.0), (1.0, 4.0)]
    pairs += [(float(a), float(b)) for a, b in zip(rng.uniform(0.1, 100, 300), rng.uniform(0.1, 100, 300))]
    for a, b in pairs:
        a, b = rounded_in(a, dtype), rounded_in(b, dtype)
        rounded, dist, above, exact = ulpwise.midpoint("div", a, dtype, b)
        assert rounded == float(NP[dtype](a) / NP[dtype](b))
        want_dist, want_above, want_exact = oracle_midpoint(Fraction(a) / Fraction(b), rounded, dtype)
        assert exact == want_exact
        if not exact:
            assert above == want_above
            assert dist == pytest.approx(want_dist, rel=1e-12)


def test_random_sqrt_against_decimal():
    rng = np.random.default_rng(1)
    for dtype in ("f32", "f64"):
        xs = np.exp(rng.uniform(-80, 80, 400)) if dtype == "f64" else np.exp(rng.uniform(-80, 80, 400)).astype(np.float32)
        for x in xs.tolist():
            rounded, dist, above, exact = ulpwise.midpoint("sqrt", x, dtype)
            want_dist, want_above, want_exact = oracle_midpoint(Fraction(Decimal(x).sqrt()), rounded, dtype)
            assert exact == want_exact and (exact or above == want_above)
            assert exact or dist == pytest.approx(want_dist, rel=1e-9)
            assert ulpwise.sqrt_cr(x, dtype) == float(np.sqrt(NP[dtype](x)))


def test_ulp_distance_matches_numpy_view():
    def numpy_ulps(a, b, t):
        arr = np.array([a, b], dtype=t)
        i = arr.view(np.int32 if t is np.float32 else np.int64).astype(np.int64)
        i = np.where(i < 0, np.iinfo(i.dtype).min - i, i)
        return int(abs(i[0] - i[1]))

    rng = np.random.default_rng(2)
    for t, dtype in ((np.float32, "f32"), (np.float64, "f64")):
        for _ in range(500):
            a = float(t(rng.normal() * 10.0 ** rng.integers(-5, 5)))
            b = float(np.nextafter(t(a), t(np.inf)) if rng.random() < 0.5 else t(a * (1 + rng.normal() * 1e-6)))
            assert ulpwise.ulp_distance(a, b, dtype) == numpy_ulps(a, b, t), (a, b, dtype)
    assert ulpwise.ulp_distance(-0.0, 0.0) == 0
    assert ulpwise.ulp_distance(float("nan"), 1.0) is None
    assert ulpwise.ulp_distance(1.0, 1.0 + 2 ** -52) == 1
    assert ulpwise.ulp_distance(1.0, 1.0 + 2 ** -23, "f32") == 1
    assert ulpwise.ulp_distances([1.0, 2.0], [1.0, 2.0 + 2 ** -51]) == [0, 1]
    with pytest.raises(ValueError):
        ulpwise.ulp_distances([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        ulpwise.ulp_distance(1.0, 2.0, "f128")


def test_float16_ulp_distance_matches_numpy_view():
    def numpy_ulps(a, b):
        i = np.array([a, b], dtype=np.float16).view(np.int16).astype(np.int64)
        o = np.where(i < 0, -(i & 0x7FFF), i)
        return int(abs(o[0] - o[1]))

    rng = np.random.default_rng(3)
    for _ in range(500):
        a = float(np.float16(rng.normal() * 10.0 ** rng.integers(-6, 5)))
        b = float(np.float16(a * (1 + rng.normal() * 10.0 ** rng.integers(-4, 0))))
        assert ulpwise.ulp_distance(a, b, "f16") == numpy_ulps(a, b), (a, b)
        x = rng.normal() * 10.0 ** rng.integers(-6, 5)  # not representable: rounded to nearest even like numpy
        assert ulpwise.ulp_distance(x, float(np.float16(x)), "f16") == 0, x
    assert ulpwise.ulp_distance(1.0, 1 + 2 ** -10, "f16") == 1
    assert ulpwise.ulp_distance(1.0, 1 + 2 ** -11, "f16") == 0  # tie, rounds to even
    assert ulpwise.ulp_distance(65504.0, 1e5, "f16") == 1  # the largest float16 and infinity
    assert ulpwise.ulp_distance(-0.0, 0.0, "f16") == 0
    assert ulpwise.ulp_distance(float("nan"), 1.0, "f16") is None
    assert ulpwise.ordered(1.0, "f16") == 0x3C00 and ulpwise.ordered(-1.0, "f16") == -0x3C00
    assert ulpwise.ulp_distances([1.0, 2.0], [1.0, 2 + 2 ** -9], "f16") == [0, 1]
    assert ulpwise.flatten(np.float16([1.0]))[1] == "f16"
    assert ulpwise.max_ulp(np.float16([1.0, 2.0]), [1.0, 2 + 2 ** -9]) == (1, 1)
    assert ulpwise.max_ulp(np.float16([1.0]), np.float64([1 + 2 ** -10])) == (1, 0)  # the less precise dtype wins
    assert ulpwise.max_ulp([1.0], [1 + 2 ** -10]) == (2 ** 42, 0)


def test_bfloat16_ulp_distance():
    assert ulpwise.ulp_distance(1.0, 1 + 2 ** -7, "bf16") == 1
    assert ulpwise.ulp_distance(1.0, 1 + 2 ** -8, "bf16") == 0  # tie, rounds to even
    assert ulpwise.ulp_distance(1.0, 1 + 3 * 2 ** -8, "bf16") == 2  # tie, rounds up to the even 1 + 2 ** -6
    assert ulpwise.ulp_distance(3.3895313892515355e38, 1e39, "bf16") == 1  # the largest bfloat16 and infinity
    assert ulpwise.ulp_distance(3.4028234663852886e38, float("inf"), "bf16") == 0  # float32 max rounds to inf
    assert ulpwise.ulp_distance(2 ** -133, 0.0, "bf16") == 1 and ulpwise.ulp_distance(2 ** -134, 0.0, "bf16") == 0
    assert ulpwise.ulp_distance(-0.0, 0.0, "bf16") == 0
    assert ulpwise.ulp_distance(1.0, float("nan"), "bf16") is None
    assert ulpwise.ordered(1.0, "bf16") == 0x3F80 and ulpwise.ordered(-1.0, "bf16") == -0x3F80
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    for _ in range(500):
        x = rng.normal() * 10.0 ** rng.integers(-6, 5)
        t = torch.tensor([x, x * (1 + rng.normal() * 10.0 ** rng.integers(-3, 0))], dtype=torch.float64)
        a, b = t.to(torch.bfloat16)
        i = a.view(torch.int16).item(), b.view(torch.int16).item()
        o = [-(v & 0x7FFF) if v < 0 else v for v in i]
        assert ulpwise.ulp_distance(a.item(), b.item(), "bf16") == abs(o[0] - o[1]), (a, b)
        assert ulpwise.ulp_distance(x, a.item(), "bf16") == 0, x  # rounding agrees with torch
    assert ulpwise.flatten(torch.ones(2, dtype=torch.bfloat16))[1] == "bf16"
    assert ulpwise.max_ulp(torch.tensor([1.0, 2.0], dtype=torch.bfloat16), [1 + 2 ** -7, 2.0]) == (1, 0)
    assert ulpwise.assert_max_ulp(torch.tensor([0.1, 0.2], dtype=torch.bfloat16), torch.tensor([0.1, 0.2])) == 0


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_special_values_do_what_their_names_say(dtype):
    t = NP[dtype]
    s = dict(ulpwise.special(dtype))
    with np.errstate(all="ignore"):
        v = t(s["square_underflows_to_zero"])
        assert v * v == 0 and np.nextafter(v, t(1)) * np.nextafter(v, t(1)) > 0
        v = t(s["square_is_subnormal"])
        assert 0 < v * v < np.finfo(t).tiny
        v = t(s["square_just_finite"])
        assert np.isfinite(v * v) and np.isinf(t(s["square_overflows"]) * t(s["square_overflows"]))
        v = t(s["reciprocal_overflows"])
        assert np.isinf(t(1) / v) and np.isfinite(t(1) / np.nextafter(v, t(1)))
        assert t(s["integer_limit"]) + t(1) == t(s["integer_limit"])
        assert t(s["one_plus_ulp"]) == np.nextafter(t(1), t(2))
        assert t(s["max_subnormal"]) == np.nextafter(np.finfo(t).tiny, t(0))
        assert math.isnan(s["nan"]) and s["inf"] == math.inf and s["neg_zero"] == 0 and math.copysign(1, s["neg_zero"]) < 0
    assert len(s) == 29


def test_neighbours_spacing_and_binades():
    assert ulpwise.neighbours(1.0, 1, "f32") == [float(np.nextafter(np.float32(1), np.float32(0))), 1.0, 1.0 + 2 ** -23]
    assert ulpwise.neighbours(1.0, 2) == [1 - 2 ** -52, 1 - 2 ** -53, 1.0, 1 + 2 ** -52, 1 + 2 ** -51]
    assert ulpwise.spacing(1.0, "f32") == 2 ** -23 and ulpwise.spacing(-8.0) == 8 * 2 ** -52
    assert ulpwise.next_up(1.0, "f32") == 1 + 2 ** -23 and ulpwise.next_down(1.0) == 1 - 2 ** -53
    assert ulpwise.binade_edges(0, 1) == [1 - 2 ** -53, 1.0, 2 - 2 ** -52, 2.0]
    assert ulpwise.all_floats(1.0, 1 + 2 ** -22, "f32") == [1.0, 1 + 2 ** -23, 1 + 2 ** -22]
    with pytest.raises(ValueError):
        ulpwise.all_floats(0.0, 1.0, "f32", limit=1000)
    assert ulpwise.ordered(-0.0) == 0 and ulpwise.ordered(1.0, "f32") == 0x3F800000


def test_unary_reference_and_scan_warning():
    for op in ulpwise.UNARY_OPS:
        rep = ulpwise.midpoint(op, 0.7, "f32")
        assert rep is not None and 0 < rep[1] <= 0.5, op
    rounded, *_ = ulpwise.midpoint("exp", 1.0, "f32")
    assert rounded == float(np.float32(np.exp(np.float64(1.0))))
    assert ulpwise.midpoint("log", -1.0, "f32") is None
    assert ulpwise.midpoint("exp", 100.0, "f32") is None
    with pytest.raises(ValueError):
        ulpwise.midpoint("exp", 1.0, "f64")
    with pytest.raises(ValueError):
        ulpwise.midpoint("div", 1.0, "f32")
    with pytest.warns(RuntimeWarning):
        ulpwise.knife_edges("exp", 0.0, 1.0, 1e-4, "f32", limit=10 ** 9, max_evals=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hits = ulpwise.knife_edges("sqrt", 0.5, 1.0, 1e-4, "f32", limit=10 ** 9)
    # Midpoint distances are close to uniform on (0, 0.5], so about 2 * tol of the 2**23 inputs hit.
    assert 0.5 * 2e-4 * 2 ** 23 < len(hits) < 2.0 * 2e-4 * 2 ** 23


def test_assert_max_ulp_on_arrays():
    torch = pytest.importorskip("torch")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = a.clone()
    b[1] = torch.nextafter(b[1], torch.tensor(10.0))
    assert ulpwise.assert_max_ulp(a, b, 1) == 1
    assert ulpwise.max_ulp(a, b) == (1, 1)
    with pytest.raises(AssertionError, match="max ulp distance 1 > 0 at flat index 1"):
        ulpwise.assert_max_ulp(a, b, 0)
    # Measured in float32 ulps because the tensors are float32, not in float64 ulps of the same numbers.
    assert ulpwise.max_ulp(a.tolist(), b.tolist())[0] > 1_000_000
    assert ulpwise.max_ulp(np.float32([1.0, np.nan]), np.float32([1.0, np.nan])) == (0, 0)
    assert ulpwise.max_ulp([1.0, float("nan")], [1.0, 1.0]) == (None, 1)
    with pytest.raises(AssertionError, match="NaN mismatch"):
        ulpwise.assert_max_ulp([float("nan")], [1.0])
    with pytest.raises(TypeError):
        ulpwise.flatten(np.array([1, 2]))
    assert ulpwise.flatten([[1.0, 2.0], [3.0]]) == ([1.0, 2.0, 3.0], "f64")
    assert ulpwise.flatten(torch.ones(2, 2, dtype=torch.float64))[1] == "f64"


def test_pytest_plugin_parametrizes_edge_values(edge_f32, edge_f64, assert_max_ulp):
    assert isinstance(edge_f32, float) and isinstance(edge_f64, float)
    assert edge_f32 == float(np.float32(edge_f32)) or math.isnan(edge_f32)
    assert assert_max_ulp is ulpwise.assert_max_ulp


def test_cli(capsys):
    from ulpwise.__main__ import main

    assert main(["midpoint", "sqrt", "0.8528626561164856", "--dtype", "f32"]) == 0
    assert "above" in capsys.readouterr().out
    assert main(["knife", "sqrt", "--lo", "0.85", "--hi", "0.86", "--limit", "3"]) == 0
    assert "knife edge" in capsys.readouterr().err
    assert main(["special", "f64"]) == 0
    assert "square_overflows" in capsys.readouterr().out
    assert main(["ulp", "1.0", "1.0000001192092896", "--dtype", "f32"]) == 0
    assert capsys.readouterr().out.strip() == "1"
    assert main(["ulp", "1.0", "1.001", "--dtype", "f16"]) == 0
    assert capsys.readouterr().out.strip() == "1"
