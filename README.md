# ulpwise

Numerical conformance testing for ML code. A Rust core with a Python API and a pytest plugin.

`ulpwise` answers three questions that come up every time a numerical test goes red on one
platform and green on another:

1. **Which inputs break first?** Named edge values computed from the float format (the largest
   `x` with `x * x == 0`, the first `x` whose square overflows, the value where `1 / x` overflows,
   binade boundaries, subnormals) and every-float enumerations of a range.
2. **What is the right answer, exactly?** Rounding oracles that compare the exact result against
   the candidate floats with integer arithmetic on significands, so the correctly rounded `sqrt`,
   reciprocal and quotient, and the distance from the exact result to the rounding midpoint, do
   not depend on any libm.
3. **Where do two implementations disagree?** Knife-edge scans: the inputs whose exact result sits
   within `tol` ulp of a rounding midpoint, so that two implementations that differ by one ulp
   return different floats. 8.4 million `f32` inputs are scanned in about 0.3 s.
4. **How accurate are the functions I call, and would the library's own tests notice?**
   `ulpwise survey` measures 61 elementary and special functions of torch, numpy, scipy and jax in
   ulps against a 200 bit mpmath reference and, for torch, checks every error against the tolerance
   of torch's own `OpInfo` reference test, default and per op override.

## Why this exists

On 2026-09-23 a kornia pull request went red on 21 of 39 CI jobs after a maintainer added a test
around the shape `(1.0, 0.9235056042671204, 0.8528626561164856)`. In float32 the exact value of
`sqrt(0.8528626561164856)` lies 0.0004 ulp below the midpoint of its two float32 neighbours. The
macOS runners and my sandbox rounded it down, the ubuntu and windows runners rounded it up, and the
estimator under test took two different branches. `ulpwise` finds that input, and the 16,995 others
like it in `[0.5, 1)`, before CI does:

```
$ ulpwise midpoint sqrt 0.8528626561164856 --dtype f32
rounded 0.9235056042671204, exact result lies above it, 3.771e-04 ulp from the rounding midpoint

$ ulpwise knife sqrt --lo 0.85 --hi 0.86 --tol 1e-3 --limit 3
                       x                  rounded  midpoint distance (ulp) exact lies
      0.8500027060508728       0.9219558835029602                2.521e-04      above
      0.8500217199325562       0.9219662547111511                5.452e-04      below
      0.8500407338142395       0.9219765067100525                5.924e-04      above
3 knife edge(s) with distance < 0.001 ulp
```

Measured with `examples/torch_sqrt_conformance.py` on one Linux x86_64 machine (AVX512, torch
2.14.0+cpu built with MKL, numpy 2.2.6), at the 16,996 float32 knife edges of `[0.5, 1)` with
`tol = 1e-3`:

| implementation | off by 1 ulp at knife edges | direction |
|---|---|---|
| `numpy.sqrt` float32 | 0 / 16,996 | correctly rounded |
| `torch.sqrt` float32 | 7,131 / 16,996 (42.0%) | always rounded down instead of up |
| `torch.pow(x, 0.5)` float32 | 7,131 / 16,996 (42.0%) | same |
| `torch.sqrt` float64 then cast | 0 / 16,996 | correctly rounded |
| `torch.sqrt` float64 (20,000 f64 knife edges) | 9,996 / 20,000 (50.0%) | always down |

On random inputs the same `torch.sqrt` differs from correct rounding at 0.70% of float32 values,
which is why this goes unnoticed until a test happens to pin one of them. Other platforms will
show other numbers; the CI of this repository prints them for ubuntu, macOS and windows on every
run.

## Install

```sh
pip install ulpwise          # or: uv add ulpwise
uvx ulpwise special f32      # run the command line tool without installing anything
```

Wheels on [PyPI](https://pypi.org/project/ulpwise/) cover Linux x86_64 and aarch64, macOS arm64 and
x86_64, and Windows x64, for Python 3.9 or newer (abi3). To build from source you need a Rust
toolchain (1.86 or newer) and [maturin](https://www.maturin.rs/):

```sh
pip install maturin
pip install .            # or: maturin develop --release  (inside a virtualenv)
```

The Rust crate is usable on its own (`cargo add --git https://github.com/Nicholas022400701/ulpwise`).

## Use

```python
import ulpwise

# ulp distances, dtype aware: a float32 tensor is measured in float32 ulps.
ulpwise.ulp_distance(0.9235056042671204, 0.9235056638717651, "f32")   # 1
ulpwise.assert_max_ulp(torch_out, reference, max_ulp_=2)               # floats, lists, numpy, torch

# the 29 named edge values of a dtype
dict(ulpwise.special("f32"))["square_underflows_to_zero"]              # 2.6469779601696886e-23, the largest x with x * x == 0

# exact placement of a result relative to its rounding midpoint
ulpwise.midpoint("sqrt", 0.8528626561164856, "f32")   # (0.9235056042671204, 0.000377, True, False)
ulpwise.midpoint("div", 1.0, "f64", 3.0)              # (0.3333333333333333, 0.1666..., True, False)

# knife edges of a function on a range: (x, rounded, distance_ulp, exact_above)
ulpwise.knife_edges("exp", 0.5, 1.0, tol_ulp=1e-3, dtype="f32", limit=100)

# a correctly rounded sqrt that does not depend on the platform
ulpwise.sqrt_cr(0.8528626561164856, "f32")            # 0.9235056042671204
```

`midpoint` and `knife_edges` accept `sqrt`, `recip` and `div` (exact integer oracle, `f32` and
`f64`) and `rsqrt exp exp2 expm1 log log2 log10 log1p sin cos tan atan tanh sigmoid softplus`
(`f32` only, `f64` libm as reference, trust the distance down to about `1e-8` ulp).

### pytest plugin

Installed automatically. A test that takes `edge_f32` or `edge_f64` runs once per named edge
value, and `assert_max_ulp` is available as a fixture:

```python
def test_my_kernel_survives_the_edges(edge_f32, assert_max_ulp):
    x = torch.tensor([edge_f32])
    assert_max_ulp(my_kernel(x), reference(x), max_ulp_=1)
```

Failures read `test_my_kernel_survives_the_edges[square_overflows]`.

## Regression corpus

`python/ulpwise/corpus/cases.json` holds upstream bugs found by the contribution pipeline this
project grew out of, each with a runnable repro, the buggy behaviour quoted from the merged pull
request, and the check that tells the two apart. `pytest tests/test_corpus.py` runs every case
whose packages are installed; a case is an expected failure while the installed release is not
known to contain the fix, so the run tells you which bugs are present in your environment.
`ulpwise corpus` does the same without pytest, one line per case with the installed version and
`present`, `fixed` or `skipped`; `ulpwise corpus --repo pytorch --fail-if-present` is the CI gate
form.

| case | kind | merged |
|---|---|---|
| kornia #4683 second derivative sign in `spatial_gradient(order=2)` | sign | 2026-09-22 |
| kornia #4767 mixed second order kernel scale, wrong `hessian_response` determinant | scale | 2026-09-23 |
| kornia #4768 `ellipse_to_laf` under-tilted every ellipse with `b != 0` | geometry | 2026-09-24 |
| pytorch #198006 `Multinomial.entropy()` evaluated in the default dtype | dtype | 2026-09-23 |
| pytorch/rl #4443 `arange(0, 1, 1/n)` gives `n + 1` positions for 140 values of `n` below 2000 | rounding | 2026-09-20 |
| pytorch/rl #4444 `min_value or -inf` drops `min_value=0` | falsy zero | 2026-09-20 |
| pytorch/rl #4445 scheduler `state_dict()` contained a module object | crash | 2026-09-20 |
| timm #2786 Mars kept a reference to `p.grad` as the previous gradient | aliasing | 2026-09-17 |
| timm #2790 AdafactorBigVision clipped updates in the wrong direction | direction | 2026-09-18 |
| timm #2791 AdaMuon conv LR scale computed from the wrong dims | scale | 2026-09-18 |
| timm #2792 Kron `__setstate__` shadowed | crash | 2026-09-18 |
| peft #3777 pointwise Conv3d took the conv2d 1x1 shortcut | shape | 2026-09-21 |
| ultralytics #26330 OBB train and val on plain box labels crashed in the validator or the loss instead of at load time | crash | 2026-09-25 |
| pytorch #198448 `torch.sqrt` float64 not correctly rounded at 27 of 64 knife edges | rounding | open |
| pytorch #198583 `bessel_j0/j1/y0/y1`, `airy_ai` float64 lose up to 12 digits (`p1evl` leading 1) | digits | open |
| kornia #4838 `axis_angle_to_rotation_matrix` drops the `theta^2` terms below 1e-3 rad | series | open |
| kornia #4897 `So3.log`, the `So3` Jacobians and `Se3.exp/log` lose all digits for small angles | series | open |
| torchvision #9676 `clamp_bounding_boxes` collapses slightly tilted rotated boxes to a point | geometry | open |
| pytorch #198663 `polygamma(1, x)` float64 keeps 9 digits (series stops at `1/42`), float32 loses all for large negative `x` | truncation, rounding | open |
| pytorch #198664 `erfcx` off by `x*x/2` ulps for negative `x` (`exp` at the rounded square) | rounding | open |

Cases marked `open` have an issue with the complete patch attached and no merged fix yet; they are
expected failures until a release contains the fix (`fixed_in_release` in `cases.json`), and the
`max_ulp` check type measures the digits directly. The ultralytics case builds its one-image dataset
in a temporary directory and needs no weights; two more ultralytics fixes (#26240, #26246) are not in
the corpus yet because their repros need model weights or the COCO evaluator.

## Accuracy survey

```sh
pip install 'ulpwise[survey]' torch scipy jax      # mpmath is the reference, the rest are backends
ulpwise survey --out survey                        # results.csv and results.md, about 3 minutes
ulpwise survey --functions bessel_j0,polygamma_1 --backends torch,scipy --dtypes f64 --points 2000
```

For every function in `ulpwise.survey.REGISTRY` (exp, log, trig and hyperbolic functions, erf and
friends, gamma family, torch.special Bessel and Airy functions, the activation functions), every
dtype and every installed backend, the survey evaluates a log spaced grid over the function's domain
plus the named edge values of the dtype, computes the exact value with mpmath at the rounded input,
and reports max, p99 and median error in ulps, the fraction of inputs beyond 1 and 10 ulps, non
finite mismatches and the worst input. For torch it also reports how many inputs the vectorized
kernel and the scalar tail disagree on, and how many inputs would fail torch's reference test under
the dtype default tolerance and under the op's `OpInfo` override, read from `op_db`.

[`studies/accuracy-survey-2026-09`](https://github.com/Nicholas022400701/ulpwise/blob/main/studies/accuracy-survey-2026-09/README.md) is the first run
(torch 2.14.0+cpu, numpy 2.2.6, scipy 1.18.1, jax 0.11.2, Linux x86_64 AVX512). The short version:

- torch's `bessel_j0/j1/y0/y1` and `airy_ai` in float64 are off by 2.6e9 to 3.9e12 ulps and the
  `precisionOverride({torch.float64: 1e-05})` on their tests hides every failing input
  (pytorch #198583).
- torch's `polygamma(1, x)` in float64 keeps about 9 digits (4.0e6 ulps, 46 percent of inputs
  beyond 10 ulps) and passes the default float64 tolerance, which at `rtol = atol = 1e-7` tolerates
  about 4.5e8 ulps; in float32 it loses every digit for large negative `x` (pytorch #198663).
- torch's `erfcx` for negative `x` is off by up to `x*x/2` ulps, 44 in float32 at `x = -8.44` and
  157 in float64 at `x = -23.25`, and scipy's float64 `erfcx` returns the same wrong values
  (pytorch #198664).
- for 12 of 61 torch functions the AVX512 kernel and the scalar tail return different floats for
  the same input, up to 246 of 619 inputs for `mish`.
- jax on CPU flushes subnormals to zero, its float64 `erfinv` loses 5 digits near the ends of the
  interval and its float64 `log_ndtr` loses 3 digits between `x = 5.4` and 8.
- scipy's float64 `lgamma` does not handle the zeros at 1 and 2, and its Bessel functions lose the
  phase at large `x`.

## How the exact oracle works

For `sqrt(x)` the candidate `r` and the midpoint `m` between `r` and its neighbour are written as
integers times a power of two, `m^2` is formed in `u128` (at most 110 bits) and compared with `x`
after aligning exponents. The sign says on which side of `m` the exact root lies, and
`|x - m^2| / (sqrt(x) + m)` is the distance to the midpoint. If the hardware result turns out to be
on the wrong side of a midpoint it is stepped one ulp toward the exact value and checked again, so
`sqrt_cr` is correctly rounded even where the platform `sqrt` is not. Division uses the same
machinery with `m * b` against `a`, and there the residual is exact. `tests/test_core.py`
cross-checks both against `fractions.Fraction` and `decimal.Decimal` at 80 digits.

## Roadmap

- Mutation scoring for numerical tests: single token mutants of the code under test (`abs`, a
  dropped `sqrt`, `/ 4` for `/ 16`) run against the test suite, reporting which survive.
- `float16` and `bfloat16` ulps and edge values.
- Exact references for transcendental functions in Rust (correctly rounded `exp`, `log`, ...) so
  the `f64` knife-edge scans do not need mpmath.
- Survey backends for CUDA and MPS, and `float16` / `bfloat16` rows.
- Zero copy paths for numpy arrays and torch tensors.
- More corpus entries, with fixtures for the cases that need data.

## AI disclosure

This project is written with an AI coding agent (Claude) working for
区梓灏 ([@Nicholas022400701](https://github.com/Nicholas022400701)), who owns the repository and
reviews what is published. The same pipeline produced the upstream fixes in the corpus; each of
those pull requests carries the same disclosure.

## License

Licensed under either of

- the MIT license ([LICENSE-MIT](https://github.com/Nicholas022400701/ulpwise/blob/main/LICENSE-MIT)), or
- the Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/Nicholas022400701/ulpwise/blob/main/LICENSE-APACHE)),

at your option. "At your option" means that whoever uses or redistributes this code picks
whichever of the two licenses they want to comply with. Nobody has to ask anyone. Unless you say
otherwise, a contribution you send is dual licensed the same way, without extra terms.
