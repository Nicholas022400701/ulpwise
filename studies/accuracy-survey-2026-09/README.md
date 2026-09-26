# Accuracy survey, September 2026

61 elementary and special functions, float32 and float64, four backends, one machine. The question
behind it: how much precision do the functions machine learning code calls every day actually
deliver, and would the test suites of the libraries notice if they lost some.

| | |
|---|---|
| command | `ulpwise survey --out studies/accuracy-survey-2026-09` (about 3 minutes, needs `pip install ulpwise[survey] torch scipy jax`) |
| backends | torch 2.14.0+cpu (CPU capability AVX512), numpy 2.2.6, scipy 1.18.1, jax 0.11.2 on CPU with `jax_enable_x64` |
| reference | mpmath at 200 bits (about 60 decimal digits), evaluated at the input the backend actually saw |
| inputs | 600 log spaced points over each function's domain plus the named edge values of the dtype, 610 to 900 inputs per row, subnormals included |
| platform | Linux x86_64, Python 3.13, mpmath 1.3.0 |
| files | `results.csv` (one row per function, backend and dtype with every metric and the worst input), `results.md` (the pivot table) |

## How to read a number

An error is `|got - exact| / spacing(dtype, exact)`. Correct rounding is 0.5, a good libm stays
below 1, careful special function code stays below 4. The reference is computed at the rounded
input, so the error is the implementation's and not the input's.

Three situations produce enormous numbers that are not implementation bugs in any useful sense. The
table does not mark them, so keep them in mind:

1. **The exact result is zero.** `sinc` at the integers: every backend evaluates `sin(pi * x) / (pi * x)`
   with a rounded `pi * x`, so `sinc(100000)` is `5.06e-8` in float32 and `-1.08e-16` in float64
   instead of 0. `lgamma` at 1 and 2 (jax float32 returns `4.77e-7` at exactly 1). The number
   reported is the error in ulps of the smallest subnormal.
2. **The exact result is subnormal or below the smallest subnormal.** `sigmoid(-92.59)` in float32 is
   `6.16e-41`, but `exp(92.59)` overflows float32, so torch, scipy and jax all return 0 (43,951 ulps).
   `erfc(27)` in float64 is `5.2e-319`: scipy and jax return 0, torch returns it.
3. **jax on CPU flushes subnormals to zero**, inputs, intermediates and results. Every jax float32 row
   with a maximum near `8.4e6` (2^23) and every float64 row near `4.5e15` (2^52) is that:
   `sqrt` of the largest float32 subnormal is 0, `asin` of the smallest normal float64 is 0 because
   an intermediate `x / 2` went subnormal. torch and numpy keep subnormals. Because the grids
   include subnormal inputs, the p99 column of a jax row can also be dominated by this.

Two more caveats. scipy has no float32 path for its special functions, it computes in double and
rounds, so its float32 rows measure a double result in float32 ulps (`ndtri` 0.5 ulp against torch's
2.6). And the `torch default fail` column uses `torch.testing`'s default tolerance for the dtype,
which for float64 is `rtol = atol = 1e-7`. That is about 4.5e8 ulps: a float64 function can lose
nine digits and pass torch's reference tests without any override at all.

## Findings

### Errors that torch's test tolerances hide

This is why the survey exists. The `OpInfo` reference test compares against scipy with a tolerance
made of the dtype default and a per op override, and the survey reads both out of `op_db`.

- **Bessel `j0`, `j1`, `y0`, `y1` and `airy_ai` in float64** are off by 2.6e9 to 3.9e12 ulps, up to 12
  lost digits, at 27 to 29 percent of the Bessel inputs and 8.6 percent of the `airy_ai` inputs
  (pytorch #198583: the Cephes `p1evl` denominators in `Math.h` are evaluated without their implied
  leading 1). The override
  `precisionOverride({torch.float64: 1e-05})` hides all of it: with the default tolerance 4, 2, 0, 4
  and 3 inputs fail for `j0`, `j1`, `y0`, `y1` and `airy_ai`, with the override none. `y0` passes
  even the default because its worst value is 0.0213 and the default `atol` of 1e-7 covers a
  relative error of 4.2e-7 there.
- **`polygamma(1, x)` (trigamma) in float64** is accurate to about 9 digits and no more: 4.0e6 ulps at
  `x = 0.9047` (got `1.9074856066876746`, exact `1.9074856057949192`), p99 3.1e6, 46 percent of the
  inputs off by more than 10 ulps. `trigamma` in `Math.h` shifts `x` by 6 and then truncates the
  asymptotic series after the `1/42` term, so the first dropped term, `-1 / (30 x^9)`, is 1e-9 at
  `x = 6.9`: a single precision recipe (pytorch #198663, patch attached). The default float64
  tolerance passes it, so no override was ever needed. In float32 the reflection
  `pi^2 / sin(pi x)^2` at large negative `x` is far worse (`x = -60619`: got 8961, exact 7285) and
  86 of 616 inputs fail the float32 tolerance, at inputs the torch tests do not sample.
- **`erfcx` in float32** is 44 ulps off at `x = -8.44` and 4 inputs fail torch's float32 tolerance.
  The negative branch computes `2 * exp(x * x) - erfcx(-x)` and `x * x` is rounded in float32
  before the `exp`. The same rounding gives 157 ulps in float64 at `x = -23.25`, where scipy's
  float64 `erfcx` returns the same wrong value (pytorch #198664, patch attached). For float32 scipy
  computes in double and rounds: 0.5 ulp.
- **`torch.sqrt` in float64 is not correctly rounded** on this build: 0.62 ulp at `7.24e215`, numpy's
  0.5 (pytorch #198448, the MKL `vdSqrt` path). `exp` 0.62 and `expm1` 0.64 in float64 are the same
  class of near miss.

### The vectorized kernel and the scalar tail disagree (torch)

For 12 of the 61 functions (`atanh`, `cosh`, `elu`, `exp2`, `gelu`, `gelu_tanh`, `mish`, `selu`,
`sigmoid`, `silu`, `sinh`, `softplus`) the AVX512 kernel and the scalar loop that handles the tail
of a tensor return different floats for the same input: 246 of 619 float64 inputs for `mish`, 239
in float32, 151 of 904 for `atanh` float64, 84 for `gelu` float32, 68 for `sinh` float64. Both
paths are within 1 or 2 ulps, so this is not a precision problem but a determinism one: an element's
result depends on whether it landed in the vector body or in the tail, that is on the tensor size
and alignment.

### Formula choices that lose the tail

- **torch's exact `gelu`** computes `0.5 * x * (1 + erf(x / sqrt 2))`. `1 + erf` is exactly 0 below
  about `x = -5.5` in float32 and `x = -8.4` in float64, so torch returns `-0.0` where the true value
  is an ordinary normal number (`gelu(-6) = -5.9e-9`, `gelu(-9) = -1.0e-18`). jax writes it as
  `0.5 * x * erfc(-x / sqrt 2)` and keeps the tail (p99 65 ulps in float32, 462 in float64). The
  `tanh` approximation has the same shape in both libraries: below `x = -7` torch returns `-0.0` and
  jax returns values 50 times too large, because `1 + tanh(t)` is quantized to multiples of the
  spacing at 1.
- **`softplus` in float64** with torch's documented `threshold = 20` returns `x` above 20 and drops
  `log1p(exp(-x))`: 1.1e5 ulps at `x = 21.6`, 0.8 percent of the inputs. Harmless in float32, where
  `exp(-20)` is below the spacing at 20, and up to 6 lost digits in float64 between 20 and about 34,
  where `exp(-x) / x` drops below the float64 spacing.
- **`log_ndtr` for large positive `x`** in torch (822 ulps at `x = 29.8`) and scipy (957 at `x = 37.1`)
  both pay for rounding `t = x / sqrt 2` before `erfc(t)`, whose relative sensitivity to `t` is
  `2 t^2`, about 900 ulps at `x = 30`.
- **`logit` near 0.5** (73 ulps float32, 41 float64), **`digamma` near its zero at 1.4616** (62 ulps
  float64 for torch, 51 scipy, 41 jax) and **`airy_ai` near its first zero at -2.338** (12,371 ulps
  float32) are absolute errors of a few ulps of the function's scale, magnified by a small result.

### jax on CPU, beyond the subnormal flush

- **`erfinv` in float64** loses up to 5 digits near the ends of the interval: more than 10 ulps for
  `|x| > 0.9994` (186 of 904 inputs), 4.8e5 ulps at `1 - |x| = 7e-9`. scipy is within 2 ulps and
  torch within 0.51 on the same grid.
- **`log_ndtr` in float64** between `x = 5.4` and 8 evaluates `log(ndtr(x))` while `ndtr(x)` is within
  a few ulps of 1: 7.2e12 ulps at `x = 7.86` (got `-1.887379e-15`, exact `-1.884547e-15`, an error
  of 0.15 percent). torch uses `log1p(-erfc(t) / 2)` there.
- **`exp2`** has an error that grows linearly in `|x|`, 308 ulps at `x = 1000` in float64 and 23 ulps
  at `x = -119` in float32, which is what `exp(x * log 2)` with a rounded product gives. **`cosh`**
  grows the same way, 458 ulps at `x = 700` in float64 and 23 ulps at `x = 35` in float32.
- **`lgamma` in float32** returns `4.77e-7` at `x = 1` and 0 at `x = 1 + 2^-23` (exact `-6.9e-8`),
  has an absolute error of 1.1e-6 at `x = -2.46` where torch is within 1 ulp, and returns `inf` for
  subnormal inputs after flushing them to zero.

### scipy

- **`lgamma` in float64** does not treat the zeros at 1 and 2: at `x = 1 - 2^-53` it returns
  `2.22e-16` for an exact `6.41e-17` (1.3e16 ulps), torch's `std::lgamma` is within 1.2 ulps there.
  jax float64 is at `8.88e-16`.
- **`bessel_y1` in float64** is 2.1e5 ulps off at `x = 3461` and 19 percent of its inputs are beyond 10
  ulps, `y0` 284, `j1` 176, `j0` 25: the phase `x - 3 pi / 4` of the asymptotic branch loses digits
  at large `x`. torch's float32 Bessel functions compute the same phase in float32 and are 7,000 to
  38,000 ulps off around `|x| = 5000`, scipy's float32 rows are 0.5 ulp because they round a double.

### Shared by everyone

`sinc` for `|x| > 10` (30 percent of the grid beyond 10 ulps in all three backends, see above),
`sigmoid` of large negative float32 inputs, and `gelu_tanh` in the tail behave the same in torch,
numpy, scipy and jax because they use the same textbook formula.

## What the survey does not measure

One machine, CPU only, one build of each library. 600 points per domain, so knife edge failures such
as the 27 of 64 `torch.sqrt` inputs found by `knife_edges` in the corpus are not what this grid is
for. Accuracy near singularities is measured at the rounded input, which excludes the conditioning
of the function itself, and the grid is log spaced, so regions of width 1e-3 around a zero hold only
a handful of points.
