# Accuracy survey, October 2026: torch in float16 and bfloat16

The 45 functions of the September survey for which torch has a float16 and a bfloat16 CPU kernel,
measured in the ulps of those dtypes, on the same machine and the same torch build. The question
this time: a half precision kernel in torch is supposed to compute in float and round once, so the
rows should all read 0.5. Which ones do not, and why.

| | |
|---|---|
| command | `ulpwise survey --backends torch --dtypes f16,bf16 --out studies/accuracy-survey-2026-10-half` on ulpwise `main` at 0b12fb8 (0.3.1 plus the `polygamma` pole references below) |
| backends | torch 2.14.0+cpu (CPU capability AVX512, MKL available) |
| reference | mpmath at 200 bits, evaluated at the float16 or bfloat16 input the kernel actually saw |
| inputs | 600 log spaced points over each function's domain, clipped to the finite range of the dtype, plus the named edge values of the dtype, 480 to 662 inputs per row, subnormals included |
| platform | Linux x86_64, Python 3.13, mpmath 1.3.0 |
| files | `results.csv` (one row per function and dtype with every metric and the worst input), `results.md` (the pivot table) |

The 16 functions without half precision kernels in torch (`bessel_*`, `modified_bessel_*`,
`scaled_modified_bessel_*`, `spherical_bessel_j0`, `airy_ai`, `erfcx`, `log_ndtr`, `ndtri`, `zeta`)
are logged and skipped, so there are 90 rows.

## How to read a number

The same metric as in September: `|got - exact| / spacing(dtype, exact)`, correct rounding is 0.5,
and an exact result of zero or in the subnormal range produces enormous numbers that are not
precision bugs. Three things are specific to the half precision grids:

1. **bfloat16 has 8 significant bits and float16 11.** Every bfloat16 value with `|x| >= 256` is
   an integer, every float16 value with `|x| >= 2048` is one. A log spaced grid over a domain that
   reaches 1e5 therefore spends a quarter of its points on integers, which is why `sinc` in
   bfloat16 reads `8.4e32` (the exact value at an integer is 0) and why `polygamma_1` in bfloat16
   has 75 non finite mismatches (every negative point past 256 is a pole).
2. **The float16 range is narrow.** Normal numbers stop at 6.1e-5, the smallest subnormal is 6e-8
   and the largest value is 65504. A tail value such as `gelu(-5.57) = -7.1e-8` is a subnormal with
   one significant bit; its ulp is 6e-8 and torch's `-0` there counts as 1.19 ulps, not as the
   millions the same formula gives in float32.
3. **The two tolerance columns** use `torch.testing`'s defaults for the dtype, `rtol = 1e-3,
   atol = 1e-5` for float16 and `rtol = 1.6e-2, atol = 1e-5` for bfloat16, and the per op override
   from `op_db`. One ulp of float16 is between 4.9e-4 and 9.8e-4 relative, so the default float16
   tolerance is about one ulp and the bfloat16 one about four.

The `polygamma` reference changed since September: at a non positive integer it is now the signed
infinity of the pole (`+inf` for `polygamma_1`, `-inf` for `polygamma_2`, the sign torch, scipy and
jax return) instead of a domain error that expected `nan`, so a kernel that returns the infinity at
a pole is no longer counted as a mismatch. The September `polygamma_1` float32 row counts 7 non
finite mismatches with the old reference and would count 4 with this one (the changelog entry on
the `polygamma` references has the per dtype numbers); the measured errors do not depend on the
reference at the poles, and the September files are left as they were measured.

## Findings

### Two kernels lose their digits before the rounding step

In both cases the half precision path is not "compute in float and round once", and the errors
are far outside what the dtype can represent.

- **`erf` in bfloat16 and float16** (pytorch #199850, patch and Draft PR #199851). 254 ulps in
  bfloat16 at `x = -1.3e-8` (torch returns `-0`), p99 241, 34.6 percent of the grid beyond 1 ulp;
  4.55 ulps in float16, 24.5 percent beyond 1 ulp. The reduced types go through
  `Vectorized<float>::erf()`, the Abramowitz and Stegun 7.1.26 formula `1 - r * t * exp(-x * x)`,
  whose absolute error of 1.5e-7 is invisible for a float32 result of order 1 but is the whole
  result when `erf(x) ~ 1.128 x` is small: below `|x| = 1.8e-7` the `1 - 1` rounds to exactly 0.
  Over every bfloat16 value in `(0, 0.5]`, 14,102 of 16,128 inputs are more than 1 ulp off and
  13,379 return 0; float32 and float64 are correct on this build because they take the MKL path.
- **`logit` in float16 and bfloat16** (pytorch #199867, patch on a branch). 72 ulps in float16 at
  `x = 0.49731`, 29 of the 575 inputs fail the default float16 tolerance; 64 ulps in bfloat16, 10
  of 480. The CPU kernel computes `1 - x` and `x / (1 - x)` in the input dtype and only the `log`
  in float, so near `x = 0.5`, where the result is small, the two roundings take most of its
  digits: `logit(0.499756)` in float16 is `-0.000488` for an exact `-0.000977`, 512 ulps, and 582 of
  the 15,358 float16 values in `(0, 1)` are beyond 1 ulp (59 of 16,254 in bfloat16). Computing in
  float and rounding once is correctly rounded at every one of those inputs, and that is what the
  CUDA kernel already does. The `OpInfo` reference test carries
  `precisionOverride({torch.bfloat16: 5e-1, torch.float16: 5e-1})`, an absolute tolerance of 0.5
  on values below 0.01, so the `op fail` column is 0 for both rows.

### `polygamma(2, x)` in float16 at the negative half integers

14 ulps on the grid at `x = -578.5` (got `-2.146e-6`, exact `-2.983e-6`, a 28 percent error). Off
the grid it is worse: at every one of the 768 float16 half integers in `(-1024, -256)` the result is
more than 1 ulp off, 1,540 ulps at `x = -1023.5` (got `-2.38e-7`, exact `-9.54e-7`), and so are 203
of the 255 half integers in `(-256, -1)`. float32 is within 1e-4 float16 ulps at the same inputs.
These are the points where the function is small: at a half integer the reflection term of
`polygamma(2, x)` vanishes and the value is `polygamma(2, 1 - x) ~ -1 / x^2`, while the Hurwitz
zeta sum behind it adds terms of order 1 that cancel. torch's `calc_polygamma<Half>` accumulates
that sum in `float` (`acc_type<Half, false>`) and `calc_polygamma<float>` in `double`, which is why
float32 keeps its digits and float16 keeps an absolute error of about 1e-6. bfloat16 does not show
it only because bfloat16 has no half integers past 256 and the values below 256 are of order 1. Not
reported upstream yet (the fix would compute the reduced types through the float path, as
`calc_digamma` does). On CUDA `acc_type<float, true>` is `float`, so by the same reading of the
source the float32 kernel there has the same problem; not measured.

### The formula tails, as in September

- **`gelu` and `gelu_tanh` in bfloat16** return `-0` from about `x = -5.5` down, where
  `1 + erf(x / sqrt 2)` and `1 + tanh(t)` are exactly 0 in float: 242 and 250 ulps at the worst grid
  point, 2.1 and 1.6 percent of the grid beyond 10 ulps. bfloat16 has the float32 exponent range, so
  `gelu(-7.875) = -1.3e-14` is an ordinary normal number there. In float16 the same inputs are
  below the subnormal range and the row reads 1.19 ulps (see above).
- **`silu(-92.5)` and `sigmoid(-92.5)` in bfloat16** return `-0` and 0 because `exp(92.5)` overflows
  float32: 67.7 and 0.73 ulps, exactly the float32 behaviour of September measured in coarser ulps.

### The vectorized kernel and the scalar tail disagree

`rsqrt` (128 of 571 float16 inputs, 184 of 615 bfloat16) and `i0e` (303 of 588, 413 of 620) return
different bits for the same input depending on whether it landed in the vector body or in the
scalar tail of the tensor; `gelu` in float16 differs at 3 inputs. Both paths are within 1.5 ulps
(`rsqrt` 0.95 and 1.08, `i0e` 1.46 and 0.65), so this is determinism and not precision: the result
of an element depends on the tensor size and alignment.

### Small things

- `entr` reads 1.19 ulps in float16 (`x = 3088`) and 1.2 in bfloat16: `-x * log(x)` with the log
  rounded to the dtype before the product.
- `polygamma_1` in bfloat16 returns a large finite number instead of `inf` at the 75 negative
  integers of its grid (`4.3e9` at `-300`), the float32 reflection `pi^2 / sin(pi x)^2` with a
  rounded argument. float16 returns `inf` at the same points because `4.3e9` overflows float16.
- The single non finite mismatch of `reciprocal` and of `digamma` in each dtype is the edge value
  `-0.0`, where torch returns an infinity and the reference has no value.

### Everything else rounds correctly

33 of the 45 float16 rows and 34 of the 45 bfloat16 rows have a maximum of 0.5 ulps over the whole
grid, 39 and 36 stay at or below 1 ulp. `exp`, `log`, `sqrt`, the trigonometric and hyperbolic
functions and their inverses, `erfc`, `erfinv`, `lgamma`, `digamma`, `i0`, `i1`, `i1e`, `elu`,
`selu`, `mish`, `softplus` and `logsigmoid` are all correctly rounded in both dtypes: they convert
to float, compute, and convert back once.

## What the survey does not measure

One machine, CPU only, one build. 600 points per domain, so the all values scans quoted above for
`erf`, `logit` and `polygamma_2` were separate runs over every value of the dtype in the stated
interval, not part of `results.csv`. CUDA kernels are not measured. The half precision grids are
coarse at the top of their range (see the integer note above), which is also why the bfloat16
`polygamma_2` row cannot show the half integer problem.
