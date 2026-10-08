# ulpwise

Numerical conformance testing for ML code. A Rust core with a Python API and a pytest plugin.

`ulpwise` answers five questions that come up every time a numerical test goes red on one
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
5. **Where should I read first in a repository I do not know?** `ulpwise scan` parses every Python
   file and reports the expressions behind the bugs in the corpus (`exp(x * x)`, `sin(pi * x)`,
   `1 - cos(x)`, `sqrt(a * a + b * b)`, `log(1 + exp(x))`, divisions by an unguarded angle, ...)
   with file, line and function, then lists the functions with the most elementary math.

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

# ulp distances, dtype aware: a float32 tensor is measured in float32 ulps, a bfloat16 tensor
# against a float64 reference in bfloat16 ulps. f64, f32, f16 and bf16.
ulpwise.ulp_distance(0.9235056042671204, 0.9235056638717651, "f32")   # 1
ulpwise.ulp_distance(1.0, 1.001, "bf16")                              # 0, both round to 1.0
ulpwise.assert_max_ulp(torch_out, reference, max_ulp_=2)               # floats, lists, numpy, torch

# the 29 named edge values of a dtype, f64 f32 f16 bf16
dict(ulpwise.special("f32"))["square_underflows_to_zero"]              # 2.6469779601696886e-23, the largest x with x * x == 0
dict(ulpwise.special("bf16"))["square_overflows"]                      # 1.8446744073709552e+19, 2 ** 64

# the floats around a value in any of the four dtypes
ulpwise.next_up(1.0, "bf16")                                           # 1.0078125
ulpwise.spacing(65504.0, "f16")                                        # 32.0, the ulp of the largest float16
ulpwise.neighbours(1.0, 2, "f16")                                      # [0.99902, 0.99951, 1.0, 1.00098, 1.00195]

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

Installed automatically. A test that takes `edge_f64`, `edge_f32`, `edge_f16` or `edge_bf16` runs
once per named edge value, and `assert_max_ulp` is available as a fixture:

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
form. `ULPWISE_CORPUS_STRICT=1 pytest tests/test_corpus.py` also fails on an unexpected pass, which
is how the weekly scheduled run of this repository (and a manual `workflow_dispatch` run) notices
that a release fixed a case before `fixed_in_release` says so. To run the corpus against an older release without touching your
environment, side-install it and put it first on the path:

```sh
pip install --no-deps --target /tmp/kornia-0.8.3 kornia==0.8.3
PYTHONPATH=/tmp/kornia-0.8.3 ulpwise corpus --repo kornia     # 31 present, 0 fixed on kornia 0.8.3
```

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
| ultralytics #26379 `scale_masks` kept a padded row when the letterbox padding was odd, the bottom of every mask faded to 0 | rounding | 2026-09-28 |
| ultralytics #26377 `verify_image_label` dropped a polygon that shared its class and box with another | aliasing | 2026-09-28 |
| ultralytics #26357 `verify_image_label` read a pose row as a polygon for a detect task and dropped the image as corrupt | crash | 2026-09-27 |
| kornia #4941 `So3.log`, the `So3` Jacobians and `Se3.exp/log` lost all digits for small angles (issue #4897) | series | 2026-09-26 |
| kornia #5124 `axis_angle_to_rotation_matrix` dropped the `theta^2` terms below 1e-3 rad (issue #4838) | series | 2026-09-30 |
| kornia #4960 `Se2.exp` and `Se2.log` lost the translation at small angles | cancellation | 2026-09-26 |
| kornia #4975 `point_line_distance` ignored the weight of a homogeneous point | scale | 2026-09-27 |
| kornia #5003 `Quaternion.__pow__` returned the zero quaternion on the negative real axis | branch cut | 2026-09-27 |
| kornia #5005 `So2` from a `(B, 1)` angle rotated `(B, 2)` points into `(B, B, 2)` | shape | 2026-09-27 |
| kornia #5024 `solve_cubic` lost the root of a cubic with a tiny leading coefficient | scale | 2026-09-28 |
| kornia #5111 `RgbToGrayscale` turned a uint8 image into zeros | dtype | 2026-09-30 |
| kornia #5134 `conv_soft_argmax2d` shifted every window's exponent by the maximum of the whole map | normalisation | 2026-09-30 |
| kornia #5376 `get_gaussian_discrete_kernel1d(1, sigma)` returned 3 taps | shape | 2026-10-03 |
| kornia #4963 `Se3.exp` at `omega = 0`: `d t / d omega` was nan, then zero, instead of `-0.5 [upsilon]_x` | gradient | 2026-09-26 |
| kornia #4981 `Quaternion.polar_angle` had a nan gradient at the identity | gradient | 2026-09-27 |
| kornia #4998 `decompose_essential_matrix` added a batch dimension to a `(3, 3)` input | shape | 2026-09-27 |
| kornia #5058 `Hyperplane.through` had nan gradients for orthogonal edges of equal length | gradient | 2026-09-29 |
| kornia #5084 `Vector3.normalized` turned a float16 zero vector into NaN | dtype | 2026-09-29 |
| kornia #5182 `otsu_threshold(return_mask=True)` compared the thresholded image with 0 | sign | 2026-10-01 |
| kornia #4972 the Hessian of `So3.exp` at the identity was nan (issue #4966) | gradient | 2026-09-27 |
| kornia #5116 `sampson_epipolar_distance` scored an exact match `sqrt(eps)` and changed with the scale of `F` (issue #4881) | eps, scale | 2026-09-30 |
| kornia #5131 `RandomHue` shifted a float64 image by the float32 pi, 8.7e-8 past a half turn (issue #5127) | constant | 2026-09-30 |
| kornia #5143 `MS_SSIMLoss` built an even window for sigma 1.3 and lost a row and a column (issue #5126) | shape | 2026-10-01 |
| kornia #5353 `MS_SSIMLoss` raised on a uint8 image (issue #5351) | crash | 2026-10-02 |
| kornia #4967 `So3.right_jacobian` in float16 was off by 0.79 above 41 rad (issue #4965) | overflow | 2026-09-26 |
| kornia #4980 `average_quaternions` counted a member stored as `3 q` nine times (issue #4974) | normalisation | 2026-09-27 |
| kornia #5104 `Hyperplane.through` flipped the normal of a large float16 triangle | overflow | 2026-09-29 |
| kornia #5301 `filter2d` with one kernel per sample raised on a channels-last input (issue #5292) | crash | 2026-10-02 |
| kornia #5303 `lovasz_hinge_loss` returned a float32 loss for a float16 prediction (issue #5289) | dtype | 2026-10-02 |
| kornia #5357 `get_box_kernel1d` returned a stride-0 view, editing one tap rewrote them all (issue #5160) | aliasing | 2026-10-02 |
| pytorch #198448 `torch.sqrt` float64 not correctly rounded at 27 of 64 knife edges | rounding | open |
| pytorch #198583 `bessel_j0/j1/y0/y1`, `airy_ai` float64 lose up to 12 digits (`p1evl` leading 1) | digits | open |
| torchvision #9676 `clamp_bounding_boxes` collapses slightly tilted rotated boxes to a point | geometry | open |
| pytorch #198663 `polygamma(1, x)` float64 keeps 9 digits (series stops at `1/42`), float32 loses all for large negative `x` | truncation, rounding | open |
| pytorch #198664 `erfcx` off by `x*x/2` ulps for negative `x` (`exp` at the rounded square) | rounding | open |
| pytorch #199850 `torch.erf` in bfloat16 and float16 on CPU returns 0 at and below 1.8e-7 and loses relative accuracy below 1e-3 (13404 bfloat16 ulps, 5 float16 ulps) | cancellation | open |
| pytorch #199867 `torch.special.logit` in float16 and bfloat16 on CPU rounds `1 - x` and `x / (1 - x)` to the input dtype before the log, logit(0.499756) is -0.000488 for -0.000977 (512 float16 ulps, 64 bfloat16 ulps) | rounding | open |
| kornia #5505 `angle_error_mat` and `angle_error_vec` returned exactly 0 for every rotation below 0.03 degrees in float32 (`acos` next to 1, issue #5500) | cancellation | 2026-10-06 |
| peft #3769 `add_weighted_adapter` with `combination_type='svd'` crashed for Conv1d and Conv3d LoRA layers | crash | open |
| peft #3830 OFT `module_dropout` was never applied, every training forward used all rotation blocks | no-op | open |

Cases marked `open` have an issue with the complete patch attached and no merged fix yet. Every case
is an expected failure until the installed release contains the fix (`fixed_in_release` in
`cases.json`; the timm and ultralytics fixes above are released, the kornia, pytorch/rl, peft and
pytorch ones are merged but not in a release yet), and the `max_ulp` check type measures the digits
directly. The ultralytics dataset and label cases build their one-image
dataset in a temporary directory and need no weights; two more ultralytics fixes (#26240, #26246) are not in
the corpus yet because their repros need model weights or the COCO evaluator.

## Repository scan

```sh
ulpwise scan kornia/kornia --report kornia.md        # clone with depth 1, write a Markdown report
ulpwise scan . --fail-on high                        # CI gate: exit 1 on a high severity finding
ulpwise scan path/to/repo --rules one-minus-cos,small-angle-division --top 40
```

The scan is static and needs nothing installed: it parses each file with `ast`, walks every
function and matches sixteen patterns, each with a severity, the reason it loses digits,
overflows or loses its gradient, or in one case does nothing at all, the usual replacement and,
where one exists, the upstream bug it comes from.

| rule | severity | pattern |
|---|---|---|
| `exp-of-square` | high | `exp(x * x)`, `exp(x ** 2)`, `(-x.pow(2)).exp()`: the rounding error of the square is multiplied by `x * x / 2` ulps (pytorch #198664) |
| `sin-of-pi-times` | high | `sin(pi * x)`, `cos(pi * x)`: the product is rounded before the argument reduction (pytorch #198663) |
| `softplus-by-hand` | high | `log(1 + exp(x))` |
| `logsumexp-by-hand` | high | `log(exp(a) + exp(b))`, `log(sum(exp(x)))`, `log(exp(x).sum(-1))`; quiet when every `exp` argument has its maximum subtracted first, `log(exp(x - x.max()).sum())`, the stable form |
| `hypot-by-hand` | high | `sqrt(a * a + b * b)`; the advice no longer says `norm`, which squares first too and overflows at 1.8e19 in float32 (kornia #5505 review) |
| `sqrt-of-difference` | medium | `sqrt(a - b)` |
| `one-minus-cos` | medium | `1 - cos(x)` (kornia #4897) |
| `log1p-by-hand`, `expm1-by-hand` | medium | `log(1 + x)`, `exp(x) - 1` |
| `atan-of-quotient` | medium | `atan(y / x)` |
| `small-angle-division` | medium | `/ theta`, `/ theta ** 2`, `/ sin(theta)` in a function that takes `sin` or `cos` of `theta` and has no `where`, `clamp`, `eps` or series in sight (kornia #4838, #4897) |
| `acos-for-angle` | medium | `acos`, `asin` used to recover an angle: the angle comes back with an absolute error of `sqrt(eps)`, 0.02 degrees in float32 (kornia #5500) |
| `eps-floor` | medium | `x * (1 - eps) + eps`: every value moves, 0 becomes `eps`, so a one-hot's entries sum to `1 + (C - 1) eps` and a perfect prediction scores a loss that grows with the image (kornia #5538, found by the kornia conventions audit) |
| `where-nan-gradient` | medium | `where(d > eps, f(d), other)` with `f` a division by `d` or a `sqrt`, `log`, `acos` or `asin` of it: `where` evaluates both branches and hands the discarded one a zero gradient, and the backward of `f` at the singularity turns that zero into `0 / 0 = nan`, so the guard protects the value and not the gradient; quiet for `numpy.where`, for a comparison against a number above 1, for a floor named `eps`, `tol` or `floor` on the other side, and once `d` is re-bound to a `where`, `clamp` or `maximum` of itself (kornia #5579, found by the kornia conventions audit) |
| `clamp-at-singularity` | medium | `clamp(x, min=0).sqrt()`, `sqrt(clamp(x, min=0))`, `clamp(c, -1, 1).acos()`: the bound is the point where the next function has an infinite derivative, and clamp's derivative at its own bound is not the same across torch versions, 1 on 2.5.1 and 2.9.1 and 0 on 2.14, so the gradient there is `inf` or `nan` on the older half of a supported range; a bound strictly inside the domain, `min=1e-8`, is a floor and is not reported, and `numpy` is quiet (kornia #4229, found by the kornia conventions audit; kornia #5500) |
| `dropout-never-applied` | medium | a `Dropout`, `DropPath` or a `ModuleDict` of them assigned to `self` and then never called, never passed on and its rate never read, anywhere in the file: the option is accepted and does nothing, so the model trains without the regularisation it reports (peft #3830) |

On kornia `main` at `e05b0ee` the scan takes 4 s for 506 files and reports 35 findings. The
`one-minus-cos` and `small-angle-division` findings are the four lines of `So3.right_jacobian` and
`So3.left_jacobian` that kornia #4897 fixes, `So3.log` (kornia #4838) is under `acos-for-angle`,
and `Se3.exp` (also #4897) is under `one-minus-cos`. The scan puts `ellipse_to_laf` (kornia #4768)
on the list too, for a `sqrt` of a difference; the bug there was a different one, so that entry is
what the scan is: a reading list, not a verdict.

The two gradient rules are reading lists in the same sense. On kornia `main` at `d15741e2` they
report seven lines: three `clamp` bounds in `_solve_cubic_real`, and four `where` guards, in
`PatchDominantGradientOrientation`, `compute_correspond_epilines`, `_crop_scale_translation` and
`_get_convex_edges`, of which the first two sit on a differentiable path and the last two draw
boxes and polygons. On ultralytics `main` at `8df3534` they report three lines, all in metrics or
inference code that is never differentiated, and nothing in torchvision.

`dropout-never-applied` reads the whole file rather than one function, since the module is built
in `__init__` and used, if at all, in `forward` or in a sibling class. Calling it, calling it
through a subscript, passing it to another module, returning it, iterating over it and reading
its rate (`scaled_dot_product_attention(dropout_p=self.dropout.p)`) all count as use; filling it
(`self.oft_dropout.update(...)`) does not, and a subclass of `Sequential` runs every attribute and
is not read. On kornia `main` at `6d579a72`, ultralytics `6d51b7e`, torchvision `9a8d545`,
diffusers `d961a38`, torchrl `648f50c`, vllm `73c742b` and detectron2 `1e3e13b` it reports
nothing. On peft `main` at `f6d8480` it reports `OFTLayer.oft_dropout`, which is #3830; on timm
`83e6eb5` it reports `FactorAttnConvRelPosEnc.attn_drop` in `coat.py`, which a comment on that
line already calls unused; on transformers `9167f73` it reports nine lines, four of them in
modular files whose `forward` is inherited from another file, which the rule cannot see, and
five attention, embedding and head dropouts built in `__init__` and used nowhere in their file.

`--run` adds the dynamic half. The module level functions with the most elementary math are
imported and called with the same 91 point grid (both signs of `1e-8` to `1e3`, and zero) for every
required argument, in float32 and in float64, torch first and numpy second, and the float32 result
is measured against the float64 one. Two numbers per function: `at scale`, the largest absolute
error in ulps of the largest output, and `elementwise`, the worst per element ulp distance with
the input where it happens. Read both. A rotation matrix has entries that should be zero, and
there the elementwise count compares float32 rounding noise with float64 rounding noise and reaches
`1e9` while the matrix is fine to one ulp at scale. A function whose output spans forty orders of
magnitude, a Bessel function, has a meaningless `at scale` number and a meaningful elementwise one.
A small angle formula without a guard, `(1 - cos(theta)) / theta ** 2`, shows `9e6` in both
columns. Static methods run; instance methods, functions that need other arguments, fail to import
or return something that is not a float array are counted with the reason and skipped, never
guessed at. This imports and runs the repository's code, from the scanned tree, or from the
package installed in the environment with `--run-installed` when the tree has unbuilt extensions.

```sh
ulpwise scan kornia/kornia --run --run-limit 200 --report kornia.md
ulpwise scan pytorch/vision --run --run-installed --run-limit 150
```

On kornia main 9 of the 200 busiest functions run as is, the rest are instance methods or want
shaped inputs. `adjust_log` is on both lists: the static scan flags its `(1 + image).log2()` as
`log1p-by-hand`, and the run shows `8.7e8` elementwise ulps at `x = 5.6e-8` next to `0.7` ulps at
scale, which is the right reading for a function defined on images in `[0, 1]`.

In ML repositories most `high` findings are learning rate schedules (`cos(pi * progress)` with
`progress` in `[0, 1]`) and pixel distances (`sqrt(dx * dx + dy * dy)` on coordinates below
`1e4`), where the argument is bounded and the pattern is harmless. The scan cannot know the bound;
the reading list is where that judgement happens.

## Accuracy survey

```sh
pip install 'ulpwise[survey]' torch scipy jax      # mpmath is the reference, the rest are backends
ulpwise survey --out survey                        # results.csv and results.md, about 3 minutes
ulpwise survey --functions bessel_j0,polygamma_1 --backends torch,scipy --dtypes f64 --points 2000
ulpwise survey --functions erf,exp --backends torch --dtypes f16,bf16   # half precision rows
```

For every function in `ulpwise.survey.REGISTRY` (exp, log, trig and hyperbolic functions, erf and
friends, gamma family, torch.special Bessel and Airy functions, the activation functions), every
dtype (`f64`, `f32`, `f16` and `bf16`; the default is `f32,f64`) and every installed backend, the
survey evaluates a log spaced grid over the function's domain, clipped to the finite range of the dtype,
plus the named edge values of the dtype, computes the exact value with mpmath at the rounded input,
and reports max, p99 and median error in ulps, the fraction of inputs beyond 1 and 10 ulps, non
finite mismatches and the worst input. For torch it also reports how many inputs the vectorized
kernel and the scalar tail disagree on, and how many inputs would fail torch's reference test under
the dtype default tolerance and under the op's `OpInfo` override, read from `op_db`. Reading `op_db` needs
`expecttest`, a test-only dependency of torch that the `survey` extra installs; without it the survey logs
one line and uses the default. torch and jax have `f16` and `bf16` rows; numpy has no bfloat16 and
scipy.special computes a float16 input in float32, so neither has `bf16` rows and scipy has no `f16` rows.
A backend without a kernel for a function in a dtype (torch's Bessel functions in half precision) is
logged and skipped.

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

[`studies/accuracy-survey-2026-10-half`](https://github.com/Nicholas022400701/ulpwise/blob/main/studies/accuracy-survey-2026-10-half/README.md)
is the second run, the 45 functions with torch float16 and bfloat16 CPU kernels measured in the ulps of
those dtypes on the same build. 33 of the 45 float16 rows and 34 of the bfloat16 rows are correctly
rounded over the whole grid; the exceptions:

- `erf` goes through a formula with 1.5e-7 absolute error, 254 bfloat16 ulps and `-0` below
  `|x| = 1.8e-7` (pytorch #199850).
- `logit` rounds `1 - x` and `x / (1 - x)` to the input dtype before the log, 512 float16 ulps at
  `x = 0.499756` while the CUDA kernel computes in float (pytorch #199867).
- `polygamma(2, x)` in float16 is off at every half integer in `(-1024, -256)`, 1,540 ulps at
  `x = -1023.5`, because the Hurwitz zeta sum accumulates in float for the reduced types and in
  double for float32.
- the `gelu`, `gelu_tanh`, `silu` and `sigmoid` tails and the `rsqrt` and `i0e` vector versus scalar
  disagreements are the float32 findings of September seen through coarser ulps.

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

- `ulpwise scan --run` for methods and functions with tensor shape requirements, by reading the
  docstring and the checks for the shapes, and an mpmath reference for the scalar functions.
- Mutation scoring for numerical tests: single token mutants of the code under test (`abs`, a
  dropped `sqrt`, `/ 4` for `/ 16`) run against the test suite, reporting which survive.
- `float16` and `bfloat16` knife edges and exact oracles (ulp distances, edge values, `spacing` and the neighbour functions are done).
- Exact references for transcendental functions in Rust (correctly rounded `exp`, `log`, ...) so
  the `f64` knife-edge scans do not need mpmath.
- Survey backends for CUDA and MPS.
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
