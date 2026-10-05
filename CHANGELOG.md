# Changelog

## Unreleased

- `ulpwise scan` read the first positional argument of a method call as the value the method is applied
  to, so `log(exp(x).sum(-1))` was not a `logsumexp-by-hand` finding while `log(exp(x).sum())` and
  `log(exp(x).sum(dim=-1))` were. A method call whose first argument is a dimension (an integer, `None`
  or a tuple of them) is now read as applying to its receiver, like the argument-less form.
- `ulpwise scan` printed a `SyntaxWarning` (a `DeprecationWarning` before Python 3.12) for every invalid
  escape sequence in the files it scanned, four lines on kornia. The parse now ignores those warnings;
  a file that does not parse is skipped as before.
- `ulpwise scan` reported Windows paths with backslashes (`kornia\geometry\conversions.py`), so the same
  report read differently from the one made on Linux or macOS. Paths are now written with forward
  slashes on every platform; `--run` still imports the module from either form. The CI pytest job on
  ubuntu, macOS and windows runs `tests/test_scan.py` now, and that file also covers the reasons
  `--run` gives for a function it does not run (a class, a constant, an instance method, a builtin
  without a signature, a nested name, a module that fails to import) and the ulp comparison of the
  results across infinities, NaNs and an all zero reference.
- `ulpwise survey` without scipy installed died at `ndtri` with a `ModuleNotFoundError` and lost every
  row computed before it: the reference started its Newton iteration from `scipy.special.ndtri`. It now
  starts from a bisection on `math.erfc`, returns an exact zero at `p = 0.5` (a Newton step there leaves a
  rounding residual, which the ulp metric would read as 1e16 ulps against the exact 0.0 of torch and scipy)
  and gives the same rows as before on the 600 point float64 and float32 grids. The survey also evaluates
  the reference only when some requested backend implements the function; a numpy only run no longer
  spends its time on references for the torch and scipy only entries. The pytest job on ubuntu, macOS and
  windows (Python 3.12 and 3.9) runs `tests/test_survey.py` now, with numpy as the only backend.
- `ulpwise survey` printed a numpy `RuntimeWarning` for every backend call that divided by zero or
  overflowed at the edge values of the grid (`reciprocal` and `log` at 0, `reciprocal` at the overflow
  edge), hundreds of lines on a full run. Those points are counted in the `nonfinite` column already, so
  the survey now evaluates the backends under `np.errstate(all="ignore")`, and a test runs `reciprocal` and
  `log` with warnings turned into errors.
- The README said that two kornia fixes in the corpus were merged but not released; the corpus has 31
  kornia cases now, and the pytorch/rl, peft and pytorch (#198006) fixes are in the same state (kornia
  0.8.3 and 0.9.0rc1, torchrl 0.14.0, peft 0.21.2 and torch 2.14.1 predate them or cherry-pick other
  changes), while the timm and ultralytics fixes are released. The sentence says so now, and
  `tests/test_corpus.py` checks that the README table has one row per case with the merge date from
  `cases.json`, so a case added without its row, or a row with a stale date, fails the suite.
- `tests/test_survey.py` checks every reference of the survey registry against the libraries it measures:
  each entry is surveyed on a 24 point float64 grid against torch, numpy and scipy and its median error must
  stay under 32 ulp, which a reference that is another function fails by fifteen orders of magnitude (torch's
  `polygamma_1` is the worst true median at 9 ulp), and the piecewise references (`selu`, `elu`, `entr`,
  `gelu_tanh`, `log_ndtr`, `lgamma`, `digamma`, `erfinv`, `ndtri`, `zeta`, `sinc`, `spherical_bessel_j0`)
  are pinned once per branch, since a slip in one branch moves a third of the grid and keeps the median.
  `survey.py` goes from 77 to 94 percent covered.
- With torch installed but `expecttest` missing, `torch.testing._internal` does not import and
  `ulpwise survey` silently used the dtype default as the `OpInfo` tolerance, so the `op tol` columns
  looked like an override that was never read. The survey now logs one line with the import error and
  the `survey` extra installs `expecttest`. The regression corpus CI job, which has torch but had no
  `expecttest`, failed on the two `op_db` tests for the same reason; it installs `expecttest` now and the
  tests skip with the reason where `op_db` is not importable.
- `ulpwise survey` looked up torch's `OpInfo` tolerance by op name and took the first `op_db` entry
  with that name. `polygamma` has one entry per order, and the first, `polygamma_n_0`, has no
  override, so the `op fail` column of `polygamma_1` and `polygamma_2` in float32 was computed with
  the dtype default (`rtol 1.3e-6`, `atol 1e-5`) instead of torch's `polygamma_n_1` and `_n_2`
  override (`rtol 0.01`, `atol 1e-4`). `Entry` gains `opinfo_variant`, `torch_opinfo_tolerance`
  takes a `variant` and otherwise prefers the base variant, and the two registry entries name
  theirs. In `studies/accuracy-survey-2026-09` the `polygamma_1` f32 row goes from 86 to 7 inputs
  failing the op's tolerance; the measured errors are unchanged (erratum in its README).
- `tests/test_survey.py` covers the survey itself now: `survey()` on `sqrt` against numpy and torch
  (numpy within half an ulp, torch under one), an inexact backend, a backend that raises or returns
  the wrong shape, the backend resolution, the `op_db` tolerance lookup including the variant case,
  the csv and markdown writers, `versions()`, `main()` and the `survey` subcommand. CI installs
  `mpmath` for the Python job so these tests run there instead of being skipped.
- Six more kornia cases, each present on kornia 0.8.3 and fixed on main: the Hessian of `So3.exp`
  at the identity (#4972, nine nan for `-I / 4` and zeros), `sampson_epipolar_distance` of a point on
  its epiline with `squared=False` (#5116, `sqrt(eps)` = 1e-4 for 0) and of the same `F` scaled by
  1e-4 (#5116, a third lower), `RandomHue` on a float64 image (#5131, 8.7e-8 rad past a half turn,
  the float32 rounding error of pi), `MS_SSIMLoss(sigmas=(0.5, 1.3), reduction="none")` (#5143, an
  even 6-pixel window and a `(1, 15, 19)` map for a `(1, 3, 16, 20)` input) and `MS_SSIMLoss` on a
  uint8 pair (#5353, `expected scalar type Byte but found Float`; with `data_range=255` it now scores
  the pair divided by 255 at the default to 1e-6). Six more: `So3.right_jacobian` of a float16
  45 rad rotation (#4967, the identity for a matrix of 0.2 to 0.4), `average_quaternions` with a member
  stored as `3 q` (#4980, 41.8 degrees for the 22.5 degree bisector), `Hyperplane.through` a float16
  triangle with legs of 300 (#5104, the normal `(0, 0, 1)` for `(0, 0, -1)`), `filter2d` with a
  per-sample kernel on a channels-last image (#5301, torch's view error), `lovasz_hinge_loss` of a
  float16 prediction (#5303, a float32 loss) and `get_box_kernel1d` (#5357, a stride-0 view where
  one write zeroed all three taps). The kornia block is 31 cases, all `present` on 0.8.3 and all
  `fixed` on main. One more ultralytics case, `scale_masks` with the dataloader's `ratio_pad` and an
  odd letterbox padding (#26379, a padded row survived the crop and the bottom rows of every mask
  faded to 0) and `verify_image_label` on two triangles that tile one square (#26377, the second
  polygon was dropped as a duplicate of the first because only the class and the box were compared);
  both carry `fixed_in_release` 8.4.165, the first tag with the fixes), and `verify_image_label` on a
  pose row for a detect task (#26357, the row was read as a polygon, its box was out of bounds and
  the image was dropped as corrupt; `fixed_in_release` 8.4.164). The corpus is 50 cases.
- The manual `workflow_dispatch` run of CI is now strict about the corpus like the Monday run (the
  changelog said so already, the workflow set the variable for `schedule` only).
- The release workflow's manual dry run (`workflow_dispatch` without `publish`) now ends in a
  `collect` job that downloads the artifacts the way the publish job does and checks that there
  are five wheels and one sdist, so a dry run covers the whole pipeline short of the upload. The
  workflow actions moved to `actions/checkout@v7`, `setup-python@v7`, `upload-artifact@v7` and
  `download-artifact@v8` (the first Dependabot pull requests), and `Cargo.lock` to pyo3 0.29.3.
- Dependabot (`.github/dependabot.yml`) opens weekly pull requests for Cargo.lock and the
  workflow actions, so pyo3 patch releases land through CI-tested pull requests instead of a
  hand-run `cargo update`; Python dependencies stay unpinned on purpose.
- Eight more kornia cases, each present on kornia 0.8.3 and fixed on main: `Se2.exp`/`Se2.log` at
  theta = 1e-8 (#4960, the translation came back as (1, 2) from exp and (1e-8, -5e-9) from log),
  `point_line_distance` of a homogeneous point with w = 2 (#4975, 4.0 for 1.5), `Quaternion.__pow__`
  of -1 (#5003, the zero quaternion), `So2` from a `(B, 1)` angle times `(B, 2)` points (#5005,
  `(B, B, 2)`), `solve_cubic` of `1e-30 x^3 + 2x - 6` (#5024, roots `[0, 0, 0]` for 3),
  `RgbToGrayscale` on uint8 (#5111, all zeros), `conv_soft_argmax2d` with a far-away peak (#5134,
  the weak window's coordinates moved from 0.8834 to 1.0) and `get_gaussian_discrete_kernel1d(1,
  sigma)` (#5376, 3 taps). Six more in the same shape: `Se3.exp`'s `d t / d omega` at the identity
  (#4963, nan on 0.8.3, zero on main before the fix), `Quaternion.polar_angle`'s gradient at the
  identity (#4981, nan), `decompose_essential_matrix` of a `(3, 3)` input (#4998, `(1, 3, 3)`),
  `Hyperplane.through`'s gradients for orthogonal equal-length edges (#5058, nan),
  `Vector3.normalized` of a float16 zero vector (#5084, NaN) and `otsu_threshold`'s mask below
  zero (#5182, all False). The whole kornia block of the corpus, 19 cases, reads `present` on 0.8.3;
  the mean_average_precision recall thresholds (#5101) were checked and left out because 0.8.3
  already scores the exact-tenth case right.
- CI runs the regression corpus every Monday (and on `workflow_dispatch`) with
  `ULPWISE_CORPUS_STRICT=1`, which turns an unexpected pass into a failure: a release fixed a case
  and `fixed_in_release` is stale. Push and pull request runs stay non strict. The corpus job also
  installs `ultralytics`, so the ultralytics case runs there instead of being skipped.
- Corpus metadata: kornia #4838 is fixed by kornia #5124 (merged 2026-09-30) and kornia #4897 by
  kornia #4941 (merged 2026-09-26); both cases now carry the `pr` and `merged_at`, and stay expected
  failures because no kornia release after 0.9.0rc1 (2026-07-19) exists yet. ultralytics #26330 sets
  `fixed_in_release` to `8.4.164`: the tag `v8.4.164` (2026-09-27, the first release after the merge)
  carries the `use_obb` check in `verify_labels` and `v8.4.163` does not, so the case is a hard
  failure, not an expected one, from 8.4.164 on. peft #3777 keeps `fixed_in_release: null`: the
  `v0.21.1` and `v0.21.2` tags still carry the `weight.size()[2:4]` shortcut in `lora/layer.py`.

## 0.3.0 (2026-09-26)

- `ulpwise scan`: static scan of a repository (a directory, a GitHub URL or `owner/repo`, cloned
  with depth 1) for the floating point patterns behind the corpus bugs. Twelve rules with severity,
  reason, replacement and upstream example: `exp-of-square`, `sin-of-pi-times`, `softplus-by-hand`,
  `logsumexp-by-hand`, `hypot-by-hand`, `sqrt-of-difference`, `one-minus-cos`, `log1p-by-hand`,
  `expm1-by-hand`, `atan-of-quotient`, `small-angle-division`, `acos-for-angle`. Findings carry
  file, line, function and the source line; the report ends with the functions that do the most
  elementary math. `--report` writes Markdown, `--rules` filters, `--fail-on` gates CI,
  `--include-tests` widens the walk. On kornia main the four `small-angle-division` and
  `one-minus-cos` lines are the ones kornia #4897 fixes.
- `ulpwise scan --run`: imports the math-heavy module level functions and calls them on the same
  grid in float32 and float64, reporting the largest error in ulps of the largest output and the
  worst elementwise ulp distance with its input. Static methods run, instance methods and other
  skips carry their reason. `--run-limit` bounds it, `--run-installed` imports the installed package
  instead of the scanned tree.
- `ulp_distance`, `ulp_distances`, `ordered`, `max_ulp` and `assert_max_ulp` take `f16` and
  `bf16`, and `flatten` reads the dtype off float16 and bfloat16 numpy arrays and torch tensors.
  Inputs that are not representable are rounded to nearest even first, so a bfloat16 tensor can be
  measured against a float64 reference; without an explicit dtype the less precise of the two
  inputs decides. Cross-checked against the numpy int16 view for float16 and the torch view for
  bfloat16.
- `special("f16")` and `special("bf16")`: the same 29 named edge values as f32 and f64, so
  `edge_values`, the `edge_f16` and `edge_bf16` pytest fixtures and `ulpwise special f16` work.
  Every value satisfies the same checks as the f32 and f64 tables, run through numpy for float16
  and torch for bfloat16.
- `ulpwise corpus`: runs the regression corpus against the installed packages without pytest and
  prints one line per case, present, fixed or skipped, with the installed version and the upstream
  reference. `--repo` filters by repository or case id, `--fail-if-present` makes a present bug exit 1.
  Any exception from a repro counts as present, like the pytest run: several corpus bugs are crashes.

## 0.2.0 (2026-09-26)

- `ulpwise survey` (`ulpwise.survey`): accuracy survey of 61 elementary and special functions of
  torch, numpy, scipy and jax against a 200 bit mpmath reference, in ulps of the dtype, with the
  worst input per row. For torch it also counts inputs where the vectorized kernel and the scalar
  tail disagree and inputs that would fail the `OpInfo` reference test tolerance, default and per op
  override, read from `op_db`. Writes `results.csv` and `results.md`. Optional extra
  `ulpwise[survey]` pulls in mpmath.
- `studies/accuracy-survey-2026-09`: the first run and its reading notes.
- Corpus: `max_ulp` check type, and five open cases with the complete patch attached to the issue:
  pytorch #198448 (`torch.sqrt` float64 rounding), pytorch #198583 (Bessel and Airy `p1evl` leading
  1), kornia #4838 and #4897 (small angle series), torchvision #9676 (rotated box clamp). Open cases
  carry `issue`, `pr: null` and `fixed_in_release: null` and stay expected failures until a release
  contains the fix.
- Corpus: three more open cases, pytorch #198663 (`polygamma(1, x)`: float64 series truncation and
  float32 reflection argument) and pytorch #198664 (`erfcx` negative branch, `exp` at the rounded
  square), both with the complete patch attached to the issue.
- Corpus: kornia #4768, `ellipse_to_laf` described the wrong ellipse whenever `b != 0`.
- Corpus: ultralytics #26330, OBB datasets with plain box labels are rejected at load time; the repro
  writes a one-image dataset to a temporary directory, so it needs ultralytics but no weights.

## 0.1.2 (2026-09-24)

- `ulpwise` console script, so `uvx ulpwise midpoint sqrt 0.85`, `pipx run ulpwise ...` and a plain
  `ulpwise ...` inside a virtualenv work without `python -m`.

## 0.1.1 (2026-09-24)

No code changes. The AI disclosure in the README was shortened and the package was republished
from a repository with a fresh history.

## 0.1.0 (2026-09-24)

First release.

- Rust core: ordered float views and ulp distances (`ulp`), named edge values computed from the
  format (`edge`), exact rounding oracles for `sqrt`, reciprocal and division with correctly rounded
  results that do not depend on the platform libm, f64 referenced midpoint reports for 15 more
  `f32` functions (`exact`), and knife-edge scans (`knife`).
- Python package built with maturin: `ulp_distance`, `assert_max_ulp` and `max_ulp` for floats,
  lists, numpy arrays and torch tensors, `special`, `neighbours`, `binade_edges`, `all_floats`,
  `midpoint`, `knife_edges`, `sqrt_cr`, a CLI (`python -m ulpwise`) and a pytest plugin
  (`edge_f32` / `edge_f64` parametrization, `assert_max_ulp` fixture).
- Regression corpus of 11 upstream bugs with runnable repros (kornia, pytorch, pytorch/rl, timm,
  peft), run by `tests/test_corpus.py` against whatever is installed.
- `examples/torch_sqrt_conformance.py`: measures how often a platform's `torch.sqrt` and
  `numpy.sqrt` disagree with correct rounding at knife-edge inputs.
