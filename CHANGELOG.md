# Changelog

## Unreleased

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
