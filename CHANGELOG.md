# Changelog

## Unreleased

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
