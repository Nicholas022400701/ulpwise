# Changelog

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
