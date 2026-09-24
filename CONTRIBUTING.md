# Contributing

Build and test locally:

```sh
cargo test --release                      # Rust core
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest numpy packaging
maturin develop --release                 # builds ulpwise._core into the venv
pytest tests -q                           # Python tests and the regression corpus
cargo fmt --check && cargo clippy --release --all-features
```

Rules that keep this project useful:

- Every oracle claim needs an independent check. The Rust `sqrt` and division oracles are
  cross-checked against `Fraction` and `Decimal` in `tests/test_core.py`; a new oracle needs the
  same treatment before it is used anywhere else.
- A corpus case must set `actual` and `expected`, quote the buggy behaviour from the merged
  upstream pull request in `before_fix`, and fail for that reason on a release that predates the
  fix. Do not add a case you have not run against both a buggy and a fixed version, or at least
  against the buggy one with the fixed behaviour taken from the upstream test.
- Numbers in the README come from a script in `examples/` so anyone can rerun them.
- The project is written with an AI coding agent. Say so in the pull request, and read every line
  you send.
- Unless you say otherwise, your contribution is licensed as the project is, MIT or Apache-2.0 at
  the recipient's option, with no extra terms.
