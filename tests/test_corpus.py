"""Run the regression corpus against whatever is installed.

A case is skipped when its packages are missing. It is an expected failure when the installed
release is not known to contain the fix, so a red run means the bug is present in what you have
installed, and a green one means it is gone. With ``ULPWISE_CORPUS_STRICT=1`` an unexpected pass
is a failure too: a release fixed the case and ``fixed_in_release`` in ``cases.json`` is stale.
"""

import os
import pathlib
import re
import sys

import pytest

from ulpwise import corpus

CASES = corpus.load()
STRICT = os.environ.get("ULPWISE_CORPUS_STRICT", "0") not in ("", "0")


def _release_has_fix(case):
    fixed = case.get("fixed_in_release")
    have = corpus.installed_version(case)
    if fixed is None or have is None:
        return False
    try:
        from packaging.version import Version

        return Version(have) >= Version(fixed)
    except Exception:  # noqa: BLE001
        return have >= fixed


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_corpus_case(case, request):
    missing = corpus.missing_requirements(case)
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")
    if not _release_has_fix(case):
        request.node.add_marker(
            pytest.mark.xfail(
                strict=STRICT,
                reason=f"installed {case['requires'][-1]} {corpus.installed_version(case)} is not known to contain "
                f"the fix from {case['repo']}#{case['pr'] or case.get('issue')}",
            )
        )
    corpus.run(case)


def test_corpus_command_reports_every_case(capsys):
    """`ulpwise corpus` prints one line per case and a summary, whatever is installed."""
    from ulpwise.__main__ import main

    assert main(["corpus"]) == 0
    out = capsys.readouterr().out.splitlines()
    ids = [line.split()[0] for line in out[:-1]]
    assert ids == [c["id"] for c in CASES]
    assert all(any(state in line.split() for state in ("present", "fixed", "skipped")) for line in out[:-1])
    present, fixed, skipped = (int(word) for word in out[-1].split() if word.isdigit())
    assert present + fixed + skipped == len(CASES)


def test_corpus_command_filter_and_exit_code(capsys):
    """--repo narrows the cases; --fail-if-present turns a present bug into exit code 1."""
    from ulpwise.__main__ import main

    code = main(["corpus", "--repo", "timm", "--fail-if-present"])
    out = capsys.readouterr().out.splitlines()
    assert all(line.startswith("timm-") for line in out[:-1])
    present = int(out[-1].split()[0])
    assert code == (1 if present else 0)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_corpus_case_declares_its_imports(case):
    """Every package a repro imports is in `requires`, so a missing one skips the case instead of failing it."""
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"ulpwise", "numpy"}
    imported = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", case["repro"], re.M))
    assert imported - stdlib <= set(case["requires"]), case["id"]
    assert case["requires"][-1] != "numpy"  # the last entry is the package the bug lived in


def test_readme_table_lists_every_case():
    """The corpus table in README.md has one row per fix, with the merge date from `cases.json`."""
    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        pytest.skip("README.md is not part of this checkout")
    text = readme.read_text(encoding="utf-8")
    table = text[text.index("| case | kind | merged |") :].split("\n\n", 1)[0].splitlines()[2:]
    rows = {}
    for line in table:
        case_cell, _kind, merged = (cell.strip() for cell in line.strip().strip("|").split("|"))
        rows[int(re.search(r"#(\d+)", case_cell).group(1))] = merged  # the first number is the PR, or the issue
    expected = {case["pr"] or case["issue"]: case["merged_at"][:10] if case["pr"] else "open" for case in CASES}
    assert rows == expected
