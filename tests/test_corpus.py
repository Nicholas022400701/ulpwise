"""Run the regression corpus against whatever is installed.

A case is skipped when its packages are missing. It is an expected failure (non strict) when the
installed release is not known to contain the fix, so a red run means the bug is present in what
you have installed, and a green one means it is gone.
"""

import pytest

from ulpwise import corpus

CASES = corpus.load()


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
                strict=False,
                reason=f"installed {case['requires'][-1]} {corpus.installed_version(case)} is not known to contain "
                f"the fix from {case['repo']}#{case['pr']}",
            )
        )
    corpus.run(case)
