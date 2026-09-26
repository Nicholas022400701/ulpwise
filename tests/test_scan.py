"""``ulpwise scan``: every rule fires on the snippet written for it, the guarded variant stays quiet,
test directories are skipped and the CLI exit code follows --fail-on."""

import textwrap

import pytest

from ulpwise import scan

SNIPPET = textwrap.dedent(
    '''
    import math
    import torch


    def erfcx_neg(x):
        return 2.0 * torch.exp(x * x) - erfcx_pos(-x)


    def trigamma_reflect(x):
        z = math.pi / torch.sin(math.pi * x)
        return -trigamma_pos(1 - x) + z * z


    def softplus(x):
        return torch.log(1 + torch.exp(x))


    def lse(a, b):
        return torch.log(torch.exp(a) + torch.exp(b))


    def norm2(a, b):
        return torch.sqrt(a ** 2 + b ** 2)


    def angle_between(u, v):
        return torch.acos((u * v).sum(-1))


    def so3_jacobian(theta, w):
        return (1 - torch.cos(theta)) / theta ** 2 * w + (theta - torch.sin(theta)) / theta ** 3


    def so3_jacobian_guarded(theta, w):
        small = theta.abs() < 1e-4
        s = torch.where(small, 1 - theta ** 2 / 6, torch.sin(theta) / theta)
        return w / s


    def misc(x, y):
        a = torch.log(1 + x)
        b = torch.exp(x) - 1
        c = torch.atan(y / x)
        d = torch.sqrt(1 - x * x)
        e = (-(x.pow(2))).exp()
        f = torch.exp(-torch.square(x))
        return a, b, c, d, e, f
    '''
)

EXPECTED = {
    ("exp-of-square", "erfcx_neg"),
    ("exp-of-square", "misc"),
    ("sin-of-pi-times", "trigamma_reflect"),
    ("softplus-by-hand", "softplus"),
    ("logsumexp-by-hand", "lse"),
    ("hypot-by-hand", "norm2"),
    ("acos-for-angle", "angle_between"),
    ("one-minus-cos", "so3_jacobian"),
    ("small-angle-division", "so3_jacobian"),
    ("log1p-by-hand", "misc"),
    ("expm1-by-hand", "misc"),
    ("atan-of-quotient", "misc"),
    ("sqrt-of-difference", "misc"),
}


def test_every_rule_fires_once_on_its_snippet_and_the_guard_silences_it():
    findings, spots = scan.scan_source(SNIPPET, "demo.py")
    assert {(f.rule, f.function) for f in findings} == EXPECTED
    assert {rule for rule, _ in EXPECTED} == set(scan.RULES)  # every rule has a snippet
    assert sum(f.rule == "exp-of-square" for f in findings) == 3  # x * x, x.pow(2), square(x)
    assert not [f for f in findings if f.function == "so3_jacobian_guarded"]
    assert all(f.snippet and f.line > 0 and f.path == "demo.py" for f in findings)
    busiest = max(spots, key=lambda s: s.calls)
    assert busiest.function == "misc" and busiest.calls == 7 and "exp" in busiest.names


def test_scan_tree_skips_tests_and_reports(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "ops.py").write_text(SNIPPET)
    (pkg / "broken.py").write_text("def f(:\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ops.py").write_text(SNIPPET)
    findings, spots, nfiles = scan.scan_tree(str(tmp_path))
    assert nfiles == 2 and {f.path for f in findings} == {"pkg/ops.py"}
    findings_all, _, nfiles_all = scan.scan_tree(str(tmp_path), include_tests=True)
    assert nfiles_all == 3 and len(findings_all) == 2 * len(findings)
    root, sha = scan.checkout(str(tmp_path))
    assert root == str(tmp_path) and sha is None
    with pytest.raises(ValueError):
        scan.checkout("not a repository at all")
    text = scan.render(findings, spots, "pkg", None, nfiles)
    assert text.startswith("ulpwise scan of pkg: 2 files") and "exp-of-square [high] x3" in text
    md = scan.render(findings, spots, "pkg", "abc123", nfiles, top=3, markdown=True)
    assert md.startswith("# ulpwise scan of pkg @ abc123") and "| calls |" in md and md.count("\n| ") == 4


def test_scan_cli(tmp_path, capsys):
    from ulpwise.__main__ import main

    (tmp_path / "ops.py").write_text(SNIPPET)
    report = tmp_path / "scan.md"
    assert main(["scan", str(tmp_path), "--report", str(report), "--fail-on", "high"]) == 1
    assert "findings" in capsys.readouterr().out and report.read_text().startswith("# ulpwise scan of")
    assert main(["scan", str(tmp_path), "--rules", "acos-for-angle", "--fail-on", "high"]) == 0
    out = capsys.readouterr().out
    assert "acos-for-angle [info] x1" in out and "exp-of-square" not in out
    assert main(["scan", str(tmp_path), "--rules", "no-such-rule"]) == 2
    assert main(["scan", str(tmp_path / "missing")]) == 2


RUN_SNIPPET = textwrap.dedent(
    '''
    import torch


    def unguarded(theta):
        return (1 - torch.cos(theta)) / theta ** 2


    def guarded(theta):
        t2 = theta * theta
        series = 0.5 - t2 / 24 + t2 * t2 / 720 - t2 * t2 * t2 / 40320
        return torch.where(theta.abs() < 0.5, series, (1 - torch.cos(theta)) / t2)


    def cancels(x):
        return torch.stack([torch.cos(0 * x), (1 + x) - 1 - x], -1)


    def two_args(a, b):
        return torch.sqrt(a * a + b * b)


    def needs_three(a, b, c):
        return torch.exp(a) + b + c


    def returns_str(x):
        return str(torch.exp(x))


    class K:
        @staticmethod
        def method(x):
            return torch.exp(x * x)

        def bound(self, x):
            return torch.exp(x)
    '''
)


def test_run_measures_float32_against_float64(tmp_path, capsys):
    pytest.importorskip("torch")
    pkg = tmp_path / "runpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "ops.py").write_text(RUN_SNIPPET)
    _, spots, _ = scan.scan_tree(str(tmp_path))
    results = {r.spot.function: r for r in scan.run_functions(str(tmp_path), spots)}
    assert results["unguarded"].status == "ran" and results["unguarded"].at_scale > 1e5
    assert results["unguarded"].worst > 1e6 and abs(results["unguarded"].at) < 1e-6
    assert results["guarded"].status == "ran" and results["guarded"].at_scale < 100 and results["guarded"].worst < 100
    assert results["cancels"].worst > 1e6 and results["cancels"].at_scale <= 2  # noise against noise, but tiny at scale
    assert results["two_args"].status == "ran" and results["two_args"].worst <= 2
    assert results["needs_three"].status == "needs 3 positional arguments"
    assert results["returns_str"].status == "did not return a float array"
    assert results["K.method"].status == "ran" and results["K.method"].worst is not None  # static methods run
    assert results["K.bound"].status == "instance method"
    text = scan.render_run(list(results.values()))
    assert text.splitlines()[1].startswith("8 functions tried, 5 ran, 3 skipped") and "unguarded" in text.splitlines()[2]
    from ulpwise.__main__ import main

    report = tmp_path / "scan.md"
    assert main(["scan", str(tmp_path), "--run", "--run-limit", "3", "--report", str(report)]) == 0
    assert "| at scale | elementwise |" in report.read_text() and "3 functions tried" in report.read_text()
