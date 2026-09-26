"""Command line entry points.

    python -m ulpwise knife sqrt --lo 0.5 --hi 1 --tol 1e-3 --dtype f32 --limit 20
    python -m ulpwise midpoint sqrt 0.8528626561164856 --dtype f32
    python -m ulpwise special f32
    python -m ulpwise ulp 0.9235056042671204 0.9235056638717651 --dtype f32
    python -m ulpwise corpus --repo pytorch
    python -m ulpwise scan kornia/kornia --report kornia.md
"""

import argparse
import sys

import ulpwise


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ulpwise")
    sub = parser.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("knife", help="scan for knife-edge inputs of a function")
    k.add_argument("op", choices=list(ulpwise.UNARY_OPS))
    k.add_argument("--lo", type=float, required=True)
    k.add_argument("--hi", type=float, required=True)
    k.add_argument("--tol", type=float, default=1e-3, help="midpoint distance in ulps")
    k.add_argument("--dtype", default="f32")
    k.add_argument("--limit", type=int, default=20)
    k.add_argument("--stride", type=int, default=1)

    m = sub.add_parser("midpoint", help="report where the exact result of op(x) sits")
    m.add_argument("op")
    m.add_argument("x", type=float)
    m.add_argument("y", type=float, nargs="?", help="divisor for op=div")
    m.add_argument("--dtype", default="f32")

    s = sub.add_parser("special", help="list the named edge values of a dtype")
    s.add_argument("dtype", nargs="?", default="f32")

    u = sub.add_parser("ulp", help="ulp distance between two values")
    u.add_argument("a", type=float)
    u.add_argument("b", type=float)
    u.add_argument("--dtype", default="f64")

    v = sub.add_parser("survey", help="accuracy survey of elementary and special functions against mpmath")
    v.add_argument("--backends", default="torch,numpy,scipy,jax", help="comma separated: torch numpy scipy jax")
    v.add_argument("--dtypes", default="f32,f64")
    v.add_argument("--points", type=int, default=600, help="grid points per function and dtype")
    v.add_argument("--functions", default=None, help="comma separated subset of function names")
    v.add_argument("--out", default="survey", help="output directory for results.csv and results.md")

    c = sub.add_parser("corpus", help="run the regression corpus against the installed packages")
    c.add_argument("--repo", default=None, help="only cases whose repository or id contains this text, e.g. timm")
    c.add_argument(
        "--fail-if-present", action="store_true", help="exit 1 when at least one case still shows its bug"
    )

    n = sub.add_parser("scan", help="read a repository for the floating point patterns behind the corpus bugs")
    n.add_argument("target", help="a directory, a GitHub URL or owner/repo (cloned with depth 1)")
    n.add_argument("--report", default=None, help="write a Markdown report here instead of printing")
    n.add_argument("--top", type=int, default=20, help="how many math-heavy functions to list")
    n.add_argument("--rules", default=None, help="comma separated subset of rule ids")
    n.add_argument("--include-tests", action="store_true", help="also scan test files and directories")
    n.add_argument("--fail-on", choices=["high", "medium", "info"], default=None, help="exit 1 when a finding of this severity or worse exists")
    n.add_argument("--workdir", default=None, help="where to clone (default: a temporary directory)")

    args = parser.parse_args(argv)
    if args.cmd == "knife":
        hits = ulpwise.knife_edges(args.op, args.lo, args.hi, args.tol, args.dtype, args.limit, args.stride)
        print(f"{'x':>24} {'rounded':>24} {'midpoint distance (ulp)':>24} {'exact lies':>10}")
        for x, rounded, dist, above in hits:
            print(f"{x!r:>24} {rounded!r:>24} {dist:>24.3e} {'above' if above else 'below':>10}")
        print(f"{len(hits)} knife edge(s) with distance < {args.tol} ulp", file=sys.stderr)
    elif args.cmd == "midpoint":
        rep = ulpwise.midpoint(args.op, args.x, args.dtype, args.y)
        if rep is None:
            print("result is not finite")
            return 1
        rounded, dist, above, exact = rep
        if exact:
            print(f"exact: {args.op}({args.x!r}) = {rounded!r} in {args.dtype}")
        else:
            side = "above" if above else "below"
            print(f"rounded {rounded!r}, exact result lies {side} it, {dist:.3e} ulp from the rounding midpoint")
    elif args.cmd == "special":
        for name, value in ulpwise.special(args.dtype):
            print(f"{name:>26}  {value!r}")
    elif args.cmd == "ulp":
        print(ulpwise.ulp_distance(args.a, args.b, args.dtype))
    elif args.cmd == "survey":
        from ulpwise import survey as _survey

        return _survey.main(args)
    elif args.cmd == "scan":
        from ulpwise import scan as _scan

        return _scan.main(args)
    elif args.cmd == "corpus":
        from ulpwise import corpus as _corpus

        cases = [k for k in _corpus.load() if not args.repo or args.repo in k["repo"] or args.repo in k["id"]]
        counts = _corpus.report(cases, sys.stdout)
        return 1 if args.fail_if_present and counts["present"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
