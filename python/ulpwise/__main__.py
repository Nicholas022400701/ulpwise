"""Command line entry points.

    python -m ulpwise knife sqrt --lo 0.5 --hi 1 --tol 1e-3 --dtype f32 --limit 20
    python -m ulpwise midpoint sqrt 0.8528626561164856 --dtype f32
    python -m ulpwise special f32
    python -m ulpwise ulp 0.9235056042671204 0.9235056638717651 --dtype f32
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
