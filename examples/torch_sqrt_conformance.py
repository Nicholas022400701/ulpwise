"""How often does this platform's torch / numpy float32 sqrt disagree with correct rounding?

Knife-edge inputs are the ones where a 1 ulp libm error flips the rounded result, so measuring
there shows an implementation's rounding behaviour in a few thousand evaluations instead of
millions. Run it on every CI platform and compare.
"""

import platform
import sys

import numpy as np

import ulpwise

LO, HI, TOL = 0.5, 1.0, 1e-3


def report(name, values, reference, dtype, exact_above):
    d = ulpwise.ulp_distances(values, reference, dtype)
    off = [i for i, v in enumerate(d) if v]
    # exact_above: the exact result lies above the correctly rounded value, so an implementation
    # that is off there returned next_up(rounded), it rounded up where it should have rounded down.
    up = sum(1 for i in off if exact_above[i])
    print(
        f"  {name:<30s} off by 1 ulp at {len(off):5d} / {len(d)} knife edges"
        f" ({100 * len(off) / len(d):4.1f}%): {up} rounded up instead of down, {len(off) - up} down instead of up"
    )


def main():
    print(f"{platform.platform()} {platform.machine()} python {sys.version.split()[0]} numpy {np.__version__}")
    hits = ulpwise.knife_edges("sqrt", LO, HI, TOL, "f32", limit=10**9)
    xs = np.array([h[0] for h in hits], dtype=np.float32)
    ref = [h[1] for h in hits]
    above = [h[3] for h in hits]
    print(f"float32 sqrt on [{LO}, {HI}): {len(hits)} knife edges within {TOL} ulp of a midpoint")
    report("numpy.sqrt float32", np.sqrt(xs).tolist(), ref, "f32", above)
    try:
        import torch
    except ImportError:
        print("  torch not installed")
        return
    print(f"  torch {torch.__version__} cpu capability {torch.backends.cpu.get_cpu_capability()} mkl {torch.backends.mkl.is_available()}")
    t = torch.from_numpy(xs)
    report("torch.sqrt float32", torch.sqrt(t).tolist(), ref, "f32", above)
    report("torch.pow(x, 0.5) float32", torch.pow(t, 0.5).tolist(), ref, "f32", above)
    report("torch.sqrt float64 -> float32", torch.sqrt(t.double()).float().tolist(), ref, "f32", above)
    hits64 = ulpwise.knife_edges("sqrt", LO, HI, TOL, "f64", limit=20000, max_evals=20_000_000)
    x64 = torch.tensor([h[0] for h in hits64], dtype=torch.float64)
    print(f"float64 sqrt: first {len(hits64)} knife edges of [{LO}, {HI})")
    report("numpy.sqrt float64", np.sqrt(x64.numpy()).tolist(), [h[1] for h in hits64], "f64", [h[3] for h in hits64])
    report("torch.sqrt float64", torch.sqrt(x64).tolist(), [h[1] for h in hits64], "f64", [h[3] for h in hits64])


if __name__ == "__main__":
    main()
