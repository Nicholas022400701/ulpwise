"""Self-contained check: is torch.sqrt bit-identical to the correctly rounded square root?

The inputs are knife edges: their exact square root lies within 1e-4 ulp of a rounding midpoint,
so any implementation that is not correctly rounded gets about half of them wrong. The expected
values come from an exact integer oracle (ulpwise) and equal numpy.sqrt on every platform tried.
"""

import numpy as np

F32 = ["0x1.0023e20000000p-1", "0x1.00b2e40000000p-1", "0x1.00c15a0000000p-1", "0x1.00cd860000000p-1", "0x1.00ef800000000p-1", "0x1.0119420000000p-1", "0x1.012e900000000p-1", "0x1.019fe60000000p-1", "0x1.01a62c0000000p-1", "0x1.01f6ec0000000p-1", "0x1.01f9680000000p-1", "0x1.021f0e0000000p-1", "0x1.0236b00000000p-1", "0x1.023c780000000p-1", "0x1.0299a00000000p-1", "0x1.02a5a40000000p-1", "0x1.02c3fc0000000p-1", "0x1.0326180000000p-1", "0x1.0336420000000p-1", "0x1.0339ba0000000p-1", "0x1.03df3e0000000p-1", "0x1.0406660000000p-1", "0x1.04283e0000000p-1", "0x1.043efa0000000p-1", "0x1.04858e0000000p-1", "0x1.048ee80000000p-1", "0x1.04a7600000000p-1", "0x1.052c660000000p-1", "0x1.056ef00000000p-1", "0x1.059f1a0000000p-1", "0x1.05b59c0000000p-1", "0x1.0661d80000000p-1"]
F32_SQRT = ["0x1.6a23460000000p-1", "0x1.6a884e0000000p-1", "0x1.6a92860000000p-1", "0x1.6a9b1c0000000p-1", "0x1.6ab31a0000000p-1", "0x1.6ad0920000000p-1", "0x1.6adf9a0000000p-1", "0x1.6b2f840000000p-1", "0x1.6b33f20000000p-1", "0x1.6b6cd60000000p-1", "0x1.6b6e960000000p-1", "0x1.6b891c0000000p-1", "0x1.6b99c00000000p-1", "0x1.6b9dd00000000p-1", "0x1.6bdf600000000p-1", "0x1.6be7d40000000p-1", "0x1.6bfd2e0000000p-1", "0x1.6c42260000000p-1", "0x1.6c4d820000000p-1", "0x1.6c4ff40000000p-1", "0x1.6cc42e0000000p-1", "0x1.6cdfaa0000000p-1", "0x1.6cf7680000000p-1", "0x1.6d075a0000000p-1", "0x1.6d38d60000000p-1", "0x1.6d3f640000000p-1", "0x1.6d508a0000000p-1", "0x1.6dadb40000000p-1", "0x1.6ddc460000000p-1", "0x1.6dfdf80000000p-1", "0x1.6e0db60000000p-1", "0x1.6e86160000000p-1"]
F64 = ["0x1.000000002b505p-1", "0x1.000000006bd8ap-1", "0x1.000000014507dp-1", "0x1.0000000185902p-1", "0x1.000000021e370p-1", "0x1.000000025ebf5p-1", "0x1.00000002f7663p-1", "0x1.00000003900d1p-1", "0x1.00000003d0956p-1", "0x1.0000000428b3fp-1", "0x1.00000004693c4p-1", "0x1.00000004c15adp-1", "0x1.0000000501e32p-1", "0x1.000000055a01bp-1", "0x1.00000005f2a89p-1", "0x1.000000064ac72p-1", "0x1.00000006e36e0p-1", "0x1.000000073b8c9p-1", "0x1.0000000793ab2p-1", "0x1.0000000884709p-1", "0x1.00000008dc8f2p-1", "0x1.0000000934adbp-1", "0x1.000000098ccc4p-1", "0x1.00000009e4eadp-1", "0x1.0000000a3d096p-1", "0x1.0000000a9527fp-1", "0x1.0000000b5cfb5p-1", "0x1.0000000bb519ep-1", "0x1.0000000cd50bdp-1", "0x1.0000000d9cdf3p-1", "0x1.0000000ed4676p-1", "0x1.00000012520dep-1"]
F64_SQRT = ["0x1.6a09e668125d6p-1", "0x1.6a09e6683fff0p-1", "0x1.6a09e668d991ap-1", "0x1.6a09e66907333p-1", "0x1.6a09e66973243p-1", "0x1.6a09e669a0c5cp-1", "0x1.6a09e66a0cb6cp-1", "0x1.6a09e66a78a7bp-1", "0x1.6a09e66aa6494p-1", "0x1.6a09e66ae498ap-1", "0x1.6a09e66b123a3p-1", "0x1.6a09e66b50899p-1", "0x1.6a09e66b7e2b2p-1", "0x1.6a09e66bbc7a8p-1", "0x1.6a09e66c286b6p-1", "0x1.6a09e66c66bacp-1", "0x1.6a09e66cd2abap-1", "0x1.6a09e66d10fb0p-1", "0x1.6a09e66d4f4a5p-1", "0x1.6a09e66df98a8p-1", "0x1.6a09e66e37d9dp-1", "0x1.6a09e66e76292p-1", "0x1.6a09e66eb4787p-1", "0x1.6a09e66ef2c7cp-1", "0x1.6a09e66f31171p-1", "0x1.6a09e66f6f666p-1", "0x1.6a09e66ffcb2cp-1", "0x1.6a09e6703b021p-1", "0x1.6a09e671069dbp-1", "0x1.6a09e67193ea0p-1", "0x1.6a09e67270335p-1", "0x1.6a09e674e81b5p-1"]


def check(name, got, expected):
    got = np.asarray(got)
    bad = np.flatnonzero(got != expected)
    print(f"  {name:<24s} {len(bad):2d} / {len(expected)} differ from the correctly rounded value")
    for i in bad[:8]:
        print(f"      x={xs_by_dtype[expected.dtype][i]!r} got {float(got[i]).hex()} want {float(expected[i]).hex()}")


xs_by_dtype = {}
x32 = np.array([float.fromhex(h) for h in F32], dtype=np.float32)
e32 = np.array([float.fromhex(h) for h in F32_SQRT], dtype=np.float32)
x64 = np.array([float.fromhex(h) for h in F64], dtype=np.float64)
e64 = np.array([float.fromhex(h) for h in F64_SQRT], dtype=np.float64)
xs_by_dtype[e32.dtype] = x32
xs_by_dtype[e64.dtype] = x64

print(f"numpy {np.__version__}")
check("numpy.sqrt float32", np.sqrt(x32), e32)
check("numpy.sqrt float64", np.sqrt(x64), e64)
try:
    import torch
except ImportError:
    print("torch not installed")
else:
    print(f"torch {torch.__version__} cpu capability {torch.backends.cpu.get_cpu_capability()} mkl {torch.backends.mkl.is_available()}")
    check("torch.sqrt float32", torch.sqrt(torch.from_numpy(x32)).numpy(), e32)
    check("torch.pow(x, 0.5) float32", torch.pow(torch.from_numpy(x32), 0.5).numpy(), e32)
    check("torch.sqrt float64", torch.sqrt(torch.from_numpy(x64)).numpy(), e64)
    check("float64 sqrt then float32", torch.sqrt(torch.from_numpy(x32).double()).float().numpy(), e32)
