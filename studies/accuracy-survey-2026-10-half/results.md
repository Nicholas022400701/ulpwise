# Accuracy survey

Max error in ulps of the dtype against mpmath at 200 bits, about 60 decimal digits (p99 in brackets), on log spaced inputs over each function's domain plus the named edge values of the dtype. The reference is evaluated at the rounded input, so the numbers measure the implementation and not the conditioning of the function. An error is |got - exact| divided by the spacing of the dtype at the exact value, which is why an exact result of zero or a result in the subnormal range gives enormous numbers: read those rows together with the notes in README.md next to this file.

Versions: ulpwise 0.3.1, numpy 2.2.6, torch 2.14.0+cpu, scipy 1.18.1, mpmath 1.3.0, torch cpu capability AVX512

torch columns: `vec!=scalar` is the number of inputs where the vectorized kernel and the scalar tail disagree; `default fail` and `op fail` count inputs that fail torch's reference test tolerance for the dtype, with the default tolerance and with the op's own override respectively. A row with `default fail > 0` and `op fail = 0` is an error that torch's test suite tolerates because of the override.

| function | dtype | torch max (p99) | torch vec!=scalar | torch default fail | torch op fail | op (rtol, atol) |
|---|---|---|---|---|---|---|
| exp | bf16 | 0.5 (0.48) | 0 | 0 | 0 | (0.016, 1e-05) |
| exp | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| exp2 | bf16 | 0.49 (0.48) | 0 | 0 | 0 | (0.016, 1e-05) |
| exp2 | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| expm1 | bf16 | 0.49 (0.47) | 0 | 0 | 0 | (0.016, 1e-05) |
| expm1 | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| log | bf16 | 0.5 (0.5) | 0 | 0 | 0 | (0.016, 0.05) |
| log | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| log2 | bf16 | 0.5 (0.5) | 0 | 0 | 0 | (0.016, 0.1) |
| log2 | f16 | 0.5 (0.5) | 0 | 0 | 0 | (0.001, 1e-05) |
| log10 | bf16 | 0.5 (0.49) | 0 | 0 | 0 | (0.016, 0.05) |
| log10 | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| log1p | bf16 | 0.5 (0.49) | 0 | 0 | 0 | (0.016, 0.1) |
| log1p | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| sqrt | bf16 | 0.5 (0.5) | 0 | 0 | 0 | (0.016, 0.07) |
| sqrt | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| rsqrt | bf16 | 1.1 (0.5) | 184 | 0 | 0 | (0.016, 1e-05) |
| rsqrt | f16 | 0.95 (0.5) | 128 | 0 | 0 | (0.001, 0.05) |
| reciprocal | bf16 | 0.5 (0.5) +1 nonfinite | 0 | 0 | 0 | (0.016, 1e-05) |
| reciprocal | f16 | 0.5 (0.5) +1 nonfinite | 0 | 0 | 0 | (0.001, 1e-05) |
| sin | bf16 | 0.5 (0.49) | 0 | 0 | 0 | (0.016, 0.01) |
| sin | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| cos | bf16 | 0.5 (0.47) | 0 | 0 | 0 | (0.016, 0.01) |
| cos | f16 | 0.5 (0.5) | 0 | 0 | 0 | (0.001, 1e-05) |
| tan | bf16 | 0.5 (0.49) | 0 | 0 | 0 | (0.016, 1e-05) |
| tan | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| asin | bf16 | 0.49 (0.48) | 0 | 0 | 0 | (0.016, 0.01) |
| asin | f16 | 0.49 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| acos | bf16 | 0.5 (0.49) | 0 | 0 | 0 | (0.016, 0.1) |
| acos | f16 | 0.5 (0.5) | 0 | 0 | 0 | (0.001, 0.01) |
| atan | bf16 | 0.5 (0.48) | 0 | 0 | 0 | (0.016, 0.01) |
| atan | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| sinh | bf16 | 0.48 (0.43) | 0 | 0 | 0 | (0.016, 1e-05) |
| sinh | f16 | 0.5 (0.47) | 0 | 0 | 0 | (0.001, 0.01) |
| cosh | bf16 | 0.5 (0.49) | 0 | 0 | 0 | (0.016, 1e-05) |
| cosh | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| tanh | bf16 | 0.49 (0.42) | 0 | 0 | 0 | (0.016, 0.01) |
| tanh | f16 | 0.5 (0.47) | 0 | 0 | 0 | (0.001, 1e-05) |
| asinh | bf16 | 0.5 (0.49) | 0 | 0 | 0 | (0.016, 0.05) |
| asinh | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| acosh | bf16 | 0.5 (0.49) | 0 | 0 | 0 | (0.016, 0.05) |
| acosh | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| atanh | bf16 | 0.5 (0.48) | 0 | 0 | 0 | (0.016, 0.01) |
| atanh | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |
| erf | bf16 | 254 (241) | 0 | 0 | 0 | (0.016, 0.01) |
| erf | f16 | 4.6 (4) | 0 | 0 | 0 | (0.001, 0.01) |
| erfc | bf16 | 0.5 (0.49) | 0 | 0 | 0 | (0.016, 0.01) |
| erfc | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 0.01) |
| erfinv | bf16 | 0.5 (0.5) | 0 | 0 | 0 | (0.016, 0.01) |
| erfinv | f16 | 0.5 (0.5) | 0 | 0 | 0 | (0.001, 0.01) |
| lgamma | bf16 | 0.5 (0.49) | 0 | 0 | 0 | (0.016, 1e-05) |
| lgamma | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 0.7) |
| digamma | bf16 | 0.5 (0.49) +1 nonfinite | 0 | 0 | 0 | (0.016, 1e-05) |
| digamma | f16 | 0.5 (0.49) +1 nonfinite | 0 | 0 | 0 | (0.001, 0.5) |
| sigmoid | bf16 | 0.73 (0.5) | 0 | 0 | 0 | (0.016, 0.01) |
| sigmoid | f16 | 0.5 (0.5) | 0 | 0 | 0 | (0.001, 0.01) |
| i0 | bf16 | 0.5 (0.48) | 0 | 0 | 0 | (0.016, 1e-05) |
| i0 | f16 | 0.5 (0.48) | 0 | 0 | 0 | (0.001, 1e-05) |
| i0e | bf16 | 0.65 (0.5) | 413 | 0 | 0 | (0.016, 0.3) |
| i0e | f16 | 1.5 (0.75) | 303 | 0 | 0 | (0.001, 0.3) |
| i1 | bf16 | 0.5 (0.49) | 0 | 0 | 0 | (0.016, 1e-05) |
| i1 | f16 | 0.5 (0.5) | 0 | 0 | 0 | (0.001, 1e-05) |
| i1e | bf16 | 0.5 (0.48) | 0 | 0 | 0 | (0.016, 1e-05) |
| i1e | f16 | 0.5 (0.5) | 0 | 0 | 0 | (0.001, 1e-05) |
| logit | bf16 | 64 (5.8) | 0 | 10 | 0 | (0.016, 0.5) |
| logit | f16 | 72 (7.1) | 0 | 29 | 0 | (0.001, 0.5) |
| sinc | bf16 | 8.4e+32 (6.6e+32) | 0 | 0 | 0 | (0.016, 1e-05) |
| sinc | f16 | 1 (1) | 0 | 0 | 0 | (0.001, 1e-05) |
| entr | bf16 | 1.2 (1) | 0 | 0 | 0 | (0.016, 0.1) |
| entr | f16 | 1.2 (0.94) | 0 | 0 | 0 | (0.001, 0.1) |
| polygamma_1 | bf16 | 0.5 (0.49) +75 nonfinite | 0 | 0 | 0 | (0.1, 10) |
| polygamma_1 | f16 | 0.53 (0.48) | 0 | 0 | 0 | (0.001, 1e-05) |
| polygamma_2 | bf16 | 1.2 (0.5) | 0 | 0 | 0 | (0.1, 10) |
| polygamma_2 | f16 | 14 (7.4) | 0 | 0 | 0 | (0.001, 1e-05) |
| gelu | bf16 | 242 (156) | 0 | 0 | 0 | (0.016, 1e-05) |
| gelu | f16 | 1.2 (0.5) | 3 | 0 | 0 | (0.001, 1e-05) |
| gelu_tanh | bf16 | 250 (163) | 0 | 0 | 0 | (0.016, 1e-05) |
| gelu_tanh | f16 | 0.91 (0.5) | 0 | 0 | 0 | (0.001, 1e-05) |
| silu | bf16 | 68 (0.49) | 0 | 0 | 0 | (0.016, 1e-05) |
| silu | f16 | 0.5 (0.5) | 0 | 0 | 0 | (0.001, 1e-05) |
| mish | bf16 | 0.49 (0.49) | 0 | 0 | 0 | (0.016, 1e-05) |
| mish | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 0.01) |
| softplus | bf16 | 0.5 (0.5) | 0 | 0 | 0 | (0.016, 0.01) |
| softplus | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.01, 0.01) |
| logsigmoid | bf16 | 0.5 (0.5) | 0 | 0 | 0 | (0.016, 0.005) |
| logsigmoid | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 0.01) |
| elu | bf16 | 0.5 (0.46) | 0 | 0 | 0 | (0.016, 1e-05) |
| elu | f16 | 0.5 (0.47) | 0 | 0 | 0 | (0.001, 1e-05) |
| selu | bf16 | 0.5 (0.5) | 0 | 0 | 0 | (0.016, 1e-05) |
| selu | f16 | 0.5 (0.49) | 0 | 0 | 0 | (0.001, 1e-05) |

## Errors tolerated by a torch override

- `logit` f16: 29 of 575 inputs fail the default tolerance, none fail the op's (rtol 0.001, atol 0.5); worst x = 0.497314453125, got -0.01129150390625, exact -0.01074229080098363 (72 ulp).
- `logit` bf16: 10 of 480 inputs fail the default tolerance, none fail the op's (rtol 0.016, atol 0.5); worst x = 0.498046875, got -0.00390625, exact -0.007812539736793652 (64 ulp).
