# lcao-grid-kernel

Fused spherical-harmonic evaluation and scatter-reduce onto a real-space
grid — the inner loop of LCAO-style grid integration, built as a standalone
microkernel with synthetic orbitals and its own harness. A grid-integration
kernel with a DFT-shaped workload, not a DFT code: no SCF, no Hamiltonian,
no pseudopotentials, no fitted basis sets.

## Vendor gap

The operation is a compute-heavy scatter-reduce with geometric constraints,
not a GEMM. cuBLAS has no mapping for it.

## Workload

Synthetic atom-centered orbitals chi(r) = R(|r-c|) * Y_lm(r-c) with

- real orthonormal harmonics Y_lm by factorial-free normalized
  associated-Legendre recurrences plus a rotation recurrence for the
  azimuth (no trig calls, no gamma functions), Condon-Shortley phase;
- contracted two-term Gaussian radial parts R(r) = r^l (c0 e^{-a0 r^2} +
  c1 e^{-a1 r^2}) with a compact support radius r_cut set at a Gaussian
  tail of 1e-12;
- density accumulation rho(g) = sum_o w_o chi_o(r_g)^2 on a uniform grid
  over the cell, points at cell centers.

All parameters come from a seeded generator (`make_synthetic_model`):
atoms_per_side^3 atoms at spacing 6.0, each with s/p/d/f shells (16
orbitals per atom). Nothing is fitted to any element.

## Layout

- `include/lcao/harmonics.hpp` — `ylm_real`, host/device shared, validated
  through l = 6.
- `include/lcao/model.hpp`, `src/model.cpp` — orbital/grid types, the
  seeded synthetic generator, the in-cutoff pair counter.
- `include/lcao/cpu_ref.hpp`, `src/cpu_ref.cpp` — the naive orbital-major
  evaluator (the oracle) and a box-free point-major variant.
- `src/grid_gpu.cu` — the two GPU paths:
  - **unfused**: per 32-orbital chunk, evaluate all chi into a global-memory
    V slab (pass 1), then reduce w*chi^2 into rho (pass 2). The slab
    round-trips 16 * n_orb * n_grid bytes per full pass.
  - **fused**: one thread per (orbital, grid point) pair; in-cutoff pairs
    evaluate chi and `atomicAdd` w*chi^2 straight into rho. V never exists;
    accumulation order is whatever the atomic unit serializes.
- `tests/` — the gates below, registered as CTest cases.
- `benchmarks/bench_grid.cpp` — timing sweep, JSON with raw per-rep timings
  and machine/toolchain metadata.

Build and run:

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/bench_grid          # writes data/lcao_grid_bench.json
```

The CMake configuration selects g++-15 for host compilation because nvcc
rejects the installed g++-16 headers (same pattern as
[kpm-tightbinding](https://github.com/FireHorseAutumn/kpm-tightbinding)).

## Correctness

- **Harmonics gate** (`test_harmonics`, must pass before anything else was
  built): hand-derived closed forms through l = 2 at 4096 seeded random
  directions (max error 4.4e-16) and on the coordinate axes (m != 0 exactly
  0 on the z axis); orthonormality of all 49 harmonics with l <= 6 under
  32-point Gauss-Legendre(theta) x 64-point uniform(phi) quadrature:
  max |diag - 1| = 5.8e-15, max |offdiag| = 3.6e-15.
- **CPU reference** (`test_cpu_density`): single isolated s, p_z, p_x, d_z2
  orbitals match hand-derived closed-form densities at linf <= 6.9e-18;
  the naive evaluator agrees bitwise with a box-free point-major
  accumulation on a full synthetic model; the in-cutoff pair counter
  matches a brute-force count exactly.
- **GPU** (`test_gpu_density`), FP64 vs the CPU oracle: at 16^3/8-atoms,
  33^3/8-atoms (multi-block, partial last block), and 64^3/64-atoms (1024
  orbitals, benchmark class) — unfused vs CPU linf <= 1.4e-15, fused vs CPU
  linf <= 1.4e-15, fused vs unfused linf <= 1.2e-15, on max densities up to
  1.4. Per-point orbital overlap is a handful of terms, so the atomic
  ordering adds little to the host/device `exp` difference. Tolerance in
  the test is 1e-11, four decades above the measured worst case.

## Measured

RTX 5060 Laptop GPU (sm_120, 26 SMs, 32 MiB L2), nvcc 13.2.51, host gcc
15.3.1, CUDA runtime 13020, Release, seed 0x5eed3, device-side timing,
median of 11 reps after one warmup. Measured FP64 STREAM-like triad on the
same GPU: 269.3 GB/s [268.8, 269.7] over 20 reps.

Device milliseconds per full density pass, median [min,max], with the
fused/unfused median-time ratio:

8 atoms, 128 orbitals (in-cutoff 38.5% of pairs):

| grid | unfused ms | fused ms | ratio |
|--:|--:|--:|--:|
| 32 | 4.620 [4.612, 4.634] | 4.541 [4.538, 4.550] | 1.02x |
| 48 | 15.367 [15.360, 15.373] | 15.186 [14.401, 15.197] | 1.01x |
| 64 | 32.587 [32.582, 32.596] | 32.039 [32.034, 32.044] | 1.02x |
| 96 | 101.578 [101.572, 103.731] | 101.463 [101.435, 101.474] | 1.00x |
| 128 | 237.027 [236.991, 245.906] | 240.133 [240.109, 240.196] | 0.99x |

64 atoms, 1024 orbitals (in-cutoff 7.6% of pairs):

| grid | unfused ms | fused ms | ratio |
|--:|--:|--:|--:|
| 32 | 17.041 [17.009, 17.084] | 16.021 [16.017, 16.032] | 1.06x |
| 48 | 52.772 [52.749, 52.895] | 50.670 [50.665, 50.678] | 1.04x |
| 64 | 107.282 [107.260, 107.289] | 101.000 [100.991, 101.024] | 1.06x |
| 96 | 333.438 [332.920, 335.050] | 319.324 [311.746, 319.392] | 1.04x |
| 128 | 898.976 [771.736, 908.162] | 800.729 [753.789, 839.858] | 1.12x |

What the numbers say:

- The unfused value slab moves 16 * n_orb * n_grid bytes per pass, but its
  measured effective rate is 14–44 GB/s against a 269 GB/s triad — the
  unfused path is not bandwidth-bound at these sizes, it is bound by the
  per-pair evaluation work it shares with the fused path. Removing the V
  round-trip therefore buys little: fused wins 1.0–1.1x at 64 atoms and is
  a wash at 8 atoms, including a slight loss at 128^3 (0.99x, overlapping
  spreads). Reported as measured; no counter data taken.
- Fused throughput tops out at 0.43 Geval/s (8 atoms, fine grids) and sits
  at 0.20 Geval/s at 64 atoms; grid-point rates are 7–9 Mpts/s and 2–3
  Mpts/s respectively. The FP64 atomicAdd per in-cutoff pair does not
  measurably hurt at these overlap densities.
- Both 128^3 configs show the widest spreads (at 64 atoms: unfused
  [772, 908] ms, fused [754, 840] ms — overlapping), so the 1.12x median
  ratio there is weaker evidence than the mid-size ratios, where spreads
  are disjoint. Wide multi-rep swings of this size appeared on both modes
  and on repeated sweeps — machine variance, not a mode effect.

Raw per-rep timings and full metadata (device, clocks, driver, runtime,
seed, timestamp, toolchain) are in
[`data/lcao_grid_bench.json`](data/lcao_grid_bench.json). Absolute times
vary with machine state; the table reports medians and observed spreads of
one sweep, not a stable speedup distribution.

Throughput units: **Mgridpts/s** = grid points per second of device time;
**Geval/s** = in-cutoff (orbital, grid-point) evaluations per second, the
honest work rate — out-of-cutoff pairs are skipped by an r^2 test in the
fused kernel and stored as zeros in the unfused slab. In-cutoff fractions
(38.5% at 8 atoms, 7.6% at 64 atoms) are a property of the synthetic cell
density, not a tuning choice.

## Status

CPU reference, GPU fused/unfused kernels, correctness gates, and measured
sweep complete 2026-10-24; suite green 3/3 from a fresh build. Natural next
gate: block-per-atom tiling with shared-memory density accumulation to cut
atomic traffic, measured against the fused baseline here.

Related: [kpm-tightbinding](https://github.com/FireHorseAutumn/kpm-tightbinding),
[pauli-hamiltonian](https://github.com/FireHorseAutumn/pauli-hamiltonian).
