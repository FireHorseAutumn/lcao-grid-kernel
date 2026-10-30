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
  seeded synthetic generator, the in-cutoff and bounding-box pair counters.
- `include/lcao/cpu_ref.hpp`, `src/cpu_ref.cpp` — the naive orbital-major
  evaluator (the oracle) and a box-free point-major variant.
- `src/grid_gpu.cu` — the three GPU paths:
  - **unfused**: per 32-orbital chunk, evaluate all chi into a global-memory
    V slab (pass 1), then reduce w*chi^2 into rho (pass 2). The slab
    round-trips 16 * n_orb * n_grid bytes per full pass.
  - **fused**: one thread per (orbital, grid point) pair; in-cutoff pairs
    evaluate chi and `atomicAdd` w*chi^2 straight into rho. V never exists;
    accumulation order is whatever the atomic unit serializes.
  - **tiled**: one block per (orbital group, 8^3 sub-tile of the group's
    union bounding box); group params staged in shared memory, w*chi^2
    accumulated into a shared tile with privatized atomics, one global
    `atomicAdd` per nonzero in-grid tile point on flush. Only bounding-box
    points are visited. Two groupings: per atom (fixed 16-orbital stride)
    and per shell (maximal runs sharing center and r_cut); tile side is
    `LCG_TILE_SIDE` (default 8).
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
  orbitals match hand-derived closed-form densities at linf <= 7e-18;
  an orbital centered exactly on a grid point (the r = 0 branch) gives
  R(0)*Y_00 for s and exactly 0 for p; the naive evaluator agrees bitwise
  with a box-free point-major accumulation on a full synthetic model; the
  in-cutoff pair counter matches a brute-force count exactly.
- **GPU** (`test_gpu_density`), FP64 vs the CPU oracle: at 6^3/8-atoms
  (grid points land exactly on atom centers, exercising the r = 0 branch
  on every mode), 16^3/8-atoms, 33^3/8-atoms (multi-block, partial last
  block; the tiled paths also get tiles overhanging the grid edge and
  boxes larger than one sub-tile), and
  64^3/64-atoms (1024 orbitals, benchmark class) — all four GPU modes vs
  CPU linf <= 1.4e-15, fused vs unfused linf <= 1.2e-15, tiled vs fused
  linf <= 5.6e-16, tiled-shell vs tiled linf <= 4.5e-16, on max
  densities up to 1.4. Per-point orbital overlap is a handful of terms, so
  the atomic ordering adds little to the host/device `exp` difference.
  Tolerance in the test is 1e-11, four decades above the measured worst
  case. The bench entry points reject n_reps < 1, and
  `count_box_pairs` rejects a zero group stride.

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

### Tiled paths (2026-10-26 sweep; pinned rerun 2026-10-30)

Same protocol as above, all four modes in one session, tile side 8. The
table below is the governor-pinned idle-machine rerun of 2026-10-30
(performance governor, triad 348.2 GB/s [347.1, 349.3]); the original
2026-10-26 evening sweep (triad 327.2 GB/s) agrees with it to the second
decimal on every ratio — the earliest 2026-10-24 sweep (triad 269.3 GB/s)
is the one that drifted. Ratios are against the fused median; u/ts is vs
unfused.

8 atoms, 128 orbitals (in-cutoff 38.5% of pairs; atom boxes visit 81–83%,
shell boxes 55–58%):

| grid | unfused ms | fused ms | tiled ms | tiled-shell ms | t/f | ts/f | u/ts |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 32 | 3.865 [3.858, 3.872] | 3.796 [3.792, 3.801] | 4.070 [4.016, 4.116] | 3.694 [3.642, 3.750] | 0.93x | 1.03x | 1.05x |
| 48 | 12.847 [12.845, 12.850] | 12.698 [12.694, 12.702] | 11.645 [11.606, 11.727] | 10.945 [10.910, 10.974] | 1.09x | 1.16x | 1.17x |
| 64 | 28.709 [28.701, 28.719] | 28.273 [28.268, 28.278] | 26.322 [26.287, 26.376] | 24.748 [24.709, 24.775] | 1.07x | 1.14x | 1.16x |
| 96 | 89.313 [89.303, 89.334] | 87.571 [87.560, 87.583] | 82.711 [82.661, 82.757] | 78.709 [78.695, 78.719] | 1.06x | 1.11x | 1.13x |
| 128 | 204.680 [204.659, 204.699] | 200.312 [200.305, 200.337] | 193.507 [193.484, 193.520] | 183.145 [183.133, 183.174] | 1.04x | 1.09x | 1.12x |

64 atoms, 1024 orbitals (in-cutoff 7.6% of pairs; atom boxes visit
19.5–21.1%, shell boxes 13.0–14.2%):

| grid | unfused ms | fused ms | tiled ms | tiled-shell ms | t/f | ts/f | u/ts |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 32 | 14.210 [14.192, 14.227] | 13.366 [13.359, 13.368] | 9.258 [9.003, 9.428] | 8.484 [8.396, 8.629] | 1.44x | 1.58x | 1.67x |
| 48 | 44.010 [43.994, 44.030] | 42.138 [42.130, 42.143] | 24.368 [24.245, 24.525] | 22.994 [22.907, 23.095] | 1.73x | 1.83x | 1.91x |
| 64 | 89.028 [89.017, 89.060] | 84.018 [84.012, 84.023] | 53.485 [53.414, 53.616] | 50.824 [50.754, 50.849] | 1.57x | 1.65x | 1.75x |
| 96 | 277.233 [277.206, 277.267] | 258.332 [258.307, 258.376] | 161.522 [161.462, 161.595] | 153.253 [153.205, 153.277] | 1.60x | 1.69x | 1.81x |
| 128 | 632.815 [632.740, 632.911] | 586.316 [586.299, 586.357] | 360.192 [360.156, 360.218] | 344.016 [343.990, 344.052] | 1.63x | 1.70x | 1.84x |

What the tiled numbers say:

- The 2026-10-24 sweep showed fused ~ unfused even though fused evaluates
  only 7.6% of pairs at 64 atoms — the cost is the per-pair visit (index
  math, parameter load, r^2 test) over the full orbital x grid product, not
  the evaluation. Tiled tests that directly: visiting only bounding-box
  pairs (19.5–21.1% at 64 atoms) wins 1.44–1.73x over fused, spreads
  disjoint at every 64-atom size.
- Per-shell boxes cut the union-box waste: visited pairs drop to
  13.0–14.2% (the remainder is the sphere-in-box factor, pi/6 ~ 0.52 of
  each shell box, against the 7.6% in-cutoff floor). Tiled-shell is the
  best path at every measured size on both atom counts: 1.03–1.16x over
  fused at 8 atoms, 1.58–1.83x at 64 atoms, 1.91x peak vs unfused.
- The win tracks the visit-count reduction, not the privatization: global
  atomic traffic drops by the group size (one flush per tile point) but
  FP64 atomics were never measurably hot at these overlap densities.
  Shared-memory atomics did not add measurable cost either.
- Tile side A/B (recorded, cross-run with ~6% triad drift — only cells
  past that margin are decisive): side 8 vs 16 wins or ties everywhere,
  decisively where boxes are small — the 32^3/8-atom atom-box loss with
  side 16 (0.59x, parallelism starved at atoms x tiles blocks) shrinks to
  0.93x with side 8 (pinned rerun; 0.94x unpinned), and the shell variant
  flips it to a 1.03x win. Side
  8 is the committed default (`LCG_TILE_SIDE` in CMake).
- Remaining losing cell: atom-box tiled at 32^3/8 atoms, 0.93x with
  disjoint spreads — boxes span the whole cell there, so tiling pays
  per-tile fixed costs for no visit reduction. The shell variant already
  covers it.

Raw per-rep timings and full metadata (device, clocks, driver, runtime,
seed, timestamp, toolchain) are in
[`data/lcao_grid_bench.json`](data/lcao_grid_bench.json) — the pinned
2026-10-30 sweep. The table reports medians and observed spreads of one
pinned sweep, not a stable speedup distribution.

Throughput units: **Mgridpts/s** = grid points per second of device time;
**Geval/s** = in-cutoff (orbital, grid-point) evaluations per second, the
honest work rate — out-of-cutoff pairs are skipped by an r^2 test in the
fused kernel and stored as zeros in the unfused slab. In-cutoff fractions
(38.5% at 8 atoms, 7.6% at 64 atoms) are a property of the synthetic cell
density, not a tuning choice.

## Status

CPU reference, GPU unfused/fused/tiled/tiled-shell kernels, correctness
gates, and measured sweeps complete 2026-10-26; suite green 3/3. The
tiled-shell path (per-shell bounding boxes, tile side 8) is the leader at
every measured size: 1.03–1.16x over fused at 8 atoms, 1.58–1.83x at 64
atoms. The governor-pinned rerun landed 2026-10-30 and confirmed the
2026-10-26 sweep to the second decimal on every ratio; the machine-state
caveat is lifted. Natural next gates: closing the remaining sphere-in-box
waste (13–14% visited vs 7.6% in-cutoff at 64 atoms) and block-size tuning.

Related: [kpm-tightbinding](https://github.com/FireHorseAutumn/kpm-tightbinding),
[pauli-hamiltonian](https://github.com/FireHorseAutumn/pauli-hamiltonian).
