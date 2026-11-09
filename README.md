# lcao-grid-kernel

Fused spherical-harmonic evaluation and scatter-reduce onto a real-space
grid: the inner loop of LCAO-style grid integration, built as a
standalone microkernel with synthetic orbitals and its own harness. A
grid-integration kernel with a DFT-shaped workload, not a DFT code: no
SCF, no Hamiltonian, no pseudopotentials, no fitted basis sets.

## Vendor gap

The operation is a compute-heavy scatter-reduce with geometric
constraints, not a GEMM. cuBLAS has no mapping for it.

## Status

- CTest 3/3: harmonics gate, CPU-reference gate, four-mode GPU gate.
- Leader at every measured size: the tiled-shell path (per-shell
  bounding boxes, tile side 8), 1.03-1.16x over fused at 8 atoms and
  1.58-1.83x at 64 atoms, 1.91x peak vs unfused.
- The headline table is a governor-pinned idle-machine rerun (triad
  348.2 GB/s); it confirms the earlier unpinned sweep to the second
  decimal on every ratio.

## Headline: four GPU paths, 64 atoms / 1024 orbitals

Device milliseconds per full density pass, median [min,max], in-cutoff
7.6% of pairs; ratios against the fused median:

| grid | unfused ms | fused ms | tiled ms | tiled-shell ms | t/f | ts/f | u/ts |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 32 | 14.210 [14.192, 14.227] | 13.366 [13.359, 13.368] | 9.258 [9.003, 9.428] | 8.484 [8.396, 8.629] | 1.44x | 1.58x | 1.67x |
| 48 | 44.010 [43.994, 44.030] | 42.138 [42.130, 42.143] | 24.368 [24.245, 24.525] | 22.994 [22.907, 23.095] | 1.73x | 1.83x | 1.91x |
| 64 | 89.028 [89.017, 89.060] | 84.018 [84.012, 84.023] | 53.485 [53.414, 53.616] | 50.824 [50.754, 50.849] | 1.57x | 1.65x | 1.75x |
| 96 | 277.233 [277.206, 277.267] | 258.332 [258.307, 258.376] | 161.522 [161.462, 161.595] | 153.253 [153.205, 153.277] | 1.60x | 1.69x | 1.81x |
| 128 | 632.815 [632.740, 632.911] | 586.316 [586.299, 586.357] | 360.192 [360.156, 360.218] | 344.016 [343.990, 344.052] | 1.63x | 1.70x | 1.84x |

## Key numbers

- Harmonics gate: closed forms through l = 2 at 4096 seeded directions,
  max error 4.4e-16; orthonormality of all 49 harmonics with l <= 6
  under quadrature: max |diag - 1| = 5.8e-15, max |offdiag| = 3.6e-15.
- GPU vs CPU oracle across all four modes: linf <= 1.4e-15 against a
  1e-11 tolerance, four decades above the measured worst case.
- Fused throughput tops out at 0.43 Geval/s (8 atoms, fine grids) and
  sits at 0.20 Geval/s at 64 atoms.
- Visiting only bounding-box pairs (19.5-21.1% at 64 atoms) wins
  1.44-1.73x over fused; per-shell boxes cut visited pairs to
  13.0-14.2%.

## Honest limits

- Fusion alone buys little: the unfused path moves 14-44 GB/s effective
  against a 269 GB/s triad, so it is not bandwidth-bound; fused wins
  only 1.0-1.1x at 64 atoms and is a wash at 8 atoms.
- The win tracks visit-count reduction, not privatization: FP64 atomics
  were never measurably hot at these overlap densities.
- One losing cell remains: atom-box tiled at 32^3/8 atoms, 0.93x with
  disjoint spreads; the shell variant covers it.
- Tables report medians and observed spreads of one pinned sweep, not a
  stable speedup distribution. Synthetic cell; nothing is fitted to any
  element.

## Build and test

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/bench_grid          # writes data/lcao_grid_bench.json
```

CMake selects g++-15 for host compilation because nvcc rejects the
installed g++-16 headers (same pattern as
[kpm-tightbinding](https://github.com/FireHorseAutumn/kpm-tightbinding)).

## Repository layout

- `include/lcao/harmonics.hpp`: `ylm_real`, host/device shared,
  validated through l = 6.
- `include/lcao/model.hpp`, `src/model.cpp`: orbital/grid types, the
  seeded synthetic generator, the pair counters.
- `include/lcao/cpu_ref.hpp`, `src/cpu_ref.cpp`: the naive orbital-major
  oracle and a box-free point-major variant.
- `src/grid_gpu.cu`: the unfused, fused, tiled, and tiled-shell paths.
- `tests/`, `benchmarks/bench_grid.cpp`: the CTest gates and the timing
  sweep (JSON with raw per-rep timings and machine metadata).

## Deeper reading

- `docs/RUNGS.md`: the full record: workload definition, gate detail,
  every sweep table, the tile-side A/B, caveats.
- Related:
  [kpm-tightbinding](https://github.com/FireHorseAutumn/kpm-tightbinding),
  [pauli-hamiltonian](https://github.com/FireHorseAutumn/pauli-hamiltonian).
