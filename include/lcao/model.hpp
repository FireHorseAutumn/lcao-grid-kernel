#pragma once

// Workload model: synthetic atom-centered orbitals on a uniform 3D grid.
//
//   chi(r) = R(|r - c|) * Y_lm(r - c)          (real harmonics, harmonics.hpp)
//   R(r)   = r^l (c0 exp(-a0 r^2) + c1 exp(-a1 r^2)),  r <= r_cut
//          = 0,                                         r > r_cut
//   density(g) = sum_o weight_o * chi_o(r_g)^2
//
// The grid spans [0, cell)^3 with n points per axis, h = cell/n, points at
// cell centers ((i+0.5)h, ...). Not fitted to any element; the generator
// draws parameters from fixed seeded distributions.
//
// r_cut is set from the smaller exponent so the Gaussian factor at the
// cutoff is e^-27.63 = 1e-12; including the r^l factor the dropped tail is
// at most r_cut^l * 1e-12 ~ 2e-9 in absolute chi units for l <= 3,
// a_min >= 0.2 — far below the FP64 accumulation noise being validated.

#include "lcao/harmonics.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace lcao {

struct Orbital {
  double cx, cy, cz; // center
  int l, m;
  double a0, a1;     // Gaussian exponents
  double c0, c1;     // contraction coefficients
  double weight;     // occupation-like weight in the density sum
  double r_cut;      // compact support radius
};

struct Grid {
  int n;       // points per axis
  double cell; // side length; h = cell / n
};

struct Model {
  Grid grid;
  std::vector<Orbital> orbitals;
};

inline int grid_points(const Grid& g) { return g.n * g.n * g.n; }

// Radial part at distance r (r <= r_cut on every call path that matters;
// the cutoff test lives in orbital_value).
template <typename Real>
LCAO_HD Real radial_value(const Orbital& o, Real r) {
  Real rp = 1;
  for (int k = 0; k < o.l; ++k)
    rp *= r;
  return rp * (o.c0 * std::exp(-o.a0 * r * r) + o.c1 * std::exp(-o.a1 * r * r));
}

// chi at displacement (dx, dy, dz) from the orbital center; 0 beyond r_cut.
// At r = 0 the direction is undefined; only l = 0 contributes there since
// R ~ r^l, so the value is R(0) * Y_00 for l = 0 and 0 otherwise.
template <typename Real>
LCAO_HD Real orbital_value(const Orbital& o, Real dx, Real dy, Real dz) {
  const Real r2 = dx * dx + dy * dy + dz * dz;
  if (r2 > o.r_cut * o.r_cut)
    return 0;
  if (r2 == 0)
    return o.l == 0 ? radial_value(o, Real(0)) *
                          Real(0.28209479177387814347) // Y_00
                    : Real(0);
  const Real r = std::sqrt(r2);
  return radial_value(o, r) * ylm_real(o.l, o.m, dx / r, dy / r, dz / r);
}

// atoms_per_side^3 atoms on a simple-cubic arrangement, spacing 6.0, cell
// side = 6.0 * atoms_per_side. Each atom carries s, p, d, f shells (16
// orbitals); radial parameters and the weight are drawn per shell from the
// seeded generator, shared across the 2l+1 m components.
Model make_synthetic_model(int atoms_per_side, int grid_n, std::uint64_t seed);

// In-cutoff (orbital, grid-point) pairs; the honest work count for
// throughput reporting.
std::uint64_t count_pairs(const Model& m);

} // namespace lcao
