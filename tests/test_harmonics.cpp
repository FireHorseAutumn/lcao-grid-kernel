// Gate for everything downstream: the Y_lm evaluation must be orthonormal
// under numerical quadrature and must match hand-derived closed forms
// pointwise before any fusion or timing work exists.
//
// Two independent checks:
//  1. Closed forms through l = 2 (s, p, d), derived by hand from the
//     recurrence stated in harmonics.hpp, at seeded random directions and at
//     the coordinate axes (the sin(theta) = 0 guard).
//  2. Orthonormality of the full l <= 6 set: |int Y_a Y_b dOm - delta_ab|
//     on a Gauss-Legendre(theta) x uniform(phi) grid. Products have degree
//     <= 12 in cos(theta) and azimuthal order <= 12, so 32 x 64 points
//     integrate them exactly; the residual is pure evaluation error.

#include "lcao/harmonics.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* name, const std::string& detail) {
  std::printf("[%s] %s: %s\n", ok ? "pass" : "FAIL", name, detail.c_str());
  if (!ok)
    ++failures;
}

// n-point Gauss-Legendre nodes and weights on [-1, 1] by Newton iteration
// on P_n with the usual three-term recurrence for P and its derivative.
void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
  x.resize(n);
  w.resize(n);
  const double pi = 3.14159265358979323846;
  for (int i = 0; i < n; ++i) {
    double z = std::cos(pi * (i + 0.75) / (n + 0.5));
    double p1 = 0, p2 = 0;
    for (int it = 0; it < 100; ++it) {
      p1 = 1.0;
      p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }
      const double dp = n * (z * p1 - p2) / (z * z - 1.0);
      const double dz = p1 / dp;
      z -= dz;
      if (std::abs(dz) < 1e-15)
        break;
    }
    const double dp = n * (z * p1 - p2) / (z * z - 1.0);
    x[i] = z;
    w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
  }
}

// Hand-derived closed forms, conventions of harmonics.hpp (Condon-Shortley
// inside A, so p_x = -Y_11 etc.). Index = l*l + l + m over l <= 2.
double closed_form(int l, int m, double nx, double ny, double nz) {
  const double y00 = 0.28209479177387814347;      // 1/sqrt(4 pi)
  const double p1 = 0.48860251190291992159;       // sqrt(3/(4 pi))
  const double d0 = 0.31539156525252005448;       // sqrt(5/(16 pi))
  const double d1 = 1.09254843059207907054;       // sqrt(15/(4 pi))
  const double d2 = 0.54627421529603953527;       // sqrt(15/(16 pi))
  if (l == 0)
    return y00;
  if (l == 1) {
    if (m == 0)
      return p1 * nz;
    return m > 0 ? -p1 * nx : -p1 * ny;
  }
  if (l == 2) {
    switch (m) {
    case 0:
      return d0 * (3.0 * nz * nz - 1.0);
    case 1:
      return -d1 * nx * nz;
    case -1:
      return -d1 * ny * nz;
    case 2:
      return d2 * (nx * nx - ny * ny);
    default:
      return d1 * nx * ny; // m = -2: sin(2 phi) = 2 sin cos
    }
  }
  return 0.0;
}

} // namespace

int main() {
  char buf[256];

  // ---- pointwise closed forms at random directions, l <= 2 ----
  std::mt19937_64 rng(0x5eed1u);
  std::normal_distribution<double> gauss(0.0, 1.0);
  double worst_point = 0.0;
  for (int t = 0; t < 4096; ++t) {
    const double vx = gauss(rng), vy = gauss(rng), vz = gauss(rng);
    const double r = std::sqrt(vx * vx + vy * vy + vz * vz);
    const double nx = vx / r, ny = vy / r, nz = vz / r;
    for (int l = 0; l <= 2; ++l)
      for (int m = -l; m <= l; ++m) {
        const double got = lcao::ylm_real(l, m, nx, ny, nz);
        const double want = closed_form(l, m, nx, ny, nz);
        worst_point = std::max(worst_point, std::abs(got - want));
      }
  }
  std::snprintf(buf, sizeof buf, "max |Y - closed form| = %.3e over l <= 2, "
                "4096 random directions", worst_point);
  check(worst_point < 1e-14, "closed forms at random directions", buf);

  // ---- axes: sin(theta) = 0 guard must give exact zeros for m != 0 ----
  double worst_axis = 0.0;
  const double axes[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                             {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  bool any_nan = false;
  double worst_axis_zero = 0.0;
  for (const auto& a : axes)
    for (int l = 0; l <= lcao::kMaxL; ++l)
      for (int m = -l; m <= l; ++m) {
        const double got = lcao::ylm_real(l, m, a[0], a[1], a[2]);
        if (!std::isfinite(got)) {
          any_nan = true;
          continue;
        }
        if (l <= 2)
          worst_axis =
              std::max(worst_axis, std::abs(got - closed_form(l, m, a[0], a[1], a[2])));
        else if (m != 0 && a[0] == 0.0 && a[1] == 0.0)
          // z axis: sin(theta) = 0 kills every m != 0 term exactly
          worst_axis_zero = std::max(worst_axis_zero, std::abs(got));
      }
  if (any_nan)
    worst_axis = 1.0;
  std::snprintf(buf, sizeof buf,
                "max |Y - closed form| on axes = %.3e, max |Y(l>2, m!=0)| on "
                "z axis = %.3e (exact 0), no NaN anywhere",
                worst_axis, worst_axis_zero);
  check(worst_axis < 1e-15 && worst_axis_zero == 0.0 && !any_nan,
        "coordinate-axis behavior", buf);

  // ---- quadrature orthonormality, l, l' <= kMaxL ----
  const int n_th = 32, n_ph = 64;
  std::vector<double> gx, gw;
  gauss_legendre(n_th, gx, gw);
  const double pi = 3.14159265358979323846;
  const double dphi = 2.0 * pi / n_ph;

  const int nf = (lcao::kMaxL + 1) * (lcao::kMaxL + 1); // sum_{l} (2l+1) = 49
  std::vector<double> gram(static_cast<size_t>(nf) * nf, 0.0);
  std::vector<double> vals(nf);
  for (int i = 0; i < n_th; ++i) {
    const double x = gx[i];
    const double s = std::sqrt(1.0 - x * x);
    for (int j = 0; j < n_ph; ++j) {
      const double phi = j * dphi;
      const double nx = s * std::cos(phi), ny = s * std::sin(phi), nz = x;
      int f = 0;
      for (int l = 0; l <= lcao::kMaxL; ++l)
        for (int m = -l; m <= l; ++m, ++f)
          vals[f] = lcao::ylm_real(l, m, nx, ny, nz);
      const double w = gw[i] * dphi;
      for (int a = 0; a < nf; ++a)
        for (int b = 0; b <= a; ++b)
          gram[static_cast<size_t>(a) * nf + b] += w * vals[a] * vals[b];
    }
  }
  double worst_off = 0.0, worst_diag = 0.0;
  for (int a = 0; a < nf; ++a)
    for (int b = 0; b <= a; ++b) {
      const double v = gram[static_cast<size_t>(a) * nf + b];
      if (a == b)
        worst_diag = std::max(worst_diag, std::abs(v - 1.0));
      else
        worst_off = std::max(worst_off, std::abs(v));
    }
  std::snprintf(buf, sizeof buf,
                "max |diag - 1| = %.3e, max |offdiag| = %.3e (%d harmonics, "
                "%d x %d quadrature)",
                worst_diag, worst_off, nf, n_th, n_ph);
  check(worst_diag < 1e-12 && worst_off < 1e-12,
        "orthonormality under Gauss-Legendre x uniform quadrature", buf);

  if (failures) {
    std::printf("%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("all harmonics checks passed\n");
  return 0;
}
