// CPU reference checks. The naive evaluator is the oracle for the GPU path,
// so it is itself pinned against hand-derived closed forms for single
// isolated orbitals (s, p_z, p_x, d_z2) and against an independent
// point-major accumulation order on a full synthetic model.

#include "lcao/cpu_ref.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* name, const std::string& detail) {
  std::printf("[%s] %s: %s\n", ok ? "pass" : "FAIL", name, detail.c_str());
  if (!ok)
    ++failures;
}

double linf_diff(const std::vector<double>& x, const std::vector<double>& y) {
  double worst = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
    worst = std::max(worst, std::abs(x[i] - y[i]));
  return worst;
}

// One orbital at the cell center of a 12^3 cell, grid 16^3 (h = 0.75; grid
// points sit at half-integer multiples of h, the center at 8h, so r = 0
// never occurs). Radial exponents/coefs and weight fixed by hand.
lcao::Model single_orbital_model(int l, int m) {
  lcao::Model model;
  model.grid = lcao::Grid{16, 12.0};
  const double a0 = 1.0, a1 = 0.3;
  model.orbitals.push_back(lcao::Orbital{6.0, 6.0, 6.0, l, m, a0, a1, 1.1,
                                         -0.4, 1.3,
                                         std::sqrt(27.631021115928547 / a1)});
  return model;
}

// Density from the closed-form angular part, no harmonics code involved:
// chi = R(r) * f(direction) with f written out per (l, m).
std::vector<double> closed_form_density(const lcao::Model& model, int l,
                                        int m) {
  const lcao::Orbital& o = model.orbitals[0];
  const int n = model.grid.n;
  const double h = model.grid.cell / n;
  std::vector<double> rho(static_cast<size_t>(n) * n * n, 0.0);
  for (int iz = 0; iz < n; ++iz)
    for (int iy = 0; iy < n; ++iy)
      for (int ix = 0; ix < n; ++ix) {
        const double dx = (ix + 0.5) * h - o.cx;
        const double dy = (iy + 0.5) * h - o.cy;
        const double dz = (iz + 0.5) * h - o.cz;
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > o.r_cut * o.r_cut)
          continue;
        const double r = std::sqrt(r2);
        double rp = 1.0;
        for (int k = 0; k < l; ++k)
          rp *= r;
        const double radial =
            rp * (o.c0 * std::exp(-o.a0 * r2) + o.c1 * std::exp(-o.a1 * r2));
        const double ux = dx / r, uz = dz / r;
        double ang = 0.0;
        if (l == 0)
          ang = 0.28209479177387814347;                    // 1/sqrt(4 pi)
        else if (l == 1 && m == 0)
          ang = 0.48860251190291992159 * uz;               // p_z
        else if (l == 1 && m == 1)
          ang = -0.48860251190291992159 * ux;              // p_x
        else if (l == 2 && m == 0)
          ang = 0.31539156525252005448 * (3.0 * uz * uz - 1.0); // d_z2
        const double chi = radial * ang;
        rho[(static_cast<size_t>(iz) * n + iy) * n + ix] += o.weight * chi * chi;
      }
  return rho;
}

} // namespace

int main() {
  char buf[256];

  // ---- single-orbital closed forms: geometry, radial part, indexing ----
  const int cases[4][2] = {{0, 0}, {1, 0}, {1, 1}, {2, 0}};
  const char* names[4] = {"s", "p_z", "p_x", "d_z2"};
  for (int c = 0; c < 4; ++c) {
    const lcao::Model model = single_orbital_model(cases[c][0], cases[c][1]);
    const std::vector<double> got = lcao::density_cpu_naive(model);
    const std::vector<double> want =
        closed_form_density(model, cases[c][0], cases[c][1]);
    const double d = linf_diff(got, want);
    std::snprintf(buf, sizeof buf, "linf = %.3e", d);
    check(d < 1e-14, (std::string("single ") + names[c] +
                      " orbital matches closed-form density")
                         .c_str(),
          buf);
  }

  // ---- full synthetic model: naive vs box-free gather path ----
  const lcao::Model model = lcao::make_synthetic_model(2, 16, 0x5eed2u);
  const std::vector<double> naive = lcao::density_cpu_naive(model);
  const std::vector<double> gather = lcao::density_cpu_gather(model);
  double rho_max = 0.0;
  for (double v : naive)
    rho_max = std::max(rho_max, v);
  const double d = linf_diff(naive, gather);
  std::snprintf(buf, sizeof buf,
                "linf = %.3e on max density %.3e (128 orbitals, 16^3 grid; "
                "bitwise agreement expected: same per-orbital visit order, "
                "box-free gather catches bounding-box/indexing bugs)",
                d, rho_max);
  check(d == 0.0, "naive vs gather accumulation order", buf);

  // ---- count_pairs bounding box vs brute-force all-pairs count ----
  std::uint64_t brute = 0;
  const int n = model.grid.n;
  const double h = model.grid.cell / n;
  for (const lcao::Orbital& o : model.orbitals)
    for (int iz = 0; iz < n; ++iz)
      for (int iy = 0; iy < n; ++iy)
        for (int ix = 0; ix < n; ++ix) {
          const double dx = (ix + 0.5) * h - o.cx;
          const double dy = (iy + 0.5) * h - o.cy;
          const double dz = (iz + 0.5) * h - o.cz;
          if (dx * dx + dy * dy + dz * dz <= o.r_cut * o.r_cut)
            ++brute;
        }
  const std::uint64_t boxed = lcao::count_pairs(model);
  std::snprintf(buf, sizeof buf, "count_pairs = %llu, brute force = %llu",
                static_cast<unsigned long long>(boxed),
                static_cast<unsigned long long>(brute));
  check(boxed == brute, "count_pairs bounding box exactness", buf);

  if (failures) {
    std::printf("%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("all CPU density checks passed\n");
  return 0;
}
