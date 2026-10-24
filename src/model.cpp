#include "lcao/model.hpp"

#include <random>

namespace lcao {

Model make_synthetic_model(int atoms_per_side, int grid_n,
                           std::uint64_t seed) {
  const double spacing = 6.0;
  Model m;
  m.grid = Grid{grid_n, spacing * atoms_per_side};
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  for (int az = 0; az < atoms_per_side; ++az)
    for (int ay = 0; ay < atoms_per_side; ++ay)
      for (int ax = 0; ax < atoms_per_side; ++ax) {
        const double cx = (ax + 0.5) * spacing;
        const double cy = (ay + 0.5) * spacing;
        const double cz = (az + 0.5) * spacing;
        for (int l = 0; l <= 3; ++l) { // s, p, d, f shells
          const double a0 = 0.8 + 1.2 * u01(rng);
          const double a1 = (0.25 + 0.45 * u01(rng)) * a0;
          const double c0 = 0.8 + 0.7 * u01(rng);
          const double c1 = (0.1 + 0.4 * u01(rng)) * (u01(rng) < 0.5 ? -1 : 1);
          const double weight = 0.5 + 1.5 * u01(rng);
          const double amin = a0 < a1 ? a0 : a1;
          // Gaussian factor at r_cut is exp(-27.631) = 1e-12.
          const double r_cut = std::sqrt(27.631021115928547 / amin);
          for (int mm = -l; mm <= l; ++mm)
            m.orbitals.push_back(
                Orbital{cx, cy, cz, l, mm, a0, a1, c0, c1, weight, r_cut});
        }
      }
  return m;
}

std::uint64_t count_pairs(const Model& m) {
  const int n = m.grid.n;
  const double h = m.grid.cell / n;
  std::uint64_t count = 0;
  for (const Orbital& o : m.orbitals) {
    const int lo[3] = {std::max(0, static_cast<int>((o.cx - o.r_cut) / h)),
                       std::max(0, static_cast<int>((o.cy - o.r_cut) / h)),
                       std::max(0, static_cast<int>((o.cz - o.r_cut) / h))};
    const int hi[3] = {std::min(n - 1, static_cast<int>((o.cx + o.r_cut) / h)),
                       std::min(n - 1, static_cast<int>((o.cy + o.r_cut) / h)),
                       std::min(n - 1, static_cast<int>((o.cz + o.r_cut) / h))};
    for (int iz = lo[2]; iz <= hi[2]; ++iz)
      for (int iy = lo[1]; iy <= hi[1]; ++iy)
        for (int ix = lo[0]; ix <= hi[0]; ++ix) {
          const double dx = (ix + 0.5) * h - o.cx;
          const double dy = (iy + 0.5) * h - o.cy;
          const double dz = (iz + 0.5) * h - o.cz;
          if (dx * dx + dy * dy + dz * dz <= o.r_cut * o.r_cut)
            ++count;
        }
  }
  return count;
}

} // namespace lcao
