#include "lcao/cpu_ref.hpp"

#include <algorithm>

namespace lcao {

std::vector<double> density_cpu_naive(const Model& m) {
  const int n = m.grid.n;
  const double h = m.grid.cell / n;
  std::vector<double> rho(static_cast<size_t>(n) * n * n, 0.0);
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
          const double chi = orbital_value(o, dx, dy, dz);
          if (chi != 0.0)
            rho[(static_cast<size_t>(iz) * n + iy) * n + ix] +=
                o.weight * chi * chi;
        }
  }
  return rho;
}

std::vector<double> density_cpu_gather(const Model& m) {
  const int n = m.grid.n;
  const double h = m.grid.cell / n;
  std::vector<double> rho(static_cast<size_t>(n) * n * n, 0.0);
  for (int iz = 0; iz < n; ++iz)
    for (int iy = 0; iy < n; ++iy)
      for (int ix = 0; ix < n; ++ix) {
        double acc = 0.0;
        for (const Orbital& o : m.orbitals) {
          const double dx = (ix + 0.5) * h - o.cx;
          const double dy = (iy + 0.5) * h - o.cy;
          const double dz = (iz + 0.5) * h - o.cz;
          const double chi = orbital_value(o, dx, dy, dz);
          acc += o.weight * chi * chi;
        }
        rho[(static_cast<size_t>(iz) * n + iy) * n + ix] = acc;
      }
  return rho;
}

} // namespace lcao
