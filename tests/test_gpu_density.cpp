// GPU density vs the CPU naive oracle, FP64. All GPU modes evaluate chi
// with the same device math; the CPU oracle uses the host libm exp and a
// sequential per-orbital accumulation order, so agreement is at rounding
// level, not bitwise. The unfused reduce visits chunk slots in a fixed
// order, so unfused vs CPU differs only through exp; fused adds global
// atomic ordering on top, and tiled adds shared-memory privatized atomic
// ordering plus one global atomic per flushed tile point.
//
// Grid coverage: 16^3 and 33^3 give multi-block launches with a partial
// last block; 64^3 with 64 atoms (1024 orbitals) is benchmark class. For
// the tiled path, 16^3/33^3 also exercise tiles overhanging the grid edge
// and bounding boxes larger than one sub-tile.

#include "lcao/cpu_ref.hpp"
#include "lcao/grid_gpu.hpp"

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

double vec_max(const std::vector<double>& x) {
  double m = 0.0;
  for (double v : x)
    m = std::max(m, v);
  return m;
}

} // namespace

int main() {
  char buf[256];
  try {
    std::printf("device: %s\n", lcao::gpu_device_name().c_str());

    struct Config {
      int atoms_per_side, grid_n;
    };
    const Config configs[3] = {{2, 16}, {2, 33}, {4, 64}};
    // Measured on this machine: all GPU modes vs CPU <= 1.4e-15 across the
    // three configs, on max densities up to 1.4; per-point overlap is a
    // handful of orbitals, so atomic ordering adds little to the exp
    // difference. Tolerance 1e-11 leaves four decades.
    const double tol = 1e-11;

    for (const Config& cfg : configs) {
      const lcao::Model model =
          lcao::make_synthetic_model(cfg.atoms_per_side, cfg.grid_n, 0x5eed3u);
      const std::vector<double> ref = lcao::density_cpu_naive(model);
      const double rho_max = vec_max(ref);

      const lcao::GpuDensityResult unfused =
          lcao::density_gpu(model, lcao::GpuMode::Unfused);
      const lcao::GpuDensityResult fused =
          lcao::density_gpu(model, lcao::GpuMode::Fused);
      const lcao::GpuDensityResult tiled =
          lcao::density_gpu(model, lcao::GpuMode::Tiled);
      const lcao::GpuDensityResult tshell =
          lcao::density_gpu(model, lcao::GpuMode::TiledShell);

      const double du = linf_diff(unfused.rho, ref);
      const double df = linf_diff(fused.rho, ref);
      const double dt = linf_diff(tiled.rho, ref);
      const double dts = linf_diff(tshell.rho, ref);
      const double dfu = linf_diff(fused.rho, unfused.rho);
      const double dtf = linf_diff(tiled.rho, fused.rho);
      const double dtst = linf_diff(tshell.rho, tiled.rho);

      std::snprintf(buf, sizeof buf,
                    "grid %d^3, %d atoms (%d orbitals): max rho = %.3e",
                    cfg.grid_n, cfg.atoms_per_side * cfg.atoms_per_side *
                                      cfg.atoms_per_side,
                    static_cast<int>(model.orbitals.size()), rho_max);
      std::printf("  %s\n", buf);

      std::snprintf(buf, sizeof buf, "linf = %.3e", du);
      check(du < tol, "unfused GPU vs CPU oracle", buf);
      std::snprintf(buf, sizeof buf, "linf = %.3e", df);
      check(df < tol, "fused GPU vs CPU oracle", buf);
      std::snprintf(buf, sizeof buf, "linf = %.3e", dt);
      check(dt < tol, "tiled GPU vs CPU oracle", buf);
      std::snprintf(buf, sizeof buf, "linf = %.3e", dts);
      check(dts < tol, "tiled-shell GPU vs CPU oracle", buf);
      std::snprintf(buf, sizeof buf, "linf = %.3e", dfu);
      check(dfu < tol, "fused vs unfused GPU", buf);
      std::snprintf(buf, sizeof buf, "linf = %.3e", dtf);
      check(dtf < tol, "tiled vs fused GPU", buf);
      std::snprintf(buf, sizeof buf, "linf = %.3e", dtst);
      check(dtst < tol, "tiled-shell vs tiled GPU", buf);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_density failed: %s\n", e.what());
    return 1;
  }

  if (failures) {
    std::printf("%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("all GPU density checks passed\n");
  return 0;
}
