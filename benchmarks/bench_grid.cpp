// Fused vs unfused density benchmark: grid-size and orbital-count sweeps,
// FP64, median of N reps with min/max spread, device-side timing, effective
// GB/s for the unfused slab traffic against a measured STREAM-like triad on
// the same GPU. JSON out to data/ with raw per-rep timings and
// device/driver/seed/toolchain metadata.
//
// Absolute times vary with machine state; report medians and observed
// spreads.

#include "lcao/grid_gpu.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ConfigEntry {
  int grid_n;
  int atoms;
  int n_orb;
  int n_grid;
  std::uint64_t pairs_total;
  std::uint64_t pairs_evaluated;
  lcao::GpuBenchResult unfused, fused;
};

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    const size_t comma = s.find(',', pos);
    out.push_back(std::stoi(s.substr(pos, comma - pos)));
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return out;
}

std::string timestamp_utc() {
  const std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

std::string hex_seed(std::uint64_t seed) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "0x%llx",
                static_cast<unsigned long long>(seed));
  return buf;
}

void json_result(std::ofstream& f, const lcao::GpuBenchResult& r, int n_grid,
                 std::uint64_t pairs_evaluated, double value_bytes,
                 int indent) {
  const std::string pad(indent, ' ');
  f << pad << "{\n"
    << pad << "  \"median_s\": " << r.median_s << ",\n"
    << pad << "  \"min_s\": " << r.min_s << ",\n"
    << pad << "  \"max_s\": " << r.max_s << ",\n"
    << pad << "  \"grid_points_per_s\": " << n_grid / r.median_s << ",\n"
    << pad << "  \"evals_per_s\": "
    << static_cast<double>(pairs_evaluated) / r.median_s << ",\n";
  if (value_bytes > 0.0)
    f << pad << "  \"value_slab_bytes\": " << value_bytes << ",\n"
      << pad << "  \"effective_gbps\": " << value_bytes / r.median_s / 1e9
      << ",\n";
  f << pad << "  \"reps_s\": [";
  for (size_t i = 0; i < r.reps_s.size(); ++i)
    f << (i ? ", " : "") << r.reps_s[i];
  f << "]\n"
    << pad << "}";
}

} // namespace

int main(int argc, char** argv) {
  std::vector<int> grids = {32, 48, 64, 96, 128};
  std::vector<int> atoms_per_side = {2, 4};
  std::uint64_t seed = 0x5eed3u;
  int n_reps = 11;
  int triad_reps = 20;
  std::string out_path = std::string(LCG_DATA_DIR) + "/lcao_grid_bench.json";
  for (int i = 1; i < argc; ++i) {
    try {
      if (std::strncmp(argv[i], "--grids=", 8) == 0)
        grids = parse_list(argv[i] + 8);
      else if (std::strncmp(argv[i], "--atoms-per-side=", 17) == 0)
        atoms_per_side = parse_list(argv[i] + 17);
      else if (std::strncmp(argv[i], "--seed=", 7) == 0)
        seed = std::stoull(argv[i] + 7, nullptr, 0);
      else if (std::strncmp(argv[i], "--reps=", 7) == 0)
        n_reps = std::stoi(argv[i] + 7);
      else if (std::strncmp(argv[i], "--triad-reps=", 13) == 0)
        triad_reps = std::stoi(argv[i] + 13);
      else if (std::strncmp(argv[i], "--out=", 6) == 0)
        out_path = argv[i] + 6;
      else
        throw std::invalid_argument(std::string("unknown argument: ") +
                                    argv[i]);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "bench_grid: %s\n", e.what());
      std::fprintf(stderr,
                   "usage: bench_grid [--grids=32,48,64,96,128] "
                   "[--atoms-per-side=2,4] [--seed=0x5eed3] [--reps=11] "
                   "[--triad-reps=20] [--out=PATH]\n");
      return 1;
    }
  }

  try {
    const lcao::GpuDeviceInfo devinfo = lcao::gpu_device_info();
    const std::string device = lcao::gpu_device_name();
    std::printf("device: %s\n", device.c_str());

    const lcao::TriadBenchResult triad = lcao::bench_triad_fp64(triad_reps);
    std::printf("STREAM-like triad FP64 (3 x 512 MiB arrays): median %.1f "
                "GB/s (min %.1f, max %.1f, %d reps)\n\n",
                triad.median_gbps, triad.min_gbps, triad.max_gbps,
                triad.n_reps);

    std::printf("median of %d reps (device time), one warmup excluded\n\n",
                n_reps);
    std::printf("%5s %6s %7s | %12s %12s | %10s %10s | %8s %8s\n", "grid",
                "atoms", "orbs", "unfus_ms", "fused_ms", "Mpts/s_u",
                "Mpts/s_f", "Ge/s_u", "Ge/s_f");

    std::vector<ConfigEntry> entries;
    for (const int a : atoms_per_side)
      for (const int g : grids) {
        const lcao::Model model = lcao::make_synthetic_model(a, g, seed);
        ConfigEntry e{g,
                      a * a * a,
                      static_cast<int>(model.orbitals.size()),
                      lcao::grid_points(model.grid),
                      static_cast<std::uint64_t>(model.orbitals.size()) *
                          lcao::grid_points(model.grid),
                      lcao::count_pairs(model),
                      lcao::bench_density_gpu(model, lcao::GpuMode::Unfused,
                                              n_reps),
                      lcao::bench_density_gpu(model, lcao::GpuMode::Fused,
                                              n_reps)};
        std::printf(
            "%5d %6d %7d | %12.3f %12.3f | %10.1f %10.1f | %8.3f %8.3f\n", g,
            e.atoms, e.n_orb, e.unfused.median_s * 1e3, e.fused.median_s * 1e3,
            e.n_grid / e.unfused.median_s / 1e6,
            e.n_grid / e.fused.median_s / 1e6,
            static_cast<double>(e.pairs_evaluated) / e.unfused.median_s / 1e9,
            static_cast<double>(e.pairs_evaluated) / e.fused.median_s / 1e9);
        std::printf("      spread ms [min,max]: u [%.3f,%.3f] f "
                    "[%.3f,%.3f]; in-cutoff pairs %llu / %llu (%.1f%%)\n",
                    e.unfused.min_s * 1e3, e.unfused.max_s * 1e3,
                    e.fused.min_s * 1e3, e.fused.max_s * 1e3,
                    static_cast<unsigned long long>(e.pairs_evaluated),
                    static_cast<unsigned long long>(e.pairs_total),
                    100.0 * e.pairs_evaluated / e.pairs_total);
        entries.push_back(e);
      }

    std::ofstream f(out_path);
    if (!f)
      throw std::runtime_error("cannot open " + out_path);
    f.precision(17);
    f << "{\n"
      << "  \"timestamp_utc\": \"" << timestamp_utc() << "\",\n"
      << "  \"seed\": \"" << hex_seed(seed) << "\",\n"
      << "  \"device\": \"" << device << "\",\n"
      << "  \"device_info\": {\n"
      << "    \"name\": \"" << devinfo.name << "\",\n"
      << "    \"compute_capability\": \"" << devinfo.major << "."
      << devinfo.minor << "\",\n"
      << "    \"sm_count\": " << devinfo.sm_count << ",\n"
      << "    \"clock_rate_khz\": " << devinfo.clock_rate_khz << ",\n"
      << "    \"memory_clock_rate_khz\": " << devinfo.memory_clock_rate_khz
      << ",\n"
      << "    \"memory_bus_width_bits\": " << devinfo.memory_bus_width_bits
      << ",\n"
      << "    \"l2_bytes\": " << devinfo.l2_bytes << ",\n"
      << "    \"global_mem_bytes\": " << devinfo.global_mem_bytes << ",\n"
      << "    \"driver_version\": " << devinfo.driver_version << ",\n"
      << "    \"runtime_version\": " << devinfo.runtime_version << "\n"
      << "  },\n"
      << "  \"toolchain\": \"" << lcao::gpu_toolchain() << "\",\n"
      << "  \"precision\": \"fp64\",\n"
      << "  \"chunk_orbitals\": 32,\n"
      << "  \"n_reps\": " << n_reps << ",\n"
      << "  \"triad_gbps\": {\"median\": " << triad.median_gbps
      << ", \"min\": " << triad.min_gbps << ", \"max\": " << triad.max_gbps
      << ", \"reps\": " << triad.n_reps << ",\n"
      << "    \"reps_gbps\": [";
    for (size_t i = 0; i < triad.reps_gbps.size(); ++i)
      f << (i ? ", " : "") << triad.reps_gbps[i];
    f << "]},\n"
      << "  \"configs\": [\n";
    for (size_t i = 0; i < entries.size(); ++i) {
      const ConfigEntry& e = entries[i];
      f << "    {\"grid_n\": " << e.grid_n << ", \"atoms\": " << e.atoms
        << ", \"orbitals\": " << e.n_orb << ", \"grid_points\": " << e.n_grid
        << ",\n     \"pairs_total\": " << e.pairs_total
        << ", \"pairs_evaluated\": " << e.pairs_evaluated << ",\n"
        << "     \"unfused\": ";
      json_result(f, e.unfused, e.n_grid, e.pairs_evaluated,
                  lcao::unfused_value_bytes(e.n_orb, e.n_grid), 16);
      f << ",\n     \"fused\": ";
      json_result(f, e.fused, e.n_grid, e.pairs_evaluated, 0.0, 16);
      f << "}" << (i + 1 < entries.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    if (!f)
      throw std::runtime_error("write failed: " + out_path);
    std::printf("\nJSON -> %s\n", out_path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_grid failed: %s\n", e.what());
    return 1;
  }
  return 0;
}
