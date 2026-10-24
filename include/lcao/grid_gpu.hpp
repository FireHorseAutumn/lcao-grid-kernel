#pragma once

#include "lcao/model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lcao {

// Which GPU density path to run.
enum class GpuMode {
  Unfused, // two passes over a 32-orbital slab: evaluate all chi into a
           // global-memory V slab, then reduce w*chi^2 into the density.
           // V round-trips through global memory for every chunk.
  Fused,   // one scatter pass: one thread per (orbital, grid point), in-cutoff
           // pairs evaluate chi and atomicAdd w*chi^2 straight into the
           // density; V never exists in memory.
};

struct GpuDensityResult {
  std::vector<double> rho;
  double seconds = 0.0; // device time of the kernel pass(es); upload excluded
};

GpuDensityResult density_gpu(const Model& m, GpuMode mode);

// Timing statistics over repeated full passes; orbitals are uploaded once,
// one warmup run is excluded.
struct GpuBenchResult {
  double median_s = 0.0;
  double min_s = 0.0;
  double max_s = 0.0;
  int n_reps = 0;
  std::vector<double> reps_s; // per-rep device seconds, run order
};

GpuBenchResult bench_density_gpu(const Model& m, GpuMode mode, int n_reps);

// STREAM-like triad result: c = a + s*b in FP64 with arrays far larger
// than L2 (512 MiB each). Roofline reference; the spec sheet number is
// never used.
struct TriadBenchResult {
  double median_gbps = 0.0;
  double min_gbps = 0.0;
  double max_gbps = 0.0;
  int n_reps = 0;
  std::vector<double> reps_gbps;
};

TriadBenchResult bench_triad_fp64(int n_reps);

// Algorithmic global-memory traffic of the unfused path's value slab:
// write plus read of n_orb * n_grid FP64 values. Orbital-parameter reads
// and the density itself are negligible next to this at workload sizes.
double unfused_value_bytes(int n_orb, int n_grid);

// Device properties plus driver/runtime versions, for benchmark metadata.
// Clock rates are the device-reported maxima, not live measurements.
struct GpuDeviceInfo {
  std::string name;
  int major = 0;
  int minor = 0;
  int sm_count = 0;
  int clock_rate_khz = 0;
  int memory_clock_rate_khz = 0;
  int memory_bus_width_bits = 0;
  int l2_bytes = 0;
  std::uint64_t global_mem_bytes = 0;
  int driver_version = 0;
  int runtime_version = 0;
};

GpuDeviceInfo gpu_device_info();

std::string gpu_device_name();

// Build toolchain of the GPU library: nvcc version, host compiler,
// C++ standard, build type.
std::string gpu_toolchain();

} // namespace lcao
