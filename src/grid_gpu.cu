// Fused and unfused GPU density kernels, FP64. Hand-written CUDA; no
// vendor library kernels anywhere in the measured path.
//
// Unfused (two passes over a kChunk-orbital slab):
//   pass 1: V[c][g] = chi_{base+c}(r_g) for every grid point (0 outside the
//           cutoff) — writes chunk * n_grid doubles
//   pass 2: rho[g] += sum_c w_c V[c][g]^2 — reads them back
//   total slab traffic per chunk = 16 * chunk * n_grid bytes.
//
// Fused (single scatter pass): one thread per (orbital, grid point) pair;
// out-of-cutoff pairs exit after the r^2 test, in-cutoff pairs evaluate chi
// and atomicAdd(w*chi^2) into rho[g]. The value array never exists. The
// accumulation order is whatever the atomic unit serializes, so fused
// output differs from any sequential order at FP64 rounding level.
//
// Both kernels use the 2D launch (x over grid points, y over orbitals or
// chunk slots) so neither needs a flat pair index; n_orb stays under the
// 65535 y-limit at every benchmark size.

#include "lcao/grid_gpu.hpp"
#include "lcao/model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#define LCG_CUDA_CHECK(call)                                                    \
  do {                                                                          \
    const cudaError_t lcg_err_ = (call);                                        \
    if (lcg_err_ != cudaSuccess)                                                \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + " in " #call ": " +   \
                               cudaGetErrorString(lcg_err_));                   \
  } while (0)

namespace {

constexpr int kBlock = 256;
constexpr int kChunk = 32; // orbitals per V slab in the unfused path

__global__ void eval_values_kernel(const lcao::Orbital* __restrict__ orbs,
                                   const int n, const double h,
                                   double* __restrict__ V) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_grid = n * n * n;
  if (g >= n_grid)
    return;
  const lcao::Orbital& o = orbs[blockIdx.y];
  const int ix = g % n;
  const int iy = (g / n) % n;
  const int iz = g / (n * n);
  const double dx = (ix + 0.5) * h - o.cx;
  const double dy = (iy + 0.5) * h - o.cy;
  const double dz = (iz + 0.5) * h - o.cz;
  V[static_cast<size_t>(blockIdx.y) * n_grid + g] =
      lcao::orbital_value(o, dx, dy, dz);
}

__global__ void reduce_values_kernel(const lcao::Orbital* __restrict__ orbs,
                                     const int orb_base, const int chunk,
                                     const int n_grid,
                                     const double* __restrict__ V,
                                     double* __restrict__ rho) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_grid)
    return;
  double acc = 0.0;
  for (int c = 0; c < chunk; ++c) {
    const double chi = V[static_cast<size_t>(c) * n_grid + g];
    acc += orbs[orb_base + c].weight * chi * chi;
  }
  rho[g] += acc;
}

__global__ void fused_scatter_kernel(const lcao::Orbital* __restrict__ orbs,
                                     const int n, const double h,
                                     double* __restrict__ rho) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_grid = n * n * n;
  if (g >= n_grid)
    return;
  const lcao::Orbital& o = orbs[blockIdx.y];
  const int ix = g % n;
  const int iy = (g / n) % n;
  const int iz = g / (n * n);
  const double dx = (ix + 0.5) * h - o.cx;
  const double dy = (iy + 0.5) * h - o.cy;
  const double dz = (iz + 0.5) * h - o.cz;
  const double r2 = dx * dx + dy * dy + dz * dz;
  if (r2 > o.r_cut * o.r_cut)
    return;
  const double chi = lcao::orbital_value(o, dx, dy, dz);
  atomicAdd(&rho[g], o.weight * chi * chi);
}

__global__ void triad_kernel(const size_t n, const double* __restrict__ a,
                             const double* __restrict__ b,
                             double* __restrict__ c, const double s) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + s * b[i];
}

// Owns all device state for one model; the density pass can then be re-run
// (warmup + timed reps) without further host<->device traffic.
class DensityRunner {
public:
  explicit DensityRunner(const lcao::Model& m)
      : n_(m.grid.n), n_grid_(m.grid.n * m.grid.n * m.grid.n),
        n_orb_(static_cast<int>(m.orbitals.size())),
        h_(m.grid.cell / m.grid.n) {
    LCG_CUDA_CHECK(cudaMalloc(&orbs_, sizeof(lcao::Orbital) * n_orb_));
    LCG_CUDA_CHECK(cudaMalloc(&rho_, sizeof(double) * n_grid_));
    LCG_CUDA_CHECK(
        cudaMalloc(&V_, sizeof(double) * static_cast<size_t>(kChunk) * n_grid_));
    LCG_CUDA_CHECK(cudaMemcpy(orbs_, m.orbitals.data(),
                              sizeof(lcao::Orbital) * n_orb_,
                              cudaMemcpyHostToDevice));
    LCG_CUDA_CHECK(cudaEventCreate(&ev0_));
    LCG_CUDA_CHECK(cudaEventCreate(&ev1_));
  }

  ~DensityRunner() {
    cudaFree(orbs_);
    cudaFree(rho_);
    cudaFree(V_);
    cudaEventDestroy(ev0_);
    cudaEventDestroy(ev1_);
  }

  DensityRunner(const DensityRunner&) = delete;
  DensityRunner& operator=(const DensityRunner&) = delete;

  lcao::GpuDensityResult run(lcao::GpuMode mode, bool want_rho) {
    const int bx = (n_grid_ + kBlock - 1) / kBlock;
    LCG_CUDA_CHECK(cudaMemset(rho_, 0, sizeof(double) * n_grid_));
    LCG_CUDA_CHECK(cudaEventRecord(ev0_));
    if (mode == lcao::GpuMode::Fused) {
      fused_scatter_kernel<<<dim3(bx, n_orb_), kBlock>>>(orbs_, n_, h_, rho_);
      LCG_CUDA_CHECK(cudaGetLastError());
    } else {
      for (int base = 0; base < n_orb_; base += kChunk) {
        const int chunk = std::min(kChunk, n_orb_ - base);
        eval_values_kernel<<<dim3(bx, chunk), kBlock>>>(orbs_ + base, n_, h_,
                                                        V_);
        LCG_CUDA_CHECK(cudaGetLastError());
        reduce_values_kernel<<<bx, kBlock>>>(orbs_, base, chunk, n_grid_, V_,
                                             rho_);
        LCG_CUDA_CHECK(cudaGetLastError());
      }
    }
    LCG_CUDA_CHECK(cudaEventRecord(ev1_));
    LCG_CUDA_CHECK(cudaEventSynchronize(ev1_));
    float ms = 0.0f;
    LCG_CUDA_CHECK(cudaEventElapsedTime(&ms, ev0_, ev1_));

    lcao::GpuDensityResult res;
    res.seconds = static_cast<double>(ms) * 1e-3;
    if (want_rho) {
      res.rho.resize(n_grid_);
      LCG_CUDA_CHECK(cudaMemcpy(res.rho.data(), rho_, sizeof(double) * n_grid_,
                                cudaMemcpyDeviceToHost));
    }
    return res;
  }

private:
  int n_, n_grid_, n_orb_;
  double h_;
  lcao::Orbital* orbs_ = nullptr;
  double* rho_ = nullptr;
  double* V_ = nullptr;
  cudaEvent_t ev0_{}, ev1_{};
};

} // namespace

namespace lcao {

GpuDensityResult density_gpu(const Model& m, GpuMode mode) {
  DensityRunner runner(m);
  return runner.run(mode, true);
}

GpuBenchResult bench_density_gpu(const Model& m, GpuMode mode, int n_reps) {
  DensityRunner runner(m);
  runner.run(mode, false); // warmup, untimed
  std::vector<double> times;
  times.reserve(n_reps);
  for (int rep = 0; rep < n_reps; ++rep)
    times.push_back(runner.run(mode, false).seconds);
  GpuBenchResult res;
  res.reps_s = times;
  std::sort(times.begin(), times.end());
  res.median_s = times[times.size() / 2];
  res.min_s = times.front();
  res.max_s = times.back();
  res.n_reps = n_reps;
  return res;
}

TriadBenchResult bench_triad_fp64(int n_reps) {
  const size_t n = size_t{64} << 20; // 64M doubles = 512 MiB per array
  double *a = nullptr, *b = nullptr, *c = nullptr;
  LCG_CUDA_CHECK(cudaMalloc(&a, sizeof(double) * n));
  LCG_CUDA_CHECK(cudaMalloc(&b, sizeof(double) * n));
  LCG_CUDA_CHECK(cudaMalloc(&c, sizeof(double) * n));
  LCG_CUDA_CHECK(cudaMemset(a, 0, sizeof(double) * n));
  LCG_CUDA_CHECK(cudaMemset(b, 0, sizeof(double) * n));
  const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  cudaEvent_t ev0, ev1;
  LCG_CUDA_CHECK(cudaEventCreate(&ev0));
  LCG_CUDA_CHECK(cudaEventCreate(&ev1));

  TriadBenchResult res;
  res.n_reps = n_reps;
  res.reps_gbps.reserve(n_reps);
  const double bytes = 3.0 * sizeof(double) * n;
  for (int rep = 0; rep < n_reps + 1; ++rep) { // first rep is warmup
    LCG_CUDA_CHECK(cudaEventRecord(ev0));
    triad_kernel<<<grid, kBlock>>>(n, a, b, c, 3.0);
    LCG_CUDA_CHECK(cudaGetLastError());
    LCG_CUDA_CHECK(cudaEventRecord(ev1));
    LCG_CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    LCG_CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    if (rep > 0)
      res.reps_gbps.push_back(bytes / (static_cast<double>(ms) * 1e-3) / 1e9);
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  std::vector<double> sorted = res.reps_gbps;
  std::sort(sorted.begin(), sorted.end());
  res.min_gbps = sorted.front();
  res.max_gbps = sorted.back();
  res.median_gbps = sorted[sorted.size() / 2];
  return res;
}

double unfused_value_bytes(int n_orb, int n_grid) {
  return 16.0 * static_cast<double>(n_orb) * n_grid;
}

GpuDeviceInfo gpu_device_info() {
  int dev = 0;
  LCG_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  LCG_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  GpuDeviceInfo info;
  info.name = prop.name;
  info.major = prop.major;
  info.minor = prop.minor;
  info.sm_count = prop.multiProcessorCount;
  LCG_CUDA_CHECK(cudaDeviceGetAttribute(&info.clock_rate_khz,
                                        cudaDevAttrClockRate, dev));
  LCG_CUDA_CHECK(cudaDeviceGetAttribute(&info.memory_clock_rate_khz,
                                        cudaDevAttrMemoryClockRate, dev));
  LCG_CUDA_CHECK(cudaDeviceGetAttribute(&info.memory_bus_width_bits,
                                        cudaDevAttrGlobalMemoryBusWidth, dev));
  info.l2_bytes = prop.l2CacheSize;
  info.global_mem_bytes = prop.totalGlobalMem;
  LCG_CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
  LCG_CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
  return info;
}

std::string gpu_device_name() {
  const GpuDeviceInfo info = gpu_device_info();
  return info.name + " (sm_" + std::to_string(info.major) +
         std::to_string(info.minor) + ", " + std::to_string(info.sm_count) +
         " SMs, L2 " + std::to_string(info.l2_bytes / (1024 * 1024)) +
         " MiB)";
}

std::string gpu_toolchain() {
  std::string s = "nvcc " + std::to_string(__CUDACC_VER_MAJOR__) + "." +
                  std::to_string(__CUDACC_VER_MINOR__) + "." +
                  std::to_string(__CUDACC_VER_BUILD__);
#if defined(__clang__)
  s += ", host clang " __VERSION__;
#elif defined(__GNUC__)
  s += ", host gcc " + std::to_string(__GNUC__) + "." +
       std::to_string(__GNUC_MINOR__) + "." +
       std::to_string(__GNUC_PATCHLEVEL__);
#endif
  s += ", c++" + std::to_string(__cplusplus / 100 % 100);
#ifdef LCG_BUILD_TYPE
  s += ", " LCG_BUILD_TYPE;
#endif
  return s;
}

} // namespace lcao
