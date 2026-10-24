#pragma once

#include "lcao/model.hpp"

#include <vector>

namespace lcao {

// The oracle. Orbital-major loop, cutoff bounding box per orbital, strictly
// sequential accumulation; simple on purpose. This is the reference every
// other path is validated against.
std::vector<double> density_cpu_naive(const Model& m);

// Point-major loop over the full grid with an inner loop over all orbitals
// (no bounding box). Per grid point the accumulation visits orbitals in the
// same order as the naive path, so agreement is bitwise; what this catches
// is bounding-box and indexing bugs in the naive path.
std::vector<double> density_cpu_gather(const Model& m);

} // namespace lcao
