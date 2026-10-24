#pragma once

// Real orthonormal spherical harmonics Y_lm (l >= 0, -l <= m <= l) from a
// unit direction, Condon-Shortley phase inside the Legendre part:
//
//   m = 0:  Y = A_l^0(x)
//   m > 0:  Y = sqrt(2) A_l^m(x) cos(m phi)
//   m < 0:  Y = sqrt(2) A_l^|m|(x) sin(|m| phi)
//
// with x = cos(theta) and A the fully normalized associated Legendre
// function, A_l^m(x) = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!) P_l^m(x), built by
// factorial-free recurrences (no gamma calls, no overflow at any l):
//
//   A_0^0 = 1/sqrt(4 pi)
//   A_m^m = -sqrt((2m+1)/(2m)) sqrt(1-x^2) A_{m-1}^{m-1}
//   A_l^m = sqrt((4l^2-1)/(l^2-m^2))
//           ( x A_{l-1}^m - sqrt(((l-1)^2-m^2)/(4(l-1)^2-1)) A_{l-2}^m )
//
// The l = m+1 case of the last line has a vanishing A_{l-2} coefficient, so
// one recurrence covers the whole column. cos(m phi) and sin(m phi) come
// from the 2D rotation recurrence off (nx/s, ny/s), s = sin(theta); no trig
// calls. On the z axis s = 0 kills every m != 0 term through the sqrt(1-x^2)
// factor in A_m^m, so the azimuth there is set to (1, 0) and never observed.
//
// Accuracy: validated through l = 6 by Gauss-Legendre(theta) x uniform(phi)
// quadrature orthonormality and by closed-form pointwise checks in
// tests/test_harmonics.cpp.

#include <cmath>

#ifdef __CUDACC__
#define LCAO_HD __host__ __device__
#else
#define LCAO_HD
#endif

namespace lcao {

constexpr int kMaxL = 6; // tests validate l, l' <= kMaxL

template <typename Real>
LCAO_HD Real ylm_real(int l, int m, Real nx, Real ny, Real nz) {
  const Real x = nz; // cos(theta)
  Real s2 = Real(1) - x * x;
  if (s2 < Real(0))
    s2 = Real(0);
  const Real s = std::sqrt(s2); // sin(theta)
  Real cph = 1, sph = 0;
  if (s > Real(0)) {
    cph = nx / s;
    sph = ny / s;
  }
  const int am = m < 0 ? -m : m;

  // A_am^am by the diagonal recurrence.
  Real a_diag = Real(0.28209479177387814347); // 1/sqrt(4 pi)
  for (int k = 1; k <= am; ++k)
    a_diag = -std::sqrt(Real(2 * k + 1) / Real(2 * k)) * s * a_diag;

  // Up the am column to l.
  Real a = a_diag;
  if (l > am) {
    Real a_prev2 = 0, a_prev1 = a_diag, a_cur = 0;
    for (int k = am + 1; k <= l; ++k) {
      const Real c1 = std::sqrt(Real(4 * k * k - 1) / Real(k * k - am * am));
      const Real c2 =
          std::sqrt(Real((k - 1) * (k - 1) - am * am) /
                    Real(4 * (k - 1) * (k - 1) - 1));
      a_cur = c1 * (x * a_prev1 - c2 * a_prev2);
      a_prev2 = a_prev1;
      a_prev1 = a_cur;
    }
    a = a_cur;
  }

  if (m == 0)
    return a;
  Real cm = 1, sm = 0;
  for (int k = 1; k <= am; ++k) {
    const Real cm_next = cm * cph - sm * sph;
    sm = sm * cph + cm * sph;
    cm = cm_next;
  }
  const Real r2 = 1.4142135623730950488; // sqrt(2)
  return m > 0 ? r2 * a * cm : r2 * a * sm;
}

} // namespace lcao
