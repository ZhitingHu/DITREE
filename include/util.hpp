#ifndef DITREE_UTIL_HPP_
#define DITREE_UTIL_HPP_

#include "common.hpp"

#include <cmath>
#include <algorithm>

namespace ditree {


// VMF log probability density function
// vMF(x | mu, kappa)
inline float LogVMFProb(const UIntFloatMap& x, const FloatVec& mu, float kappa);

inline double fastLogBesselI(double d, double kappa);

// Adapted from D. Blei's lda-c code
inline double log_sum(double log_a, double log_b);
inline double trigamma(double x);
inline double digamma(double x);
inline double log_gamma(double x);
inline int argmax(double* x, int n);

} // namespace ditree

#endif
