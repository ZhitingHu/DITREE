#ifndef DITREE_UTIL_HPP_
#define DITREE_UTIL_HPP_

#include "common.hpp"

#include <cmath>
#include <algorithm>

namespace ditree {


class Triple {
 public:
  Triple(uint32 x, uint32 y, float w): x_(x), y_(y), w_(w) { }

  inline uint32 x() { return x_; }
  inline uint32 y() { return y_; }
  inline float w() { return w_; }
 private:
  uint32 x_;
  uint32 y_;
  float w_;
}

inline void CopyUIntFloatMap(const UIntFloatMap& source, 
    const float coeff, UIntFloatMap& target);
inline void AccumUIntFloatMap(const UIntFloatMap& source, 
    const float coeff, UIntFloatMap& target);
// Z = aX + bY
inline void AddFloatVectors(const float a, const FloatVec& X, 
    const float b, const FloatVec& Y, FloatVec& Z);

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
