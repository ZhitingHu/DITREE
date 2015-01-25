#ifndef DITREE_UTIL_HPP_
#define DITREE_UTIL_HPP_

#include "common.hpp"
#include <boost/foreach.hpp>

#include <cmath>
#include <algorithm>

namespace ditree {

class Triple {
 public:
  Triple(uint32 x, uint32 y, float w): x_(x), y_(y), w_(w) { }

  inline uint32 x() const { return x_; }
  inline uint32 y() const { return y_; }
  inline float w() const { return w_; }
 private:
  uint32 x_;
  uint32 y_;
  float w_;
};

inline void PrintFloatVec(const FloatVec& v) {
  ostringstream oss;
  for (const auto v_ele : v) {
    oss << v_ele << " ";
  }
  oss << "\n";
  LOG(INFO) << oss.str();
}
inline void PrintUIntFloatMap(const UIntFloatMap& v) {
  ostringstream oss;
  BOOST_FOREACH(const UIntFloatPair& v_ele, v) {
    oss << v_ele.first << ":" << v_ele.second << " ";
  }
  oss << "\n";
  LOG(INFO) << oss.str();
}

inline void ResetUIntFloatMap(UIntFloatMap& target) {
  BOOST_FOREACH(UIntFloatPair& t_ele, target) {
    t_ele.second = 0;
  }
}
inline void CopyUIntFloatMap(const UIntFloatMap& source, 
    const float coeff, UIntFloatMap& target) {
  BOOST_FOREACH(const UIntFloatPair& s_ele, source) {
#ifdef DEBUG
    CHECK(target.find(s_ele.first) != target.end());
#endif
    target[s_ele.first] = s_ele.second * coeff;
  }
}
inline void AccumUIntFloatMap(const UIntFloatMap& source, 
    const float coeff, UIntFloatMap& target) {
  BOOST_FOREACH(const UIntFloatPair& s_ele, source) {
#ifdef DEBUG
    CHECK(target.find(s_ele.first) != target.end());
#endif
    target[s_ele.first] += s_ele.second * coeff;
  }
}
inline void CopyFloatVec(const FloatVec& source, 
    const float coeff, FloatVec& target) {
#ifdef DEBUG
    CHECK_EQ(source.size(), target.size());
#endif
  for (int s_idx = 0; s_idx < source.size(); ++s_idx) {
    target[s_idx] = coeff * source[s_idx];
  }
}
inline void AccumFloatVec(const FloatVec& source, 
    const float coeff, FloatVec& target) {
#ifdef DEBUG
    CHECK_EQ(source.size(), target.size());
#endif
  for (int s_idx = 0; s_idx < source.size(); ++s_idx) {
    target[s_idx] += coeff * source[s_idx];
  }
}

// Z = aX + bY
inline void AddFloatVectors(const float a, const FloatVec& X, 
    const float b, const FloatVec& Y, FloatVec& Z) {
#ifdef DEBUG
  CHECK_EQ(X.size(), Y.size());
  CHECK_EQ(X.size(), Z.size());
#endif
  for (int i = 0; i < X.size(); ++i) {
    Z[i] = a * X[i] + b * Y[i];
  }
}

// Z = X'*Y
inline float DotProdFloatVectors(const FloatVec& X, const FloatVec& Y) {
#ifdef DEBUG
  CHECK_EQ(X.size(), Y.size());
#endif
  float prod = 0;
  for (int i = 0; i < X.size(); ++i) {
    prod += X[i] * Y[i];
  }
  return prod;
}

template <class I, class T>
void FreeMap(std::map<I, T*>& delete_map) {
  for (auto& ele : delete_map) {
    delete ele.second;
  }
  delete_map.clear();
}

// Fast approximate log modified Bessel function of the first kind
inline double fastLogBesselI(double d, double kappa) {
  static const double pi = atan(1.)*4;
  double frac = kappa/d;
  double square = 1 + frac*frac;
  double root = pow(square,0.5);
  double eta = root + log(frac) - log(1+root);
  double approx = -log(pow(2*pi*d,0.5)) + d*eta - 0.25*log(square);
  return approx;
}

// High-precision Bessel function (slow)
// Requires libntl
/*
RR BesselI(double s, double x) {
  if (x == 0) return to_RR("0.0");
  if (s == 0) return to_RR("1.0");

  RR scale_term;
  RR srr = to_RR(s);

  scale_term = pow ( to_RR(HALF_ED*x / s), srr);
  RR tmp;
  tmp = 1 + 1.0/(12*s) + 1.0/(288*s*s) - 139.0/(51840*s*s*s);
  scale_term *= sqrt(s) * INV_2ROOTPI / tmp;

  double ratio;
  RR tol = TOL;

  RR aterm;
  aterm = 1.0/s;
  RR sum; sum = aterm;
  int k = 1;

  while (true) {
    ratio = ((0.25*x*x) / (k * (s+k)));
    aterm *= ratio;
    if (aterm  < tol*sum) break;
    sum += aterm;
    ++k;
  }
  sum *= scale_term;
  return sum;
}
*/

// VMF log probability density function
// vMF(x | mu, kappa)
inline float LogVMFProb(const UIntFloatMap& x, const FloatVec& mu, float kappa) {
  //static const float pi = atan(1.) * 4;
  float result = 0.0;
  BOOST_FOREACH(const UIntFloatPair& ele, x) {
    result += ele.second * mu[ele.first];
  }
  result *= kappa;
  //std::cout << kappa << "\t" << (V_/2.0 - 1)*log(kappa) << "\t" << - (V_/2.0)*log(2*pi) << "\t" << - fastLogBesselI(V_/2.0 - 1, kappa)
  //          << "\t\t\t" << (V_/2.0 - 1)*log(kappa) - (V_/2.0)*log(2*pi) - fastLogBesselI(V_/2.0 - 1, kappa) << std::endl;
  
  // vMF normalizer
  // This normalizer is known to be inaccurate in high dimensions. If you want
  // to disable it, make sure to set all node emission Kappas identically!
  //result += (V_/2.0 - 1)*log(kappa) - (V_/2.0)*log(2*pi) - to_double(log(BesselI(V_/2.0 - 1, kappa)));
  //result += (V_/2.0 - 1)*log(kappa) - (V_/2.0)*log(2*pi) - fastLogBesselI(V_/2.0 - 1, kappa);
  return result;
}

inline double LogVMFProbNormalizer(const double V, const double kappa) {
  static const double pi = atan(1.)*4;
  double result
      = (V/2.0 - 1)*log(kappa) - (V/2.0)*log(2*pi) - fastLogBesselI(V/2.0 - 1, kappa);
  return result;
}

/*
 * given log(a) and log(b), return log(a + b)
 *
 */
inline double log_sum(double log_a, double log_b) {
  double v;

  if (log_a < log_b) {
      v = log_b + log(1 + exp(log_a - log_b));
  }
  else {
      v = log_a + log(1 + exp(log_b - log_a));
  }
  return v;
}

 /**
   * Proc to calculate the value of the trigamma, the second
   * derivative of the loggamma function. Accepts positive matrices.
   * From Abromowitz and Stegun.  Uses formulas 6.4.11 and 6.4.12 with
   * recurrence formula 6.4.6.  Each requires workspace at least 5
   * times the size of X.
   *
   **/
inline double trigamma(double x) {
    double p;
    int i;

    x=x+6;
    p=1/(x*x);
    p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
         *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
    for (i=0; i<6 ;i++) {
        x=x-1;
        p=1/(x*x)+p;
    }
    return(p);
}


/*
 * taylor approximation of first derivative of the log gamma function
 *
 */
inline double digamma(double x) {
  double p;
  x=x+6;
  p=1/(x*x);
  p=(((0.004166666666667*p-0.003968253986254)*p+
      0.008333333333333)*p-0.083333333333333)*p;
  p=p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
  return p;
}


inline double log_gamma(double x) {
     double z=1/(x*x);

    x=x+6;
    z=(((-0.000595238095238*z+0.000793650793651)
	*z-0.002777777777778)*z+0.083333333333333)/x;
    z=(x-0.5)*log(x)-x+0.918938533204673+z-log(x-1)-
	log(x-2)-log(x-3)-log(x-4)-log(x-5)-log(x-6);
    return z;
}


/*
 * argmax
 *
 */
inline int argmax(double* x, int n) {
    int i;
    double max = x[0];
    int argmax = 0;
    for (i = 1; i < n; i++)
    {
        if (x[i] > max)
        {
            max = x[i];
            argmax = i;
        }
    }
    return argmax;
}


} // namespace ditree

#endif
