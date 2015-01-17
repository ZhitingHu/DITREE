#ifndef DITREE_RANDOM_HPP_
#define DITREE_RANDOM_HPP_

#include <cmath>
#include <algorithm>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include "common.hpp"

namespace ditree {

// Random number generator
class Random {
  boost::mt19937 generator;
  boost::uniform_real<> zero_one_dist;
  boost::variate_generator<boost::mt19937&,
                           boost::uniform_real<> > zero_one_generator;
  
public:
  Random(unsigned int seed) :
    generator(seed),
    zero_one_dist(0,1),
    zero_one_generator(generator, zero_one_dist)
  { }
  
  // Draws a random unsigned integer
  unsigned int randInt() {
    return generator();
  }
  
  // Draws a random real number in [0,1)
  float rand() {
    return zero_one_generator();
  }
  
  // Draws a random number from a Gamma(a) distribution
  float randGamma(float a) {
    boost::gamma_distribution<> gamma_dist(a);
    boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> >
      gamma_generator(generator, gamma_dist);
    return gamma_generator();
  }
  
  // Draws a random number from an Exponential(a) distribution
  float randExponential(float a) {
    boost::exponential_distribution<> exponential_dist(a);
    boost::variate_generator<boost::mt19937&,
                             boost::exponential_distribution<> >
      exponential_generator(generator, exponential_dist);
    return exponential_generator();
  }
  
  // Draws a random number from a Beta(a,b) disstribution.
  float randBeta(float a, float b) {
    float x = randGamma(a);
    float y = randGamma(b);
    return x/(x+y);
  }
  
  // Draws a random vector from a symmetric Dirichlet(a) distribution.
  // The dimension of the vector is determined from output.size().
  void randSymDirichlet(float a, FloatVec& output) {
    boost::gamma_distribution<> gamma_dist(a);
    boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> >
      gamma_generator(generator, gamma_dist);
    float total    = 0;
    for (unsigned int i = 0; i < output.size(); ++i) {
      output[i]   = gamma_generator();
      total       += output[i];
    }
    for (unsigned int i = 0; i < output.size(); ++i) {
      output[i]   /= total;
    }
  }
  
  // Samples from an unnormalized discrete distribution, in the range
  // [begin,end)
  size_t randDiscrete(const FloatVec& distrib, size_t begin, size_t end) {
    float totprob  = 0;
    for (size_t i = begin; i < end; ++i) {
      totprob += distrib[i];
    }
    float r        = totprob * zero_one_generator();
    float cur_max  = distrib[begin];
    size_t idx      = begin;
    while (r > cur_max) {
      cur_max += distrib[++idx];
    }
    return idx;
  }
  
  // Converts the range [begin,end) of a vector of log-probabilities into
  // relative probabilities
  static void logprobsToRelprobs(FloatVec &distrib, size_t begin, size_t end) {
    // Find the maximum element in [begin,end)
    float max_log = *std::max_element(distrib.begin()+begin,
                                       distrib.begin()+end);
    for (size_t i = begin; i < end; ++i) {
      // Avoid over/underflow by centering log-probabilities to their max
      distrib[i] = exp(distrib[i] - max_log);
    }
  }
  
  // Log-gamma function
  static float lnGamma(float xx) {
    int j;
    float x,y,tmp1,ser;
    static const float cof[6]={76.18009172947146,-86.50532032941677,
                                24.01409824083091,-1.231739572450155,
                                0.1208650973866179e-2,-0.5395239384953e-5};
    y = xx;
    x = xx;
    tmp1 = x + 5.5;
    tmp1 -= (x+0.5)*log(tmp1);
    ser = 1.000000000190015;
    for (j=0;j<6;j++) ser += cof[j]/++y;
    return -tmp1 + log(2.5066282746310005*ser/x);
  }
}; // Random

} // namespace ditree

#endif
