#ifndef DITREE_VERTEX_HPP_
#define DITREE_VERTEX_HPP_

#include "common.hpp"
#include "proto/ditree.pb.h"

namespace ditree {

class Vertex {
 public:

  void RecursiveComputeVarZPrior();

  inline const FloatVec& mean() const { return mean_; }
  inline float kappa() const { return kappa_; }

  inline float tau(const int idx) const { 
#ifdef DEBUG
    CHECK_LT(idx, 2);
#endif
    return tau_[idx]; 
  }
  inline float sigma(const int idx) const { 
#ifdef DEBUG
    CHECK_LT(idx, 2);
#endif
    return sigma_[idx]; 
  }

  inline float var_z_prior_part_for_child() {
    return var_z_prior_part_for_child_; 
  }
  inline float var_z_prior_part_for_sibling() {
    return var_z_prior_part_for_sibling_; 
  }
  inline float exp_var_z_prior() { return exp_var_z_prior_; }

 private: 

  inline void ComputeVarZPrior();

 private:
  VertexParameter vertex_param_;

  /// tree structure
  uint32 idx_;
  Vertex* parent_;
  Vertex* left_sibling_;
  vector<Vertex*> children_;
  // a vextex's children are guaranteed to be in the same table
  uint32 child_table_idx_;
  
  // depth of this node (root has depth 1)
  int depth_;

  /// (global) parameters
  // emission ~ vMF(mean_, kappa_)
  FloatVec mean_;
  float kappa_;
  // prior
  float tau_[2]; // prior of \nu
  float sigma_[2]; // prior of \psi

  /// (global) memoized VI
  float n_; // \sum_{n} q(z_n = z)
  FloatVec s_; // \sum_{n} q(z_n = z) * x_n
  //float h_; // \sum_{n} q(z_n = z) * log q(z_n = z)

  /// history
  FloatVec tau_0_history_; // f(z) = \sum_{t'}{n} q(z = z_n) 
  FloatVec tau_1_history_; // f(z) = \sum_{t'}{n} q(z < z_n) 
  FloatVec sigma_0_history_; // f(z.i) = \sum_{t'}{n} q(z.i <= z_n)
  FloatVec sigma_1_history_; // f(z.i) = \sum_{t'}{n}{j} q(z.j <= z_n)


  /// (local) VI
  // \sum_{z' <= z} E[ log (1 - \nu_z') ] + E[ log \phi_z' ]
  float var_z_prior_part_for_child_;
  // \sum_{j <= i} E[ log (1 - \psi_{j}) ]
  float var_z_prior_part_for_sibling_;
  // exp{ E[ log p(z | \nu, \psi) ] }
  float exp_var_z_prior_;

  
};

} // namespace ditree

#endif
