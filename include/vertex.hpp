#ifndef DITREE_VERTEX_HPP_
#define DITREE_VERTEX_HPP_

#include "common.hpp"
#include "proto/ditree.pb.h"

namespace ditree {

class Vertex {
 public:
  explicit Vertex(const VertexParameter& param);

  void RecursConstructParam();
  void RecursComputeVarZPrior();

  void UpdateParamTable();
  void ReadParamTable();

  inline const FloatVec& mean() const { return mean_; }
  inline const FloatVec& mean_history() const { 
#ifdef DEBUG
    CHECK_EQ(new_born_, false);
#endif
    return mean_history_; 
  }
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
  inline bool new_born() const {
    return new_born_;
  }

  inline float var_n_sum_for_parent() const {
    return var_n_sum_for_parent_;
  }
  inline float var_n_sum_for_sibling() const {
    return var_n_sum_for_sibling_;
  }
  inline float var_z_prior_part_for_child() const {
    return var_z_prior_part_for_child_; 
  }
  inline float var_z_prior_part_for_sibling() const {
    return var_z_prior_part_for_sibling_; 
  }
  inline float var_z_prior() const { return var_z_prior_; }

  inline int idx() const { return idx_; }

  inline void add_child(Vertex* child) {
    children_.push_back(child);
    child->set_parent(this);
    if (children_.size() > 1) {
      Vertex* child_left_sibling = children_[children_.size() - 2];
      child->set_left_sibling(child_left_sibling);
      child_left_sibling->set_right_sibling(child);
    }
  }
  inline void set_parent(Vertex* parent) { parent_ = parent; }
  inline void set_left_sibling(Vertex* left_sibling) { 
    left_sibling_ = left_sibling; 
  }
  inline void set_right_sibling(Vertex* right_sibling) { 
    right_sibling_ = right_sibling; 
  }
  //inline void set_depth(const int depth) { depth_ = depth; }

 private:
  
  inline void ComputeVarZPrior();
  inline void ComputeParam();
  inline float ComputeTaylorApprxCoeff(const float rho_apprx);

 private:
  //VertexParameter vertex_param_;

  /// tree structure
  uint32 idx_;
  Vertex* parent_;
  Vertex* left_sibling_;
  Vertex* right_sibling_;
  vector<Vertex*> children_;
  // a vextex's children are guaranteed to be in the same table
  uint32 child_table_idx_;
  
  // depth of this node (root has depth 1)
  //int depth_;

  /// (global) parameters
  // emission ~ vMF(mean_, kappa_)
  FloatVec mean_;
  float kappa_;
  // prior
  float tau_[2]; // prior of \nu
  float sigma_[2]; // prior of \psi

  // (local)
  float var_n_sum_for_parent_; // \sum_{z <= z.i} n_(z.i)
  float var_n_sum_for_sibling_; //\sum_{j >= i}{z.j <= z.j.k} n_{z.j.k} 

  /// (local) history
  float tau_fixed_part_[2]; // 1 + history, \alpha + history
  float sigma_fixed_part_[2]; // 1 + history, \gamma + history
  // f(z) = \sum_{t'}{n} < q(z = z_n), q(z < z_n) >
  FloatVec tau_history_[2]; 
  // f(z.i) = \sum_{t'}{n} < q(z.i <= z_n), \sum_{j} q(z.j <= z_n) >
  FloatVec sigma_history_[2];
  // mean at time stamp (t-1)
  FloatVec mean_history_;
  bool new_born_;

  /// (global) memoized VI
  float n_; // \sum_{n} q(z_n = z)
  FloatVec s_; // \sum_{n} q(z_n = z) * x_n
  //float h_; // \sum_{n} q(z_n = z) * log q(z_n = z)

  /// (local) VI
  // \sum_{z' <= z} E[ log (1 - \nu_z') ] + E[ log \phi_z' ]
  float var_z_prior_part_for_child_;
  // \sum_{j <= i} E[ log (1 - \psi_{j}) ]
  float var_z_prior_part_for_sibling_;
  // E[ log p(z | \nu, \psi) ]
  float var_z_prior_;
  
  /// fixed parameters
  float kappa_0_;
  float kappa_1_;
  float kappa_2_;
  float beta_;
};

} // namespace ditree

#endif
