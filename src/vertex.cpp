
#include "vertex.hpp"

namespace ditree {

Vertex::Vertex() {

}

void Vertex::RecursiveComputeVarZPrior() {
  ComputeVarZPrior();

  // recursion
  for (int c_idx = 0; c_idx < children_.size(); ++c_idx) {
    children_[c_idx]->RecursiveComputeVarZPrior();
  }
}

inline void Vertex::ComputeVarZPrior() {
  // E[ log \phi_z ]
  const float digamma_sigma_sum = digamma(sigma_[0] + sigma_[1]);
  const float expt_log_phi_z 
      = digamma(sigma_[0]) - digamma_sigma_sum
      + (left_sibling_ ? left_sibling_->var_z_prior_part_for_sibling() : 0);
  // E [ log \nu_z ]
  const float digamma_tau_sum = digamma(tau_[0] + tau_[1]);
  const float expt_log_nu_z
      = digamma(tau_[0]) - digamma_tau_sum;
  // exp{ E[ log p(z | \nu, \psi) ] }
  exp_var_z_prior_ = exp(
      parent_->var_z_prior_part_for_child() 
      + expt_log_phi_z + expt_log_nu_z);
  
  var_z_prior_part_for_sibling_ 
      = (left_sibling_ ? left_sibling_->var_z_prior_part_for_sibling() : 0)
      + digamma(sigma_[1]) - digamma_sigma_sum; 
  var_z_prior_part_for_child_
      = parent_->var_z_prior_part_for_child()
      + digamma(tau_[1]) - digamma_tau_sum
      + expt_log_phi_z;
}


}
