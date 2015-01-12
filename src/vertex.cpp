
#include "vertex.hpp"
#include <cmath>

namespace ditree {

Vertex::Vertex(const VertexParameter& param): parent_(NULL), 
    left_sibling_(NULL), right_sibling_(NULL), children_(0) {
  idx_ = param.index();
  //TODO init other params
}

void Vertex::RecursConstructParam() {
  // Construct from leaves to root
  for (int c_idx = 0; c_idx < children_.size(); ++c_idx) {
    children_[c_idx]->RecursConstructParam(); 
  }
  
  ConstructParam(data_batch_n_new[idx_], data_batch_n_old[idx_], 
      data_batch_s_new[idx_], data_batch_s_old[idx_]);
}

void Vertex::ConstructParam() {
  // tau
  tau_[0] = tau_fixed_part_[0] + n_; 
  float children_n_sum = 0;
  for (int c_idx = 0; c_idx < children_.size(); ++c_idx) {
    children_n_sum += children_[c_idx]->var_n_sum_for_parent();
  } 
  tau_[1] = tau_fixed_part_[1] + children_n_sum;
  var_n_sum_for_parent_ = children_n_sum + n_;
  // sigma
  sigma_[0] = sigma_fixed_part_[0] + var_n_sum_for_parent_;
  var_n_sum_for_sibling_ = var_n_sum_for_parent_
      + (right_sibling_ ? right_sibling_->var_n_sum_for_sibling() : 0);
  sigma_[1] = sigma_fixed_part_[1] + var_n_sum_for_sibling_ 
      - var_n_sum_for_parent_;
  
  // mean
  const FloatVec mean_prev(mean_);
  std::fill(mean_.begin(), mean_.end(), 0); 
  for (int c_idx = 0; c_idx < children_.size(); ++c_idx) {
#ifdef
    CHECK_EQ(new_born_, false);
#endif
    if (!children_[c_idx]->new_born()) { 
      // child's history
      const FloatVec& child_mean_history = children_[c_idx]->mean_history();
      float rho_apprx = 0;
      for (int i = 0; i < mean_prev.size(); ++i) {
        float ele = kappa_1_ * mean_prev[i] + kappa_2_ * child_mean_history[i];
        rho_apprx += ele * ele;
      }
      rho_apprx = sqrt(rho_apprx) * kappa_0_;
      float taylor_coeff = ComputeTaylorApprxCoeff(rho_apprx);
      for (int i = 0; i < mean_.size(); ++i) {
        mean_[i] += taylor_coeff * chile_mean_history[i];
      }
    }
    const FloatVec& child_mean = children_[c_idx]->mean();
    for (int i = 0; i < mean_.size(); ++i) {
      mean_[i] += kappa_0_ * kappa_1_ * child_mean[i];
    }
  } // end of children
  const FloatVec& parent_mean = parent_->mean();
  if (!new_born_) {
    for (int i = 0; i < mean_.size(); ++i) {
      mean_[i] += beta_ * s_[i] + kappa_0_ * (kappa_1_ * parent_mean[i] 
          + kappa_2_ * mean_history_[i]);
    }
  } else {
    for (int i = 0; i < mean_.size(); ++i) {
      mean_[i] += beta_ * s_[i] + kappa_0_ * kappa_1_ * parent_mean[i];
    }
  }
  // kappa
  kappa_ = 0;
  for (int i = 0; i < mean_.size(); ++i) {
    kappa_ += mean_[i] * mean_[i];
  }
  kappa_ = sqrt(kappa_);
#ifdef DEBUG
  CHECK_GE(kappa_, kFloatEpsilon);
#endif
  for (int i = 0; i < mean_.size(); ++i) {
    mean_[i] /= kappa_;
  }
}

void Vertex::UpdateParamTable(
    const float data_batch_n_z_new, 
    const float data_batch_n_z_old, 
    const UIntFloatMap& data_batch_s_z_new,
    const UIntFloatMap& data_batch_s_z_old) {
  petuum::Table<float>* param_table = Context::param_table();
  petuum::DenseUpdateBatch<float> update_batch(
      0, kColIdxParamTableSStart + s_.size());
  update_batch[kColIdxParamTableN] 
      = data_batch_n_z_new - data_batch_n_z_old;
  BOOST_FOREACH(const UIntFloatPair& ele, data_batch_s_z_new) {
    update_batch[kColIdxParamTableStart + mean_.size() + ele.first] 
        = ele.second - data_batch_s_z_old[ele.first];
  }
  param_table->DenseBatchInc(idx_, update_batch);
}

void Vertex::ReadParamTable() {
  petuum::Table<float>* param_table = Context::param_table();
  petuum::RowAccessor row_acc;
  vector<float> row_cache(kColIdxParamTableSStart + s_.size());
  const auto& r 
      = param_table->Get<petuum::DenseRow<float> >(idx_, &row_acc);
  r.CopyToVector(&row_cache);
  n_ = row_cache[kColIdxParamTableN];
  for (int i = 0; i < s_.size(); ++i) {
    s_[i] = row_cache[kColIdxParamTableStart + i];
  }
}

#if 0
void Vertex::RecursUpdateParamTable(
    const UIntFloatMap& data_batch_n_new, 
    const UIntFloatMap& data_batch_n_old, 
    const map<uint32, UIntFloatMap>& data_batch_s_new,
    const map<uint32, UIntFloatMap>& data_batch_s_old) {
  // Update from leaves to root
  for (int c_idx = 0; c_idx < children_.size(); ++c_idx) {
    children_[c_idx]->RecursUpdateParam(data_batch_n_new, 
        data_batch_n_old, data_batch_s_new, data_batch_s_old);
  }
  
  UpdateParam(data_batch_n_new[idx_], data_batch_n_old[idx_], 
      data_batch_s_new[idx_], data_batch_s_old[idx_]);
}

void Vertex::UpdateParam(
    const float data_batch_n_z_new, 
    const float data_batch_n_z_old, 
    const UIntFloatMap& data_batch_s_z_new,
    const UIntFloatMap& data_batch_s_z_old,
    const float learning_rate) {
  /// sufficient statistics
  n_ = n_ - data_batch_n_z_old + data_batch_n_z_new;
  BOOST_FOREACH(const UIntFloatPair& ele, data_batch_s_z_new) {
#ifdef DEBUG
    CHECK(data_batch_s_z_old[ele.first] != data_batch_s_z_old.end());
#endif
    s_[ele.first] = s_[ele_first] - data_batch_s_z_old[ele.first] 
        + ele.second;
  }
  
  /// topic parameters
  // tau
  tau_[0] = tau_fixed_part_[0] + n_; 
  float children_n_sum = 0;
  for (int c_idx = 0; c_idx < children_.size(); ++c_idx) {
    children_n_sum += children_[c_idx]->var_n_sum_for_parent();
  } 
  tau_[1] = tau_fixed_part_[1] + children_n_sum;
  var_n_sum_for_parent_ = children_n_sum + n_;
  // sigma
  sigma_[0] = sigma_fixed_part_[0] + var_n_sum_for_parent_;
  var_n_sum_for_sibling_ = var_n_sum_for_parent_
      + (right_sibling_ ? right_sibling_->var_n_sum_for_sibling() : 0);
  sigma_[1] = sigma_fixed_part_[1] + var_n_sum_for_sibling_ 
      - var_n_sum_for_parent_;
  
  // mean
  // TODO: use the latest mean_ from PS Table
  const FloatVec mean_prev(mean_);
  std::fill(mean_.begin(), mean_.end(), 0); 
  for (int c_idx = 0; c_idx < children_.size(); ++c_idx) {
#ifdef
    CHECK_EQ(new_born_, false);
#endif
    if (!children_[c_idx]->new_born()) { 
      // child's history
      const FloatVec& child_mean_history = children_[c_idx]->mean_history();
      float rho_apprx = 0;
      for (int i = 0; i < mean_prev.size(); ++i) {
        float ele = kappa_1_ * mean_prev[i] + kappa_2_ * child_mean_history[i];
        rho_apprx += ele * ele;
      }
      rho_apprx = sqrt(rho_apprx) * kappa_0_;
      float taylor_coeff = ComputeTaylorApprxCoeff(rho_apprx);
      for (int i = 0; i < mean_.size(); ++i) {
        mean_[i] += taylor_coeff * chile_mean_history[i];
      }
    }
    const FloatVec& child_mean = children_[c_idx]->mean();
    for (int i = 0; i < mean_.size(); ++i) {
      mean_[i] += kappa_0_ * kappa_1_ * child_mean[i];
    }
  } // end of children
  const FloatVec& parent_mean = parent_->mean();
  if (!new_born_) {
    for (int i = 0; i < mean_.size(); ++i) {
      mean_[i] += beta_ * s_[i] + kappa_0_ * (kappa_1_ * parent_mean[i] 
          + kappa_2_ * mean_history_[i]);
    }
  } else {
    for (int i = 0; i < mean_.size(); ++i) {
      mean_[i] += beta_ * s_[i] + kappa_0_ * kappa_1_ * parent_mean[i];
    }
  }

  // kappa
  kappa_ = 0;
  for (int i = 0; i < mean_.size(); ++i) {
    kappa_ += mean_[i] * mean_[i];
  }
  kappa_ = sqrt(kappa_);
#ifdef DEBUG
  CHECK_GE(kappa_, kFloatEpsilon);
#endif
  for (int i = 0; i < mean_.size(); ++i) {
    mean_[i] /= kappa_;
  }

  /// Update param tables
  petuum::Table<float>* param_table = Context::param_table();
  petuum::DenseUpdateBatch<float> update_batch(
      kColIdxParamTableMeanStart, 
      kColIdxParamTableMeanStart + mean_.size() * 2);
  update_batch[kColIdxParamTableLr] = learning_rate;
  update_batch[kColIdxParamTableN] 
      = data_batch_n_z_new - data_batch_n_z_old;
  update_batch[kColIdxParamTableTau0] = tau_[0];
  update_batch[kColIdxParamTableTau1] = tau_[1];
  update_batch[kColIdxParamTableSigma0] = sigma_[0];
  update_batch[kColIdxParamTableSigma1] = sigma_[1];
  update_batch[kColIdxParamTableKappa] = kappa_;
  for (int i = 0; i < mean_.size(); ++i) {
    update_batch[kColIdxParamTableStart + i] = mean_[i];
  }
  BOOST_FOREACH(const UIntFloatPair& ele, data_batch_s_z_new) {
    update_batch[kColIdxParamTableStart + mean_.size() + ele.first] 
        = ele.second - data_batch_s_z_old[ele.first];
  }
  param_table->DenseBatchInc(idx_, update_batch);
}

void Vertex::ReadParamTable() {
  petuum::Table<float>* param_table = Context::param_table();
  petuum::RowAccessor row_acc;
  vector<float> row_cache(kColIdxParamTableMeanStart + mean_.size() * 2);
  const auto& r 
      = param_table->Get<petuum::DITreeDenseRow<float> >(idx_, &row_acc);
  r.CopyToVector(&row_cache);
  n_ = row_cache[kColIdxParamTableN];
  tau_[0] = row_cache[kColIdxParamTableTau0];
  tau_[1] = row_cache[kColIdxParamTableTau1];
  sigma_[0] = row_cache[kColIdxParamTableSigma0];
  sigma_[1] = row_cache[kColIdxParamTableSigma1];
  kappa_ = row_cache[kColIdxParamTableKappa];
  for (int i = 0; i < mean_.size(); ++i) {
    mean_[i] = row_cache[kColIdxParamTableStart + i];
  }
  for (int i = 0; i < mean_.size(); ++i) {
    s_[i] = row_cache[kColIdxParamTableStart + mean_.size() + i];
  }
}
#endif

void Vertex::RecursComputeVarZPrior() {
  ComputeVarZPrior();

  // recursion: from root to leaves
  for (int c_idx = 0; c_idx < children_.size(); ++c_idx) {
    children_[c_idx]->RecursComputeVarZPrior();
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
  // E [ log p(z | \nu, \psi) ]
  var_z_prior_ = parent_->var_z_prior_part_for_child() 
      + expt_log_phi_z + expt_log_nu_z;
  
  var_z_prior_part_for_sibling_ 
      = (left_sibling_ ? left_sibling_->var_z_prior_part_for_sibling() : 0)
      + digamma(sigma_[1]) - digamma_sigma_sum; 
  var_z_prior_part_for_child_
      = parent_->var_z_prior_part_for_child()
      + digamma(tau_[1]) - digamma_tau_sum
      + expt_log_phi_z;
}

inline float ComputeTaylorApprxCoeff(const float rho_apprx) {
#ifdef DEBUG
  CHECK_GE(rho_apprx, kFloatEpsilon);
#endif
  const float V = mean_size();
  const float bessel_1 = exp(ditree::fastLogBesselI(V / 2.0, rho_apprx));
  const float bessel_2 = exp(ditree::fastLogBesselI(V / 2.0 - 1, rho_apprx));
#ifdef DEBUG
  CHECK(!isinf(bessel_1));
  CHECK(!isinf(bessel_2));
  CHECK(!isnan(bessel_1));
  CHECK(!isnan(bessel_2));
  CHECK_GE(bessel_2, kFloatEpsilon);
#endif
  return ((V - 2.0) + (bessel_1 / bessel_2 + (V / 2.0 - 1.0) / rho_apprx)) 
      * kappa_0 * kappa_0 * kappa_1 * kappa_2 / rho_apprx;
}

} // namespace ditree
