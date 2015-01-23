
#include "context.hpp"
#include "common.hpp"
#include "util.hpp"
#include "vertex.hpp"
#include <cmath>
#include <boost/foreach.hpp>

namespace ditree {

Vertex::Vertex(const VertexParameter& param, const TreeParameter& tree_param) {
  Init();
  idx_ = param.index();
#ifdef DEBUG
  CHECK_GT(Context::vocab_size(), 0);
#endif
  kappa_0_ = tree_param.kappa_0();
  kappa_1_ = tree_param.kappa_1(); 
  kappa_2_ = tree_param.kappa_2(); 
  beta_ = tree_param.beta(); 
  // TODO: history
  kappa_ = kappa_0_ * kappa_1_;
  alpha_ = tree_param.alpha();
  gamma_ = tree_param.gamma();
  tau_[0] = tau_fixed_part_[0] = 1.0;
  tau_[1] = tau_fixed_part_[1] = alpha_;
  sigma_fixed_part_[0] = 1.0;
  sigma_[0] = sigma_fixed_part_[0];
  sigma_fixed_part_[1] = tree_param.gamma();
  sigma_[1] = sigma_fixed_part_[1];
  // if have history, new_born_ = false
}

void Vertex::Init() {
  parent_ = NULL;
  left_sibling_ = NULL;
  right_sibling_ = NULL;
  root_ = false;
  new_born_ = true;
  children_.clear();

  var_z_prior_part_for_child_ = 0;
  var_z_prior_part_for_sibling_ = 0;
  var_n_sum_for_parent_ = 0;
  var_n_sum_for_sibling_ = 0;

  mean_.resize(Context::vocab_size());
  s_.resize(Context::vocab_size());
  n_ = 0;
}

void Vertex::RecursSetDepth(const int parent_depth) {
  depth_ = parent_depth + 1;
  for (int c_idx = 0; c_idx < children_.size(); ++c_idx) {
    children_[c_idx]->RecursSetDepth(depth_); 
  }
}

void Vertex::RecursConstructParam() {
  // Construct from leaves to root
  // and from right to left
  for (int c_idx = children_.size() - 1; c_idx >= 0; --c_idx) {
    children_[c_idx]->RecursConstructParam(); 
  }
  
  ConstructParam();
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
  LOG(INFO) << idx_ << " sigma_[0] " << sigma_[0] << " " 
      << (sigma_fixed_part_[0] + var_n_sum_for_parent_) << " " 
      << children_n_sum << " " << n_;

  sigma_[0] = sigma_fixed_part_[0] + var_n_sum_for_parent_;
  var_n_sum_for_sibling_ = var_n_sum_for_parent_
      + (right_sibling_ ? right_sibling_->var_n_sum_for_sibling() : 0);

  LOG(INFO) << idx_ << " sigma_[1] " << sigma_[1] << " " 
      << (sigma_fixed_part_[1] + var_n_sum_for_sibling_ - var_n_sum_for_parent_);

  sigma_[1] = sigma_fixed_part_[1] 
      + (right_sibling_ ? right_sibling_->var_n_sum_for_sibling() : 0);
  
  // mean
  const FloatVec mean_prev(mean_);
  std::fill(mean_.begin(), mean_.end(), 0); 

  LOG(INFO) << "here";

  for (int c_idx = 0; c_idx < children_.size(); ++c_idx) {
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
        mean_[i] += taylor_coeff * child_mean_history[i];
      }
    }
    const FloatVec& child_mean = children_[c_idx]->mean();
    for (int i = 0; i < mean_.size(); ++i) {
      mean_[i] += kappa_0_ * kappa_1_ * child_mean[i];
    }
  } // end of children

  LOG(INFO) << "here .1";

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

  LOG(INFO) << "here 2";

  // kappa
  kappa_ = 0;
  for (int i = 0; i < mean_.size(); ++i) {
    kappa_ += mean_[i] * mean_[i];
  }
  kappa_ = sqrt(kappa_);
#ifdef DEBUG
  CHECK_GE(kappa_, kFloatEpsilon);
#endif
  ostringstream oss;
  oss << "mean of " << idx_ << ": ";
  for (int i = 0; i < mean_.size(); ++i) {
    mean_[i] /= kappa_;
    oss << mean_[i] << " ";
  }
  oss << std::endl;
  LOG(INFO) << oss.str();
  LOG(INFO) << "kappa_ = " << kappa_;
}

void Vertex::InitParam(const float n_init, const FloatVec& s_init) {
  // Init param table
//#ifdef DEBUG
//  CHECK_EQ(s_init.size(), s_.size());
//#endif
//  petuum::Table<float>* param_table = Context::param_table();
//  petuum::DenseUpdateBatch<float> update_batch(
//      0, kColIdxParamTableSStart + s_.size());
//  update_batch[kColIdxParamTableN] = n_init; 
//  for (int w_idx = 0; w_idx < s_.size(); ++w_idx) {
//    //update_batch[kColIdxParamTableSStart + w_idx] = s_init[w_idx];
//    update_batch[kColIdxParamTableSStart + w_idx] = Context::rand();
//  }
//  param_table->DenseBatchInc(idx_, update_batch);
 
  /// Init params
  // Must do this, since a vertex's param depends on both parent 
  // and children in ConstructParam()
//  float s_init_norm = 0;
//  for (int s_idx = 0; s_idx < s_init.size();; ++s_idx) {
//    s_init_norm = s_init[s_idx] * s_init[s_idx];
//  }
//  s_init_norm = sqrt(s_init_norm);
//#ifdef DEBUG
//    CHECK_EQ(s_init.size(), mean_size());
//#endif
//  for (int s_idx = 0; s_idx < s_init.size(); ++s_idx) {
//    mean_[s_idx] = s_init[s_idx] / s_init_norm;
//  }
  //TODO: history
  for (int s_idx = 0; s_idx < s_init.size(); ++s_idx) {
    mean_[s_idx] = s_init[s_idx]; 
  }
}

void Vertex::UpdateParamTable(const float data_batch_n_z_new, 
    const float data_batch_n_z_old, const UIntFloatMap& data_batch_s_z_new,
    const UIntFloatMap& data_batch_s_z_old) {
  petuum::Table<float>* param_table = Context::param_table();
  petuum::DenseUpdateBatch<float> update_batch(
      0, kColIdxParamTableSStart + s_.size());
  update_batch[kColIdxParamTableN] 
      = data_batch_n_z_new - data_batch_n_z_old;

  BOOST_FOREACH(const UIntFloatPair& ele, data_batch_s_z_new) {
    update_batch[kColIdxParamTableSStart + ele.first] 
        = ele.second - data_batch_s_z_old.find(ele.first)->second;
  }
  param_table->DenseBatchInc(idx_, update_batch);
}

void Vertex::UpdateParamTable(const float data_batch_n_z_new, 
    const float data_batch_n_z_old, const FloatVec& data_batch_s_z_new,
    const FloatVec& data_batch_s_z_old) {
  petuum::Table<float>* param_table = Context::param_table();
  petuum::DenseUpdateBatch<float> update_batch(
      0, kColIdxParamTableSStart + s_.size());
  update_batch[kColIdxParamTableN] 
      = data_batch_n_z_new - data_batch_n_z_old;

  for (int sz_i = 0; sz_i < data_batch_s_z_new.size(); ++sz_i) {
    update_batch[kColIdxParamTableSStart + sz_i] 
        = data_batch_s_z_new[sz_i] - data_batch_s_z_old[sz_i];
  }
  param_table->DenseBatchInc(idx_, update_batch);
}

void Vertex::UpdateParamTableByInc(const float n_z, const UIntFloatMap& s_z,
    const float coeff) {
  petuum::Table<float>* param_table = Context::param_table();
  petuum::DenseUpdateBatch<float> update_batch(
      0, kColIdxParamTableSStart + s_.size());
  update_batch[kColIdxParamTableN] = coeff * n_z;

  BOOST_FOREACH(const UIntFloatPair& ele, s_z) {
    update_batch[kColIdxParamTableSStart + ele.first] = coeff * ele.second;
  }
  param_table->DenseBatchInc(idx_, update_batch);
}

void Vertex::UpdateParamTableByInc(const float n_z, const FloatVec& s_z,
    const float coeff) {
  petuum::Table<float>* param_table = Context::param_table();
  petuum::DenseUpdateBatch<float> update_batch(
      0, kColIdxParamTableSStart + s_.size());
  update_batch[kColIdxParamTableN] = coeff * n_z;

  for (int sz_i = 0; sz_i < s_z.size(); ++sz_i) {
    update_batch[kColIdxParamTableSStart + sz_i] = coeff * s_z[sz_i];
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
    s_[i] = row_cache[kColIdxParamTableSStart + i];
  }

  // TODO
  LOG(INFO) << "Read PS Table - thread " << idx_;
  ostringstream oss;
  oss << "n_ " << n_ << "\n";
  for (int i = 0; i < s_.size(); ++i) {
    oss << s_[i] << " ";
  }
  oss << "\n";
  LOG(INFO) << oss.str();
}

void Vertex::UpdateParamLocal(const float n_z_new, const float n_z_old,
    const UIntFloatMap& s_z_new, const UIntFloatMap& s_z_old) {
  BOOST_FOREACH(const UIntFloatPair& ele, s_z_new) {
    s_[ele.first] += ele.second - s_z_old.find(ele.first)->second;
  }
  n_ += n_z_new - n_z_old;
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
  // and from left to right
  for (int c_idx = 0; c_idx < children_.size(); ++c_idx) {
    children_[c_idx]->RecursComputeVarZPrior();
  }
}

inline void Vertex::ComputeVarZPrior() {
  // E[ log \phi_z ]
  float digamma_sigma_sum = digamma(sigma_[0] + sigma_[1]);
  float expt_log_phi_z 
      = digamma(sigma_[0]) - digamma_sigma_sum
      + (left_sibling_ ? left_sibling_->var_z_prior_part_for_sibling() : 0);
  if (root_) {
    // root node has phi=1
    expt_log_phi_z = 0;
  }
  LOG(INFO) << idx_ << " expt_log_phi_z " << expt_log_phi_z;

  //TODO
  CHECK(!isnan(expt_log_phi_z)) << digamma(sigma_[0]) << " " << sigma_[0] << " "
      << digamma_sigma_sum << " " << (sigma_[0] + sigma_[1]);
  
  // E [ log \nu_z ]
  const float digamma_tau_sum = digamma(tau_[0] + tau_[1]);
  const float expt_log_nu_z
      = digamma(tau_[0]) - digamma_tau_sum;

  LOG(INFO) << idx_ << " expt_log_nu_z " << expt_log_nu_z 
      << " parent_->var_z_prior_part_for_child():" << parent_->var_z_prior_part_for_child();

  // E [ log p(z | \nu, \psi) ]
  var_z_prior_ = parent_->var_z_prior_part_for_child() 
      + expt_log_phi_z + expt_log_nu_z;

  //TODO
  CHECK(!isnan(var_z_prior_)) << parent_->var_z_prior_part_for_child() << " "
      << expt_log_phi_z << " " << expt_log_nu_z;
  
  var_z_prior_part_for_sibling_ 
      = (left_sibling_ ? left_sibling_->var_z_prior_part_for_sibling() : 0)
      + digamma(sigma_[1]) - digamma_sigma_sum; 
  var_z_prior_part_for_child_
      = parent_->var_z_prior_part_for_child()
      + digamma(tau_[1]) - digamma_tau_sum
      + expt_log_phi_z;
}

inline float Vertex::ComputeTaylorApprxCoeff(const float rho_apprx) {
#ifdef DEBUG
  CHECK_GE(rho_apprx, kFloatEpsilon);
#endif
  const float V = mean_.size();
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
      * kappa_0_ * kappa_0_ * kappa_1_ * kappa_2_ / rho_apprx;
}

float Vertex::ComputeELBO() const {
  float elbo = 0;
  elbo += beta_ * DotProdFloatVectors(mean_, s_) 
      + n_ * LogVMFProbNormalizer(mean_.size(), beta_)
      + n_ * var_z_prior_;

  CHECK(!isnan(elbo));
  LOG(INFO) << "index = " << idx_ << "\tELBO = " << elbo << "\t" 
      << beta_ * DotProdFloatVectors(mean_, s_) << "\t" << n_ 
      << "\t" << n_ * var_z_prior_ << "\t" 
      << n_ * LogVMFProbNormalizer(mean_.size(), beta_);

  elbo += (1 - tau_[0]) * (digamma(tau_[0]) - digamma(tau_[0] + tau_[1]))
      + (alpha_ - tau_[1]) * (digamma(tau_[1]) - digamma(tau_[0] + tau_[1]));
  if (!root_) {
    elbo 
        += (1 - sigma_[0]) 
        * (digamma(sigma_[0]) - digamma(sigma_[0] + sigma_[1]))
        + (gamma_ - sigma_[1]) 
        * (digamma(sigma_[1]) - digamma(sigma_[0] + sigma_[1]));
  }
      
  CHECK(!isnan(elbo));
  LOG(INFO) << "index = " << idx_ << "\tELBO = " << elbo;

  float rho = 0;
  const FloatVec& parent_mean = parent_->mean(); 
  if (!new_born_) {
    for (int i = 0; i < parent_mean.size(); ++i) {
      float ele = kappa_1_ * parent_mean[i] + kappa_2_ * mean_history_[i];
      rho += ele * ele;
      elbo += (kappa_1_ * parent_mean[i] + kappa_2_ * mean_history_[i] 
          - kappa_ * mean_[i]) * mean_[i];
    }
    rho = sqrt(rho) * kappa_0_;
  } else {
    for (int i = 0; i < parent_mean.size(); ++i) {
      elbo += (kappa_1_ * parent_mean[i] - kappa_ * mean_[i]) * mean_[i];
    }
    rho = kappa_0_ * kappa_1_;
  }

  CHECK(!isnan(elbo));
  LOG(INFO) << "index = " << idx_ << "\tELBO = " << elbo;

  elbo += LogVMFProbNormalizer(mean_.size(), rho)
      - LogVMFProbNormalizer(mean_.size(), kappa_);

  CHECK(!isnan(elbo));
  LOG(INFO) << "index = " << idx_ << "\tELBO = " << elbo 
      << " rho=" << rho << " kappa=" << kappa_ << " " 
      << LogVMFProbNormalizer(mean_.size(), rho) << " "
      << LogVMFProbNormalizer(mean_.size(), kappa_);

  return elbo;
}

void Vertex::CopyParamsFrom(const Vertex* source) {
  (*this) = (*source);
  Init();
  std::fill(s_.begin(), s_.end(), 0); 
  n_ = 0;
  //s_ = source->s();
  //n_ = source->n();
}

/// Merge params from host and guest, but do not change the tree structure
void Vertex::MergeFrom(const Vertex* host, const Vertex* guest) {
  LOG(INFO) << "Merge from " << host->idx() << " " << guest->idx();
  CopyParamsFrom(host);
  parent_ = host->parent();
  left_sibling_ = host->left_sibling();
  right_sibling_ = host->right_sibling();
  root_ = host->root();
  new_born_ = host->new_born();
  for (const auto host_child : host->children()) {
    // Note: do not use add_child(), since it will change the link 
    //   structure of childs
    children_.push_back(host_child);
  }
  for (const auto guest_child : guest->children()) {
    // Note: do not use add_child(), since it will change the link 
    //   structure of childs
    children_.push_back(guest_child);
  }
  // Suff stat
  n_ = host->n() + guest->n();     
  CopyFloatVec(host->s(), 1.0, s_);
  AccumFloatVec(guest->s(), 1.0, s_);
  //
  ConstructParam();

  //LOG(INFO) << "Merge from param checking";
  //LOG(INFO) << tau_[0] << " " << host->tau(0) + guest->tau(0) - 1.0; 
  //LOG(INFO) << tau_[1] << " " << host->tau(1) + guest->tau(1) - alpha_; 
  //LOG(INFO) << sigma_[0] << " " << host->sigma(0) + guest->sigma(0) - 1.0; 
  //LOG(INFO) << sigma_[1] << " " << host->sigma(1);

  //TODO: history
}

} // namespace ditree
