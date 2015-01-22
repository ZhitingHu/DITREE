#ifndef DITREE_TARGET_DATASET_HPP_
#define DITREE_TARGET_DATASET_HPP_

#include "datum.hpp"
#include "common.hpp"
#include "context.hpp"
#include <boost/foreach.hpp>

namespace ditree {

class TargetDataset {
 public:
  explicit TargetDataset(const uint32 target_vertex_idx) 
      : target_vertex_idx_(target_vertex_idx), success_(false) {
    n_parent_ = 0;
    n_child_ = 0;
  }

  inline void AddDatum(const Datum* datum, const UIntFloatMap& log_weights, 
      const float log_weights_sum) {
    data_.push_back(datum);
    BOOST_FOREACH(const UIntFloatPair& lw_ele, log_weights) {
      origin_n_[lw_ele.first] += exp(lw_ele.second - log_weights_sum);
    }
    BOOST_FOREACH(const UIntFloatPair& d_ele, datum->data()) {
      s_parent_[d_ele.first] = 0;
      s_child_[d_ele.first] = 0;
    }
#ifdef DEBUG
    CHECK_EQ(origin_n_.size(), log_weights.size());
#endif
  }
 
  inline bool success() const { return success_; }
  inline void set_success() { success_ = true; }

  inline uint32 target_vertex_idx() { return target_vertex_idx_; }
  inline uint32 child_vertex_idx() { return child_vertex_idx_; }
  inline void set_child_vertex_idx(const uint32 child_vertex_idx) {
    child_vertex_idx_ = child_vertex_idx;
  } 
  inline const UIntFloatMap& origin_n() { return origin_n_; }
 
  inline const Datum* datum(const int idx) {
#ifdef DEBUG
    CHECK_LT(idx, data_.size());
#endif
    return data_[idx];
  }
  inline const vector<const Datum*>& data() { return data_; }
  inline int size() const { return data_.size(); }

  inline float& n_parent() { return n_parent_; }
  inline float& n_child() { return n_child_; }
  inline UIntFloatMap& s_parent() { return s_parent_; }  
  inline UIntFloatMap& s_child() { return s_child_; }  

 private:

 private:
  vector<const Datum*> data_;
 
  uint32 target_vertex_idx_;
  uint32 child_vertex_idx_;
  bool success_;
  
  // Used in restricted update
  float n_parent_;
  float n_child_;
  UIntFloatMap s_parent_;
  UIntFloatMap s_child_;

  UIntFloatMap origin_n_;
};

} // namespace ditree

#endif
