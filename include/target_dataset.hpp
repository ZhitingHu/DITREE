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
      : target_vertex_idx_(target_vertex_idx) {
    n_parent_ = 0;
    n_child_ = 0;
  }

  inline void AddDatum(const Datum* datum, const UIntFloatMap& log_weights, 
      const float log_weights_sum) {
    data_.push_back(datum);
    BOOST_FOREACH(const UIntFloatPair& lw_ele, log_weights) {
      origin_n_[lw_ele.first] += exp(lw_ele.second - log_weights_sum);
    }
#ifdef DEBUG
    CHECK_EQ(origin_n_.size(), log_weights.size());
#endif
    BOOST_FOREACH(const UIntFloatPair& d_ele, datum->data()) {
      if (word_idxes_.find(d_ele.first) == word_idxes_.end()) {
        int index = word_idxes_.size();
        word_idxes_[d_ele.first] = index;
      } 
    }
    s_parent_.resize(word_idxes_.size());
    s_child_.resize(word_idxes_.size());
  }
 
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

  inline UIntUIntMap& word_idxes() { return word_idxes_; }
  inline float n_parent() const { return n_parent_; }
  inline float n_child() const { return n_child_; }
  inline float& mutable_n_parent() { return n_parent_; }
  inline float& mutable_n_child() { return n_child_; }
  inline FloatVec& s_parent() { return s_parent_; }  
  inline FloatVec& s_child() { return s_child_; }  

 private:

 private:
  vector<const Datum*> data_;
  UIntUIntMap word_idxes_; 

  uint32 target_vertex_idx_;
  uint32 child_vertex_idx_;
  
  // Used in restricted update
  float n_parent_;
  float n_child_;
  FloatVec s_parent_;
  FloatVec s_child_;

  UIntFloatMap origin_n_;
};

} // namespace ditree

#endif
