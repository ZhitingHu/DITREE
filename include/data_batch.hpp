#ifndef DITREE_DATA_BATCH_HPP_
#define DITREE_DATA_BATCH_HPP_

#include "tree.hpp"
#include "datum.hpp"
#include "common.hpp"
#include "context.hpp"

namespace ditree {

class DataBatch {
 public:
  explicit DataBatch(const int data_idx_begin, const int size)
  : data_idx_begin_(data_idx_begin), size_(size) { }
 
  void UpdateSuffStatStruct(const Tree* tree, const Context::Phase phase);
  void InitSuffStatStruct(const Tree* tree, const vector<Datum*>& data);

  inline UIntUIntMap& word_idxes() { return word_idxes_; }
  inline UIntFloatMap& n() { return n_; }
  inline map<uint32, FloatVec>& s() { return s_; }
  inline int size() const { return size_; }
  inline int data_idx_begin() const { return data_idx_begin_; } 

  void UpdateSuffStatStructBySplit(
      const vector<pair<uint32, uint32> >& vertex_split_records);
  void UpdateSuffStatStructByMerge(
      const vector<pair<uint32, uint32> >& vertex_merge_records);

 private:


 private:
  int data_idx_begin_;
  int size_;

  /// sufficient statistics
  /// one entry for each topic component
  UIntFloatMap n_;

  // word_idx => index in s_'s FloatVec
  UIntUIntMap word_idxes_;
  // vertex_id => (weights of words in word_idxes_)
  map<uint32, FloatVec> s_;
};

} // namespace ditree

#endif
