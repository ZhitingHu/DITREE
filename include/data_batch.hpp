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
 
  void UpdateSuffStatStruct(const Tree* tree);
  void InitSuffStatStruct(const Tree* tree, const vector<Datum*>& data);

  inline UIntFloatMap& n() { return n_; }
  inline map<uint32, UIntFloatMap>& s() { return s_; }
  inline int size() const { return size_; }
  inline int data_idx_begin() const { return data_idx_begin_; } 

 private:

  void UpdateSuffStatStructBySplit(
      const vector<Triple>& vertex_split_records);
  void UpdateSuffStatStructByMerge(
      const vector<Triple>& vertex_merge_records);

 private:
  int data_idx_begin_;
  int size_;

  /// sufficient statistics
  /// one entry for each topic component
  UIntFloatMap n_;
  // vertex_id => (word_id => weight)
  map<uint32, UIntFloatMap> s_;
};

} // namespace ditree

#endif
