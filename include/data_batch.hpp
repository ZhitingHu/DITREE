#ifndef DITREE_DATA_BATCH_HPP_
#define DITREE_DATA_BATCH_HPP_

#include "common.hpp"
#include "context.hpp"


namespace ditree {

class DataBatch {
 public:
  explicit DataBatch();
 
  void UpdateSuffStatStruct(
      const vector<Triple>& vertex_split_records, 
      const vector<Triple>& vertex_merge_records);

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
  int size_;
  int data_idx_begin_;

  /// sufficient statistics
  /// one entry for each topic component
  UIntFloatMap n_;
  // vertex_id => (word_id => weight)
  map<uint32, UIntFloatMap> s_;
  //UIntFloatMap h_;
};

} // namespace ditree

#endif
