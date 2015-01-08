#ifndef DITREE_DATA_BATCH_HPP_
#define DITREE_DATA_BATCH_HPP_

#include "common.hpp"
#include "context.hpp"


namespace ditree {

class DataBatch {
 public:
  explicit DataBatch();
 
  inline int size() { return data_.size(); }
  inline Datum* datum(int idx) {
#ifdef DEBUG
    CHECK_LT(idx, data_.size());
#endif
    return data_[idx];
  }

  inline const UIntFloatMap& n() { return n_; }
  inline const map<uint32, UIntFloatMap>& s() { return s_; }

 private:

 private:
  vector<Datum*> data_;

  /// one entry for each topic component
  UIntFloatMap n_;
  // topic_id => (word_id => weight)
  map<uint32, UIntFloatMap> s_;
  //UIntFloatMap h_;
};

} // namespace ditree

#endif
