#ifndef DITREE_DATASET_HPP_
#define DITREE_DATASET_HPP_

#include "common.hpp"
#include "context.hpp"
#include "data_batch.hpp"

namespace ditree {

class Dataset {
 public:
  explicit Dataset();
  
  inline DataBatch* GetNextDataBatch(); 
  inline Datum* datum(int idx) {
#ifdef DEBUG
    CHECK_LT(idx, data_.size());
#endif
    return data_[idx];
  }

  inline vector<Datum*>& data() { return data_; }
  inline int vocabulary_size() { return vocabulary_size_; }

 private:

 private:
  vector<DataBatch*> data_batches_;
  vector<Datum*> data_;
  int vocabulary_size_;
};

} // namespace ditree

#endif
