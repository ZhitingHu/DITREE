#ifndef DITREE_DATASET_HPP_
#define DITREE_DATASET_HPP_

#include "common.hpp"
#include "context.hpp"
#include "data_batch.hpp"
#include "datum.hpp"

namespace ditree {

class Dataset {
 public:
  explicit Dataset() {};
  
  DataBatch* GetNextDataBatch();
  void Restart(); 
 
  const inline Datum* datum(int idx) {
#ifdef DEBUG
    CHECK_LT(idx, data_.size());
#endif
    return data_[idx];
  }

  inline bool epoch_end() { return (iter_ >= data_batches_.size()); }
  inline int size() { return data_.size(); }
  inline int batch_num() { return data_batches_.size(); }
  inline vector<Datum*>& data() { return data_; }

  void Init(const string& filename);

 private:
  void ReadData(const string& filename);

 private:
  vector<DataBatch*> data_batches_;
  vector<Datum*> data_;
  // 
  int iter_;
  vector<int> data_batch_queue_;
};

} // namespace ditree

#endif
