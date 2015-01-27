#ifndef DITREE_DATASET_HPP_
#define DITREE_DATASET_HPP_

#include "common.hpp"
#include "context.hpp"
#include "data_batch.hpp"
#include "datum.hpp"
#include <atomic>

namespace ditree {

class Dataset {
 public:
  explicit Dataset() {};
  
  DataBatch* GetNextDataBatch();
  DataBatch* GetRandomDataBatch();
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
  inline void set_need_restart() { need_restart_ = true; }
  inline const map<int, string>& vocab() { return vocab_; }

  void Init(const string& doc_file, const string& vocab_file = "",
      const bool train = false);

  void RecordLastIterBeforeMerge() {
    last_iter_before_merge_ = iter_;
  };
  DataBatch* GetNextBatchToApplyMerge();

 private:
  void ReadData(const string& filename);
  void ReadVocab(const string& filename);

 private:
  vector<DataBatch*> data_batches_;
  vector<Datum*> data_;
 
  map<int, string> vocab_;

  // 
  int iter_;
  vector<int> data_batch_queue_;

  bool need_restart_;

  std::atomic<int> last_iter_before_merge_;
  int iter_for_merge_;
};

} // namespace ditree

#endif
