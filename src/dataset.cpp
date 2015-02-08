
#include "dataset.hpp"
#include "data_batch.hpp"
#include "util.hpp"
#include "common.hpp"
#include <boost/foreach.hpp>
#include <fstream>
#include <algorithm>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

namespace ditree {

boost::mutex data_access_mutex;

void Dataset::Init(const string& doc_file, const string& vocab_file,
    const bool train) {
  if (vocab_file.size()) {
    ReadVocab(vocab_file);
  }
  ReadData(doc_file);
  CHECK_GT(data_.size(), 0);

  // initialize data batches
  ditree::Context& context = ditree::Context::Get();
  int batch_size = 0;
  if (train) { 
    batch_size = context.get_int32("train_batch_size");
  } else {
    batch_size = context.get_int32("test_batch_size");
  }
  int num_batches = (data_.size() + batch_size - 1) / batch_size;
  CHECK_GT(num_batches, 0);
  data_batches_.resize(num_batches);
  int data_idx_begin = 0, db_idx;
  for (db_idx = 0; db_idx < num_batches - 1; ++db_idx) {
    data_batches_[db_idx] = new DataBatch(data_idx_begin, batch_size);
    data_idx_begin += batch_size;
  }
  data_batches_[db_idx] 
      = new DataBatch(data_idx_begin, data_.size() - data_idx_begin);

  //TODO
  LOG(INFO) << "Size of last batch: " << data_.size() - data_idx_begin;

  iter_ = 0;
  data_batch_queue_.resize(num_batches);
  for (db_idx = 0; db_idx < num_batches; ++db_idx) {
    data_batch_queue_[db_idx] = db_idx;
  }
  std::random_shuffle(data_batch_queue_.begin(), data_batch_queue_.end());

  need_restart_ = false;
  last_iter_before_merge_ = 0;
  iter_for_merge_ = 0;
}

void Dataset::ReadData(const string& filename) {
  //TODO
  //FloatVec tot_word(Context::vocab_size());

  fstream input(filename.c_str(), ios::in | ios::binary);
  CHECK(input.is_open()) << "File not found: " << filename;
  int num_doc, doc_len, word_id;
  float word_weight;
  int counter = 0;
  input.read((char*)&num_doc, sizeof(int)); 
  LOG(INFO) << "Total number of docs: " << num_doc;
  for (int d_idx = 0; d_idx < num_doc; ++d_idx) {
    Datum* datum = new Datum();
    // format: doc_len [word_id word_cnt]+
    input.read((char*)&doc_len, sizeof(int));
    for (int w_idx = 0; w_idx < doc_len; ++w_idx) {
      input.read((char*)&word_id, sizeof(int));
#ifdef DEBUG
      CHECK_LT(word_id, Context::vocab_size());
#endif
      input.read((char*)&word_weight, sizeof(float));
      datum->AddWord(word_id, word_weight);
    
      //TODO 
      //tot_word[word_id] += word_weight;
    }
    data_.push_back(datum);
    counter++;
    if (counter % (num_doc / 5) == 0) {
      LOG(INFO) << "Finish reading " << (counter / (num_doc / 5)) 
          << "/5 docs"; 
    }
  }
  input.close();

  //TODO 
  //ostringstream oss;
  //oss << "============================\n";
  //for (int i=0; i<tot_word.size(); ++i) {
  //  oss << tot_word[i] << " ";
  //}
  //oss << "\n";
  //LOG(INFO) << oss.str();
}

void Dataset::ReadVocab(const string& filename) {
  LOG(INFO) << "Read vocab " << filename;
  vocab_.clear();

  fstream input(filename.c_str(), ios::in);
  CHECK(input.is_open()) << "File not found: " << filename;
  int word_id = 0;
  string word_str;
  while (getline(input, word_str)) {
    vocab_[word_id] = word_str;
    ++word_id;
  }
  input.close();
  CHECK_EQ(vocab_.size(), Context::vocab_size());
}

DataBatch* Dataset::GetNextDataBatch() {
  boost::mutex::scoped_lock lock(data_access_mutex);
  LOG(INFO) << "Get next batch iter_=" << iter_ 
    << " " << data_batch_queue_[iter_] << " #batches " << data_batches_.size();
  if (iter_ < data_batches_.size()) {
    DataBatch* next_batch = data_batches_[data_batch_queue_[iter_]];
    ++iter_;
    return next_batch;
  } else {
    return NULL;
  }
}

DataBatch* Dataset::GetRandomDataBatch() {
  int db_idx = Context::randInt() % data_batches_.size();
  DataBatch* next_batch = data_batches_[db_idx];
  return next_batch;
}

DataBatch* Dataset::GetNextBatchToApplyMerge() {
  boost::mutex::scoped_lock lock(data_access_mutex);

  //LOG(ERROR) << "iter_for_merge_ " << iter_for_merge_ 
  //    << " " << last_iter_before_merge_;

  if (iter_for_merge_ < last_iter_before_merge_) {
    DataBatch* next_batch = data_batches_[data_batch_queue_[iter_for_merge_]];
    ++iter_for_merge_;
    return next_batch;
  } else {
    return NULL;
  }
}

void Dataset::Restart() {
  boost::mutex::scoped_lock lock(data_access_mutex);
  if (need_restart_) {
    LOG(INFO) << "Restart!!! ";
    need_restart_ = false;
    // Only one thread on each client will execute this code
    iter_ = 0;
    std::random_shuffle(data_batch_queue_.begin(), data_batch_queue_.end());
    iter_for_merge_ = 0;
  }
}

} // namespace ditree
