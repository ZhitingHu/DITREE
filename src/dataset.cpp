
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

void Dataset::Init(const string& filename) {
  ReadData(filename);
  CHECK_GT(data_.size(), 0);

  // initialize data batches
  ditree::Context& context = ditree::Context::Get();
  int batch_size = context.get_int32("batch_size");
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
}

void Dataset::ReadData(const string& filename) {
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
      input.read((char*)&word_weight, sizeof(float));
      datum->AddWord(word_id, word_weight);
    }
    data_.push_back(datum);
    counter++;
    if (counter % (num_doc / 5) == 0) {
      LOG(INFO) << "Finish reading " << (counter / (num_doc / 5)) 
          << "/5 docs"; 
    }
  }
  input.close();
}

DataBatch* Dataset::GetNextDataBatch() {
  LOG(INFO) << "get next batch iter_=" << iter_ << " size " << data_batches_.size();
  boost::mutex::scoped_lock lock(data_access_mutex);
  DataBatch* next_batch = data_batches_[iter_++];
  if (iter_ >= data_batches_.size()) {
    iter_ = 0;
  }
  return next_batch;
}

} // namespace ditree