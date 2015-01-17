
#include "data_batch.hpp"
#include "util.hpp"
#include <boost/foreach.hpp>

namespace ditree {

void DataBatch::UpdateSuffStatStruct(const Tree* tree) {
  if (Context::phase() == Context::Phase::kVIAfterSplit) {
    UpdateSuffStatStructBySplit(tree->vertex_split_records());
    UpdateSuffStatStructByMerge(tree->vertex_merge_records());
  } else if (Context::phase() == Context::Phase::kVIAfterMerge) {
    UpdateSuffStatStructByMerge(tree->vertex_merge_records());
    UpdateSuffStatStructBySplit(tree->vertex_split_records());
  } else {
    LOG(FATAL) << "Phase must be kVIAfterSplit or kVIAfterMerge != " 
        << Context::phase();
  }
}

void DataBatch::InitSuffStatStruct(const Tree* tree, 
    const vector<Datum*>& data) {
  LOG(INFO) << "iNIT";
  UIntFloatMap batch_words;
  for (int i = 0; i < size_; ++i) {
    const UIntFloatMap& datum_words 
        = data[data_idx_begin_ + i]->data();
    BOOST_FOREACH(const UIntFloatPair& ele, datum_words) {
      batch_words[ele.first] = 0;
    }
  }
  n_.clear();
  s_.clear();
  const map<uint32, Vertex*>& vertexes = tree->vertexes(); 
  BOOST_FOREACH(const UIntVertexPair& ele, vertexes) {
    n_[ele.first] = 0;
    s_[ele.first] = batch_words;
  }
}

void DataBatch::UpdateSuffStatStructBySplit(
    const vector<Triple>& vertex_split_records) { 

  LOG(INFO) <<"iM HEJRE";
  for (int idx = 0; idx < vertex_split_records.size(); ++idx) {
    const Triple& rec = vertex_split_records[idx];
#ifdef DEBUG
    CHECK(n_.find(rec.x()) != n_.end());
    CHECK(n_.find(rec.y()) == n_.end());
    CHECK(s_.find(rec.x()) != s_.end());
    CHECK(s_.find(rec.y()) == s_.end());
#endif
    // split n_
    const float ori_n = n_[rec.x()];
    n_[rec.y()] = ori_n * rec.w();
    n_[rec.x()] = ori_n * (1.0 - rec.w());
    // split s_
    UIntFloatMap& y_s = s_[rec.y()];
    UIntFloatMap& x_s = s_[rec.x()];
    y_s = x_s;
    BOOST_FOREACH(UIntFloatPair& x_s_ele, x_s) {
      y_s[x_s_ele.first] = x_s_ele.second * rec.w();
      x_s_ele.second = x_s_ele.second * (1.0 - rec.w());
    }
  }
}

void DataBatch::UpdateSuffStatStructByMerge(
    const vector<Triple>& vertex_merge_records) { 
  for (int idx = 0; idx < vertex_merge_records.size(); ++idx) {
    const Triple& rec = vertex_merge_records[idx];
#ifdef DEBUG
    CHECK(n_.find(rec.x()) != n_.end());
    CHECK(n_.find(rec.y()) != n_.end());
    CHECK(s_.find(rec.x()) != s_.end());
    CHECK(s_.find(rec.y()) != s_.end());
#endif
    // merge n_
    n_[rec.x()] = n_[rec.x()] + n_[rec.y()];
    n_.erase(rec.y());
    // merge s_ 
    UIntFloatMap& y_s = s_[rec.y()];
    UIntFloatMap& x_s = s_[rec.x()];
    BOOST_FOREACH(UIntFloatPair& x_s_ele, x_s) {
#ifdef DEBUG
      CHECK(y_s.find(x_s_ele.first) != y_s.end());
#endif
      x_s_ele.second += y_s[x_s_ele.first];
    }
    s_.erase(rec.y());
  }
}

} // namespace ditree
