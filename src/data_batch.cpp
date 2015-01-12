
#include "data_batch.hpp"
#include "util.hpp"
#include <boost/foreach.hpp>

namespace ditree {

void DataBatch::UpdateSuffStatStruct(
    const vector<Triple>& vertex_split_records, 
    const vector<Triple>& vertex_merge_records) {
  if (Context::phase() == VI_AFTER_SPLIT) {
    UpdateSuffStatStructBySplit(vertex_split_records);
    UpdateSuffStatStructByMerge(vertex_merge_records);
  } else if (Context::phase() == VI_AFTER_MERGE) {
    UpdateSuffStatStructByMerge(vertex_merge_records);
    UpdateSuffStatStructBySplit(vertex_split_records);
  } else {
    LOG(FATAL) << "Phase must be VI_AFTER_SPLIT or VI_AFTER_MERGE != " 
        << Context::phase();
  }
}

void DataBatch::UpdateSuffStatStructBySplit(
    const vector<Triple>& vertex_split_records) { 
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
    n_[rec.y()] = ori_n * rec.w(). 
    n_[rec.x()] = ori_n * (1.0 - rec.w()). 
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
      CHECK(y_s.find(x_s_ele) != y_s.end());
#endif
      x_s_ele.second += y_s[x_s_ele.first];
    }
    s_.erase(rec.y());
  }
}

} // namespace ditree
