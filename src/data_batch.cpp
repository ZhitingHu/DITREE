
#include "data_batch.hpp"
#include "util.hpp"
#include <boost/foreach.hpp>

namespace ditree {

void DataBatch::UpdateSuffStatStruct(const Tree* tree,
    const Context::Phase phase) {
  //if (phase == Context::Phase::kVIAfterSplit) {
  //  UpdateSuffStatStructByMerge(tree->vertex_merge_records());
  //  UpdateSuffStatStructBySplit(tree->vertex_split_records());
  //} else if (phase == Context::Phase::kVIAfterMerge) {
  //  UpdateSuffStatStructBySplit(tree->vertex_split_records());
  //  UpdateSuffStatStructByMerge(tree->vertex_merge_records());
  //} else {
  //  LOG(FATAL) << "Phase must be kVIAfterSplit or kVIAfterMerge != " 
  //      << phase;
  //}
  UpdateSuffStatStructBySplit(tree->vertex_split_records());
  if (phase == Context::Phase::kVIAfterMerge) {
    UpdateSuffStatStructByMerge(tree->vertex_merge_records());
  }
}

void DataBatch::InitSuffStatStruct(const Tree* tree, 
    const vector<Datum*>& data) {
  UIntFloatMap batch_words;
  for (int i = 0; i < size_; ++i) {
    const UIntFloatMap& datum_words 
        = data[data_idx_begin_ + i]->data();
    BOOST_FOREACH(const UIntFloatPair& ele, datum_words) {
      batch_words[ele.first] = 0;
    }
  }
  LOG(INFO) << "SIZE:" << batch_words.size();

  n_.clear();
  s_.clear();
  const map<uint32, Vertex*>& vertexes = tree->vertexes(); 
  BOOST_FOREACH(const UIntVertexPair& ele, vertexes) {
    n_[ele.first] = 0;
    s_[ele.first] = batch_words;
  }
}

void DataBatch::UpdateSuffStatStructBySplit(
    const vector<pair<uint32, uint32> >& vertex_split_records) { 
  for (const auto& rec : vertex_split_records) {
    const uint32 parent_idx = rec.first;
    const uint32 child_idx = rec.second;
#ifdef DEBUG
    CHECK(n_.find(parent_idx) != n_.end());
    CHECK(n_.find(child_idx) == n_.end())
        << "child_idx " << child_idx << " already exists. n_.size="
        << n_.size();
    CHECK(s_.find(parent_idx) != s_.end());
    CHECK(s_.find(child_idx) == s_.end());
#endif
    n_[child_idx] = 0;
    s_[child_idx] = s_[parent_idx];
    ResetUIntFloatMap(s_[child_idx]);
  }
}

void DataBatch::UpdateSuffStatStructByMerge(
    const vector<pair<uint32, uint32> >& vertex_merge_records) {
  
  //LOG(ERROR) << "Merging struct " << data_idx_begin_;
 
  for (const auto& rec : vertex_merge_records) {
    const uint32 merging_idx = rec.first;
    const uint32 merged_idx = rec.second;
#ifdef DEBUG
    CHECK(n_.find(merging_idx) != n_.end());
    CHECK(n_.find(merged_idx) != n_.end()) 
        << "merged_idx=" << merged_idx << ", n_.size=" << n_.size();
    CHECK(s_.find(merging_idx) != s_.end());
    CHECK(s_.find(merged_idx) != s_.end());
#endif
    // merge n_
    n_[merging_idx] += n_[merged_idx];
    n_.erase(merged_idx);
    // merge s_ 
    AccumUIntFloatMap(s_[merged_idx], 1, s_[merging_idx]);
    s_.erase(merged_idx);
  }
}

} // namespace ditree
