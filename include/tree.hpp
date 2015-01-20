#ifndef DITREE_TREE_HPP_
#define DITREE_TREE_HPP_

#include "common.hpp"
#include "vertex.hpp"
#include "util.hpp"
#include "mutable_heap.hpp"
#include "data_batch.hpp"
#include "dataset.hpp"
#include "proto/ditree.pb.h"

namespace ditree {

class Tree {
 public:
  explicit Tree(const TreeParameter& param, const int thread_id);
  explicit Tree(const string& param_file, const int thread_id);

  void Init(const TreeParameter& param);
  void InitParam();

  void UpdateParamTable(
      const UIntFloatMap& data_batch_n_new, 
      const UIntFloatMap& data_batch_n_old, 
      const map<uint32, UIntFloatMap>& data_batch_s_new,
      const map<uint32, UIntFloatMap>& data_batch_s_old);

  void ReadParamTable();

  void SyncStructure();

  void ConstructTableMetaInfo();

  // TODO
  const vector<uint32>& SampleVertexToSplit();
  void SplitVertex(vector<DataBatch>& split_reference_data,
      Dataset* train_data);

  float ComputeELBO();

  // number of nodes
  inline int size() { return vertexes_.size(); }
  inline Vertex* root() { return root_; }

  inline Vertex* vertex(uint32 idx) {
#ifdef DEBUG
    CHECK(vertexes_.find(idx) != vertexes_.end());
#endif
    return vertexes_[idx];
  }
  inline const map<uint32, Vertex*>& vertexes() const {
    return vertexes_;
  }
  inline const vector<Triple>& vertex_split_records() const {
    return vertex_split_records_;
  }
  inline const vector<Triple>& vertex_merge_records() const {
    return vertex_merge_records_;
  }
  
 private: 

  void UpdateParentChildTableRelation(const uint32 parent_table_idx, 
      const uint32 child_table_idx, const int child_table_size);
  bool GetNextCandidateChildTable(const uint32 table_idx,
      IdxCntPair& child_table);

 private:
  TreeParameter param_;

  Vertex* root_;
  // pseudo parent of root node, with fixed params
  Vertex* root_parent_;
  map<uint32, Vertex*> vertexes_;

  // table_idx => { vertex idx governed by this table when spliting }
  map<uint32, set<uint32> > table_idx_governed_vertex_idxes_;
  // table_idx => { parent vertex idx }
  //Note: can be inferred from parent_, do not need storing in PS
  //map<uint32, set<uint32> > table_idx_parent_vertex_idxes_;

  // parent table idx => < child table (idx, size) >
  // Note: one table is garuanteed to have only one parent table
  map<uint32, IdxCntPairMutableMinHeap*> table_idx_child_table_sizes_; 
  // used as the handler of the min-heap
  map<uint32, set<uint32> > table_idx_child_tables_;
  // 
  static const uint32 kTempParentTableIdx;
  int max_table_num_;
  uint32 max_table_idx_;

  ///
  vector<uint32> vertexes_to_split_;
  // parent_idx => <child_idx, weight for child>
  vector<Triple> vertex_split_records_;
  // <vertex_idx_remained, vertex_idx_removed>
  vector<Triple> vertex_merge_records_;

  int client_id_;
  int thread_id_;

  //struct SortByFirstOfPair {
  //  bool operator() (const IdxCntPair& lhs, const IdxCntPair& rhs) {
  //    return (lhs.first < rhs.first)
  //        || (lhs.first == rhs.first && lhs.second < rhs.second);
  //  }
  //};
  //struct SortBySecondOfPair {
  //  bool operator() (const IdxCntPair& lhs, const IdxCntPair& rhs) {
  //    return (lhs.second < rhs.second) 
  //        || (lhs.second == rhs.second && lhs.first < rhs.first);
  //  }
  //};
};

} // namespace ditree

#endif
