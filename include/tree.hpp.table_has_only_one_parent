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

  void ConstructTableMetaInfo();

  void SampleVertexToSplit(vector<uint32>& vertex_to_split);
  // Return idx of the new_vertex
  uint32 AcceptSplitVertex(Vertex* new_vertex, const Vertex* parent_vertex_copy);
  void UpdateTreeStructAfterSplit();
  void UpdateStructTableAfterSplit();
  void ClearSplitRecords();

  void SampleVertexPairsToMerge(
      vector<pair<uint32, uint32> >& vertex_pair_to_merge);
  void AcceptMergeVertexes(const uint32 host_v_idx, const uint32 guest_v_idx);
  void UpdateTreeStructAfterMerge();
  void MergeTwoVector(const uint32 host_v_idx, const uint32 guest_v_idx);

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
  inline const vector<pair<uint32, uint32> >& vertex_split_records() const {
    return vertex_split_records_;
  }
  inline const vector<pair<uint32, uint32> >& vertex_merge_records() const {
    return vertex_merge_records_;
  }
  
 private: 

  void UpdateParentChildTableRelation(const uint32 parent_table_idx, 
      const uint32 child_table_idx, const int child_table_size);
  bool GetNextCandidateChildTable(const uint32 table_idx,
      IdxCntPair& child_table);
  void AllocateChildTable(const uint32 table_idx);

 private:
  TreeParameter param_;

  Vertex* root_;
  // pseudo parent of root node, with fixed params
  Vertex* root_parent_;
  map<uint32, Vertex*> vertexes_;

  // table_idx => { vertex idx governed by this table when spliting }
  map<uint32, set<uint32> > table_idx_governed_vertex_idxes_;
  // table_idx => { vertex idx in this table, governed by this table when merging }
  map<uint32, set<uint32> > table_idx_vertex_idxes_;
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

  // parent_idx => <child_idx, weight for child>
  vector<pair<uint32, uint32> > vertex_split_records_;
  // <vertex_idx_remained, vertex_idx_removed>
  vector<pair<uint32, uint32> > vertex_merge_records_;

  int client_id_;
  int thread_id_;
  int global_worker_id_;
  int tot_num_threads_;

  struct SortBySecondOfPair {
    bool operator() (const pair<uint32, uint32>& lhs,
        const pair<uint32, uint32>& rhs) {
      return (lhs.second < rhs.second) 
          || (lhs.second == rhs.second && lhs.first < rhs.first);
    }
  };
};

} // namespace ditree

#endif
