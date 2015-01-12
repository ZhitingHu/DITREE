#ifndef DITREE_TREE_HPP_
#define DITREE_TREE_HPP_

#include "common.hpp"
#include "vertex.hpp"
#include "util.hpp"
#include "proto/ditree.pb.h"

namespace ditree {

class Tree {
 public:
  explicit Tree(const TreeParameter& param, const int thread_id);
  explicit Tree(const string& param_file, const int thread_id);

  void Init(const TreeParameter& param);

  void UpdateParamTable(
      const UIntFloatMap& data_batch_n_new, 
      const UIntFloatMap& data_batch_n_old, 
      const map<uint32, UIntFloatMap>& data_batch_s_new,
      const map<uint32, UIntFloatMap>& data_batch_s_old);

  void ReadParamTable();

  void SyncStructure();

  // number of nodes
  inline int size() { return vertexes_.size(); }

  inline Vertex* vertex(uint32 idx) {
#ifdef DEBUG
    CHECK(vertexes_.find(idx) != vertexes_.end());
#endif
    return vertexes_[idx];
  }

 private: 

 private:
  Vertex* root_;
  // pseudo parent of root node, with fixed params
  Vertex* root_parent_;
  map<uint32, Vertex*> vertexes_;

  ///
  // parent_idx => <child_idx, weight for child>
  vector<Triple> vertex_split_records_;
  // <vertex_idx_remained, vertex_idx_removed>
  vector<Triple> vertex_merge_records_;

  int client_id_;
  int thread_id_;

};

} // namespace ditree

#endif
