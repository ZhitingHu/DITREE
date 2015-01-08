#ifndef DITREE_TREE_HPP_
#define DITREE_TREE_HPP_

#include "common.hpp"
#include "vertex.hpp"

namespace ditree {

class Tree {
 public:

  void SyncParameter();
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
  // pseudo parent of root node, with fixed parameter
  Vertex* root_parent_;
  map<uint32, Vertex*> vertexes_;

};

} // namespace ditree

#endif
