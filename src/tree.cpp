
#include "tree.hpp"

namespace ditree {

Tree::Tree(const TreeParameter& param, const int thread_id)
    : root_(NULL), root_parent_(NULL), thread_id_(thread_id) {
  Init(param);
}

Tree::Tree(const string& param_file, const int thread_id)
    : thread_id_(thread_id) {
  TreeParameter param;
  ReadProtoFromTextFile(param_file, &param);
  Init(param);
}

void Tree::Init(const TreeParameter& param) {
  ditree::Context& context = ditree::Context::Get();
  client_id_ = context.get_int32("client_id");
  if (client_id_ == 0 && thread_id_ == 0) {
    LOG(INFO) << "Init tree from parameters";
  }

  // Add vertexes
  for (int v_idx_i = 0; v_idx_i < param.vertexes_size(); ++v_idx_i) {
    const VertexParameter& vertex_param = param.vertexes(v_idx_i);
    // vertex index cannot be duplicated
    CHECK(vertexes_.find(vertex_param.index()) == vertexes_.end());
    vertexes_[vertex_param.index()] = new Vertex(vertex_param);
    if (vertex_param.root()) {
      CHECK(root_parent_ != NULL);
      root_ = vertexes_[vertex_param.index()];
      root_->set_parent(root_parent_);
    }
  }
  CHECK(root_ != NULL) << " Cannot find root.";
  // Connect the vertexes
  for (int v_idx_i = 0; v_idx_i < param.vertexes_size(); ++v_idx_i) {
    const VertexParameter& vertex_param = param.vertexes(v_idx_i);
    Vertex* vertex = vertexes_[vertex_param.index()];
    for (int c_idx_i = 0; c_idx_i < param.child_indexes_size(); ++c_idx_i) {
      CHECK(vertexes_.find(vertex_param.parent_index()) != vertexes_.end());
      vertex->add_child(vertexes_[vertex_param.child_indexes(c_idx_i)]);
    } 
  }
}

void Tree::UpdateParamTable(
    const UIntFloatMap& data_batch_n_new, 
    const UIntFloatMap& data_batch_n_old, 
    const map<uint32, UIntFloatMap>& data_batch_s_new,
    const map<uint32, UIntFloatMap>& data_batch_s_old) {
  BOOST_FOREACH(pair<const uint32, Vertex*>& ele, vertexes_) {
    uint32 vertex_id = ele.first;
#ifdef DEBUG
    CHECK(data_batch_n_new.find(vertex_id) != data_batch_n_new.end());
    CHECK(data_batch_n_old.find(vertex_id) != data_batch_n_old.end());
    CHECK(data_batch_s_new.find(vertex_id) != data_batch_s_new.end());
    CHECK(data_batch_s_old.find(vertex_id) != data_batch_s_old.end());
    CHECK_EQ(vertex_id, ele.second->idx());
#endif
    ele.second->UpdateParamTable(
        data_batch_n_new[vertex_id], data_batch_n_old[vertex_id],
        data_batch_s_new[vertex_id], data_batch_s_old[vertex_id]);
  }
}

void Tree::ReadParamTable() {
  for (int v_idx = 0; v_idx < vertexes_.size(); ++v_idx) {
    vertexes_[v_idx]->ReadParamTable();
  }
}

} // namespace ditree
