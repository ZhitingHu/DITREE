
#include "tree.hpp"
#include "common.hpp"
#include "context.hpp"
#include "io.hpp"
#include <boost/foreach.hpp>

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
 
  param_ = param;
  root_parent_ = new Vertex();

  // Add vertexes
  for (int v_idx_i = 0; v_idx_i < param.vertexes_size(); ++v_idx_i) {
    const VertexParameter& vertex_param = param.vertexes(v_idx_i);
    // vertex index cannot be duplicated
    CHECK(vertexes_.find(vertex_param.index()) == vertexes_.end()) 
        << " Vertex index duplicates " << vertex_param.index();
    vertexes_[vertex_param.index()] = new Vertex(vertex_param, param_);
    if (vertex_param.root()) {
      CHECK(root_parent_ != NULL);
      root_ = vertexes_[vertex_param.index()];
      root_->set_root();
      root_->set_parent(root_parent_);
    }
  }
  CHECK(root_ != NULL) << " Cannot find root.";
  // Connect the vertexes
  for (int v_idx_i = 0; v_idx_i < param.vertexes_size(); ++v_idx_i) {
    const VertexParameter& vertex_param = param.vertexes(v_idx_i);
    Vertex* vertex = vertexes_[vertex_param.index()];
    for (int c_idx_i = 0; c_idx_i < vertex_param.child_indexes_size(); 
        ++c_idx_i) {
      int child_index = vertex_param.child_indexes(c_idx_i);
      CHECK(vertexes_.find(child_index) != vertexes_.end()) 
          << " Child vertex not find " << child_index;
      vertex->add_child(vertexes_[child_index]);
    } 
  }
}

// Init mean_ of each vertex
void Tree::InitParam() {
  if (client_id_ == 0 && thread_id_ == 0) {
    LOG(INFO) << "Init vertexes' mean.";
  }
#ifdef DEBUG
  CHECK(root_parent_ != NULL);
#endif
  FloatVec& root_parent_mean = root_parent_->mutable_mean();
  root_parent_mean.clear();
  // Initialize pesudo parent of root_
  ditree::Context& context = ditree::Context::Get();
  const string& mean_file = context.get_string("mean");
  fstream input(mean_file.c_str(), ios::in);
  CHECK(input.is_open()) << "File not found: " << mean_file;
  float weight;
  while (input >> weight) {
    root_parent_mean.push_back(weight);
  }
  input.close();
  CHECK_EQ(root_parent_mean.size(), Context::vocab_size());

  if (client_id_ == 0 && thread_id_ == 0) {
    // Use the data mean to initialize
    BOOST_FOREACH(UIntVertexPair& ele, vertexes_) {
      ele.second->InitParamTable(0, root_parent_mean);
    }
    LOG(INFO) << "Init vertexes' mean done.";
  }
  //petuum::PSTableGroup::GlobalBarrier();
}

void Tree::UpdateParamTable(
    const UIntFloatMap& data_batch_n_new, 
    const UIntFloatMap& data_batch_n_old, 
    const map<uint32, UIntFloatMap>& data_batch_s_new,
    const map<uint32, UIntFloatMap>& data_batch_s_old) {
  BOOST_FOREACH(UIntVertexPair& ele, vertexes_) {
    uint32 vertex_id = ele.first;
#ifdef DEBUG
    CHECK(data_batch_n_new.find(vertex_id) != data_batch_n_new.end());
    CHECK(data_batch_n_old.find(vertex_id) != data_batch_n_old.end());
    CHECK(data_batch_s_new.find(vertex_id) != data_batch_s_new.end());
    CHECK(data_batch_s_old.find(vertex_id) != data_batch_s_old.end());
    CHECK_EQ(vertex_id, ele.second->idx());
#endif
    ele.second->UpdateParamTable(
        data_batch_n_new.find(vertex_id)->second, 
        data_batch_n_old.find(vertex_id)->second,
        data_batch_s_new.find(vertex_id)->second, 
        data_batch_s_old.find(vertex_id)->second);
  }
}

void Tree::ReadParamTable() {
  BOOST_FOREACH(UIntVertexPair& ele, vertexes_) {
    vertexes_[ele.first]->ReadParamTable();
  }
}

float Tree::ComputeELBO() {
  float elbo = 0;
  BOOST_FOREACH(UIntVertexPair& ele, vertexes_) {
    elbo += vertexes_[ele.first]->ComputeELBO();
  }
  return elbo;
}

} // namespace ditree
