
#include "tree.hpp"
#include "common.hpp"
#include "context.hpp"
#include "io.hpp"
#include <algorithm>
#include <boost/foreach.hpp>

namespace ditree {

const uint32 Tree::kTempParentTableIdx = -1;

Tree::Tree(const TreeParameter& param, const int thread_id)
    : root_(NULL), root_parent_(NULL), max_table_idx_(-1),
    thread_id_(thread_id) {
  Init(param);
}

Tree::Tree(const string& param_file, const int thread_id)
    : thread_id_(thread_id) {
  TreeParameter param;
  ReadProtoFromTextFile(param_file, &param);
  Init(param);
}

void Tree::Init(const TreeParameter& param) {
  client_id_ = Context::get_int32("client_id");
  if (client_id_ == 0 && thread_id_ == 0) {
    LOG(INFO) << "Init tree from parameters";
  }

  param_ = param;
  // -1 for kTempParentTableIdx (-1)
  max_table_num_ = (1 << Context::get_int32("num_table_id_bits")) - 1;

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
  // Initialize depths
  root_->RecursSetDepth(0);
  // Contruct structure info
  ConstructTableMetaInfo();
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
  // Assume the mean vector has norm = 1
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
      ele.second->InitParam(0, root_parent_mean);
    }
    LOG(INFO) << "Init vertexes' mean done.";
  }
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

// Construct table_idx_governed_vertex_idxes_
// IMPORTANT: the result of the function must be deterministic 
//   given the current tree structure, esp., all threads 
//   produce exactly the same results
void Tree::ConstructTableMetaInfo() {
  // Count the number of governed vertexes of each table
  BOOST_FOREACH(UIntVertexPair& ele, vertexes_) {
    uint32 vertex_idx = ele.first;
    uint32 table_idx = Context::table_id(vertex_idx);
    table_idx_vertex_size_[table_idx]++;

    const vector<Vertex*> children = ele.second->children();
    if (children.size() > 0) {
      // Only check the first child, since one vertex's children 
      // have the same table id 
      uint32 child_table_idx = Context::table_id(children[0]->idx());
      if (child_table_idx != table_idx) {
        table_idx_governed_vertex_idxes_[child_table_idx].insert(vertex_idx);
        // Record the parent-child relation
        UpdateParentChildTableRelation(table_idx, child_table_idx, 0);
      } else {
        table_idx_governed_vertex_idxes_[table_idx].insert(vertex_idx);
      }
      ele.second->set_child_table_idx(child_table_idx);
    }
  }
  // Count child table sizes
  // Used for child allocation
  typedef pair<const uint32, IdxCntPairMutableMinHeap*> IdxChildSizesPair;
  BOOST_FOREACH(IdxChildSizesPair& ele, table_idx_child_table_sizes_) {
#ifdef DEBUG
    CHECK(table_idx_child_tables_.find(ele.first) 
        != table_idx_child_tables_.end());
#endif
    for (const auto child_table_idx : table_idx_child_tables_[ele.first]) {
#ifdef DEBUG
    CHECK(table_idx_governed_vertex_idxes_.find(child_table_idx) 
        != table_idx_governed_vertex_idxes_.end());
#endif
      const int child_table_size 
          = table_idx_governed_vertex_idxes_[child_table_idx].size();
      UpdateParentChildTableRelation(
          ele.first, child_table_idx, child_table_size);
    }
  }
  // Create empty tables so that as many threads as possible
  // will have at least one table allocated
  const int num_clients = Context::get_int32("num_clients");
  const int num_threads = Context::get_int32("num_app_threads");
  const int temp_max_table_num = min(num_clients * num_threads, max_table_num_);
  CHECK_GT(temp_max_table_num, 0);
  for (uint32 t_idx = max_table_idx_ + 1; t_idx < temp_max_table_num; ++t_idx) {
    table_idx_governed_vertex_idxes_[t_idx] = set<uint32>();
    // set the parent table as kTempParentTableIdx(-1)
    UpdateParentChildTableRelation(kTempParentTableIdx, t_idx, 0);
    table_idx_vertex_size_[t_idx] = 0;
  }
  // For vertexes with no children,
  //  allocate tables for their future children, may create new table
  // Allocate children to smallest table first, but garuantee that
  //  one table has only one parent table
  BOOST_FOREACH(UIntVertexPair& ele, vertexes_) {
    if (ele.second->children().size() > 0 || 
        ele.second->depth() >= Context::get_int32("max_depth")) {
      continue;
    }
    AllocateChildTable(ele.first);
  }
}

void Tree::UpdateParentChildTableRelation(const uint32 parent_table_idx,
    const uint32 child_table_idx, const int child_table_size) {
  IdxCntPairMutableMinHeap* child_table_idx_sizes;
  if (table_idx_child_table_sizes_[parent_table_idx] == NULL) {
    child_table_idx_sizes = GetIdxCntPairMutableMinHeap(max_table_num_);
    child_table_idx_sizes->push(IdxCntPair(child_table_idx, child_table_size));
    table_idx_child_table_sizes_[parent_table_idx] = child_table_idx_sizes;
  } else {
    child_table_idx_sizes = table_idx_child_table_sizes_[parent_table_idx];
    if (table_idx_child_tables_[parent_table_idx].find(child_table_idx)
        == table_idx_child_tables_[parent_table_idx].end()) {
      child_table_idx_sizes->push(IdxCntPair(child_table_idx, child_table_size));
    } else {
      child_table_idx_sizes->update(IdxCntPair(child_table_idx, child_table_size));
    }
  }
  table_idx_child_tables_[parent_table_idx].insert(child_table_idx);
}

bool Tree::GetNextCandidateChildTable(const uint32 table_idx,
    IdxCntPair& child_table) {
  // First, check empty tables under kTempParentTableIdx
  if (table_idx_child_tables_[kTempParentTableIdx].size() > 0) {
    child_table = table_idx_child_table_sizes_[kTempParentTableIdx]->top();
#ifdef DEBUG
    CHECK_EQ(child_table.second, 0);
#endif
    table_idx_child_table_sizes_[kTempParentTableIdx]->pop();
    table_idx_child_tables_[kTempParentTableIdx].erase(child_table.first);
    return true;
  } else if (table_idx_child_tables_[table_idx].size() > 0) {
    child_table = table_idx_child_table_sizes_[table_idx]->top();
    return true;
  } else {
    // No candidate child tables, need create a new table
    return false;
  }
}

void Tree::AllocateChildTable(const uint32 vertex_idx) {
  const uint32 table_idx = Context::table_id(vertex_idx);
  IdxCntPair child_table;
  if (!GetNextCandidateChildTable(table_idx, child_table) ||
      child_table.second >= Context::get_int32("max_size_per_table")) {
    // Create new table
    child_table.first = max_table_idx_;
    child_table.second = 1;
    table_idx_governed_vertex_idxes_[child_table.first].insert(vertex_idx);
    table_idx_vertex_size_[child_table.first] = 0;
    ++max_table_idx_;
  } else {
    // Use existing table
#ifdef DEBUG
    CHECK(table_idx_governed_vertex_idxes_.find(child_table.first)
        != table_idx_governed_vertex_idxes_.end());
#endif
    table_idx_governed_vertex_idxes_[child_table.first].insert(vertex_idx);
    child_table.second++;
#ifdef DEBUG
  CHECK_EQ(table_idx_governed_vertex_idxes_[child_table.first].size(),
      child_table.second);
#endif
  }
  UpdateParentChildTableRelation(table_idx, child_table.first, 
      child_table.second);
  vertexes_[vertex_idx]->set_child_table_idx(child_table.first);
}

//TODO: add more random factors
const vector<uint32>& Tree::SampleVertexToSplit() {
  vertex_split_records_.clear();
  vertexes_to_split_.clear();
  // to prevent duplication
  set<uint32> vertexes_set;
  const int max_num_split = Context::get_int32("max_split_per_table");
  // Sample on each table allocated to this thread
  const int step_size = Context::get_int32("num_clients") 
      * Context::get_int32("num_app_threads");
  uint32 table_idx = thread_id_;
  for (; table_idx <= max_table_idx_; table_idx += step_size) {
#ifdef DEBUG
    CHECK(table_idx_governed_vertex_idxes_.find(table_idx)
        != table_idx_governed_vertex_idxes_.end());
#endif
    const set<uint32>& governed_vertexes 
        = table_idx_governed_vertex_idxes_[table_idx];
    vector<int> vertex_idxes;
    FloatVec vertex_weights;
    for (uint32 v_idx : governed_vertexes) {
      vertex_idxes.push_back(v_idx);
#ifdef DEBUG
      CHECK(vertexes_.find(v_idx) != vertexes_.end());
#endif
      vertex_weights.push_back(vertexes_[v_idx]->n());
    } 
    // Sampling
    int num_split = 0; 
    while(num_split < min(max_num_split, (int)governed_vertexes.size())) {
      uint32 cand_v_idx = vertex_idxes[
          Context::randDiscrete(vertex_weights, 0, vertex_idxes.size())];
      if (vertexes_set.find(cand_v_idx) == vertexes_set.end()) {
        vertexes_set.insert(cand_v_idx);
        vertexes_to_split_.push_back(cand_v_idx);
        ++num_split;
      }
    }
  } // end of tables
  return vertexes_to_split_;
}

void Tree::AcceptSplitVertex(Vertex* new_vertex,
    const Vertex* parent_vertex_copy) {
  /// Allocate vertex idx
  uint32 child_table_idx = parent_vertex_copy->child_table_idx();
  int size = table_idx_vertex_size_[child_table_idx];
  uint32 child_idx = Context::make_vertex_id(child_table_idx, size);
  new_vertex->set_idx(child_idx);
  table_idx_vertex_size_[child_table_idx]++;
  /// Add to the tree structure
#ifdef DEBUG
  CHECK(vertexes_.find(child_idx) == vertexes_.end());
#endif
  vertexes_[child_idx] = new_vertex;
  Vertex* parent_vertex = vertexes_[parent_vertex_copy->idx()];
  parent_vertex->add_child(new_vertex);
  /// Allocate child table for future children 
  if (new_vertex->depth() < Context::get_int32("max_depth")) {
    AllocateChildTable(child_idx);
  }
  ///
  /// Update param table
  // child's param
  petuum::Table<float>* param_table = Context::param_table();
  const FloatVec& new_vertex_s = new_vertex->s();
  petuum::DenseUpdateBatch<float> update_batch(
      0, kColIdxParamTableSStart + new_vertex_s.size());
  update_batch[kColIdxParamTableN] = new_vertex->n();
  for (int s_idx = 0; s_idx < new_vertex_s.size(); ++s_idx) {
    update_batch[kColIdxParamTableSStart + s_idx] = new_vertex_s[s_idx];
  }
  param_table->DenseBatchInc(child_idx, update_batch);
  // parent's param
  update_batch[kColIdxParamTableN] = parent_vertex->n() - parent_vertex->n();
  const FloatVec& parent_s = parent_vertex->s(); 
  const FloatVec& parent_copy_s = parent_vertex_copy->s(); 
  for (int s_idx = 0; s_idx < parent_s.size(); ++s_idx) {
    update_batch[kColIdxParamTableSStart + s_idx] 
        = parent_copy_s[s_idx] - parent_s[s_idx];
  }
  param_table->DenseBatchInc(parent_vertex->idx(), update_batch);

  // 
  vertex_split_records_.push_back(make_pair(parent_vertex->idx(), child_idx));
}

float Tree::ComputeELBO() {
  float elbo = 0;
  BOOST_FOREACH(UIntVertexPair& ele, vertexes_) {
    elbo += vertexes_[ele.first]->ComputeELBO();
  }
  return elbo;
}

} // namespace ditree
