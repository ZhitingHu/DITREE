
#include "tree.hpp"
#include "common.hpp"
#include "context.hpp"
#include "io.hpp"
#include <algorithm>
#include <boost/foreach.hpp>

namespace ditree {

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

  int num_threads = Context::get_int32("num_app_threads");
  global_worker_id_ = client_id_ * num_threads + thread_id_; 
  tot_num_threads_ = Context::get_int32("num_clients") * num_threads;

  param_ = param;
  max_table_num_ = (1 << Context::get_int32("num_table_id_bits"));
  table_idx_governed_size_ = GetIdxCntPairMutableMinHeap(max_table_num_); 

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
  CHECK(root_ != NULL) << "Cannot find root.";
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
//
// The only garuantee of the table organization is: all the 
//   children of a vertex is in the same table
void Tree::ConstructTableMetaInfo() {
  // Count the number of governed vertexes of each table
  BOOST_FOREACH(UIntVertexPair& ele, vertexes_) {
    uint32 vertex_idx = ele.first;
    uint32 table_idx = Context::table_id(vertex_idx);
    // insert tables
    table_idx_vertex_idxes_[table_idx].insert(vertex_idx);
    table_idx_governed_vertex_idxes_[table_idx];

    const vector<Vertex*> children = ele.second->children();
    if (children.size() > 0) {
      // Only check the first child, since one vertex's children 
      // have the same table id 
      uint32 child_table_idx = Context::table_id(children[0]->idx());
      table_idx_governed_vertex_idxes_[child_table_idx].insert(vertex_idx);
      ele.second->set_child_table_idx(child_table_idx);
    }
    max_table_idx_ = max(max_table_idx_, table_idx);

    LOG(INFO) << "MAX_table_idx_ " << max_table_idx_ << " " 
        << table_idx << " " << client_id_ << " " << thread_id_;
  }
  // Table idx is expected to be continuous
  CHECK_EQ(max_table_idx_ + 1, table_idx_vertex_idxes_.size());

  // Count table governing size
  // Used for child allocation
  for (const auto ele : table_idx_governed_vertex_idxes_) {
    table_idx_governed_size_->push(IdxCntPair(ele.first, ele.second.size()));
  }

  // Create empty tables so that as many threads as possible
  // will have at least one table allocated
  const int temp_max_table_num = min(tot_num_threads_, max_table_num_);
  // TODO
  LOG(INFO) << "temp_max_table_num " << temp_max_table_num;
  CHECK_GT(temp_max_table_num, 0);
  for (uint32 t_idx = max_table_idx_ + 1; t_idx < temp_max_table_num; ++t_idx) {
    LOG(INFO) << "CREATE EMPTY " << t_idx;
    table_idx_governed_vertex_idxes_[t_idx] = set<uint32>();
    table_idx_vertex_idxes_[t_idx] = set<uint32>();
    table_idx_governed_size_->push(IdxCntPair(t_idx, 0));
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

  // TODO
  LOG(INFO) << "Meta table info " << client_id_ << " " << thread_id_;
  for (auto& t : table_idx_vertex_idxes_ ) {
    LOG(INFO) << "v " << t.first << " " << t.second.size();
    LOG(INFO) << "g " << t.first << " " << table_idx_governed_vertex_idxes_[t.first].size();
  }
  LOG(INFO) << "MAX_TABLE_IDX_ " << max_table_idx_;
}

//void Tree::UpdateParentChildTableRelation(const uint32 parent_table_idx,
//    const uint32 child_table_idx, const int child_table_size) {
//  LOG(INFO) << "Update Parent Child table relation " << parent_table_idx << " " 
//      << child_table_idx << " " << child_table_size; 
//  IdxCntPairMutableMinHeap* child_table_idx_sizes;
//  if (table_idx_child_table_sizes_[parent_table_idx] == NULL) {
//    child_table_idx_sizes = GetIdxCntPairMutableMinHeap(max_table_num_);
//    child_table_idx_sizes->push(IdxCntPair(child_table_idx, child_table_size));
//    table_idx_child_table_sizes_[parent_table_idx] = child_table_idx_sizes;
//  } else {
//    child_table_idx_sizes = table_idx_child_table_sizes_[parent_table_idx];
//    if (table_idx_child_tables_[parent_table_idx].find(child_table_idx)
//        == table_idx_child_tables_[parent_table_idx].end()) {
//      child_table_idx_sizes->push(IdxCntPair(child_table_idx, child_table_size));
//    } else {
//      child_table_idx_sizes->update(IdxCntPair(child_table_idx, child_table_size));
//    }
//  }
//  table_idx_child_tables_[parent_table_idx].insert(child_table_idx);
//}
//
//bool Tree::GetNextCandidateChildTable(const uint32 table_idx,
//    IdxCntPair& child_table) {
//  LOG(INFO) << "gETNEX " << table_idx;
//  // First, check empty tables under kTempParentTableIdx
//  if (table_idx_child_tables_[kTempParentTableIdx].size() > 0) {
//    LOG(INFO) << "HAVE empty";
//    child_table = table_idx_child_table_sizes_[kTempParentTableIdx]->top();
//#ifdef DEBUG
//    CHECK_EQ(child_table.second, 0);
//#endif
//    table_idx_child_table_sizes_[kTempParentTableIdx]->pop();
//    table_idx_child_tables_[kTempParentTableIdx].erase(child_table.first);
//    return true;
//  } else if (table_idx_child_tables_[table_idx].size() > 0) {
//    LOG(INFO) << "HERe";
//    child_table = table_idx_child_table_sizes_[table_idx]->top();
//    return true;
//  } else {
//    LOG(INFO) << "HERe 1";
//    // No candidate child tables, need create a new table
//    return false;
//  }
//}
//
//void Tree::AllocateChildTable(const uint32 vertex_idx) {
//  LOG(INFO) << "Alloc child table for " << vertex_idx;
//  const uint32 table_idx = Context::table_id(vertex_idx);
//  IdxCntPair child_table;
//  if (!GetNextCandidateChildTable(table_idx, child_table) ||
//      child_table.second >= Context::get_int32("max_size_per_table")) {
//    LOG(INFO) << "Create table for child table for " << vertex_idx;
//    // Create new table
//    child_table.first = max_table_idx_;
//    child_table.second = 1;
//    table_idx_governed_vertex_idxes_[child_table.first].insert(vertex_idx);
//    table_idx_vertex_idxes_[child_table.first] = set<uint32>();
//    UpdateParentChildTableRelation(child_table.first, child_table.first, 
//      child_table.second);
//    ++max_table_idx_;
//  } else {
//    // Use existing table
//#ifdef DEBUG
//    CHECK(table_idx_governed_vertex_idxes_.find(child_table.first)
//        != table_idx_governed_vertex_idxes_.end());
//#endif
//    LOG(INFO) << table_idx_governed_vertex_idxes_[child_table.first].size()
//        << " " <<  child_table.second << " v_idx: " << vertex_idx << " " << child_table.first;
//    for (uint32 ele : table_idx_governed_vertex_idxes_[child_table.first]) {
//      LOG(INFO) << ele;
//    }
//    table_idx_governed_vertex_idxes_[child_table.first].insert(vertex_idx);
//    child_table.second++;
//#ifdef DEBUG
//    CHECK_EQ(table_idx_governed_vertex_idxes_[child_table.first].size(),
//        child_table.second);
//#endif
//  }
//  UpdateParentChildTableRelation(table_idx, child_table.first, 
//      child_table.second);
//  vertexes_[vertex_idx]->set_child_table_idx(child_table.first);
//}

void Tree::AllocateChildTable(const uint32 vertex_idx) {
  LOG(INFO) << "Alloc child table for " << vertex_idx;
#ifdef DEBUG
  CHECK_GT(table_idx_governed_size_->size(), 0);
#endif
  IdxCntPair child_table = table_idx_governed_size_->top();
  if (child_table.second >= Context::get_int32("max_size_per_table")) {
    LOG(INFO) << "Create table for children of " << vertex_idx;
    // Create new table
    child_table.first = max_table_idx_;
    child_table.second = 1;
    table_idx_governed_vertex_idxes_[child_table.first].insert(vertex_idx);
    table_idx_governed_size_->push(
        IdxCntPair(child_table.first, child_table.second));
    table_idx_vertex_idxes_[child_table.first] = set<uint32>();
    ++max_table_idx_;
  } else {
    // Use existing table
#ifdef DEBUG
    CHECK(table_idx_governed_vertex_idxes_.find(child_table.first)
        != table_idx_governed_vertex_idxes_.end());
#endif

    LOG(INFO) << table_idx_governed_vertex_idxes_[child_table.first].size()
        << " " <<  child_table.second << " v_idx: " << vertex_idx << " " << child_table.first;
    for (uint32 ele : table_idx_governed_vertex_idxes_[child_table.first]) {
      LOG(INFO) << ele;
    }

    table_idx_governed_vertex_idxes_[child_table.first].insert(vertex_idx);
    child_table.second++;
#ifdef DEBUG
    CHECK_EQ(table_idx_governed_vertex_idxes_[child_table.first].size(),
        child_table.second);
#endif
    table_idx_governed_size_->update(
        IdxCntPair(child_table.first, child_table.second));
  }
  vertexes_[vertex_idx]->set_child_table_idx(child_table.first);
}


//TODO: add more random factors
void Tree::SampleVertexToSplit(vector<uint32>& vertexes_to_split) {
  LOG(INFO) << "Sample to split max " << max_table_idx_ << " " 
      << client_id_ << " " << thread_id_;

  vertexes_to_split.clear();
  // to prevent duplication
  set<uint32> vertexes_set;
  const int max_num_split = Context::get_int32("max_split_per_table");
  // Sample on each table allocated to this thread
  const int step_size = tot_num_threads_;
  uint32 table_idx = thread_id_;
  for (; table_idx <= max_table_idx_; table_idx += step_size) {
    LOG(INFO) << "Sample to split table: " << table_idx << "  by " << client_id_ << " " << thread_id_;
#ifdef DEBUG
    CHECK(table_idx_governed_vertex_idxes_.find(table_idx)
        != table_idx_governed_vertex_idxes_.end());
#endif
    const set<uint32>& governed_vertexes 
        = table_idx_governed_vertex_idxes_[table_idx];
    vector<uint32> vertex_idxes;
    FloatVec vertex_weights;
    for (uint32 v_idx : governed_vertexes) {
      vertex_idxes.push_back(v_idx);
#ifdef DEBUG
      CHECK(vertexes_.find(v_idx) != vertexes_.end()) 
          << "v_idx: " << v_idx;
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
        vertexes_to_split.push_back(cand_v_idx);
        ++num_split;
      }
    }
  } // end of tables

  LOG(INFO) << "Sample to split #v=" << vertexes_to_split.size() << " " << client_id_ << " " << thread_id_;
}

uint32 Tree::AcceptSplitVertex(Vertex* new_vertex,
    const Vertex* parent_vertex_copy) {
  /// Allocate vertex idx
  uint32 child_table_idx = parent_vertex_copy->child_table_idx();
#ifdef DEBUG
  CHECK(table_idx_vertex_idxes_.find(child_table_idx)
      != table_idx_vertex_idxes_.end());
#endif
  // Do not assign vertex_idx based on vertex_idxes.size(),
  // since the vertex idxes might not be continuous. Use 
  // (max vertex idx + 1) instead
  uint32 child_idx = *(--table_idx_vertex_idxes_[child_table_idx].end()) + 1;
  CHECK_EQ(Context::table_id(child_idx), child_table_idx) 
      << "Table size exceeds limit. child_idx=" << child_idx;
  //int row_id = table_idx_vertex_idxes_[child_table_idx].size();
  //uint32 child_idx = Context::make_vertex_id(child_table_idx, row_id);
  new_vertex->set_idx(child_idx);
  table_idx_vertex_idxes_[child_table_idx].insert(child_idx);

  Vertex* parent_vertex = vertexes_[parent_vertex_copy->idx()];

  /// Update param table
  new_vertex->UpdateParamTableByInc(new_vertex->n(), new_vertex->s(), 1.0);
  //parent_vertex->ReadParamTable();
  //parent_vertex->UpdateParamTableByInc(parent_vertex_copy->n(), parent_vertex->n(),
  //    parent_vertex_copy->s(), parent_vertex->s());
  /// Record 
  vertex_split_records_.push_back(make_pair(parent_vertex->idx(), child_idx));

  return child_idx;
}

/// The result is deterministic
void Tree::UpdateTreeStructAfterSplit() {
  set<uint32> local_new_vertex_idxes;
  for (const auto& record : vertex_split_records_) {
    local_new_vertex_idxes.insert(record.second);
  }

  /// Read struct table
  petuum::Table<int>* struct_table = Context::struct_table();
  petuum::RowAccessor row_acc;
  // Meta info
  vector<int> row_cache_meta(tot_num_threads_);
  const auto& r_meta 
      = struct_table->Get<petuum::DenseRow<int> >(0, &row_acc);
  r_meta.CopyToVector(&row_cache_meta);
  
  LOG(INFO) << "UpdateTreeStructAfterSplit " << row_cache_meta[0] << " " << row_cache_meta[1] 
      << " " << client_id_ << " " << thread_id_ ;

  // Records
  vector<int> row_cache(Context::struct_table_row_length());
  for (int t_idx = 0; t_idx < tot_num_threads_; ++t_idx) {
    int num_records = row_cache_meta[t_idx];
    if (num_records == 0 || t_idx == global_worker_id_) {
      continue;
    }
    const auto& r 
        = struct_table->Get<petuum::DenseRow<int> >(t_idx + 1, &row_acc);
    r.CopyToVector(&row_cache);
    for (int r_idx = 0; r_idx < num_records; ++r_idx) {
      vertex_split_records_.push_back(
          make_pair(row_cache[r_idx * 2], row_cache[r_idx * 2 + 1]));
    }
  } // end of rows

  // TODO
  LOG(INFO) << "#records: " << vertex_split_records_.size()
      << " " << client_id_ << " " << thread_id_ ;
  for (auto r_tmp : vertex_split_records_) {
    LOG(INFO) << r_tmp.first << " vs " << r_tmp.second;
  }

  //TODO: just for test
  set<uint32> new_vertex_set;

  // IMPORTANT: this is an deterministic operation 
  //   (all threads produce exactly the same results) 
  //
  // Sort to ensure deterministic
  std::sort(vertex_split_records_.begin(), vertex_split_records_.end(),
      SortBySecondOfPair());
  /// Update local structure
  // Add to the tree structure
  for (const auto& record : vertex_split_records_) {
    uint32 new_vertex_idx = record.second;
    uint32 parent_idx = record.first;
#ifdef DEBUG
    CHECK(vertexes_.find(new_vertex_idx) == vertexes_.end()) 
        << " new_vertex_idx=" << new_vertex_idx << " parent_idx=" << parent_idx;
#endif
    VertexParameter vertex_param;
    vertex_param.set_index(new_vertex_idx);
    Vertex* new_vertex = new Vertex(vertex_param, param_);
    vertexes_[new_vertex_idx] = new_vertex;
    vertexes_[parent_idx]->add_child(new_vertex);

    //TODO: just for test
    CHECK(new_vertex_set.find(new_vertex_idx) == new_vertex_set.end());
    new_vertex_set.insert(new_vertex_idx);
  }
  // Allocate child table for future children
  for (const auto& record : vertex_split_records_) {
    const uint32 new_vertex_idx = record.second; 
    uint32 new_vertex_table_idx = Context::table_id(record.second);
#ifdef DEBUG
    CHECK(vertexes_.find(new_vertex_idx) != vertexes_.end());
#endif
    if (local_new_vertex_idxes.find(new_vertex_idx)
        == local_new_vertex_idxes.end()) {
      table_idx_vertex_idxes_[new_vertex_table_idx].insert(new_vertex_idx);
    }
    if (vertexes_[new_vertex_idx]->depth() < Context::get_int32("max_depth")) {
      AllocateChildTable(new_vertex_idx);
    }
  }
  // TODO: build a sparse vertex idx to dense row idx map
}

void Tree::UpdateStructTableAfterSplit() {
  petuum::Table<int>* struct_table = Context::struct_table();
  /// Read struct table meta info
  petuum::RowAccessor row_acc;
  vector<int> row_cache_meta(tot_num_threads_);
  const auto& r_meta 
      = struct_table->Get<petuum::DenseRow<int> >(0, &row_acc);
  r_meta.CopyToVector(&row_cache_meta);

  const int num_records = vertex_split_records_.size();
  // Put num_records in the first row of struct table
  petuum::DenseUpdateBatch<int> update_batch_meta(global_worker_id_, 1);

  update_batch_meta[global_worker_id_] 
      = num_records - row_cache_meta[global_worker_id_];
  struct_table->DenseBatchInc(0, update_batch_meta);

  // Put records in the corr. row of struct table
  if (num_records > 0) {
    const int num_cols = num_records * kNumStructTableRecordCols;
    CHECK_LE(num_cols, Context::struct_table_row_length());
    petuum::DenseUpdateBatch<int> update_batch(0, num_cols);
    for (int r_idx = 0; r_idx < num_records; ++r_idx) {
      update_batch[r_idx * 2] = vertex_split_records_[r_idx].first;
      update_batch[r_idx * 2 + 1] = vertex_split_records_[r_idx].second;
    }
    struct_table->DenseBatchInc((global_worker_id_ + 1), update_batch);
  }
}

void Tree::InitSplit() {
  // clear split records
  vector<pair<uint32, uint32> >().swap(vertex_split_records_);
  // update table idx governed sizes
#ifdef DEBUG
  CHECK_EQ(table_idx_governed_size_->size(),
      table_idx_governed_vertex_idxes_.size());
#endif
  for (const auto& ele : table_idx_governed_vertex_idxes_) {
    table_idx_governed_size_->update(IdxCntPair(ele.first, ele.second.size()));
  }
}

void Tree::SampleVertexPairsToMerge(
    vector<pair<uint32, uint32> >& vertex_pair_to_merge) {
  vertex_pair_to_merge.clear();
  // to avoid repeatedly checking failed vertexes
  set<uint32> failed_vertexes_set;
  const int max_num_merge = Context::get_int32("max_merge_per_table");
  // Sample on each table allocated to this thread
  const int step_size = tot_num_threads_;
  uint32 table_idx = thread_id_;
  for (; table_idx <= max_table_idx_; table_idx += step_size) {
#ifdef DEBUG
    CHECK(table_idx_vertex_idxes_.find(table_idx)
        != table_idx_vertex_idxes_.end());
#endif
    const set<uint32>& vertex_idxes = table_idx_vertex_idxes_[table_idx];
    const int table_size = vertex_idxes.size();
    int num_merge = 0;
    while (num_merge < min(max_num_merge, table_size / 2)) {
      if (failed_vertexes_set.size() >= table_size - 1) {
        break;
      }
      // Sample host vertex uniformly at random
      set<uint32>::const_iterator it = vertex_idxes.begin();
      advance(it, Context::randInt() % table_size);
      uint32 host_v_idx = *it;
      if (failed_vertexes_set.find(host_v_idx) 
          != failed_vertexes_set.end()) {
        continue;
      }
#ifdef DEBUG
      CHECK(vertexes_.find(host_v_idx) != vertexes_.end());
#endif
      const Vertex* host_v = vertexes_[host_v_idx];
      // Sample guest vertex from siblings by softmax probabilities
      vector<uint32> cand_vertex_idxes;
      FloatVec cand_weights;
      float cand_weight_sum = 0;
      for (Vertex* cand_v : host_v->parent()->children()) {
        if (failed_vertexes_set.find(cand_v->idx())
            != failed_vertexes_set.end() || cand_v->idx() == host_v_idx) {
          continue;
        }
        cand_vertex_idxes.push_back(cand_v->idx());
#ifdef DEBUG
        CHECK(vertexes_.find(cand_v->idx()) != vertexes_.end());
#endif
        // the dot product is in [0, 1], since vertex mean is unit vector
        float cur_weight
            = exp(DotProdFloatVectors(host_v->mean(), cand_v->mean()));
        cand_weights.push_back(cur_weight);
        cand_weight_sum += cur_weight;
      }
      if (cand_vertex_idxes.size() == 0) {
        failed_vertexes_set.insert(host_v_idx);
        continue;
      } 
#ifdef DEBUG
      //TODO: set a threshold here, if < threshold, do not merge
      CHECK_GE(cand_weight_sum, kFloatEpsilon);
#endif
      for (auto& weight : cand_weights) {
        weight /= cand_weight_sum;
      }
      uint32 guest_v_idx = cand_vertex_idxes[
          Context::randDiscrete(cand_weights, 0, cand_vertex_idxes.size())];
      // Set left child as host, right child as guest
      bool change_order = true;
      Vertex* v_cursor = host_v->right_sibling();
      while (v_cursor != NULL) {
        if (v_cursor->idx() == guest_v_idx) {
          change_order = false;
          break;
        }
        v_cursor = v_cursor->right_sibling();
      }
      if (change_order) {
        vertex_pair_to_merge.push_back(make_pair(guest_v_idx, host_v_idx));
      } else {
        vertex_pair_to_merge.push_back(make_pair(host_v_idx, guest_v_idx));
      }
      ++num_merge;
    } // end of sampling
  } // end of tables
}

void Tree::UpdateStructTableAfterMerge() {
  petuum::Table<int>* struct_table = Context::struct_table();
  /// Read struct table meta info
  petuum::RowAccessor row_acc;
  vector<int> row_cache_meta(tot_num_threads_);
  const auto& r_meta 
      = struct_table->Get<petuum::DenseRow<int> >(0, &row_acc);
  r_meta.CopyToVector(&row_cache_meta);

  const int num_records = vertex_merge_records_.size();
  // Put num_records in the first row of struct table
  petuum::DenseUpdateBatch<int> update_batch_meta(global_worker_id_, 1);

  update_batch_meta[global_worker_id_] 
      = num_records - row_cache_meta[global_worker_id_];
  struct_table->DenseBatchInc(0, update_batch_meta);

  // Put records in the corr. row of struct table
  if (num_records > 0) {
    const int num_cols = num_records * kNumStructTableRecordCols;
    CHECK_LE(num_cols, Context::struct_table_row_length());
    petuum::DenseUpdateBatch<int> update_batch(0, num_cols);
    for (int r_idx = 0; r_idx < num_records; ++r_idx) {
      update_batch[r_idx * 2] = vertex_merge_records_[r_idx].first;
      update_batch[r_idx * 2 + 1] = vertex_merge_records_[r_idx].second;
    }
    struct_table->DenseBatchInc((global_worker_id_ + 1), update_batch);
  }
}

void Tree::UpdateTreeStructAfterMerge() {
  /// Read struct table
  petuum::Table<int>* struct_table = Context::struct_table();
  petuum::RowAccessor row_acc;
  // Meta info
  vector<int> row_cache_meta(tot_num_threads_);
  const auto& r_meta 
      = struct_table->Get<petuum::DenseRow<int> >(0, &row_acc);
  r_meta.CopyToVector(&row_cache_meta);
  
  // Records
  vector<int> row_cache(Context::struct_table_row_length());
  for (int t_idx = 0; t_idx < tot_num_threads_; ++t_idx) {
    int num_records = row_cache_meta[t_idx];
    if (num_records == 0 || t_idx == global_worker_id_) {
      continue;
    }
    const auto& r 
        = struct_table->Get<petuum::DenseRow<int> >(t_idx + 1, &row_acc);
    r.CopyToVector(&row_cache);
    for (int r_idx = 0; r_idx < num_records; ++r_idx) {
      vertex_merge_records_.push_back(
          make_pair(row_cache[r_idx * 2], row_cache[r_idx * 2 + 1]));
    }
  } // end of rows

  // TODO
  LOG(INFO) << "#records: " << vertex_merge_records_.size();

  /// Update local structure
  // IMPORTANT: this is an deterministic operation 
  //   (all threads produce exactly the same results) 
  //
  // Sort to ensure deterministic
  std::sort(vertex_merge_records_.begin(), vertex_merge_records_.end(),
      SortBySecondOfPair());
  // old vertex idx => new vertex idx
  map<uint32, uint32> vertex_idx_update;
  map<uint32, uint32>::const_iterator it;
  for (const auto& record : vertex_merge_records_) {
    uint32 host_v_idx = record.first;
    it = vertex_idx_update.find(host_v_idx);
    if (it != vertex_idx_update.end()) {
      host_v_idx = it->second;
    } 
    uint32 guest_v_idx = record.second;
    it = vertex_idx_update.find(guest_v_idx);
    if (it != vertex_idx_update.end()) {
      guest_v_idx = it->second;
    } 
    MergeTwoVertex(host_v_idx, guest_v_idx, vertex_idx_update);
  }
}

// The result is deterministic
void Tree::MergeTwoVertex(const uint32 host_v_idx, const uint32 guest_v_idx,
  map<uint32, uint32>& vertex_idx_update) {
#ifdef DEBUG
  CHECK(vertexes_.find(host_v_idx) != vertexes_.end());
  CHECK(vertexes_.find(guest_v_idx) != vertexes_.end());
#endif
  Vertex* host_v = vertexes_[host_v_idx];
  Vertex* guest_v = vertexes_[guest_v_idx];
  
  /// Update param table 
  host_v->UpdateParamTableByInc(guest_v->n(), guest_v->s(), 1.0);

  /// Update local tree structure
  /// Remove guest_v
  // *) child list of host_v & parent ptr of guest_v's children
  // *) child list of guest_v's parent
  // *) sibling ptr of guest_v's neibors
  // *) vertexes_: remove guest_v
  // *) table_idx_vertex_idxes_: remove guest_v
  // *) table_idx_governed_vertex_idxes_: remove guest_v
  //
  /// IF host_child_table != guest_child_table:
  /// Merge table
  // *) idx of vertexes in guest_child_table (including guest's children)
  // *) vertexes_ accordingly
  // *) child_table_idx of the parents of vertexes in guest_child_table
  // *) table_idx_vertex_idxes_
  // *) table_idx_governed_vertex_idxes_
  // *) vertex_idx_update map
  //
  // *) table_idx_governed_size_ is updated in a batch manner
   
  /// IF host_child_table != guest_child_table:
  /// Merge table
  const uint32 host_child_table_idx = host_v->child_table_idx();
  const uint32 guest_child_table_idx = guest_v->child_table_idx();
  if (host_child_table_idx != guest_child_table_idx) {
    LOG(INFO) << "Merge table " << host_child_table_idx << " and " << guest_child_table_idx
       << " with host_v " << host_v_idx << " guest_v " << guest_v_idx;  
#ifdef DEBUG
    CHECK(table_idx_vertex_idxes_.find(guest_child_table_idx) 
        != table_idx_vertex_idxes_.end());
#endif
    //TODO: merge the smaller table to the bigger one
    set<uint32>& host_vertex_idxes
        = table_idx_vertex_idxes_[host_child_table_idx];
    const set<uint32>& guest_vertex_idxes
        = table_idx_vertex_idxes_[guest_child_table_idx];
    // Expected to iterate in order
    // guest_vertex_idxes will not be modified in this loop
    for (auto vertex_idx : guest_vertex_idxes) {
      //const uint32 vertex_new_idx
      //    = Context::make_vertex_id(host_child_table_idx,
      //    host_vertex_idxes.size());
      const uint32 vertex_new_idx = *(--host_vertex_idxes.end()) + 1;
      CHECK_EQ(Context::table_id(vertex_new_idx), host_child_table_idx) 
          << "Table size exceeds limit. vertex_new_id=" << vertex_new_idx;
      host_vertex_idxes.insert(vertex_new_idx);
#ifdef DEBUG
      CHECK(vertex_idx_update.find(vertex_new_idx)
          == vertex_idx_update.end());
#endif
      vertex_idx_update[vertex_idx] = vertex_new_idx;
      
      Vertex* vertex = vertexes_[vertex_idx];
      vertex->set_idx(vertex_new_idx);
#ifdef DEBUG
      CHECK(vertexes_.find(vertex_new_idx) == vertexes_.end());
#endif
      vertexes_[vertex_new_idx] = vertex;
      vertexes_.erase(vertex_idx);

      vertex->parent()->set_child_table_idx(host_child_table_idx);
      if (!vertex->root()) {
#ifdef DEBUG
        CHECK(table_idx_governed_vertex_idxes_.find(host_child_table_idx) 
            != table_idx_governed_vertex_idxes_.end());
#endif
        table_idx_governed_vertex_idxes_[host_child_table_idx].insert(
            vertex->parent()->idx());
      }
      uint32 governing_table_idx = vertex->child_table_idx();
#ifdef DEBUG
      CHECK(table_idx_governed_vertex_idxes_.find(governing_table_idx) 
          != table_idx_governed_vertex_idxes_.end());
#endif
      // Otherwise, do not need to do this, since the 
      // governed_vertex_idxes of guest_child_table will be cleared at last
      if (guest_child_table_idx != governing_table_idx) {
        table_idx_governed_vertex_idxes_[governing_table_idx].erase(
            vertex_idx);
        table_idx_governed_vertex_idxes_[governing_table_idx].insert(
            vertex_new_idx);
      }
    }
    //guest_vertex_idxes.clear();
    table_idx_vertex_idxes_[guest_child_table_idx].clear();

#ifdef DEBUG
    CHECK(table_idx_governed_vertex_idxes_.find(guest_child_table_idx) 
        != table_idx_governed_vertex_idxes_.end());
#endif
    table_idx_governed_vertex_idxes_[guest_child_table_idx].clear();
  } // end of merging table
 
  LOG(INFO) << "NEW Merge " << host_v->idx() << " and " << guest_v->idx();

  /// Remove guest_v
  // Note: do not use host/guest_v_idx, use ->idx(), since their idx might 
  //   have been changed in the table merging process
  for (Vertex* guest_child : guest_v->children()) {
    host_v->add_child(guest_child);
  }
  guest_v->parent()->remove_child(guest_v);
  vertexes_.erase(guest_v->idx());

  const uint32 guest_v_table_idx = Context::table_id(guest_v->idx());
#ifdef DEBUG
    CHECK(table_idx_vertex_idxes_.find(guest_v_table_idx) 
        != table_idx_vertex_idxes_.end());
#endif
  table_idx_vertex_idxes_[guest_v_table_idx].erase(guest_v->idx());
  // guest_v's governing table has been changed to host_child_table_idx 
  // in the table merging process, (if no table merging, then 
  // host_child_table_idx == guest_*, which is still correct)
  table_idx_governed_vertex_idxes_[host_child_table_idx].erase(guest_v->idx());
  // release the memory
  delete guest_v;
}

void Tree::AcceptMergeVertexes(
    const uint32 host_v_idx, const uint32 guest_v_idx) {
#ifdef DEBUG
  CHECK(vertexes_.find(host_v_idx) != vertexes_.end());
  CHECK(vertexes_.find(guest_v_idx) != vertexes_.end());
#endif
  /// Record 
  vertex_merge_records_.push_back(make_pair(host_v_idx, guest_v_idx));
}

void Tree::InitMerge() {
 vector<pair<uint32, uint32> >().swap(vertex_merge_records_);
}

float Tree::ComputeELBO() {
  float elbo = 0;
  BOOST_FOREACH(UIntVertexPair& ele, vertexes_) {
    elbo += vertexes_[ele.first]->ComputeELBO();
  }
  return elbo;
}

} // namespace ditree
