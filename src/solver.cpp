
#include "util.hpp"
#include "io.hpp"
#include "solver.hpp"
#include <cmath>
#include <algorithm>
#include <boost/foreach.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>

namespace ditree {

Solver::Solver(const SolverParameter& param, const int thread_id, 
    Dataset* train_data) : thread_id_(thread_id), train_data_(train_data) {
  Init(param);
}

Solver::Solver(const string& param_file, const int thread_id, 
    Dataset* train_data) : thread_id_(thread_id), train_data_(train_data) {
  SolverParameter param;
  ReadProtoFromTextFile(param_file, &param);
  Init(param);
}

void Solver::Init(const SolverParameter& param) {
  client_id_ = Context::get_int32("client_id");
  param_ = param;
  collect_target_data_ = false;
  log_target_data_threshold_ = log(param_.split_target_data_threshold());
  max_target_data_size_ = param_.split_max_target_data_size();

  // Initialize tree
  const string& model = Context::get_string("model"); 
  tree_ = new Tree(model, thread_id_);
  root_ = tree_->root();
  tree_->InitParam();
  train_loss_table_ 
      = petuum::PSTableGroup::GetTableOrDie<float>(kTrainLossTableID);
  test_loss_table_ 
      = petuum::PSTableGroup::GetTableOrDie<float>(kTestLossTableID);
}

void Solver::Solve(const char* resume_file) {
  if (client_id_ == 0 && thread_id_ == 0) {
    LOG(INFO) << "Solving DITree";
  }

  epoch_ = 0;
  iter_ = 0;
  if (resume_file) {
    if (client_id_ == 0 && thread_id_ == 0) {
      LOG(INFO) << "Restoring previous solver status from " << resume_file;
    }
    Restore(resume_file);
    if (client_id_ == 0 && thread_id_ == 0) {
      LOG(INFO) << "Restoration done.";
    }
  }
  petuum::PSTableGroup::GlobalBarrier();

  // Remember the initial iter_ value; will be non-zero if we loaded from a
  // resume_file above.
  //const int start_epoch = epoch_;
  int start_iter = iter_;

  display_counter_ = 0;
  test_counter_ = 0;
  total_timer_.restart();
 
  train_loss_table_ 
      = petuum::PSTableGroup::GetTableOrDie<float>(kTrainLossTableID);
  test_loss_table_ 
      = petuum::PSTableGroup::GetTableOrDie<float>(kTestLossTableID);
  ditree::Context& context = ditree::Context::Get();
  int loss_table_staleness = context.get_int32("loss_table_staleness");
  if (param_.display()) {
    display_gap_ = loss_table_staleness / param_.display() + 1;
  }
  if (param_.test_interval()) {
    test_display_gap_ = loss_table_staleness / param_.test_interval() + 1; 
  }

  //
  for (; epoch_ < param_.max_epoch(); ++epoch_) {
    LOG(INFO) << "=================================================== epoch " << epoch_;
    train_data_->Restart();
    /// VI
    while (!train_data_->epoch_end()) {
      // Save a snapshot if needed.
      if (param_.snapshot() && iter_ > start_iter &&
          iter_ % param_.snapshot() == 0) {
        Snapshot();
      }
      
      // Test if needed.
      if (param_.test_interval() && iter_ % param_.test_interval() == 0
          && (iter_ > 0 || param_.test_initialization())) {
        Test();
      }
      
      // Sample target vertexes to split  
      if (iter_ == param_.split_sample_start_iter()) {
        ClearLastSplit();
        LOG(INFO) << "SampleVertexToSplit. ";
        SampleVertexToSplit();
        LOG(INFO) << "SampleVertexToSplit Done.";
        collect_target_data_ = true;
      }

      // Main VI
      Update();
      
      petuum::PSTableGroup::Clock();
 
      // Display
      if (param_.display() && iter_ % param_.display() == 0) {
        //TODO: display
      }

      ++iter_;
    }

    if (epoch_ < param_.max_epoch() - 1) {
      /// Split
      petuum::PSTableGroup::GlobalBarrier();
      Context::set_phase(Context::Phase::kSplit, thread_id_);

      LOG(INFO) << "!!!!!!! Split";
      Split();
      LOG(INFO) << "!!!!!!! Split Done.";

      tree_->UpdateStructTableAfterSplit();
      LOG(INFO) << "!!!!!!! Split Update STable Done.";

      petuum::PSTableGroup::GlobalBarrier();
      tree_->UpdateTreeStructAfterSplit();
      LOG(INFO) << "!!!!!!! Split Update Local struc Done.";

      tree_->ReadParamTable();
      LOG(INFO) << "!!!!!!! read param table done. ";
      root_->RecursConstructParam();
      LOG(INFO) << "!!!!!!! recurs param done. ";

      Context::set_phase(Context::Phase::kVIAfterSplit, thread_id_); 
      petuum::PSTableGroup::GlobalBarrier();
    }

    /// 
    start_iter = 0;
    iter_ = 0;
  } // end of epoch

  if (param_.snapshot_after_train()) { Snapshot(); }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    Test();
  }
}

void Solver::Update() {
  DataBatch* data_batch = train_data_->GetNextDataBatch();
  // End of epoch
  if (data_batch == NULL) {
    return;
  }

  //if (Context::phase() == Context::Phase::kInit)
  if (data_batch->n().size() == 0) {
    data_batch->InitSuffStatStruct(tree_, train_data_->data());
  } else {
    data_batch->UpdateSuffStatStruct(tree_, Context::phase(thread_id_));
  }
 
  // sum_{n,z} lambda_nz log (lambda_nz) 
  float h = 0;
  /// e-step (TODO: openmp)
  UIntFloatMap n_old(data_batch->n());
  UIntFloatMap& n_new = data_batch->n();
  map<uint32, UIntFloatMap> s_old(data_batch->s());
  map<uint32, UIntFloatMap>& s_new = data_batch->s();
  BOOST_FOREACH(UIntUIntFloatMapPair& s_new_ele, s_new) {
    ResetUIntFloatMap(s_new_ele.second);
  }

#ifdef DEBUG
  CHECK_EQ(n_new.size(), tree_->size());
  CHECK_EQ(s_new.size(), tree_->size());
#endif
  // z prior
  root_->RecursComputeVarZPrior();
  for (const auto v : tree_->vertexes()) {
    LOG(INFO) << v.second->idx() << " var_z_prior: " << v.second->var_z_prior();
  }

  float tree_elbo_tmp_start = tree_->ComputeELBO();
  LOG(INFO) << "ELBO before training " << tree_elbo_tmp_start;

  LOG(INFO) << "recurs var z prior done. ";
  // likelihood
  LOG(INFO) << "databatch " << data_batch->data_idx_begin() << " " << data_batch->size();

  int d_idx = data_batch->data_idx_begin();
  int data_idx_end = d_idx + data_batch->size();
  for (; d_idx < data_idx_end; ++d_idx) {
    LOG(INFO) << "data " << d_idx;
    const Datum* datum = train_data_->datum(d_idx);
    UIntFloatMap log_weights;
    float max_log_weight = -1;
    float log_weight_sum = 0;
    // update on each vertex
    BOOST_FOREACH(const UIntFloatPair& n_new_ele, n_new) {
      const Vertex* vertex = tree_->vertex(n_new_ele.first);
      float cur_log_weight = vertex->var_z_prior() 
          + LogVMFProb(datum->data(), vertex->mean(), vertex->beta());
      log_weights[n_new_ele.first] = cur_log_weight; 
      max_log_weight = max(cur_log_weight, max_log_weight);
    }
    BOOST_FOREACH(const UIntFloatPair& log_weight_ele, log_weights) {
      log_weight_sum += exp(log_weight_ele.second - max_log_weight);
    }
    log_weight_sum = log(log_weight_sum) + max_log_weight;
    
#ifdef DEBUG
    //CHECK_GT(log_weight_sum, kFloatEpsilon);
    CHECK(!isnan(log_weight_sum));
    CHECK(!isinf(log_weight_sum));
#endif
    float n_prob = 0;
    if (d_idx == data_batch->data_idx_begin()) {
      LOG(INFO) << "here first datum of the batch";
      BOOST_FOREACH(UIntFloatPair& n_new_ele, n_new) {
        float datum_z_prob = exp(log_weights[n_new_ele.first] - log_weight_sum);
        n_new_ele.second = datum_z_prob;

        LOG(INFO) << "n_new_ele " << n_new_ele.first << " " 
            << n_new_ele.second << " " << exp(log_weights[n_new_ele.first] - log_weight_sum);

        // have reset s_new at the beginning
        AccumUIntFloatMap(datum->data(), datum_z_prob, s_new[n_new_ele.first]);

        h += datum_z_prob * (log_weights[n_new_ele.first] - log_weight_sum);
        LOG(INFO) << "h " << h << " = " << n_new_ele.second << " * " 
           << (log_weights[n_new_ele.first] - log_weight_sum) << " " 
           << datum_z_prob;

        n_prob += datum_z_prob;
      }
    } else {
      BOOST_FOREACH(UIntFloatPair& n_new_ele, n_new) {
        float datum_z_prob = exp(log_weights[n_new_ele.first] - log_weight_sum);
        n_new_ele.second += datum_z_prob;
        
        LOG(INFO) << "n_new_ele " << n_new_ele.first << " " 
            << n_new_ele.second << " " << exp(log_weights[n_new_ele.first] - log_weight_sum);

        AccumUIntFloatMap(datum->data(), datum_z_prob, s_new[n_new_ele.first]);

        h += datum_z_prob * (log_weights[n_new_ele.first] - log_weight_sum);
        LOG(INFO) << "h " << h << " = " 
           << datum_z_prob  << " * "
           << (log_weights[n_new_ele.first] - log_weight_sum);

        n_prob += datum_z_prob;
      }
    }
    
    if (collect_target_data_) {
      CollectTargetData(log_weights, log_weight_sum, datum);
    }
  } // end of datum

  //TODO
  BOOST_FOREACH(const UIntFloatPair& ele, n_old) {
    LOG(INFO) << "n_old " << ele.first << " " << ele.second;
  }
  BOOST_FOREACH(const UIntFloatPair& ele, n_new) {
    LOG(INFO) << "n_new " << ele.first << " " << ele.second;
  }

  tree_->UpdateParamTable(n_new, n_old, s_new, s_old);
  tree_->ReadParamTable();
  LOG(INFO) << "read param table done. ";
  root_->RecursConstructParam();
  LOG(INFO) << "recurs param done. ";
 
  // TODO
  float tree_elbo_tmp = tree_->ComputeELBO();
  const float elbo = tree_elbo_tmp
      - h / train_data_->size() * data_batch->size();

  CHECK(!isnan(h));
  CHECK(!isnan(elbo));
  //CHECK_LT(elbo, 0) << elbo << " " << tree_elbo_tmp << " " <<  h / train_data_->size() * data_batch->size();

  LOG(ERROR) << epoch_ << " " << iter_ << "," << elbo << "\t" << tree_elbo_tmp << "\t" << h / train_data_->size() * data_batch->size();
}

void Solver::Test() {
  
}

void Solver::ClearLastSplit() {
  LOG(INFO) << "Clear last split";
  for (auto target_data : split_target_data_) {
    if (!target_data->success()) {
      continue;
    }
    uint32 parent_vertex_idx = target_data->target_vertex_idx();
    uint32 child_vertex_idx = target_data->child_vertex_idx();
    tree_->vertex(parent_vertex_idx)->UpdateParamTable(
        target_data->n_parent(), target_data->s_parent(), -1.0);
    tree_->vertex(child_vertex_idx)->UpdateParamTable(
        target_data->n_child(), target_data->s_child(), -1.0);
  }
  LOG(INFO) << "Clear last split done.";
}

void Solver::SampleVertexToSplit() {
  vertexes_to_split_.clear();
  tree_->SampleVertexToSplit(vertexes_to_split_);

  vector<TargetDataset*>().swap(split_target_data_);
  for (int i = 0; i < vertexes_to_split_.size(); ++i) {
    split_target_data_.push_back(new TargetDataset(vertexes_to_split_[i]));
  }

  for(uint32 v_t_p : vertexes_to_split_) {
    LOG(INFO) << "To split " << v_t_p;
  }
}

void Solver::CollectTargetData(const UIntFloatMap& log_weights,
    const float log_weight_sum, const Datum* datum) {
  bool complete = true;
  for (int v_idx_i = 0; v_idx_i < vertexes_to_split_.size(); ++v_idx_i) {
    if (split_target_data_[v_idx_i]->size() >= max_target_data_size_) {
      continue;
    }
    complete = false;
    uint32 v_idx = vertexes_to_split_[v_idx_i];
    if ((log_weights.find(v_idx)->second - log_weight_sum) >
        log_target_data_threshold_) {
      split_target_data_[v_idx_i]->AddDatum(datum, log_weights, log_weight_sum);
    }
  } // end of target vertexes
  collect_target_data_ = (complete ? false : true);
}

void Solver::Split() {
  tree_->ClearSplitRecords();
  for (int i = 0; i < vertexes_to_split_.size(); ++i) {
    uint32 v_idx = vertexes_to_split_[i];
    Vertex parent_vertex_copy = (*tree_->vertex(v_idx));
    Vertex* child_vertex = new Vertex();
    TargetDataset* target_data = split_target_data_[i];

    LOG(INFO) << "target data size " << target_data->size() << " v=" << v_idx;

    // Initialize
    SplitInit(&parent_vertex_copy, child_vertex, target_data);
    // TODO
    for (int rs_iter = 0; rs_iter < 10; ++rs_iter) {
      RestrictedUpdate(&parent_vertex_copy, child_vertex, target_data);
    }
    // TODO: validate
    
    // Accept the new child vertex
    uint32 child_idx 
        = tree_->AcceptSplitVertex(child_vertex, &parent_vertex_copy);
    target_data->set_child_vertex_idx(child_idx);
    target_data->set_success();
  }
}

void Solver::RestrictedUpdate(Vertex* parent, Vertex* new_child,
    TargetDataset* target_data) {
  LOG(INFO) << "Restricted Update ";
  LOG(INFO) << "child param: " << new_child->n() << " "
      << new_child->var_z_prior() << " ";
  LOG(INFO) << "parent param: " << parent->n() << " "
      << parent->var_z_prior() << " ";
  /// e-step
  parent->ComputeVarZPrior();
  new_child->ComputeVarZPrior();

  LOG(INFO) << "child var_z_prior: " << new_child->var_z_prior() << " ";
  LOG(INFO) << "parent var_z_prior: " << parent->var_z_prior() << " ";

  const float n_parent_old = target_data->n_parent();
  float& n_parent_new = target_data->n_parent();
  const UIntFloatMap s_parent_old(target_data->s_parent());
  UIntFloatMap& s_parent_new = target_data->s_parent();

  const float n_child_old = target_data->n_child();
  float& n_child_new = target_data->n_child();
  const UIntFloatMap s_child_old(target_data->s_child());
  UIntFloatMap& s_child_new = target_data->s_child();

  n_parent_new = n_child_new = 0;
  ResetUIntFloatMap(s_parent_new);
  ResetUIntFloatMap(s_child_new);
  float h = 0;
  for (auto datum : target_data->data()) {
    float parent_log_weight = parent->var_z_prior()
        + LogVMFProb(datum->data(), parent->mean(), parent->beta());
    float child_log_weight = new_child->var_z_prior()
        + LogVMFProb(datum->data(), new_child->mean(), new_child->beta());
    float max_log_weight = max(parent_log_weight, child_log_weight);
    float log_weight_sum = log(exp(parent_log_weight - max_log_weight)
        + exp(child_log_weight - max_log_weight)) + max_log_weight;
#ifdef DEBUG
    CHECK(!isnan(log_weight_sum));
    CHECK(!isinf(log_weight_sum));
#endif
    float d_n_parent = exp(parent_log_weight - log_weight_sum);
    float d_n_child = exp(child_log_weight - log_weight_sum);
    AccumUIntFloatMap(datum->data(), d_n_parent, s_parent_new);
    AccumUIntFloatMap(datum->data(), d_n_child, s_child_new);
    h += d_n_parent * (parent_log_weight - log_weight_sum)
        + d_n_child * (child_log_weight - log_weight_sum);
    
    n_parent_new += exp(parent_log_weight - log_weight_sum);
    n_child_new += exp(child_log_weight - log_weight_sum);

    LOG(INFO) << n_parent_new  << " " <<  n_child_new << " " 
        << exp(parent_log_weight - log_weight_sum) << " "
        << exp(child_log_weight - log_weight_sum);
    //PrintUIntFloatMap(datum->data());
  } // end of datum

  LOG(INFO) << "s_parent_new";
  PrintUIntFloatMap(s_parent_new);
  LOG(INFO) << "s_child_new";
  PrintUIntFloatMap(s_child_new);
  
  parent->UpdateParamLocal(
      n_parent_new, n_parent_old, s_parent_new, s_parent_old);
  new_child->UpdateParamLocal(
      n_child_new, n_child_old, s_child_new, s_child_old);
  new_child->ConstructParam();
  parent->ConstructParam();

  LOG(INFO) << "Compute ELBO";
  float elbo = parent->ComputeELBO() + new_child->ComputeELBO() - h;
  LOG(INFO) << "res update: " << elbo << " " << parent->ComputeELBO() << " "
      << new_child->ComputeELBO() << " " << -h;

  LOG(INFO) << "Restricted Update Done.";
}

void Solver::SplitInit(Vertex* parent, Vertex* new_child,
    TargetDataset* target_data) {
  LOG(INFO) << "Split Init";
  // Initialize new_child
  new_child->CopyParamsFrom(parent);
  FloatVec& new_child_mean = new_child->mutable_mean();
  //float new_child_split_weight = 0;
  const vector<Vertex*>& children = parent->children();
  if (children.size() > 0) {
    // 
    const FloatVec& parent_mean = parent->mean();
    FloatVec ave_children_mean(parent->mean().size());
    // Compute average children mean
    const UIntFloatMap& n = target_data->origin_n(); 
    //float n_target_data_parent = n.find(parent->idx())->second;
    float n_target_data_children = 0;
    float n_whole_data_children = 0;
    for (auto child_vertex: children) {
#ifdef DEBUG
      CHECK(n.find(child_vertex->idx()) != n.end()) << " " << child_vertex->idx();
#endif
      n_target_data_children += n.find(child_vertex->idx())->second;
      n_whole_data_children += child_vertex->n();
    }
    for (auto child_vertex: children) {
      AccumFloatVec(child_vertex->mean(),
          child_vertex->n() / n_whole_data_children, ave_children_mean);
    }
    float ave_children_mean_norm = 0;
    for (auto w : ave_children_mean) {
      ave_children_mean_norm += w * w;
    }
    ave_children_mean_norm = sqrt(ave_children_mean_norm);
    // Compute new child's mean
    float new_child_mean_norm = 0;
    for (int w_idx = 0; w_idx < ave_children_mean.size(); ++w_idx) {
      ave_children_mean[w_idx] /= ave_children_mean_norm;
      new_child_mean[w_idx] = parent_mean[w_idx] - ave_children_mean[w_idx];
      new_child_mean_norm += new_child_mean[w_idx];
    }
    new_child_mean_norm = sqrt(new_child_mean_norm);
    for (int w_idx = 0; w_idx < new_child_mean.size(); ++w_idx) {
      new_child_mean[w_idx] /= new_child_mean_norm;
    }
    // Compute split weight 
    //new_child_split_weight = n_target_data_parent
    //    / (n_target_data_parent + n_target_data_children);
  } else {
    // Randomly initialize new child's mean
    int rand_datum_idx = Context::randInt() % target_data->size();
    const UIntFloatMap& rand_datum_words
        = target_data->datum(rand_datum_idx)->data();
    float new_child_mean_norm = 0;
    BOOST_FOREACH(const UIntFloatPair& rdw_ele, rand_datum_words) {
      new_child_mean[rdw_ele.first] = rdw_ele.second;
      new_child_mean_norm += rdw_ele.second * rdw_ele.second;
    }
    new_child_mean_norm = sqrt(new_child_mean_norm);
#ifdef DEBUG
    CHECK_GT(new_child_mean_norm, 0);
#endif
    for (int w_idx = 0; w_idx < new_child_mean.size(); ++w_idx) {
      new_child_mean[w_idx] /= new_child_mean_norm;
    }
    // Compute split weight
    //new_child_split_weight = 0.3; //TODO
  }

  new_child->set_idx(-1); //TODO
  parent->add_temp_child(new_child);

  //TODO
  std::fill(parent->mutable_s().begin(), parent->mutable_s().end(), 0); 
  //parent->mutable_n() = 0; 
  
  LOG(INFO) << "Parent idx " << parent->idx();

  ostringstream oss;
  oss << "mean of new child \n";
  for (int i = 0; i < new_child_mean.size(); ++i) {
    oss << new_child_mean[i] << " ";
  }
  oss << std::endl;
  LOG(INFO) << oss.str();
}

//void Solver::SplitInit(Vertex* parent, Vertex* new_child,
//    DataBatch* reference_data) {
//  const UIntFloatMap& n = reference_data->n(); 
//  const map<uint32, UIntFloatMap>& s = reference_data->s();
//#ifdef DEBUG
//  CHECK(n.find(parent->idx()) != n.end());
//#endif
//  FloatVec& new_child_mean = new_child->mutable_mean();
//  float new_child_split_weight = 0;
//  float n_parent = n[parent->idx()];
//  vector<Vertex*>& children = parent->children();
//  if (children.size() > 0) {
//    new_child->set_left_sibling(children.back());
//    FloatVec ave_children_mean(parent->mean().size());
//    float n_children = 0;
//    float n_whole_children = 0;
//    for (auto child_vertex: children) {
//#ifdef DEBUG
//      CHECK(n.find(child_vertex->idx()) != n.end());
//#endif
//      n_children += n[child_vertex->idx()];
//      n_whole_children += child_vertex->n();
//    }
//    for (auto child_vertex: children) {
//      AccumFloatVec(child_vertex->mean(),
//          child_vertex->n() / n_whole_children, ave_children_mean);
//    }
//    float ave_children_mean_norm = 0;
//    for (auto w : ave_children_mean) {
//      ave_children_mean_norm += w * w;
//    }
//    ave_children_mean_norm = sqrt(ave_children_mean_norm);
//    CopyFloatVec(ave_children_mean,
//        1.0 / ave_children_mean_norm, parent->mutable_mean());
//    
//    new_child_split_weight = n_parent / (n_parent + n_children);
//  } else {
//    new_child_split_weight = 0.3; // TODO
//  }
//}


//void Solver::RegisterPSTables() {
//  if (thread_id_ == 0) {
//    // param table 
//    for (int r_idx = 0; r_idx < num_rows; ++r_idx) {
//      outputs_global_table_.GetAsyncForced(ridx);
//    }   
//  }
//}

} // namespace ditree


