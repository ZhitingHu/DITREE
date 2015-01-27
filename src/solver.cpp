
#include "util.hpp"
#include "io.hpp"
#include "solver.hpp"
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <boost/foreach.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>

namespace ditree {

Solver::Solver(const SolverParameter& param, const int thread_id, 
    Dataset* train_data, Dataset* test_data) : thread_id_(thread_id), 
    train_data_(train_data), test_data_(test_data) {
  Init(param);
}

Solver::Solver(const string& param_file, const int thread_id, 
    Dataset* train_data, Dataset* test_data) : thread_id_(thread_id), 
    train_data_(train_data), test_data_(test_data) {
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
  tree_->InitParam(train_data_->data());
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

  const int num_clocks_per_epoch = Context::get_int32("num_clocks_per_epoch");
  const int num_threads = Context::get_int32("num_app_threads");
  const int param_table_staleness = Context::get_int32("param_table_staleness");
  CHECK_GE(num_clocks_per_epoch, param_table_staleness
      + (train_data_->batch_num() + num_threads - 1) / num_threads);
  CHECK_GT(param_.split_sample_start_iter(), param_.merge_iter());
  
  //
  tree_->PrintTreeStruct(train_data_->vocab());
  for (; epoch_ < param_.max_epoch(); ++epoch_) {
    LOG(INFO) << "==================================== epoch " << epoch_ 
        << " " << client_id_ << " " << thread_id_;
    /// VI
    int num_clocks = 0;
    while (!train_data_->epoch_end()) {
      // Save a snapshot if needed.
      if (param_.snapshot() && iter_ > start_iter &&
          iter_ % param_.snapshot() == 0) {
        Snapshot();
      }
      
      // Test if needed.
      if (param_.test_interval() && (iter_ % param_.test_interval() == 0)
          && (iter_ > 0 || param_.test_initialization())) {
        Test();
      }
     
      if (iter_ == param_.merge_iter()) {
        train_data_->RecordLastIterBeforeMerge();

        Context::set_phase(Context::Phase::kMerge, thread_id_);
        LOG(INFO) << "***** Merge " << client_id_ << " " << thread_id_;
        Merge();
        LOG(INFO) << "***** Merge Done." << client_id_ << " " << thread_id_;

        petuum::PSTableGroup::GlobalBarrier(); // Need to read struct table, so barrier here
        tree_->UpdateStructTableAfterMerge();
        LOG(INFO) << "***** Merge Update STable Done." << client_id_ << " " << thread_id_;

        petuum::PSTableGroup::GlobalBarrier(); // Need to read struct table, so barrier here
        tree_->UpdateTreeStructAfterMerge();
        LOG(INFO) << "***** Merge Update Local struc Done."<< client_id_ << " " << thread_id_;

        ConstructMergeMap();

        petuum::PSTableGroup::GlobalBarrier();
        tree_->ReadParamTable();
        LOG(INFO) << "***** read param table done. "<< client_id_ << " " << thread_id_;
        root_->RecursConstructParam();
        LOG(INFO) << "***** recurs param done. "<< client_id_ << " " << thread_id_;

        Context::set_phase(Context::Phase::kVIAfterMerge, thread_id_); 
        tree_->PrintTreeStruct(train_data_->vocab());
      }
 
      // Sample target vertexes to split  
      if (iter_ == param_.split_sample_start_iter()) {
        ClearLastSplit(); //TODO: clear until epoch finishes
        LOG(INFO) << "SampleVertexToSplit. ";
        SampleVertexToSplit();
        LOG(INFO) << "SampleVertexToSplit Done. "<< client_id_ << " " << thread_id_;
        collect_target_data_ = true;
      }

      // Main VI
      Update();
      
      petuum::PSTableGroup::Clock();
      ++num_clocks;
      //LOG(INFO) << "#clock " << num_clocks << " by " <<client_id_ << " " << thread_id_;;
 
      // Display
      if (param_.display() && iter_ % param_.display() == 0) {
        //TODO: display
        tree_->PrintTreeStruct(train_data_->vocab());
      }

      ++iter_;
    }
    for (; num_clocks <= num_clocks_per_epoch; ++num_clocks) {
      petuum::PSTableGroup::Clock();
    }
    train_data_->set_need_restart();

    if (epoch_ < param_.max_epoch() - 1) {
      /// Split
      Context::set_phase(Context::Phase::kSplit, thread_id_);

      LOG(INFO) << "!!!!!!! Split " << client_id_ << " " << thread_id_;
      Split();
      LOG(INFO) << "!!!!!!! Split Done." << client_id_ << " " << thread_id_;

      petuum::PSTableGroup::GlobalBarrier(); // Need to read struct table, so barrier here
      tree_->UpdateStructTableAfterSplit();
      LOG(INFO) << "!!!!!!! Split Update STable Done." << client_id_ << " " << thread_id_;

      petuum::PSTableGroup::GlobalBarrier(); // Need to read struct table, so barrier here
      tree_->UpdateTreeStructAfterSplit();
      LOG(INFO) << "!!!!!!! Split Update Local struc Done."<< client_id_ << " " << thread_id_;

      tree_->ReadParamTable();
      LOG(INFO) << "!!!!!!! read param table done. "<< client_id_ << " " << thread_id_;
      root_->RecursConstructParam();
      LOG(INFO) << "!!!!!!! recurs param done. "<< client_id_ << " " << thread_id_;

      Context::set_phase(Context::Phase::kVIAfterSplit, thread_id_); 
    }

    LOG(INFO) << "Tree size: " << tree_->size();
    tree_->PrintTreeStruct(train_data_->vocab());

    LOG(INFO) << "FinishDataBatchMergeMove ";
    FinishDataBatchMergeMove();
    LOG(INFO) << "FinishDataBatchMergeMove Done.";

    Context::Wait();
    LOG(INFO) << "Wait Done.";
    // Sync
    //Context::IncRestartThreadCounter();
    //LOG(ERROR) << " Im here 1 " << thread_id_;
    //Context::WaitToRestart();
    //LOG(ERROR) << " Im here 2 " << thread_id_;
    //if (thread_id_ == 0) {
    //  Context::ResetRestartThreadCounter();
    //}
    //LOG(ERROR) << " Im here 3 " << thread_id_;

    train_data_->Restart();
    LOG(INFO) << "Restart Done.";
    petuum::PSTableGroup::GlobalBarrier();

    //LOG(ERROR) << " Im here " << thread_id_;

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
    //LOG(INFO) << "Update suff " << data_batch->data_idx_begin() << " tid=" << thread_id_;
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
  //LOG(INFO) << "recurs var z prior done. ";
  //for (const auto v : tree_->vertexes()) {
  //  LOG(INFO) << v.second->idx() << " var_z_prior: " << v.second->var_z_prior();
  //}

  //float tree_elbo_tmp_start = tree_->ComputeELBO();
  //LOG(INFO) << "ELBO before training " << tree_elbo_tmp_start;

  // likelihood
  int d_idx = data_batch->data_idx_begin();
  int data_idx_end = d_idx + data_batch->size();
  for (; d_idx < data_idx_end; ++d_idx) {
    //LOG(INFO) << "data " << d_idx;
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
      //LOG(INFO) << "here first datum of the batch";
      BOOST_FOREACH(UIntFloatPair& n_new_ele, n_new) {
        float datum_z_prob = exp(log_weights[n_new_ele.first] - log_weight_sum);
        n_new_ele.second = datum_z_prob;

        //LOG(INFO) << "n_new_ele " << n_new_ele.first << " " 
        //    << n_new_ele.second << " " << exp(log_weights[n_new_ele.first] - log_weight_sum);

        // have reset s_new at the beginning
        AccumUIntFloatMap(datum->data(), datum_z_prob, s_new[n_new_ele.first]);

        h += datum_z_prob * (log_weights[n_new_ele.first] - log_weight_sum);

        //LOG(INFO) << "h " << h << " = " << n_new_ele.second << " * " 
        //   << (log_weights[n_new_ele.first] - log_weight_sum) << " " 
        //   << datum_z_prob;

        n_prob += datum_z_prob;
      }
    } else {
      BOOST_FOREACH(UIntFloatPair& n_new_ele, n_new) {
        float datum_z_prob = exp(log_weights[n_new_ele.first] - log_weight_sum);
        n_new_ele.second += datum_z_prob;
        
        //LOG(INFO) << "n_new_ele " << n_new_ele.first << " " 
        //    << n_new_ele.second << " " << exp(log_weights[n_new_ele.first] - log_weight_sum);

        AccumUIntFloatMap(datum->data(), datum_z_prob, s_new[n_new_ele.first]);

        h += datum_z_prob * (log_weights[n_new_ele.first] - log_weight_sum);
        //LOG(INFO) << "h " << h << " = " 
        //   << datum_z_prob  << " * "
        //   << (log_weights[n_new_ele.first] - log_weight_sum);

        n_prob += datum_z_prob;
      }
    }
    
    if (collect_target_data_) {
      CollectTargetData(log_weights, log_weight_sum, datum);
    }
  } // end of datum

  //TODO
  if (client_id_ == 0 && thread_id_ == 0) {
    BOOST_FOREACH(const UIntFloatPair& ele, n_old) {
      LOG(INFO) << "n_old " << ele.first << " " << ele.second;
    }
    BOOST_FOREACH(const UIntFloatPair& ele, n_new) {
      LOG(INFO) << "n_new " << ele.first << " " << ele.second;
    }
  }

  tree_->UpdateParamTable(n_new, n_old, s_new, s_old);
  tree_->ReadParamTable();
  //LOG(INFO) << "read param table done. ";
  root_->RecursConstructParam();
  //LOG(INFO) << "recurs param done. ";
 
  // TODO
  float tree_elbo_tmp = tree_->ComputeELBO();
  const float elbo = tree_elbo_tmp
      - h / data_batch->size() * train_data_->size();

  CHECK(!isnan(h));
  CHECK(!isnan(elbo));
  //CHECK_LT(elbo, 0) << elbo << " " << tree_elbo_tmp << " " <<  h / train_data_->size() * data_batch->size();

  LOG(ERROR) << epoch_ << " " << iter_ << "," << elbo << "\t" << tree_elbo_tmp 
      << "\t" << h / data_batch->size() * train_data_->size();
}

void Solver::ClearLastSplit() {
  LOG(INFO) << "Clear last split";
  for (const auto& ele : tree_->vertex_split_records()) {
    uint32 parent_vertex_idx = ele.first;
    uint32 child_vertex_idx = ele.second;

    if(split_target_data_.find(parent_vertex_idx) 
        == split_target_data_.end()) {
      continue;
    }

    TargetDataset* target_data = split_target_data_[parent_vertex_idx]; 
    if (!target_data->success()) {
      continue;
    }
    // The vertexes might have been merged, if so, update 
    // the host vertex
    map<uint32, uint32>::const_iterator it;
    //    = merged_vertexes_host_idx_.find(parent_vertex_idx);
    //if (it != merged_vertexes_host_idx_.end()) {
    //  parent_vertex_idx = it->second;
    //}
    //tree_->vertex(parent_vertex_idx)->UpdateParamTableByInc(
    //    target_data->n_parent(), target_data->s_parent(), -1.0);

    it = merged_vertexes_host_idx_.find(child_vertex_idx);
    if (it != merged_vertexes_host_idx_.end()) {
      child_vertex_idx = it->second;
    }
    LOG(INFO) << "Clear last split stats on vertex " << child_vertex_idx 
        << " by substract n_=" << target_data->n_child();
    tree_->vertex(child_vertex_idx)->UpdateParamTableByInc(
        target_data->n_child(), target_data->s_child(), -1.0);
  }
  FreeMap<uint32, TargetDataset>(split_target_data_);
  LOG(INFO) << "Clear last split done.";
}

void Solver::SampleVertexToSplit() {
  vertexes_to_split_.clear();
  tree_->SampleVertexToSplit(vertexes_to_split_);

  for (const auto v_to_split : vertexes_to_split_) { 
#ifdef DEBUG
    CHECK(split_target_data_.find(v_to_split) == split_target_data_.end());
#endif
    split_target_data_[v_to_split] = new TargetDataset(v_to_split);
  }

  // TODO
  for(uint32 v_t_p : vertexes_to_split_) {
    LOG(INFO) << "To split " << v_t_p;
  }
}

void Solver::CollectTargetData(const UIntFloatMap& log_weights,
    const float log_weight_sum, const Datum* datum) {
  bool complete = true;
  for (auto& ele : split_target_data_) {
    TargetDataset* target_data = ele.second; 
    if (target_data->size() >= max_target_data_size_) {
      continue;
    }
    complete = false;
    if ((log_weights.find(ele.first)->second - log_weight_sum) >
        log_target_data_threshold_) {
      target_data->AddDatum(datum, log_weights, log_weight_sum);
    }
  } // end of target vertexes
  collect_target_data_ = (complete ? false : true);
}

void Solver::Split() {
  collect_target_data_ = false;
  tree_->InitSplit();
 
  //LOG(FATAL) << "Solver->Split " << vertexes_to_split_.size() << " " << thread_id_ << " " << client_id_;

  for (const auto v_idx : vertexes_to_split_) {
    Vertex parent_vertex_copy = (*tree_->vertex(v_idx));
    Vertex* child_vertex = new Vertex();
    TargetDataset* target_data = split_target_data_[v_idx];

    LOG(INFO) << "#target_data on v=" << v_idx << ": " << target_data->size();
    if (target_data->size() < max_target_data_size_ / 50) {
      LOG(INFO) << "Abandom splitting " << v_idx 
          << " target data size too small: " << target_data->size();
      continue;
    }
    //CHECK_GT(target_data->size(), 0) << " v_idx=" << v_idx;

    // Initialize
    SplitInit(&parent_vertex_copy, child_vertex, target_data);
    // TODO
    for (int rs_iter = 0; rs_iter < 10; ++rs_iter) {
      RestrictedUpdate(&parent_vertex_copy, child_vertex, target_data);
    }
    // TODO: validate
    
    // Accept the new child vertex
    child_vertex->mutable_n() -= child_vertex->temp_n();
    tree_->AcceptSplitVertex(child_vertex, &parent_vertex_copy);
    target_data->set_success();
  }
}

void Solver::RestrictedUpdate(Vertex* parent, Vertex* new_child,
    TargetDataset* target_data) {
  //LOG(INFO) << "Restricted Update ";
  //LOG(INFO) << "child param: " << new_child->n() << " "
  //    << new_child->var_z_prior() << " ";
  //LOG(INFO) << "parent param: " << parent->n() << " "
  //    << parent->var_z_prior() << " ";
  /// e-step
  parent->ComputeVarZPrior();
  new_child->ComputeVarZPrior();

  //LOG(INFO) << "child var_z_prior: " << new_child->var_z_prior() << " ";
  //LOG(INFO) << "parent var_z_prior: " << parent->var_z_prior() << " ";

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
    CHECK(!isnan(log_weight_sum)) << parent->idx() << " " << client_id_ << " " << thread_id_ 
        << " " << parent_log_weight << " " << child_log_weight << " " << max_log_weight << " "
        << parent->beta() << " " << new_child->beta();
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

    //LOG(INFO) << n_parent_new  << " " <<  n_child_new << " " 
    //    << exp(parent_log_weight - log_weight_sum) << " "
    //    << exp(child_log_weight - log_weight_sum);
    //PrintUIntFloatMap(datum->data());
  } // end of datum

  //LOG(INFO) << "s_parent_new";
  //PrintUIntFloatMap(s_parent_new);
  //LOG(INFO) << "s_child_new";
  //PrintUIntFloatMap(s_child_new);
  //ostringstream oss;
  //parent->PrintTopWords(oss, train_data_->vocab());
  //LOG(INFO) << oss.str();
  //oss.str("");
  //oss.clear(); 
  //new_child->PrintTopWords(oss, train_data_->vocab());
  //LOG(INFO) << oss.str();
  
  parent->UpdateParamLocal(
      n_parent_new, n_parent_old, s_parent_new, s_parent_old);
  new_child->UpdateParamLocal(
      n_child_new, n_child_old, s_child_new, s_child_old);
  new_child->ConstructParam();
  parent->ConstructParam();

  float elbo = parent->ComputeELBO() + new_child->ComputeELBO() - h;
  //LOG(ERROR) << "res update: " << elbo << " " << parent->ComputeELBO() << " "
  //    << new_child->ComputeELBO() << " " << -h;

  //LOG(INFO) << "Restricted Update Done.";
}

void Solver::SplitInit(Vertex* parent, Vertex* new_child,
    TargetDataset* target_data) {
  LOG(INFO) << "Split Init";
  // Initialize new_child
  new_child->CopyParamsFrom(parent);
  new_child->mutable_n() = parent->n();
  new_child->mutable_temp_n() = parent->n();
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
      CHECK(n.find(child_vertex->idx()) != n.end()) 
          << " " << child_vertex->idx() << " " << n.size();
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
#ifdef DEBUG
    CHECK_GT(ave_children_mean_norm, 0);
#endif
    ave_children_mean_norm = sqrt(ave_children_mean_norm);
    // Compute new child's mean
    float new_child_mean_norm = 0;
    for (int w_idx = 0; w_idx < ave_children_mean.size(); ++w_idx) {
      ave_children_mean[w_idx] /= ave_children_mean_norm;
      new_child_mean[w_idx] = parent_mean[w_idx] - ave_children_mean[w_idx];
      new_child_mean_norm += new_child_mean[w_idx] * new_child_mean[w_idx];
      //new_child_mean_norm += new_child_mean[w_idx];
    }
#ifdef DEBUG
    CHECK_GT(new_child_mean_norm, 0);
#endif
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

  //ostringstream oss;
  //oss << "mean of new child \n";
  //for (int i = 0; i < new_child_mean.size(); ++i) {
  //  oss << new_child_mean[i] << " ";
  //}
  //oss << std::endl;
  //LOG(INFO) << oss.str();
}

void Solver::Merge() {
  tree_->InitMerge();
  /// Sample
  vertex_pairs_to_merge_.clear();
  LOG(INFO) << "SampleVertexPairsToMerge ";
  tree_->SampleVertexPairsToMerge(vertex_pairs_to_merge_);
  LOG(INFO) << "SampleVertexPairsToMerge Done. size=" 
    << vertex_pairs_to_merge_.size() << " c=" << client_id_ << " t=" << thread_id_;
  for (auto& ele : vertex_pairs_to_merge_) { 
    LOG(INFO) << ele.first << " + " << ele.second;
  }
  
  // Candidate vertexes pairs can be duplicated, so we need to 
  //   record which vertexes have been merged
  set<uint32> merged_vertex_idxes;
  /// Merge
  for (const auto& vertex_pair : vertex_pairs_to_merge_) {
    // vertex has been merged
    if (merged_vertex_idxes.find(vertex_pair.first)
        != merged_vertex_idxes.end()) {
      continue;
    }
    if (merged_vertex_idxes.find(vertex_pair.second)
        != merged_vertex_idxes.end()) {
      continue;
    }

    const Vertex* host_v = tree_->vertex(vertex_pair.first);
    const Vertex* guest_v = tree_->vertex(vertex_pair.second);

    LOG(INFO) << "Merging " << host_v->idx() << " " << guest_v->idx();

    const float ori_elbo = host_v->ComputeELBO() + guest_v->ComputeELBO();
    Vertex merged_vertex;
    merged_vertex.MergeFrom(host_v, guest_v);
    const float new_elbo = merged_vertex.ComputeELBO();

    // if ori_elbo < new_elbo, then ELBO is garuanteed to increase after 
    //  merging, so just accept it
    if (ori_elbo < new_elbo) {
      LOG(ERROR) << "Accept Merge " << vertex_pair.first << " " 
          << vertex_pair.second << " cid=" << client_id_ << " tid=" << thread_id_;
      tree_->AcceptMergeVertexes(vertex_pair.first, vertex_pair.second);

      merged_vertex_idxes.insert(vertex_pair.first);
      merged_vertex_idxes.insert(vertex_pair.second);
    }

    LOG(INFO) << "elbo: " << ori_elbo << " " << new_elbo;
  }
}

void Solver::ConstructMergeMap() {
  merged_vertexes_host_idx_.clear();
  for (const auto& merge_record : tree_->vertex_merge_records()) {
    merged_vertexes_host_idx_[merge_record.second] = merge_record.first;
  }
}

void Solver::FinishDataBatchMergeMove() {
  while(true) {
    //LOG(ERROR) << "t=" << thread_id_;
    DataBatch* data_batch = train_data_->GetNextBatchToApplyMerge();
    if (data_batch == NULL) {
      //LOG(INFO) << "t=" << thread_id_ << " done.";
      break;
    }
    //LOG(ERROR) << "t=" << thread_id_;
    data_batch->UpdateSuffStatStructByMerge(tree_->vertex_merge_records());
    //LOG(ERROR) << "t=" << thread_id_;
  }
}

//void Solver::RegisterPSTables() {
//  if (thread_id_ == 0) {
//    // param table 
//    petuum::Table<float>* temp_param_table = Context::temp_param_table();
//    int max_num_vertexes = Context::get_int32("max_num_vertexes");
//    for (int r_idx = 0; r_idx < max_num_vertexes + 5; ++r_idx) {
//      outputs_global_table_.GetAsyncForced(r_idx);
//    }   
//  }
//}

void Solver::Test() {
  DataBatch* data_batch = test_data_->GetRandomDataBatch();

  float log_likelihood = 0;
  // z prior
  root_->RecursComputeVarZPrior();

  int d_idx = data_batch->data_idx_begin();
  int data_idx_end = d_idx + data_batch->size();
  for (; d_idx < data_idx_end; ++d_idx) {
    const Datum* datum = train_data_->datum(d_idx);
    UIntFloatMap log_unnorm_vmf_probs;
    UIntFloatMap log_weights;
    float max_log_weight = -1;
    float log_weight_sum = 0;
    for (const auto& idx_vertex: tree_->vertexes()) {
      const Vertex* vertex = idx_vertex.second;

      float log_unnorm_vmf_prob
          = LogVMFProb(datum->data(), vertex->mean(), vertex->beta());
      log_unnorm_vmf_probs[idx_vertex.first] = log_unnorm_vmf_prob;

      float cur_log_weight = vertex->var_z_prior() 
          + log_unnorm_vmf_prob;
      log_weights[idx_vertex.first] = cur_log_weight; 
      max_log_weight = max(cur_log_weight, max_log_weight);
    }
    BOOST_FOREACH(const UIntFloatPair& log_weight_ele, log_weights) {
      log_weight_sum += exp(log_weight_ele.second - max_log_weight);
    }
    log_weight_sum = log(log_weight_sum) + max_log_weight;
    
#ifdef DEBUG
    CHECK(!isnan(log_weight_sum));
    CHECK(!isinf(log_weight_sum));
#endif
    // Marginalize over z
    float max_datum_z_log_likelihood = FLT_MIN;
    UIntFloatMap datum_z_log_likelihoods;
    for (const auto& idx_vertex: tree_->vertexes()) {
      const Vertex* vertex = idx_vertex.second;
      float datum_z_log_likelihood
          = log_weights[idx_vertex.first] - log_weight_sum 
          + log_unnorm_vmf_probs[idx_vertex.first]
          /* + LogVMFProbNormalizer(vertex->mean().size(), vertex->beta())*/;
      datum_z_log_likelihoods[idx_vertex.first] = datum_z_log_likelihood;
      max_datum_z_log_likelihood = max(
          max_datum_z_log_likelihood, datum_z_log_likelihood);
    }
    float datum_log_likelihood = 0;
    for (const auto& idx_vertex: tree_->vertexes()) {
      datum_log_likelihood += exp(datum_z_log_likelihoods[idx_vertex.first]
          - max_datum_z_log_likelihood);
#ifdef DEBUG
      const Vertex* vertex = idx_vertex.second;
      CHECK(!isnan(datum_log_likelihood)) 
          << "log_prior=" << log_weights[idx_vertex.first] - log_weight_sum
          << " log_unnorm_vmf=" << log_unnorm_vmf_probs[idx_vertex.first]
          << " log_vmf_norm=" 
          << LogVMFProbNormalizer(vertex->mean().size(), vertex->beta())
          << " max_datum_z_log_likelihood=" << max_datum_z_log_likelihood
          << " index=" << idx_vertex.first << " n=" << vertex->n();
      CHECK(!isinf(datum_log_likelihood))
          << "log_prior=" << log_weights[idx_vertex.first] - log_weight_sum
          << " log_unnorm_vmf=" << log_unnorm_vmf_probs[idx_vertex.first]
          << " log_vmf_norm=" 
          << LogVMFProbNormalizer(vertex->mean().size(), vertex->beta())
          << " max_datum_z_log_likelihood=" << max_datum_z_log_likelihood
          << " index=" << idx_vertex.first << " n=" << vertex->n();
#endif
    }
    datum_log_likelihood
        = log(datum_log_likelihood) + max_datum_z_log_likelihood;
    log_likelihood += datum_log_likelihood;
#ifdef DEBUG
    CHECK(!isnan(log_likelihood)) << " datum_likelihood=" << datum_log_likelihood;
    CHECK(!isinf(log_likelihood)) << " datum_likelihood=" << datum_log_likelihood;
#endif
  } // end of datum

  UpdateTestLossTable(log_likelihood, data_batch->size());

  if (client_id_ == 0 && thread_id_ == 0) {
    LOG(INFO) << "Epoch " << epoch_ << ",Iteration " << iter_ << ":" 
        << log_likelihood / data_batch->size() << ","
        << total_timer_.elapsed();

    if (test_counter_ > test_display_gap_) {
      vector<float> output_cache(kNumLossTableCols);
      petuum::RowAccessor row_acc;
      const auto& r = test_loss_table_.Get<petuum::DenseRow<float> >(
          test_counter_ - test_display_gap_ - 1, &row_acc);
      r.CopyToVector(&output_cache);
#ifdef DEBUG
      CHECK_GT(output_cache[kColIdxLossTableNumDatum], 0);
#endif
      LOG(ERROR) << "test,"
          << output_cache[kColIdxLossTableEpoch]
          << "," << output_cache[kColIdxLossTableIter]
          << "," << output_cache[kColIdxLossTableTime]
          << "," << output_cache[kColIdxLossTableLoss] 
          / output_cache[kColIdxLossTableNumDatum];
    }
  }

  ++test_counter_;
}

void Solver::UpdateTestLossTable(const float log_likelihood,
    const int num_datum) {
  if (client_id_ == 0 && thread_id_ == 0) {
    test_loss_table_.Inc(test_counter_, kColIdxLossTableEpoch, epoch_);
    test_loss_table_.Inc(test_counter_, kColIdxLossTableIter, iter_);
    test_loss_table_.Inc(test_counter_, kColIdxLossTableTime, 
        total_timer_.elapsed());
  }
  test_loss_table_.Inc(test_counter_, kColIdxLossTableLoss, 
      log_likelihood);
  test_loss_table_.Inc(test_counter_, kColIdxLossTableNumDatum, 
      num_datum);
}

} // namespace ditree


