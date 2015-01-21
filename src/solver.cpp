
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
  ditree::Context& context = ditree::Context::Get();
  client_id_ = context.get_int32("client_id");
  param_ = param;
  
  // Initialize tree
  const string& model = context.get_string("model"); 
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
  const int start_iter = iter_;

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
  for (; iter_ < param_.max_iter(); ++iter_) {
    LOG(INFO) << "start iter " << iter_;
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

    // main VI
    Update();
    LOG(INFO) << "update done. ";
    
    petuum::PSTableGroup::Clock();
 

    // Display
    if (param_.display() && iter_ % param_.display() == 0) {
      //TODO: display
    }
  } // end of iter

  if (param_.snapshot_after_train()) { Snapshot(); }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    Test();
  }
}

void Solver::Update() {
  DataBatch* data_batch = train_data_->GetNextDataBatch();
  //if (Context::phase() == Context::Phase::kInit) {
  if (data_batch->n().size() == 0) {
    data_batch->InitSuffStatStruct(tree_, train_data_->data());
  } else {
    data_batch->UpdateSuffStatStruct(tree_);
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

  float tree_elbo_tmp_start = tree_->ComputeELBO();
  LOG(INFO) << "ELBO before training " << tree_elbo_tmp_start;

  LOG(INFO) << "recurs var z prior done. ";
  // likelihood
  LOG(INFO) << "databatch " << data_batch->data_idx_begin() << " " << data_batch->size();

  //TODO
  UIntFloatMap data_batch_words(s_new[0]);
  
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
    //CHECK_GE(n_prob, 1.0 - kFloatEpsilon) << n_prob;    
    //CHECK_LE(n_prob, 1.0 + kFloatEpsilon) << n_prob;    
  } // end of datum

  //TODO
  BOOST_FOREACH(const UIntFloatPair& ele, n_old) {
    LOG(INFO) << "n_old " << ele.first << " " << ele.second;
  }
  BOOST_FOREACH(const UIntFloatPair& ele, n_new) {
    LOG(INFO) << "n_new " << ele.first << " " << ele.second;
  }
  LOG(INFO) << "s_old ";
  ostringstream oss;
  for (int i=0; i<8; ++i) {
    float sum = 0;
    for (int v=0; v<3; ++v) {
      sum += s_old[v][i];
    }
    oss << sum << "\t";
  }
  //BOOST_FOREACH(const UIntUIntFloatMapPair& s_ele, s_old) {
  //  oss << s_ele.first << "\t";
  //  BOOST_FOREACH(const UIntFloatPair& s_ele_ele, s_ele.second) {
  //    oss << s_ele_ele.first << ":" << s_ele_ele.second << "\t";
  //  }
  //  oss << "\n"; 
  //}
  LOG(INFO) << oss.str();

  LOG(INFO) << "s_new ";
  oss.str("");
  oss.clear();
  for (int i=0; i<8; ++i) {
    float sum = 0;
    for (int v=0; v<3; ++v) {
      sum += s_new[v][i];
    }
    oss << sum << "\t";
  }
  //BOOST_FOREACH(const UIntUIntFloatMapPair& s_ele, s_new) {
  //  oss << s_ele.first << "\t";
  //  BOOST_FOREACH(const UIntFloatPair& s_ele_ele, s_ele.second) {
  //    oss << s_ele_ele.first << ":" << s_ele_ele.second << "\t";
  //  }
  //  oss << "\n"; 
  //}
  LOG(INFO) << oss.str();

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

  LOG(ERROR) << iter_ << "," << elbo << "\t" << tree_elbo_tmp << "\t" << h / train_data_->size() * data_batch->size();
}

void Solver::Test() {
  
}

void Solver::Split(const vector<uint32>& vertexes_to_split) {
  for (int s_idx = 0; s_idx < vertexes_to_split.size(); ++s_idx) {
    uint32 v_idx = vertexes_to_split[s_idx];
    Vertex parent_vertex_copy = (*tree_->vertex(v_idx));
    Vertex* child_vertex = new Vertex();
    TargetDataset* target_data = split_target_data_[v_idx];
    // Initialize
    SplitInit(&parent_vertex_copy, child_vertex, target_data);
    // TODO
    for (int rs_iter = 0; rs_iter < 10; ++rs_iter) {
      RestrictedUpdate(&parent_vertex_copy, child_vertex, target_data);
    }
    // TODO: validate
    
    // Accept the new child vertex
    tree_->AcceptSplitVertex(child_vertex, &parent_vertex_copy);
  }
}

void Solver::RestrictedUpdate(Vertex* parent, Vertex* new_child,
    TargetDataset* target_data) {
  /// e-step
  parent->ComputeVarZPrior();
  new_child->ComputeVarZPrior();

  const float n_parent_old = target_data->n_parent();
  float& n_parent_new = target_data->n_parent();
  const UIntFloatMap s_parent_old(target_data->s_parent());
  UIntFloatMap& s_parent_new = target_data->s_parent();

  const float n_child_old = target_data->n_parent();
  float& n_child_new = target_data->n_parent();
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
    n_parent_new += exp(parent_log_weight - log_weight_sum);
    n_child_new += exp(child_log_weight - log_weight_sum);
    AccumUIntFloatMap(datum->data(), n_parent_new, s_parent_new);
    AccumUIntFloatMap(datum->data(), n_child_new, s_child_new);
    h += n_parent_new * (parent_log_weight - log_weight_sum)
        + n_child_new * (child_log_weight - log_weight_sum);
    
    CHECK_GE(n_parent_new + n_child_new, 1.0 - kFloatEpsilon);
    CHECK_LE(n_parent_new + n_child_new, 1.0 + kFloatEpsilon);
  } // end of datum
  
  parent->UpdateParamLocal(
      n_parent_new, n_parent_old, s_parent_new, s_parent_old);
  new_child->UpdateParamLocal(
      n_child_new, n_child_old, s_child_new, s_child_old);
  new_child->ConstructParam();
  parent->ConstructParam();

  float elbo = parent->ComputeELBO() + new_child->ComputeELBO() - h;
  LOG(INFO) << "res update: " << elbo << " " << parent->ComputeELBO()
      << new_child->ComputeELBO() << " " << -h;
}

void Solver::SplitInit(Vertex* parent, Vertex* new_child,
    TargetDataset* target_data) {
  // Initialize new_child
  new_child->CopyParamsFrom(parent);
  parent->add_temp_child(new_child);
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
      CHECK(n.find(child_vertex->idx()) != n.end());
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


