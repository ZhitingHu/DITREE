
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

//void Solver::RegisterPSTables() {
//  if (thread_id_ == 0) {
//    // param table 
//    for (int r_idx = 0; r_idx < num_rows; ++r_idx) {
//      outputs_global_table_.GetAsyncForced(ridx);
//    }   
//  }
//}

} // namespace ditree


