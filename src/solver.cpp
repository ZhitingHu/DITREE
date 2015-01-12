
#include "solver.hpp"

namespace ditree {

Solver::Solver(const SolverParameter& param, const int thread_id)
    : thread_id_(thread_id) {

}

Solver::Solver(const string& param_file, const int thread_id)
    : thread_id_(thread_id) {

}

void Solver::Init() {
  ditree::Context& context = ditree::Context::Get();
  client_id_ = context.get_int32("client_id");

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
 
  loss_table_ = petuum::PSTableGroup::GetTableOrDie<float>(kLossTableID);
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

    tree_->ReadParamTable();
    root_->ConstructParam();

    // main VI
    Update();
    
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
  DataBatch* data_batch = dataset_->GetNextDataBatch();
  data_batch->UpdateSuffStatStruct();
  
  /// e-step (TODO: openmp)
  UIntFloatMap n_old(data_batch->n());
  map<uint32, UIntFloatMap> s_old(data_batch->s());
  UIntFloatMap& n_new = data_batch->n();
  map<uint32, UIntFloatMap>& s_new = data_batch->s();
#ifdef DEBUG
  CHECK_EQ(n_new.size(), tree_->size());
  CHECK_EQ(s_new.size(), tree_->size());
#endif
  // Z prior
  root_->RecursComputeVarZPrior();
  // likelihood
  int d_idx = data_batch->data_idx_begin();
  for (; d_idx < data_batch->size(); ++d_idx) {
    Datum* datum = dataset_->datum(d_idx);
    float weight[n_new.size()];
    float weight_sum = 0;
    // update on each vertex
    BOOST_FOREACH(const UIntFloatPair& n_new_ele, n_new) {
      const Vertex* vertex = tree_->vertex(n_new_ele.first);
      weight[n_new_ele.first] = exp(vertex->var_z_prior() 
          + LogVMFProb(datum->data(), vertex->mean(), vertex->kappa()));
      weight_sum += weight;
    }
#ifdef DEBUG
    CHECK_GT(weight_sum, kFloatEpsilon);
#endif
    if (d_idx == data_batch->data_idx_begin()) {
      BOOST_FOREACH(UIntFloatPair& n_new_ele, n_new) {
        n_new_ele.second = weight / weight_sum;
        CopyUIntFloatMap(datum->data(), weight[n_new_ele.first] / weight_sum,
            s_new[n_new_ele.first]);
      }
    } else {
      BOOST_FOREACH(UIntFloatPair& n_new_ele, n_new) {
        n_new_ele.second += weight / weight_sum;
        AccumUIntFloatMap(datum->data(), weight[n_new_ele.first] / weight_sum,
            s_new[n_new_ele.first]);
      }
    }
  } // end of datum

  tree_->UpdateParamTable(n_old, n_new, s_old, s_new);
}

void Solver::Test() {

}

} // namespace ditree


