
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
  
  // TODO initialize display_gap_ 
 
  //
  for (; iter_ < param_.max_iter(); ++iter_) {
    //tree_->SyncParameter();
    
    // Save a snapshot if needed.
    if (param_.snapshot() && iter_ > start_iter &&
        iter_ % param_.snapshot() == 0) {
      Snapshot();
    }
    
    // Test if needed 
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {
      Test();
    }

    // main VI
    
    //TODO: display

  }
}


void Solver::Update() {
  DataBatch* data_batch = dataset_->GetNextDataBatch();
  
  /// e-step (TODO: openmp)

  UIntFloatMap n_new(data_batch->n());
  map<uint32, UIntFloatMap> s_new(data_batch->s());
  // Z prior
  root_->RecursiveComputeVarZPrior();
  // likelihood
  for (int d_idx = 0; d_idx < data_batch->size(); ++d_idx) {
    Datum* datum = data_batch->datum(d_idx);
    float weight_sum = 0;
    // update on each node
    BOOST_FOREACH(const UIntFloatPair& ele, n_new) {
      const Vertex* vertex = tree_->vertex(ele.first);
      const float weight 
          = LogVMFProb(datum->data(), vertex->mean(), vertex->kappa());
      weight_sum += weight;
      
    }
  } 
}

void Solver::Test() {

}

} // namespace ditree


