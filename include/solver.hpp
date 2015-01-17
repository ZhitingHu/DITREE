#ifndef DITREE_SOLVER_HPP_
#define DITREE_SOLVER_HPP_

#include "common.hpp"
#include "context.hpp"
#include "tree.hpp"
#include "dataset.hpp"

namespace ditree {

class Solver {
 public:
  explicit Solver(const SolverParameter& param, const int thread_id,
      Dataset* train_data);
  explicit Solver(const string& param_file, const int thread_id,
      Dataset* train_data);

  void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
 
 private: 
  void Init(const SolverParameter& param);
  void Test();
  void Update();

  //TODO
  void Snapshot() {}
  void Restore(const char* resume_file) {}

  //void RegisterPSTables();
 private:
  SolverParameter param_;

  int client_id_;
  int thread_id_;

  Tree* tree_;
  Vertex* root_;

  Dataset* train_data_;

  int iter_;

  petuum::Table<float> train_loss_table_;
  petuum::Table<float> test_loss_table_;
  int display_counter_;
  int test_counter_;
  // print results on the fly
  int display_gap_;
  int test_display_gap_;
  petuum::HighResolutionTimer total_timer_;
  
};

} // namespace ditree

#endif
