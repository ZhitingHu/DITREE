#ifndef DITREE_SOLVER_HPP_
#define DITREE_SOLVER_HPP_

#include "common.hpp"
#include "context.hpp"
#include "tree.hpp"
#include "dataset.hpp"

namespace ditree {

class Solver {
 public:
  explicit Solver(const SolverParameter& param, const int thread_id);
  explicit Solver(const string& param_file, const int thread_id);
  void Init();

  void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
 
 private: 
  void Test();
  void Update();

  void Snapshot();
  void Restore(const char* resume_file);

 private:
  SolverParameter param_;

  int client_id_;
  int thread_id_;

  Tree* tree_;
  Vertex* root_;

  Dataset* dataset_;

  int iter_;

  int display_counter_;
  int test_counter_;
  // print results on the fly
  int display_gap_;
  int test_display_gap_;
  petuum::HighResolutionTimer total_timer_;
  
};

} // namespace ditree

#endif
