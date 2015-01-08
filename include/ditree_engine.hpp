#ifndef DITREE_DITREE_ENGINE_HPP_
#define DITREE_DITREE_ENGINE_HPP_

#include "common.hpp"
#include "context.hpp"
#include "dataset.hpp"
#include <atomic>

namespace ditree {

class DITreeEngine {
 public:
  explicit DITreeEngine();
  
  void Init();

  void ReadData();

  // Can be called concurrently.  
  void Start();
 
 private:

 private:

  Dataset train_data_;

  // 
  std::atomic<int> thread_counter_;
};

} // namespace ditree

#endif
