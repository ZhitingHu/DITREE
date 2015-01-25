#ifndef DITREE_CONTEXT_HPP_
#define DITREE_CONTEXT_HPP_

#include "common.hpp"
#include "random.hpp"
#include "thread_barrier.hpp"
#include <unordered_map>
#include <boost/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>

namespace ditree {

// An extension of google flags. It is a singleton that stores 1) google flags
// and 2) other lightweight global flags. Underlying data structure is map of
// string and string, similar to google::CommandLineFlagInfo.
class Context {
 public:
  static Context& Get();
  void Init();

  static inline int get_int32(std::string key) {
    return atoi(get_string(key).c_str());
  }
  static inline double get_double(std::string key) {
    return atof(get_string(key).c_str());
  }
  static inline bool get_bool(std::string key) {
    return get_string(key).compare("true") == 0;
  }
  static inline std::string get_string(std::string key) {
    Get();
    auto it = Get().ctx_.find(key);
    LOG_IF(FATAL, it == Get().ctx_.end())
        << "Failed to lookup " << key << " in context.";
    return it->second;
  }

  static inline void set(std::string key, int value) {
    Get().ctx_[key] = std::to_string(value);
  }
  static inline void set(std::string key, double value) {
    Get().ctx_[key] = std::to_string(value);
  }
  static inline void set(std::string key, bool value) {
    Get().ctx_[key] = (value) ? "true" : "false";
  }
  static inline void set(std::string key, std::string value) {
    Get().ctx_[key] = value;
  }

  enum Phase { kInit, kSplit, kMerge, kVIAfterSplit, kVIAfterMerge };
  inline static Phase phase(const int thread_id) { 
    return Get().phases_[thread_id]; 
  }
  inline static void set_phase(Phase phase, const int thread_id) {
    Get().phases_[thread_id] = phase; 
  }

  inline static int num_app_threads() {
    return Get().num_app_threads_; 
  }
  inline static int vocab_size() {
    return Get().vocab_size_;
  }
  inline static int max_num_split_per_table() {
    return Get().max_num_split_per_table_;
  }

  // For sync
  //inline static void IncRestartThreadCounter() {
  //  Get().num_thread_ready_to_restart_++;
  //#ifdef DEBUG
  //  CHECK_LE(Get().num_thread_ready_to_restart_, Get().num_app_threads_);
  //#endif
  //}
  //inline static void ResetRestartThreadCounter() {
  //  Get().num_thread_ready_to_restart_ = 0;
  //}
  //inline static void WaitToRestart() {
  //  while(Get().num_thread_ready_to_restart_ 
  //      < Get().num_app_threads_) { }
  //}
 
  inline static void Wait() {
    Get().spanning_barrier_->Wait();
  }

  inline static float rand() {
    return Get().random_generator_->rand();
  }
  inline static unsigned int randInt() {
    return Get().random_generator_->randInt();
  }
  inline static size_t randDiscrete(const FloatVec& distrib, 
      size_t begin, size_t end) {
    return Get().random_generator_->randDiscrete(distrib, begin, end);
  }

  inline static petuum::Table<float>* param_table() {
    return &(Get().param_table_);
  }
  inline static petuum::Table<float>* temp_param_table() {
    return &(Get().temp_param_table_);
  }
  inline static petuum::Table<int>* struct_table() {
    return &(Get().struct_table_);
  }
  inline static void SetTables() {
    Get().param_table_ 
        = petuum::PSTableGroup::GetTableOrDie<float>(kParamTableID);
    Get().temp_param_table_ 
        = petuum::PSTableGroup::GetTableOrDie<float>(kTempParamTableID);
    Get().struct_table_
        = petuum::PSTableGroup::GetTableOrDie<int>(kStructTableID);
  }
  inline static int struct_table_row_length() {
    return Get().struct_table_row_length_;
  }
  inline static void set_struct_table_row_length(const int row_length) {
    Get().struct_table_row_length_ = row_length;
  }

  inline static uint32 table_id(const uint32 index) {
    return index >> Get().num_row_id_bits_;
  }
  inline static uint32 row_id(const uint32 index) {
    return (index << Get().num_table_id_bits_) >> Get().num_table_id_bits_;
  }
  inline static uint32 make_vertex_id(const uint32 table_id,
      const uint32 row_id) {
    CHECK_LT(table_id, 1 << Get().num_table_id_bits_);
    CHECK_LT(row_id, 1 << Get().num_row_id_bits_);
    return (table_id << Get().num_row_id_bits_) + row_id;
  }
 private:
  // Private constructor. Store all the gflags values.
  Context();

 private:
  // Underlying data structure
  std::unordered_map<std::string, std::string> ctx_;

  vector<Phase> phases_;
  int num_app_threads_;
  int num_table_id_bits_;
  int num_row_id_bits_;
  int vocab_size_;
  int max_num_split_per_table_;

  Random* random_generator_;

  petuum::Table<float> param_table_;
  petuum::Table<float> temp_param_table_;
  petuum::Table<int> struct_table_;
  int struct_table_row_length_;

  // for synchronization
  SpinningBarrier* spanning_barrier_;
};

}   // namespace ditree

#endif
