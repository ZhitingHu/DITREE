#ifndef DITREE_CONTEXT_HPP_
#define DITREE_CONTEXT_HPP_

#include "common.hpp"

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

  int get_int32(std::string key);
  double get_double(std::string key);
  bool get_bool(std::string key);
  std::string get_string(std::string key);

  void set(std::string key, int value);
  void set(std::string key, double value);
  void set(std::string key, bool value);
  void set(std::string key, std::string value);

  enum Phase { SPLIT, MERGE, VI_AFTER_SPLIT, VI_AFTER_MERGE };
//  inline static Phase phase(const int thread_id) { 
//#ifdef DEBUG
//    CHECK(Get().phases_ != NULL);
//#endif
//    return Get().phases_[thread_id]; 
//  }
  inline static Phase phase() {
    return Get().phase_;
  }
  inline static void set_phase(Phase phase) { Get().phase_ = phase; }

  inline static int num_app_threads() {
    return Get().num_app_threads_; 
  }
  inline static float kappa_0() {
    return Get().kappa_0_;
  }
  inline static float kappa_1() {
    return Get().kappa_1_;
  }
  inline static float beta() {
    return Get().beta_;
  }

  inline static petuum::Table<float>* param_table() {
    return Get().param_table_;
  }
  inline static petuum::Table<float>* struct_table() {
    return Get().struct_table_;
  }
  inline static petuum::Table<float>* param_table_meta_table() {
    return Get().param_table_meta_table_;
  }
  inline static void SetTables() {
    Get().param_table_ 
        = petuum::PSTableGroup::GetTableOrDie<float>(kParamTableID);
    Get().struct_table_
        = petuum::PSTableGroup::GetTableOrDie<int>(kStructTableID);
    Get().param_table_meta_table_
        = petuum::PSTableGroup::GetTableOrDie<int>(kParamTableMetaTableID);
  }

  inline static uint32 table_id(const uint32 index) {
    return index >> Get().num_row_id_bits_;
  }
  inline static uint32 row_id(const uint32 index) {
    return (index << Get().num_table_id_bits_) >> Get().num_table_id_bits_;
  }

private:
  // Private constructor. Store all the gflags values.
  Context();
  Init();
  // Underlying data structure
  std::unordered_map<std::string, std::string> ctx_;

  //Phase* phases_;
  Phase phase_;
  int num_app_threads_;
  int num_table_id_bits_;
  int num_row_id_bits_;
  float kappa_0_;
  float kappa_1_;
  float kappa_2_;
  float beta_;

  petuum::Table<float>* param_table_;
  petuum::Table<float>* struct_table_;
  petuum::Table<float>* param_table_meta_table_;
};

}   // namespace ditree

#endif
