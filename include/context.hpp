#ifndef DITREE_CONTEXT_HPP_
#define DITREE_CONTEXT_HPP_

#include "common.hpp"

#include <unordered_map>
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
  
  inline static int num_app_threads() {
    return Get().num_app_threads_; 
  }

  inline static uint32 table_id(const uint32 index) {
    return index >> Get().row_id_bit_size_;
  }
  inline static uint32 row_id(const uint32 index) {
    return (index << Get().table_id_bit_size_) >> Get().table_id_bit_size_;
  }

private:
  // Private constructor. Store all the gflags values.
  Context();
  // Underlying data structure
  std::unordered_map<std::string, std::string> ctx_;

  int num_app_threads_;
  int table_id_bit_size_;
  int row_id_bit_size_;
};

}   // namespace ditree

#endif
