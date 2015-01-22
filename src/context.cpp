#include "context.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vector>
#include <time.h>

namespace ditree {

Context& Context::Get()
{
  static Context instance;
  return instance;
}

Context::Context() {
  std::vector<google::CommandLineFlagInfo> flags;
  google::GetAllFlags(&flags);
  for (size_t i = 0; i < flags.size(); i++) {
    google::CommandLineFlagInfo& flag = flags[i];
    ctx_[flag.name] = flag.is_default ? flag.default_value : flag.current_value;
  }
}

void Context::Init() {
  num_app_threads_ = get_int32("num_app_threads");
  phases_ = new Phase[num_app_threads_];
  for (int t_idx = 0; t_idx < num_app_threads_; ++t_idx) {
    set_phase(Context::Phase::kVIAfterSplit, t_idx);
  }

  num_table_id_bits_ = get_int32("num_table_id_bits");
  num_row_id_bits_ = kNumIntBits - num_table_id_bits_;

  vocab_size_ = get_int32("vocab_size");

  int rand_seed = get_int32("random_seed");
  if (rand_seed >= 0) {
    random_generator_ = new Random(rand_seed);
  } else {
    random_generator_ = new Random(time(NULL));
  }
}

// -------------------- Getters ----------------------

//int Context::get_int32(std::string key) {
//  return atoi(get_string(key).c_str());
//}
//
//double Context::get_double(std::string key) {
//  return atof(get_string(key).c_str());
//}
//
//bool Context::get_bool(std::string key) {
//  return get_string(key).compare("true") == 0;
//}
//
//std::string Context::get_string(std::string key) {
//  auto it = ctx_.find(key);
//  LOG_IF(FATAL, it == ctx_.end())
//      << "Failed to lookup " << key << " in context.";
//  return it->second;
//}

// -------------------- Setters ---------------------

//void Context::set(std::string key, int value) {
//  ctx_[key] = std::to_string(value);
//}
//
//void Context::set(std::string key, double value) {
//  ctx_[key] = std::to_string(value);
//}
//
//void Context::set(std::string key, bool value) {
//  ctx_[key] = (value) ? "true" : "false";
//}
//
//void Context::set(std::string key, std::string value) {
//  ctx_[key] = value;
//}

}   // namespace ditree
