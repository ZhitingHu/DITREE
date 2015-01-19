#ifndef DITREE_DATUM_HPP_
#define DITREE_DATUM_HPP_

#include "common.hpp"
#include "context.hpp"

namespace ditree {

class Datum {
 public:
  explicit Datum() { }
  
  inline void AddWord(const int word_id, const float word_weight) {
#ifdef DEBUG
    CHECK(data_.find(word_id) == data_.end());
#endif
    data_[word_id] = word_weight;
  }
  const UIntFloatMap& data() const { return data_; }
 private:

 private:
  UIntFloatMap data_;
};

} // namespace ditree

#endif
