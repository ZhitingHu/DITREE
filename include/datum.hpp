#ifndef DITREE_DATUM_HPP_
#define DITREE_DATUM_HPP_

#include "common.hpp"
#include "context.hpp"

namespace ditree {

class Datum {
 public:
  explicit Datum();
  
  const IdxWeightMap& data() { return data_; }
 private:

 private:
  IdxWeightMap data_;

};

} // namespace ditree

#endif
