#ifndef DITREE_DATASET_HPP_
#define DITREE_DATASET_HPP_

#include "common.hpp"
#include "context.hpp"
#include "data_batch.hpp"

namespace ditree {

class Dataset {
 public:
  explicit Dataset();
  
  DataBatch* GetNextDataBatch(); 

 private:

 private:

  vector<DataBatch*> data_; 

};

} // namespace ditree

#endif
