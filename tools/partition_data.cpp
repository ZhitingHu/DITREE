
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "ditree.hpp"
#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <time.h>

using namespace std;
using namespace ditree;

DEFINE_string(path, "", "Path of the dataset");
DEFINE_string(docs, "", "Filename of the docs");
DEFINE_int32(num_partitions, 2, "The number of partitions");

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  const int np = FLAGS_num_partitions;
  CHECK_GT(np, 0);
 
  string input_file = FLAGS_path + "/" + FLAGS_docs;
  fstream input(input_file.c_str(), ios::in | ios::binary);
  CHECK(input.is_open()) << "File not found: " << input_file;

  string output_file = input_file;
  fstream outputs[np];
  for (int p = 0; p < np; ++p) {
    ostringstream oss;
    oss << output_file << "_" << p;
    outputs[p].open(oss.str().c_str(), ios::out | ios::binary);
    CHECK(outputs[p].is_open()) << "Fail to create file: " << oss.str();
  }

  int num_doc, doc_len, word_id;
  float word_weight;
  int counter = 0;
  input.read((char*)&num_doc, sizeof(int)); 
  LOG(INFO) << "Total number of docs: " << num_doc;
  int num_doc_per_part = (num_doc + np - 1) / np;
  int num_doc_last_part = num_doc - (num_doc_per_part * (np - 1));
  for (int p = 0; p < np - 1; ++p) {
    LOG(INFO) << "num docs on partition " << p << ": " << num_doc_per_part;
    outputs[p].write((char*)&num_doc_per_part, sizeof(int));
  }
  LOG(INFO) << "num docs on partition " << (np-1) << ": " << num_doc_last_part;
  outputs[np - 1].write((char*)&num_doc_last_part, sizeof(int));

  int d_idx = 0;
  while (d_idx < num_doc) {
    for (int p = 0; p < np; ++p) {
      UIntFloatMap doc;
      // format: doc_len [word_id word_cnt]+
      input.read((char*)&doc_len, sizeof(int));
      for (int w_idx = 0; w_idx < doc_len; ++w_idx) {
        input.read((char*)&word_id, sizeof(int));
        input.read((char*)&word_weight, sizeof(float));
        doc[word_id] = word_weight;
      }

      outputs[p].write((char*)&doc_len, sizeof(int));
      for (const auto& ele : doc) {
        outputs[p].write((char*)&ele.first, sizeof(int));
        outputs[p].write((char*)&ele.second, sizeof(float));
      }

      counter++;
      if (counter % (num_doc / 5) == 0) {
        LOG(INFO) << "Finish " << (counter / (num_doc / 5)) 
            << "/5 docs"; 
      }

      ++d_idx;
      if (d_idx >= num_doc) {
        break;
      }
    }
  }
  input.close();
  for (int p = 0; p < np; ++p) {
    outputs[p].close();
  } 

  LOG(INFO) << "Data partition done."; 

  return 0;
}
