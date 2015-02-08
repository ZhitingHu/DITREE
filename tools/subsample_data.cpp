
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
DEFINE_int32(subsample, 25, "Percent to be subsampled");

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  string input_file = FLAGS_path + "/" + FLAGS_docs;
  fstream input(input_file.c_str(), ios::in | ios::binary);
  CHECK(input.is_open()) << "File not found: " << input_file;

  fstream output;
  ostringstream oss;
  oss << input_file << ".sub" << FLAGS_subsample;
  output.open(oss.str().c_str(), ios::out | ios::binary);
  CHECK(output.is_open()) << "Fail to create file: " << oss.str();

  int num_doc, doc_len, word_id;
  float word_weight;
  int counter = 0;
  input.read((char*)&num_doc, sizeof(int)); 
  LOG(INFO) << "Total number of docs: " << num_doc;
  int num_subsample_doc = num_doc * FLAGS_subsample / 100;
  LOG(INFO) << "num subsample docs " << num_subsample_doc;
  output.write((char*)&num_subsample_doc, sizeof(int));

  for (int p = 0; p < num_subsample_doc; ++p) {
    UIntFloatMap doc;
    // format: doc_len [word_id word_cnt]+
    input.read((char*)&doc_len, sizeof(int));
    for (int w_idx = 0; w_idx < doc_len; ++w_idx) {
      input.read((char*)&word_id, sizeof(int));
      input.read((char*)&word_weight, sizeof(float));
      doc[word_id] = word_weight;
    }

    output.write((char*)&doc_len, sizeof(int));
    for (const auto& ele : doc) {
      output.write((char*)&ele.first, sizeof(int));
      output.write((char*)&ele.second, sizeof(float));
    }

    counter++;
    if (counter % (num_subsample_doc / 5) == 0) {
      LOG(INFO) << "Finish " << (counter / (num_subsample_doc / 5)) 
          << "/5 subsample docs"; 
    }
  }
  input.close();
  output.close();

  LOG(INFO) << "Data subsample done."; 

  return 0;
}
