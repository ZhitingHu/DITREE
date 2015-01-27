/**
 * Normalize each doc
 * Transform to binary files
 * Compute vocabulary size
 */

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "ditree.hpp"
#include <cmath>
#include <fstream>

using namespace std;
using namespace ditree;

DEFINE_string(path, "", "Path of the dataset");
DEFINE_string(docs, "", "Filename of the docs");

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  string input_file = FLAGS_path + "/" + FLAGS_docs;
  fstream input(input_file.c_str(), ios::in);
  CHECK(input.is_open()) << "File not found: " << input_file;

  string output_file = FLAGS_path + "/" + FLAGS_docs + ".bin";
  fstream output(output_file.c_str(), ios::out | ios::binary);
  CHECK(output.is_open()) << "Fail to create file: " << output_file;

  set<int> vocab;
 
  int num_doc;
  input >> num_doc;
  LOG(INFO) << "#doc " << num_doc;
  output.write((char*)&num_doc, sizeof(int)); 

  int doc_len, word_id;
  float word_weight_norm;
  int counter = 0;
  for (int d=0; d<num_doc; ++d) {
    input >> doc_len;
    CHECK_GT(doc_len, 0);
    vector<int> word_ids(doc_len);
    vector<float> word_weights(doc_len);
    word_weight_norm = 0;
    for (int i=0; i<doc_len; ++i) {
      input >> word_ids[i] >> word_weights[i];
      word_weight_norm += word_weights[i] * word_weights[i];
      vocab.insert(word_ids[i]);
    }
    CHECK_GT(word_weight_norm, 0);
    word_weight_norm = sqrt(word_weight_norm);
    output.write((char*)&doc_len, sizeof(int)); 
    for (int i=0; i<doc_len; ++i) {
      float normed_word_weight = word_weights[i] / word_weight_norm;
      output.write((char*)&word_ids[i], sizeof(int)); 
      output.write((char*)&normed_word_weight, sizeof(float)); 
    }
  }
  LOG(INFO) << "vocab size: " << vocab.size();

  input.close();
  output.close();
  
  LOG(INFO) << "Data transforming done."; 

  return 0;
}
