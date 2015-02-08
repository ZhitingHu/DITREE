
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "ditree.hpp"
#include <cmath>
#include <fstream>

using namespace std;
using namespace ditree;

DEFINE_string(path, "", "Path of the dataset");
DEFINE_string(docs, "", "Filename of the docs");
DEFINE_int32(vocab_size, 0, "Size of the vocabulary");

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  string input_file = FLAGS_path + "/" + FLAGS_docs;
  fstream input(input_file.c_str(), ios::in | ios::binary);
  CHECK(input.is_open()) << "File not found: " << input_file;

  string output_file = FLAGS_path + "/" + FLAGS_docs + "_mean.txt";
  fstream output(output_file.c_str(), ios::out);
  CHECK(output.is_open()) << "Fail to create file: " << output_file;

  CHECK_GT(FLAGS_vocab_size, 0); 
  FloatVec mean(FLAGS_vocab_size);
 
  int num_doc, doc_len, word_id;
  float word_weight;
  int counter = 0;
  input.read((char*)&num_doc, sizeof(int)); 
  LOG(INFO) << "Total number of docs: " << num_doc;
  for (int d_idx = 0; d_idx < num_doc; ++d_idx) {
    // format: doc_len [word_id word_cnt]+
    input.read((char*)&doc_len, sizeof(int));
    for (int w_idx = 0; w_idx < doc_len; ++w_idx) {
      input.read((char*)&word_id, sizeof(int));
      input.read((char*)&word_weight, sizeof(float));
      mean[word_id] += word_weight;
    }
    counter++;
    if (counter % (num_doc / 5) == 0) {
      LOG(INFO) << "Finish reading " << (counter / (num_doc / 5)) 
          << "/5 docs"; 
    }
  }
  input.close();

  // average & normalize
  float mean_norm = 0;
  for (int v = 0; v < FLAGS_vocab_size; ++v) {
    mean[v] /= num_doc;
    mean_norm += mean[v] * mean[v];
  }
  mean_norm = sqrt(mean_norm);
  for (int v = 0; v < FLAGS_vocab_size; ++v) {
    output << mean[v] / mean_norm << endl;
  }
  output.flush();
  output.close();
  
  LOG(INFO) << "Mean vector generation done."; 

  return 0;
}
