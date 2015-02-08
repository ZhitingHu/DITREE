
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

  int max_word_id = -1;
  set<int> vocab;
  float num_tokens = 0;
 
  int num_doc, vocab_size, num_triples;
  input >> num_doc >> vocab_size >> num_triples;
  LOG(INFO) << "#doc " << num_doc << "; vocab " 
      << vocab_size << " #triples " << num_triples;
  output.write((char*)&num_doc, sizeof(int)); 

  int max_doc_id = 1;
  int doc_id, word_id, count;
  float word_weight_norm;
  int counter = 0;

  bool end = false;
  input >> doc_id >> word_id >> count;
  for (int d=0; d<num_doc; ++d) {
    vector<int> word_ids;
    vector<float> word_weights;
    word_weight_norm = 0;
    while (true) {
      // The original dataset use 1 as start index
      word_ids.push_back(word_id - 1);
      word_weights.push_back(count);
      word_weight_norm += count * count;
      if (input >> doc_id >> word_id >> count) {
        CHECK_GE(doc_id, max_doc_id);
      } else {
        end = true;
      }
      if (doc_id > max_doc_id || end) {
        // write cur doc
        CHECK_GT(word_weight_norm, 0);
        word_weight_norm = sqrt(word_weight_norm);
        int doc_len = word_ids.size();
        output.write((char*)&doc_len, sizeof(int)); 
        for (int i=0; i < doc_len; ++i) {
          float normed_word_weight = word_weights[i] / word_weight_norm;
          output.write((char*)&word_ids[i], sizeof(int)); 
          output.write((char*)&normed_word_weight, sizeof(float)); 
        }
        // start next doc
        max_doc_id = doc_id;
        break;
      }
    }

    if (d % (num_doc / 10) == 0) {
      LOG(INFO) << "Finish reading " << (d / (num_doc / 10)) 
          << "/10 docs"; 
    }
    if (end) {
      break;
    } 
  } // end of reading
  CHECK_EQ(max_doc_id, num_doc);

  input.close();
  output.close();
  
  LOG(INFO) << "Pubmed transforming done."; 

  return 0;
}
