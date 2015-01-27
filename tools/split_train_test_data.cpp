
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
DEFINE_int32(percent, 10, "Takes percent% data as test data");

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  srand(time(NULL));
 
  string input_file = FLAGS_path + "/" + FLAGS_docs;
  fstream input(input_file.c_str(), ios::in | ios::binary);
  CHECK(input.is_open()) << "File not found: " << input_file;

  string train_file = FLAGS_path + "/" + FLAGS_docs + ".train";
  fstream train(train_file.c_str(), ios::out | ios::binary);
  CHECK(train.is_open()) << "Fail to create file: " << train_file;
  string test_file = FLAGS_path + "/" + FLAGS_docs + ".test";
  fstream test(test_file.c_str(), ios::out | ios::binary);
  CHECK(test.is_open()) << "Fail to create file: " << test_file;

  float test_prob = FLAGS_percent / 100.0;
  
  int num_doc, doc_len, word_id;
  float word_weight;
  int counter = 0;
  input.read((char*)&num_doc, sizeof(int)); 

  int num_test = num_doc * test_prob;
  int num_train = num_doc - num_test;
  test.write((char*)&num_test, sizeof(int));
  train.write((char*)&num_train, sizeof(int));

  LOG(INFO) << "Total number of docs: " << num_doc 
      << " #train: " << num_train << " #test: " << num_test;

  int cur_num_test = 0;
  int cur_num_train = 0;
  for (int d_idx = 0; d_idx < num_doc; ++d_idx) {
    UIntFloatMap doc;
    // format: doc_len [word_id word_cnt]+
    input.read((char*)&doc_len, sizeof(int));
    for (int w_idx = 0; w_idx < doc_len; ++w_idx) {
      input.read((char*)&word_id, sizeof(int));
      input.read((char*)&word_weight, sizeof(float));
      doc[word_id] = word_weight;
    }

    if (((float)rand() / RAND_MAX < test_prob && cur_num_test < num_test) 
        || cur_num_train >= num_train) {
      // set as test doc
      test.write((char*)&doc_len, sizeof(int));
      for (const auto& ele : doc) {
        test.write((char*)&ele.first, sizeof(int));
        test.write((char*)&ele.second, sizeof(float));
      }
      cur_num_test++;
    } else {
      // set as train doc
      train.write((char*)&doc_len, sizeof(int));
      for (const auto& ele : doc) {
        train.write((char*)&ele.first, sizeof(int));
        train.write((char*)&ele.second, sizeof(float));
      }
      cur_num_train++;
    } 

    counter++;
    if (counter % (num_doc / 5) == 0) {
      LOG(INFO) << "Finish " << (counter / (num_doc / 5)) 
          << "/5 docs"; 
    }
  }
  input.close();
  test.close();
  train.close();

  CHECK_EQ(cur_num_test, num_test);
  CHECK_EQ(cur_num_train, num_train);

  LOG(INFO) << "Splitting train-test done."; 

  return 0;
}
