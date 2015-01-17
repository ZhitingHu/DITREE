
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "ditree.hpp"
#include <cmath>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace ditree;

DEFINE_string(path, "", "Path of the dataset");
DEFINE_string(docs, "", "Filename of the docs");

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  string input_file = FLAGS_path + "/" + FLAGS_docs;
  fstream input(input_file.c_str(), ios::in | ios::binary);
  CHECK(input.is_open()) << "File not found: " << input_file;

  string output_file = FLAGS_path + "/" + FLAGS_docs + ".shuffled";
  fstream output(output_file.c_str(), ios::out | ios::binary);
  //fstream output(output_file.c_str(), ios::out);
  CHECK(output.is_open()) << "Fail to create file: " << output_file;

  vector<UIntFloatMap> docs;
  int num_doc, doc_len, word_id;
  float word_weight;
  int counter = 0;
  input.read((char*)&num_doc, sizeof(int)); 
  LOG(INFO) << "Total number of docs: " << num_doc;
  for (int d_idx = 0; d_idx < num_doc; ++d_idx) {
    UIntFloatMap doc;
    // format: doc_len [word_id word_cnt]+
    input.read((char*)&doc_len, sizeof(int));
    for (int w_idx = 0; w_idx < doc_len; ++w_idx) {
      input.read((char*)&word_id, sizeof(int));
      input.read((char*)&word_weight, sizeof(float));
      doc[word_id] = word_weight;
    }
    counter++;
    if (counter % (num_doc / 5) == 0) {
      LOG(INFO) << "Finish reading " << (counter / (num_doc / 5)) 
          << "/5 docs"; 
    }
    docs.push_back(doc);
  }
  input.close();

  random_shuffle(docs.begin(), docs.end());  
  
  output.write((char*)&num_doc, sizeof(int));
  for (int d_idx = 0; d_idx < num_doc; ++d_idx) {
    const UIntFloatMap& doc = docs[d_idx];
    int size = doc.size();
    output.write((char*)&size, sizeof(int));
    BOOST_FOREACH(const UIntFloatPair& ele, doc) {
      output.write((char*)&ele.first, sizeof(int)); 
      output.write((char*)&ele.second, sizeof(float)); 
      //output << ele.first << ":" << ele.second << " ";
    }
    //output << endl;
  }
  output.close();
  
  LOG(INFO) << "Data shuffling done."; 

  return 0;
}
