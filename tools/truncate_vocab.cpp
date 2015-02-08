
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "ditree.hpp"
#include <cmath>
#include <fstream>

using namespace std;
using namespace ditree;

DEFINE_string(path, "", "Path of the dataset");
DEFINE_string(docs, "", "Filename of the docs");
DEFINE_string(vocab, "", "Filename of the vocab");
DEFINE_int32(topk, 0, "Truncate level of vocab");

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  //  
  map<int, string> word_strs;
  string vocab_file = FLAGS_path + "/" + FLAGS_vocab;
  fstream vocab_input(vocab_file.c_str(), ios::in);
  CHECK(vocab_input.is_open()) << "File not found: " << vocab_input;
  int word_id = 0;
  string word_str;
  while (getline(vocab_input, word_str)) {
    word_strs[word_id] = word_str;
    ++word_id;
  }
  vocab_input.close();
  LOG(INFO) << "Size of vocab: " << word_id << " (" << word_strs.size() << ")";
  CHECK_LE(FLAGS_topk, word_id);

  //
  int max_word_id = -1; 
  map<int, float> word_weights;
  string input_file = FLAGS_path + "/" + FLAGS_docs;
  fstream input(input_file.c_str(), ios::in | ios::binary);
  CHECK(input.is_open()) << "File not found: " << input_file;
  int num_doc, doc_len;
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
      word_weights[word_id] += word_weight;
      max_word_id = max(max_word_id, word_id);
    }
    counter++;
    if (counter % (num_doc / 5) == 0) {
      LOG(INFO) << "Finish reading " << (counter / (num_doc / 5)) 
          << "/5 docs"; 
    }
  }
  input.close();

  CHECK_LT(max_word_id, word_strs.size());
  
  // Sort
  vector<pair<int, float> > word_weights_sort;
  for (const auto& ele : word_weights) {
    word_weights_sort.push_back(ele);
  } 
  std::partial_sort(word_weights_sort.begin(), word_weights_sort.begin() + FLAGS_topk, 
      word_weights_sort.end(), DesSortBySecondOfIntFloatPair());
  LOG(INFO) << "Most freq word " << word_strs[word_weights_sort[0].first]
      << " (" << word_weights_sort[0].second << ")";
  LOG(INFO) << "Least freq word " << word_strs[word_weights_sort[word_weights_sort.size() - 1].first]
      << " (" << word_weights_sort[word_weights_sort.size() - 1].second << ")";

  // old_word_id => new_word_id
  map<int, int> word_id_map;
  for (int i = 0; i < FLAGS_topk; ++i) {
    word_id_map[word_weights_sort[i].first] = i;
  }

  // Output
  string out_vocab_file = FLAGS_path + "/" + FLAGS_vocab + ".trunc";
  fstream out_vocab(out_vocab_file.c_str(), ios::out);
  CHECK(out_vocab.is_open()) << "Fail to create file: " << out_vocab_file;
  for (int i = 0; i < FLAGS_topk; ++i) {
    CHECK(word_strs.find(word_weights_sort[i].first) != word_strs.end())
        << word_weights_sort[i].first << " " << i;
    out_vocab << word_strs[word_weights_sort[i].first] << endl;
  }
  out_vocab.close();

  string out_doc_file = FLAGS_path + "/" + FLAGS_docs + ".trunc.txt";
  fstream out_doc(out_doc_file.c_str(), ios::out);
  CHECK(out_doc.is_open()) << "Fail to create file: " << out_doc_file;

  input.open(input_file.c_str(), ios::in | ios::binary);
  CHECK(input.is_open()) << "File not found: " << input_file;
  counter = 0;
  input.read((char*)&num_doc, sizeof(int)); 
  LOG(INFO) << "Total number of docs: " << num_doc;
  for (int d_idx = 0; d_idx < num_doc; ++d_idx) {
    map<int, float> doc;
    // format: doc_len [word_id word_cnt]+
    input.read((char*)&doc_len, sizeof(int));
    for (int w_idx = 0; w_idx < doc_len; ++w_idx) {
      input.read((char*)&word_id, sizeof(int));
      input.read((char*)&word_weight, sizeof(float));
      if (word_id_map.find(word_id) != word_id_map.end()) {
        doc[word_id_map[word_id]] = word_weight;
      }
    } // end of reading doc

    if (doc.size() > 0) {
      // normalize
      float doc_norm = 0;
      for (const auto& ele : doc) {
        doc_norm += ele.second * ele.second;
      }
      doc_norm = sqrt(doc_norm);
      CHECK_GT(doc_norm, 0);
      out_doc << doc.size() << " ";
      for (auto& ele : doc) {
        out_doc << ele.first << " " << ele.second << " ";
      }
      out_doc << endl;
      //doc_len = doc.size();
      //out_doc.write((char*)&doc_len, sizeof(int));
      //for (auto& ele : doc) {
      //  out_doc.write((char*)&ele.first, sizeof(int));
      //  out_doc.write((char*)&ele.second, sizeof(float));
      //}
    }

    counter++;
    if (counter % (num_doc / 5) == 0) {
      LOG(INFO) << "Finish reading " << (counter / (num_doc / 5)) 
          << "/5 docs"; 
    }
  }
  input.close();
  out_doc.close();
   
  LOG(INFO) << "Vocab truncation done."; 

  return 0;
}
