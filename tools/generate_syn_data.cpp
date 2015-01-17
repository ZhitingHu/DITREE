
#include <fstream>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "ditree.hpp"
#include <cmath>

using namespace std;
using namespace ditree;

DEFINE_string(path, "", "Path of the dataset");
DEFINE_string(docs, "", "Filename of the docs");

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  Random* rand_generator = new Random(clock());

  // tree param
  int vocab_size = 8;
  FloatVec root = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; 
  FloatVec v0 = {5.0, 5.0, 5.0, 5.0, 0.1, 0.1, 0.1, 0.1}; 
  FloatVec v1 = {0.1, 0.1, 0.1, 0.1, 5.0, 5.0, 5.0, 5.0};
  CHECK_EQ(vocab_size, root.size());

  vector<FloatVec> nodes;
  nodes.push_back(root);
  nodes.push_back(v0);
  nodes.push_back(v1);

  string output_file = FLAGS_path + "/" + FLAGS_docs;
  LOG(INFO) << "Generating dataset " << output_file;
  fstream output(output_file.c_str(), ios::out | ios::binary);
  int num_docs_per_node = 1000;
  int num_words_per_doc = 100;
  int tot_num_docs = num_docs_per_node * nodes.size();
  output.write((char*)&tot_num_docs, sizeof(int));

  map<int, int> doc;
  for (int v=0; v<nodes.size(); ++v) {
    for (int i=0; i<num_docs_per_node; ++i) {
      doc.clear();
      for (int w=0; w<num_words_per_doc; ++w) {
        int word = rand_generator->randDiscrete(nodes[v], 0, vocab_size);
        doc[word]++;
      }
      map<int, int>::const_iterator it = doc.begin();
      float norm = 0;
      for (; it != doc.end(); ++it) {
        norm += it->second * it->second;
      }
      norm = sqrt(norm);
      int size = doc.size();
      output.write((char*)&size, sizeof(int));
      it = doc.begin();
      for (; it != doc.end(); ++it) {
        //output << it->first << ":" << it->second << " ";
        int word_id = it->first;
        float word_weight = it->second / norm;
        output.write((char*)&word_id, sizeof(int));
        output.write((char*)&word_weight, sizeof(float));
      } 
      //output << endl;
    }
  }
  output.close();

  LOG(INFO) << "Syn data generation done."; 

  return 0;
}
