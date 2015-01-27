#ifndef DITREE_COMMON_HPP_
#define DITREE_COMMON_HPP_

#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>
#include <limits>
#include <algorithm>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <petuum_ps_common/include/petuum_ps.hpp>

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFAGS_H_ to detect if it is version
// 2.1. If yes , we will add a temporary solution to redirect the namespace.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

namespace ditree {

using std::fstream;
using std::ios;
using std::isnan;
using std::iterator;
using std::make_pair;
using std::vector;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::max;
using std::min;

class DataBatch;
class Dataset;
class Tree;
class Vertex;

// Constants
const int kNumIntBits = 32;
const float kFloatEpsilon = 1e-6;
enum RowTypes {
  //kDITreeDenseRowDtypeID = 0,
  kIntDenseRowDtypeID = 0,
  kFloatDenseRowDtypeID
};
enum TableIds {
  kParamTableID = 0,
  kTempParamTableID,
  kStructTableID,
  kTrainLossTableID,
  kTestLossTableID
};
  // param table row organization
enum ParamTableCols {
  //kColIdxParamTableLr = 0,
  kColIdxParamTableN = 0,
  //kColIdxParamTableTau0,
  //kColIdxParamTableTau1,
  //kColIdxParamTableSigma0,
  //kColIdxParamTableSigma1,
  //kColIdxParamTableKappa,
  //kColIdxParamTableMeanStart
  kColIdxParamTableSStart
};

const int kNumLossTableCols = 5;
enum LossTableCols {
  kColIdxLossTableEpoch = 0,
  kColIdxLossTableIter,
  kColIdxLossTableTime,
  kColIdxLossTableLoss,
  kColIdxLossTableNumDatum,
};

const int kNumStructTableRecordCols = 2;
enum StructTableSplitRecordCols {
  kColIdxParentVertexIdx = 0,
  kColIdxChildVertexIdx,
};

// Typedefs
typedef unsigned short uint16; // Should work for all x86/x64 compilers
typedef unsigned int   uint32; // Should work for all x86/x64 compilers
typedef vector<float> FloatVec;
typedef map<uint32, float> UIntFloatMap;
typedef pair<const uint32, float> UIntFloatPair;
typedef pair<const uint32, UIntFloatMap> UIntUIntFloatMapPair; 
typedef pair<const uint32, Vertex*> UIntVertexPair;
typedef pair<uint32, int> IdxCntPair;

} // namespace ditree

#endif
