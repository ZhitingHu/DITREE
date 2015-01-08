
#include "ditree.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <petuum_ps_common/include/petuum_ps.hpp>

// Petuum Parameters
DEFINE_string(hostfile, "",
    "Path to file containing server ip:port.");
DEFINE_int32(num_clients, 1, 
    "Total number of clients");
DEFINE_int32(num_app_threads, 1, 
    "Number of app threads in this client");
DEFINE_int32(client_id, 0, 
    "Client ID");
DEFINE_string(consistency_model, "SSPPush", 
    "SSP or SSPPush");
DEFINE_string(stats_path, "", 
    "Statistics output file");
DEFINE_int32(num_comm_channels_per_client, 1,
    "number of comm channels per client");
DEFINE_int32(staleness, 0, 
    "staleness for weight tables.");
DEFINE_int32(loss_table_staleness, 5, 
    "staleness for loss tables.");
DEFINE_int32(row_oplog_type, petuum::RowOpLogType::kDenseRowOpLog,
    "row oplog type");
DEFINE_bool(oplog_dense_serialized, true, 
    "True to not squeeze out the 0's in dense oplog.");
DEFINE_string(process_storage_type, "BoundedDense", 
    "process storage type");

// PS Table Organization Paremeters
DEFINE_int32(max_depth, 10,
    "Maximum depth of a vertex.");
DEFINE_int32(max_num_child_per_node, 20,
    "Maximum number of children of a vertex.");
DEFINE_int32(max_num_parent_per_table, 5,
    "Maximum number of parent of a table.");
DEFINE_int32(num_layer_per_table, 2,
    "Number of node layers of a table.");
DEFINE_int32(num_table_id_digit, 8,
    "Number of digit for representing table id, must be in (0, 32).");

// DITree Parameters
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(ditree_outputs, "",
    "The prefix of the ditree output file.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_int32(history, 1,
    "Number of history time slice to consider.");


int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);


  LOG(INFO) << "DITree finished and shut down!";

  return 0;
}
