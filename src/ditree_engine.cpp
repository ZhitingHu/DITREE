
#include "common.hpp"
#include "context.hpp"
#include "ditree_engine.hpp"
#include "solver.hpp"

namespace ditree {

DITreeEngine::DITreeEngine(const SolverParameter& param)
    : thread_counter_(0) {
  solver_param_ = param;
}

void DITreeEngine::Init() {
  LOG(INFO) << "Init PS environment";
  petuum::TableGroupConfig table_group_config;
  table_group_config.num_comm_channels_per_client
      = Context::get_int32("num_comm_channels_per_client");
  table_group_config.num_total_clients 
      = Context::get_int32("num_clients");
  // + 1 for main() thread.
  table_group_config.num_local_app_threads 
      = Context::get_int32("num_app_threads") + 1;
  table_group_config.client_id = Context::get_int32("client_id");
  table_group_config.stats_path = Context::get_string("stats_path");
  petuum::GetHostInfos(Context::get_string("hostfile"), 
      &table_group_config.host_map);
  string consistency_model = Context::get_string("consistency_model");
  if (std::string("SSP").compare(consistency_model) == 0) {
    table_group_config.consistency_model = petuum::SSP;
  } else if (
    std::string("SSPPush").compare(consistency_model) == 0) {
    table_group_config.consistency_model = petuum::SSPPush;
  } else if (
    std::string("LocalOOC").compare(consistency_model) == 0) {
    table_group_config.consistency_model = petuum::LocalOOC;
  } else {
    LOG(FATAL) << "Unkown consistency model: " << consistency_model;
  }
  //TODO snapshot / resume
  // ...
  // param_table, struct_table, param_table_meta_table, loss_table
  table_group_config.num_tables = 5;
  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<float> >
    (kFloatDenseRowDtypeID);
  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<int> >
    (kIntDenseRowDtypeID);
  // Use false to not let main thread access table API.
  petuum::PSTableGroup::Init(table_group_config, false);
  LOG(INFO) << "Init table group done.";

  CreateTables();
}

void DITreeEngine::CreateTables() {
  LOG(INFO) << "Create tables.";
  int param_table_staleness = Context::get_int32("param_table_staleness");
  int loss_table_staleness = Context::get_int32("loss_table_staleness");
  int row_oplog_type = Context::get_int32("row_oplog_type");
  bool oplog_dense_serialized = Context::get_bool("oplog_dense_serialized");
  int max_num_vertexes = Context::get_int32("max_num_vertexes");
  int max_num_tables = (1 << Context::get_int32("num_table_id_bits"));
  int num_threads = Context::get_int32("num_app_threads");
  int tot_num_threads = Context::get_int32("num_clients") * num_threads;
  int max_num_split_per_table = Context::get_int32("max_split_per_table");
  // common table config
  petuum::ClientTableConfig table_config;
  table_config.table_info.row_oplog_type = row_oplog_type;
  table_config.table_info.oplog_dense_serialized 
      = oplog_dense_serialized;
  //TODO: build a sparse vertex_idx to dense row_idx map
  table_config.process_storage_type = petuum::BoundedSparse;

  // param table
  int param_row_length
      = kColIdxParamTableSStart + Context::vocab_size();
  table_config.table_info.row_type = ditree::kFloatDenseRowDtypeID;
  table_config.table_info.table_staleness = param_table_staleness;
  table_config.table_info.row_capacity = param_row_length;
  table_config.process_cache_capacity = max_num_vertexes + 5; //TODO
  table_config.table_info.dense_row_oplog_capacity = param_row_length;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kParamTableID, table_config);
  LOG(INFO) << "Created param table " << kParamTableID;

  // struct table
  int struct_row_length
      = (max_num_tables + tot_num_threads - 1) / tot_num_threads 
      * max_num_split_per_table * kNumStructTableRecordCols + 1;
    // the 1st row has length of tot_num_threads
  struct_row_length = max(struct_row_length, tot_num_threads + 1);
  Context::set_struct_table_row_length(struct_row_length);
  LOG(INFO) << "Struct table row capacity " << struct_row_length;
  table_config.table_info.row_type = ditree::kIntDenseRowDtypeID;
  table_config.table_info.table_staleness = 0; 
  table_config.table_info.row_capacity = struct_row_length;
  table_config.process_cache_capacity = tot_num_threads + 100;
  table_config.table_info.dense_row_oplog_capacity = 10;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kStructTableID, table_config);
  LOG(INFO) << "Created struct table " << kStructTableID;
  // param table meta table
  table_config.table_info.row_type = ditree::kIntDenseRowDtypeID;
  table_config.table_info.table_staleness = 0;
  table_config.table_info.row_capacity = 10; //TODO
  table_config.process_cache_capacity = max_num_vertexes; //TODO
  table_config.table_info.dense_row_oplog_capacity = 10;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kParamTableMetaTableID, table_config);
  LOG(INFO) << "Created param table meta table " << kParamTableMetaTableID;
  // train loss table
  const int max_iter_per_epoch 
      = (train_data_.batch_num() + num_threads - 1) / num_threads
      + param_table_staleness + 1; 
  const int num_rows_train_loss_table
      = solver_param_.max_epoch() * max_iter_per_epoch 
      / solver_param_.display() + 1;
  CHECK_GT(num_rows_train_loss_table, 0);
  table_config.table_info.row_type = ditree::kFloatDenseRowDtypeID;
  table_config.table_info.table_staleness = loss_table_staleness;
  table_config.table_info.row_capacity = kNumLossTableCols;
  table_config.process_cache_capacity = num_rows_train_loss_table + 5;
  table_config.table_info.dense_row_oplog_capacity = 10;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kTrainLossTableID, table_config);
  LOG(INFO) << "Created train loss table " << kTrainLossTableID;
  // test loss table
  const int num_rows_test_loss_table
      = solver_param_.max_epoch() * max_iter_per_epoch 
      / solver_param_.test_interval() + 1;
  CHECK_GT(num_rows_test_loss_table, 0);
  table_config.table_info.row_capacity = kNumLossTableCols;
  table_config.process_cache_capacity = num_rows_test_loss_table + 5;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kTestLossTableID, table_config);
  LOG(INFO) << "Created test loss table " << kTestLossTableID;
  
  petuum::PSTableGroup::CreateTableDone(); 

  Context::SetTables();
}

void DITreeEngine::ReadData() {
  const string& data_file = Context::get_string("data");
  LOG(INFO) << "Reading data " << data_file;
  train_data_.Init(data_file); 
}

void DITreeEngine::Start() {
  petuum::PSTableGroup::RegisterThread();

  // Initialize local thread data structures.
  int thread_id = thread_counter_++;

  int client_id = Context::get_int32("client_id");
  //const string& solver_path = Context::get_string("solver");
  const string& snapshot_path = Context::get_string("snapshot");
  const string& params_path = Context::get_string("params");
  //const string& tree_outputs_prefix = Context::get_string("tree_outputs");

  Solver* solver = new Solver(solver_param_, thread_id, &train_data_); 

  LOG(INFO) << "Start " << client_id << " " << thread_id << " " << thread_counter_; 

  if (snapshot_path.size()) {
    if (client_id == 0 && thread_id == 0) {
      LOG(INFO) << "Resuming from " << snapshot_path;
    }
    solver->Solve(snapshot_path);
  } else if (params_path.size()) {
    if (client_id == 0 && thread_id == 0) {
      LOG(INFO) << "Finetuning from " << params_path;
      //TODO
      NOT_IMPLEMENTED;
      //solver->net()->CopyTrainedLayersFrom(weights_path, true);
    }
    solver->Solve();
  } else {
    solver->Solve();
  }
  
  petuum::PSTableGroup::GlobalBarrier();
  if (client_id == 0 && thread_id == 0) {
    LOG(INFO) << "Output training results.";
    //solver->PrintNetOutputs(net_outputs_prefix + ".netoutputs");
  }
  petuum::PSTableGroup::GlobalBarrier();

  petuum::PSTableGroup::DeregisterThread();
}

}
