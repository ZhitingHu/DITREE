
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
  // param_table, struct_table, temp_param_table, loss_tables
  table_group_config.num_tables = kNumPSTables;
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
  int max_num_merge_per_table = Context::get_int32("max_merge_per_table");
  // common table config
  petuum::ClientTableConfig table_config;
  table_config.table_info.row_oplog_type = row_oplog_type;
  table_config.table_info.oplog_dense_serialized 
      = oplog_dense_serialized;
  table_config.process_storage_type = petuum::BoundedSparse;

  // param table
  int param_row_length
      = kColIdxParamTableSStart + Context::vocab_size();
  table_config.table_info.row_type = ditree::kFloatDenseRowDtypeID;
  table_config.table_info.table_staleness = param_table_staleness;
  table_config.table_info.row_capacity = param_row_length;
  table_config.process_cache_capacity = max_num_vertexes * 10 + 5;
  table_config.table_info.dense_row_oplog_capacity = param_row_length + 10;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kParamTableID, table_config);
  LOG(INFO) << "Created param table " << kParamTableID;
  // temp param table
  int temp_param_table_row_num 
    = max(max_num_split_per_table, max_num_merge_per_table) * max_num_tables;
  table_config.table_info.table_staleness = 0;
  table_config.process_cache_capacity = temp_param_table_row_num + 5;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kTempParamTableID, table_config);
  LOG(INFO) << "Created temp param table " << kTempParamTableID;

  // struct table
  int struct_row_length
      = (max_num_tables + tot_num_threads - 1) / tot_num_threads 
      * max_num_split_per_table * kNumStructTableRecordCols + 1;
  // the 1st row has length of tot_num_threads
  struct_row_length = max(struct_row_length, tot_num_threads);
  //int struct_row_length
  //    = max(max_num_split_per_table * kNumStructTableRecordCols);
  //  // the 1st row has length of tot_num_threads
  //struct_row_length = max(struct_row_length, max_num_tables);
  Context::set_struct_table_row_length(struct_row_length);
  LOG(INFO) << "Struct table row capacity " << struct_row_length;
  table_config.table_info.row_type = ditree::kIntDenseRowDtypeID;
  table_config.table_info.table_staleness = 0; 
  table_config.table_info.row_capacity = struct_row_length;
  table_config.process_cache_capacity = tot_num_threads + 10;
  //table_config.process_cache_capacity = max_num_tables + 10;
  table_config.table_info.dense_row_oplog_capacity = struct_row_length + 10;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kStructTableID, table_config);
  LOG(INFO) << "Created struct table " << kStructTableID;

  // train loss table
  const int max_iter_per_epoch 
      = (train_data_.batch_num() + num_threads - 1) / num_threads
      + param_table_staleness + 1; 
  //const int num_rows_train_loss_table
  //    = solver_param_.max_epoch() * max_iter_per_epoch 
  //    / solver_param_.display() + 1;
  //CHECK_GT(num_rows_train_loss_table, 0);
  table_config.table_info.row_type = ditree::kFloatDenseRowDtypeID;
  table_config.table_info.table_staleness = loss_table_staleness;
  //table_config.table_info.row_capacity = kNumLossTableCols;
  //table_config.process_cache_capacity = num_rows_train_loss_table + 5;
  table_config.table_info.dense_row_oplog_capacity = kNumLossTableCols + 5;
  //table_config.oplog_capacity = table_config.process_cache_capacity;
  //petuum::PSTableGroup::CreateTable(kTrainLossTableID, table_config);
  //LOG(INFO) << "Created train loss table " << kTrainLossTableID;
  
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
  LOG(INFO) << "Reading data.";
  int num_clients = Context::get_int32("num_clients");
  int client_id = Context::get_int32("client_id");

  const string& train_data_file = Context::get_string("train_data");
  const string& vocab_file = Context::get_string("vocab");
  if (num_clients == 1) {
    train_data_.Init(train_data_file, vocab_file, true);
  } else {
    ostringstream oss;
    oss << train_data_file << "_" << client_id;
    string oss_str = oss.str();
    train_data_.Init(oss_str, vocab_file, true);
  }
 
  const string& test_data_file = Context::get_string("test_data");
  if (num_clients == 1) {
    test_data_.Init(test_data_file); 
  } else {
    ostringstream oss;
    oss << test_data_file << "_" << client_id;
    string oss_str = oss.str();
    test_data_.Init(oss_str);
  }
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

  Solver* solver = new Solver(solver_param_, thread_id, &train_data_, &test_data_); 

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
    //LOG(INFO) << "Output training results.";
    //solver->PrintNetOutputs(net_outputs_prefix + ".netoutputs");
  }
  //petuum::PSTableGroup::GlobalBarrier();

  petuum::PSTableGroup::DeregisterThread();
}

}
