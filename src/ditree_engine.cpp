
#include "common.hpp"
#include "context.hpp"
#include "ditree_engine.hpp"
#include "solver.hpp"

namespace ditree {

DITreeEngine::DITreeEngine(const SolverParameter& param) {
  solver_param_ = param;
  Init();
}

void DITreeEngine::Init() {
  LOG(INFO) << "Init PS environment";
  ditree::Context& context = ditree::Context::Get();
  //context.set_phase(ditree::Context::Phase::kInit);
  context.set_phase(ditree::Context::Phase::kVIAfterSplit);
  petuum::TableGroupConfig table_group_config;
  table_group_config.num_comm_channels_per_client
      = context.get_int32("num_comm_channels_per_client");
  table_group_config.num_total_clients 
      = context.get_int32("num_clients");
  // + 1 for main() thread.
  table_group_config.num_local_app_threads 
      = context.get_int32("num_app_threads") + 1;
  table_group_config.client_id = context.get_int32("client_id");
  table_group_config.stats_path = context.get_string("stats_path");
  petuum::GetHostInfos(context.get_string("hostfile"), 
      &table_group_config.host_map);
  string consistency_model = context.get_string("consistency_model");
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
  ditree::Context& context = ditree::Context::Get();
  int param_table_staleness = context.get_int32("param_table_staleness");
  int loss_table_staleness = context.get_int32("loss_table_staleness");
  int row_oplog_type = context.get_int32("row_oplog_type");
  bool oplog_dense_serialized = context.get_bool("oplog_dense_serialized");
  int max_num_vertexes = context.get_int32("max_num_vertexes");
  int param_row_length
      = kColIdxParamTableSStart + context.vocab_size();
  // common table config
  petuum::ClientTableConfig table_config;
  table_config.table_info.row_oplog_type = row_oplog_type;
  table_config.table_info.oplog_dense_serialized 
      = oplog_dense_serialized;
  table_config.process_storage_type = petuum::BoundedSparse;
  // param table
  table_config.table_info.row_type = ditree::kFloatDenseRowDtypeID;
  table_config.table_info.table_staleness = param_table_staleness;
  table_config.table_info.row_capacity = param_row_length;
  table_config.process_cache_capacity = max_num_vertexes; //TODO
  table_config.table_info.dense_row_oplog_capacity = param_row_length;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kParamTableID, table_config);
  LOG(INFO) << "Created param table " << kParamTableID;
  // struct table
  table_config.table_info.row_type = ditree::kIntDenseRowDtypeID;
  table_config.table_info.table_staleness = 0;
  table_config.table_info.row_capacity = 10; //TODO
  table_config.process_cache_capacity = max_num_vertexes; //TODO: num_client * num_thread
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
  const int num_rows_train_loss_table
      = solver_param_.max_iter() / solver_param_.display() + 1;
  CHECK_GT(num_rows_train_loss_table, 0);
  table_config.table_info.row_type = ditree::kFloatDenseRowDtypeID;
  table_config.table_info.table_staleness = loss_table_staleness;
  table_config.table_info.row_capacity = kNumLossTableCols;
  table_config.process_cache_capacity = num_rows_train_loss_table;
  table_config.table_info.dense_row_oplog_capacity = 10;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kTrainLossTableID, table_config);
  LOG(INFO) << "Created train loss table " << kTrainLossTableID;
  // test loss table
  const int num_rows_test_loss_table
      = solver_param_.max_iter() / solver_param_.test_interval() + 1;
  CHECK_GT(num_rows_test_loss_table, 0);
  table_config.table_info.row_capacity = kNumLossTableCols;
  table_config.process_cache_capacity = num_rows_test_loss_table;
  table_config.oplog_capacity = table_config.process_cache_capacity;
  petuum::PSTableGroup::CreateTable(kTestLossTableID, table_config);
  LOG(INFO) << "Created test loss table " << kTestLossTableID;
  
  petuum::PSTableGroup::CreateTableDone(); 

  context.SetTables();
}

void DITreeEngine::ReadData() {
  ditree::Context& context = ditree::Context::Get();
  const string& data_file = context.get_string("data");
  train_data_.Init(data_file); 
}

void DITreeEngine::Start() {
  petuum::PSTableGroup::RegisterThread();

  // Initialize local thread data structures.
  int thread_id = thread_counter_++;

  ditree::Context& context = ditree::Context::Get();
  int client_id = context.get_int32("client_id");
  //const string& solver_path = context.get_string("solver");
  const string& snapshot_path = context.get_string("snapshot");
  const string& params_path = context.get_string("params");
  //const string& tree_outputs_prefix = context.get_string("tree_outputs");

  Solver* solver = new Solver(solver_param_, thread_id, &train_data_); 

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
