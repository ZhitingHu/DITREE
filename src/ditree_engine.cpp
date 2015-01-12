
#include "common.hpp"
#include "context.hpp"
#include "ditree_engine.hpp"

namespace ditree {

DITreeEngine::DITreeEngine() {

}

void DITreeEngine::Init() {
  LOG(INFO) << "Init PS environment";
  ditree::Context& context = ditree::Context::Get();
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
  table_group_config.num_tables = 4;
  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<float> >
    (kFloatDenseRowDtypeID);
  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<int> >
    (kIntDenseRowDtypeID);
  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<float> >
    (kFloatDenseRowDtypeID);
  // Use false to not let main thread access table API.
  petuum::PSTableGroup::Init(table_group_config, false);
  LOG(INFO) << "Init table group done.";

  CreateTables();
}

void DITreeEngine::CreateTables() {
  ditree::Context& context = ditree::Context::Get();
  int param_table_staleness = context.get_int32("param_table_staleness");
  int loss_table_staleness = context.get_int32("loss_table_staleness");
  int row_oplog_type = context.get_int32("row_oplog_type");
  bool oplog_dense_serialized = context.get_bool("oplog_dense_serialized");
  const string& process_storage_type
      = context.get_string("process_storage_type");
#ifdef DEBUG
  CHECK_GT(train_data_->vocabulary_size(), 0);
#endif
  int param_row_length
      = kColIdxParamTableSStart + train_data_->vocabulary_size();
  // common table config
  petuum::ClientTableConfig table_config;
  table_config.table_info.row_oplog_type = row_oplog_type;
  table_config.table_info.oplog_dense_serialized 
      = oplog_dense_serialized;
  table_config.process_storage_type = petuum::BoundedDense;
  // param table
  table_config.table_info.row_type = ditree::kFloatDenseRowDtypeID;
  table_config.table_info.table_staleness = param_table_staleness;
  table_config.table_info.row_capacity = param_row_length;
  table_config.process_cache_capacity = 10000; //TODO
  table_config.table_info.dense_row_oplog_capacity = param_row_length;
  table_config.oplog_capacity = table_config.process_cache_capacity * 5;
  petuum::PSTableGroup::CreateTable(kParamTableID, table_config);
  LOG(INFO) << "Created param table.";
  // struct table
  table_config.table_info.row_type = ditree::kIntDenseRowDtypeID;
  table_config.table_info.table_staleness = 0;
  table_config.table_info.row_capacity = 10; //TODO
  table_config.process_cache_capacity = 10000;
  table_config.table_info.dense_row_oplog_capacity = 10;
  table_config.oplog_capacity = table_config.process_cache_capacity * 5;
  petuum::PSTableGroup::CreateTable(kStructTableID, table_config);
  LOG(INFO) << "Created struct table.";
  // param table meta table
  table_config.table_info.row_type = ditree::kIntDenseRowDtypeID;
  table_config.table_info.table_staleness = 0;
  table_config.table_info.row_capacity = 10; //TODO
  table_config.process_cache_capacity = 10000;
  table_config.table_info.dense_row_oplog_capacity = 10;
  table_config.oplog_capacity = table_config.process_cache_capacity * 5;
  petuum::PSTableGroup::CreateTable(kParamTableMetaTableID, table_config);
  LOG(INFO) << "Created param table meta table.";
  // loss table
  table_config.table_info.row_type = ditree::kFloatDenseRowDtypeID;
  table_config.table_info.table_staleness = loss_table_staleness;
  table_config.table_info.row_capacity = 10; //TODO
  table_config.process_cache_capacity = 10000;
  table_config.table_info.dense_row_oplog_capacity = 10;
  table_config.oplog_capacity = table_config.process_cache_capacity * 5;
  petuum::PSTableGroup::CreateTable(kLossTableID, table_config);
  LOG(INFO) << "Created loss table.";
  
  petuum::PSTableGroup::CreateTableDone(); 

  context.SetTables();
}

void DITreeEngine::ReadData() {

}

void DITreeEngine::Start() {

}

}
