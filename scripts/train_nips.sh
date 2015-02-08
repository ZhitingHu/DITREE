#!/usr/bin/env bash

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
progname=ditree_main
prog_path=${app_dir}/build/tools/${progname}

host_filename="machinefiles/localserver_pnas"
host_file=$(readlink -f $host_filename)

dataset=nips

##=====================================
## Parameters
##=====================================

# Input files:
solver_filename="${app_dir}/models/nips/solver.prototxt"
#model_filename="${app_dir}/models/nips/model.prototxt"
model_filename="${app_dir}/models/nips/big_model.prototxt"

history_filename="${app_dir}/models/nips/ditree_trainditreemodel.epoch.20.T10"
#snapshot_filename="${app_dir}/"
#params_filename="${app_dir}/"

# Petuum Parameters
num_app_threads=2
param_table_staleness=0
loss_table_staleness=0
num_clocks_per_epoch=12
num_comm_channels_per_client=1
consistency_model="SSPPush"

# PS Table Organization Paremeters
max_depth=3
max_num_children_per_vertex=3
max_num_vertexes=100
max_size_per_table=20
max_split_per_table=4
max_merge_per_table=0
num_table_id_bits=12

num_history=1

# Datset Parameters
train_data="${app_dir}/data/nips/nips_year_ALL_train.dat.bin.shuffled"
test_data="${app_dir}/data/nips/nips_year_ALL_test.dat.bin"
mean="${app_dir}/data/nips/nips_year_ALL_train.dat.bin_mean.txt"
#
#train_data="${app_dir}/data/nips/nips_year_T1_train.dat.bin.shuffled"
#test_data="${app_dir}/data/nips/nips_year_T17_test.dat.bin"
#mean="${app_dir}/data/nips/nips_year_T1_train.dat.bin_mean.txt"
vocab="${app_dir}/data/nips/vocab.txt"
vocab_size=14036
train_batch_size=400
test_batch_size=20
top_k=10

##=====================================

ssh_options="-oStrictHostKeyChecking=no \
-oUserKnownHostsFile=/dev/null \
-oLogLevel=quiet"

# Parse hostfile
host_list=`cat $host_file | awk '{ print $2 }'`
unique_host_list=`cat $host_file | awk '{ print $2 }' | uniq`
num_unique_hosts=`cat $host_file | awk '{ print $2 }' | uniq | wc -l`

output_dir=$app_dir/output
output_dir="${output_dir}/${dataset}.S${param_table_staleness}"
output_dir="${output_dir}.M${num_unique_hosts}"
output_dir="${output_dir}.T${num_app_threads}"
mkdir -p ${output_dir}

log_dir=$output_dir/logs
ditree_outputs_prefix="${output_dir}/${dataset}"

# Kill previous instances of this program
echo "Killing previous instances of '$progname' on servers, please wait..."
for ip in $unique_host_list; do
  ssh $ssh_options $ip \
    killall -q $progname
done
echo "All done!"

# Spawn program instances
client_id=0
for ip in $unique_host_list; do
  echo Running client $client_id on $ip
  log_path=${log_dir}.${client_id}

  cmd="mkdir -p ${log_path}; \
      GLOG_logtostderr=false \
      GLOG_stderrthreshold=0 \
      GLOG_log_dir=$log_path \
      GLOG_v=-1 \
      GLOG_minloglevel=0 \
      GLOG_vmodule="" \
      $prog_path train \
      --hostfile $host_file \
      --client_id ${client_id} \
      --num_clients $num_unique_hosts \
      --num_app_threads $num_app_threads \
      --num_clocks_per_epoch ${num_clocks_per_epoch} \
      --param_table_staleness $param_table_staleness \
      --loss_table_staleness $loss_table_staleness \
      --num_comm_channels_per_client $num_comm_channels_per_client \
      --stats_path ${output_dir}/ditree_stats.yaml \
      --solver ${solver_filename} \
      --model ${model_filename} \
      --ditree_outputs ${ditree_outputs_prefix} \
      --consistency_model ${consistency_model} \
      --max_depth ${max_depth} \
      --max_num_children_per_vertex ${max_num_children_per_vertex} \
      --max_num_vertexes ${max_num_vertexes} \ 
      --max_size_per_table ${max_size_per_table} \
      --max_split_per_table ${max_split_per_table} \
      --max_merge_per_table $max_merge_per_table \
      --num_table_id_bits $num_table_id_bits \
      --train_data $train_data \
      --test_data $test_data \
      --mean $mean \
      --vocab $vocab \
      --vocab_size $vocab_size \
      --top_k ${top_k} \
      --train_batch_size $train_batch_size \
      --test_batch_size $test_batch_size
      --history_size $num_history" #\
      #--history ${history_filename}" #\ 
      #--snapshot ${snapshot_filename} \
      #--params ${params_filename}"

  ssh $ssh_options $ip $cmd &
  #eval $cmd  # Use this to run locally (on one machine).

  # Wait a few seconds for the name node (client 0) to set up
  if [ $client_id -eq 0 ]; then
    echo $cmd   # echo the cmd for just the first machine.
    echo "Waiting for name node to set up..."
    sleep 3
  fi
  client_id=$(( client_id+1 ))
done
