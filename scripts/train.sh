#!/usr/bin/env bash

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
progname=ditree_main
prog_path=${app_dir}/build/tools/${progname}

host_filename="machinefiles/localserver"
host_file=$(readlink -f $host_filename)

dataset=syn

##=====================================
## Parameters
##=====================================

# Input files:
solver_filename="${app_dir}/models/syn/solver.prototxt"
model_filename="${app_dir}/models/syn/model.prototxt"
#snapshot_filename="${app_dir}/"
#params_filename="${app_dir}/"

# Petuum Parameters
num_app_threads=1
param_table_staleness=0
loss_table_staleness=0
num_comm_channels_per_client=1

# PS Table Organization Paremeters
max_depth=10
max_num_children_per_vertex=20
max_num_vertexes=10000
max_size_per_table=10
max_split_per_table=1
num_table_id_bits=8

num_history=1

# Datset Parameters
data="${app_dir}/data/syn/docs_train.bin.shuffled"
mean="${app_dir}/data/syn/docs_train.bin_mean.txt"
vocab_size=8
batch_size=100

##=====================================

ssh_options="-oStrictHostKeyChecking=no \
-oUserKnownHostsFile=/dev/null \
-oLogLevel=quiet"

# Parse hostfile
host_list=`cat $host_file | awk '{ print $2 }'`
unique_host_list=`cat $host_file | awk '{ print $2 }' | uniq`
num_unique_hosts=`cat $host_file | awk '{ print $2 }' | uniq | wc -l`

output_dir=$app_dir/output
output_dir="${output_dir}/${dataset}.S${staleness}"
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
      --param_table_staleness $param_table_staleness \
      --loss_table_staleness $loss_table_staleness \
      --num_comm_channels_per_client $num_comm_channels_per_client \
      --stats_path ${output_dir}/caffe_stats.yaml \
      --solver ${solver_filename} \
      --model ${model_filename} \
      --ditree_outputs ${ditree_outputs_prefix} \
      --consistency_model "SSPPush" \
      --max_depth ${max_depth} \
      --max_num_children_per_vertex ${max_num_children_per_vertex} \
      --max_num_vertexes ${max_num_vertexes} \      
      --max_size_per_table ${max_size_per_table} \
      --max_split_per_table ${max_split_per_table} \ 
      --num_table_id_bits $num_table_id_bits \
      --history $num_history \
      --data $data \
      --mean $mean \
      --vocab_size $vocab_size \
      --batch_size $batch_size" #\
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
