#!/usr/bin/env bash

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
progname=generate_syn_data
prog_path=${app_dir}/build/tools/${progname}

dataset_path="${app_dir}/data/syn/"
docs_file="docs_train.bin"

echo "Generate synthetic dataset"

GLOG_logtostderr=1
   $prog_path \
   --path ${dataset_path} \
   --docs ${docs_file}
