#!/usr/bin/env bash

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
progname=compute_mean
prog_path=${app_dir}/build/tools/${progname}

dataset_path="${app_dir}/data/syn/"
docs_file="docs_train.bin"
vocab_size=8

echo "Compute mean vector"

GLOG_logtostderr=1
   $prog_path \
   --path ${dataset_path} \
   --docs ${docs_file} \
   --vocab_size ${vocab_size}
