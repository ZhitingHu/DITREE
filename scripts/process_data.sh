#!/usr/bin/env bash

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
prog_path=${app_dir}/build/tools

dataset_path="${app_dir}/data/nyt/"
docs_file="nytimes.dat.bin"
vocab_size=102660

echo "Compute mean vector"

GLOG_logtostderr=1 \
   $prog_path/compute_mean \
   --path ${dataset_path} \
   --docs ${docs_file} \
   --vocab_size ${vocab_size}

echo "Shuffle data"

GLOG_logtostderr=1 \
   $prog_path/shuffle_data \
   --path ${dataset_path} \
   --docs ${docs_file}

echo "Split train/test data"

GLOG_logtostderr=1 \
   $prog_path/split_train_test_data\
   --path ${dataset_path} \
   --docs ${docs_file} \
   --percent ${percent}
