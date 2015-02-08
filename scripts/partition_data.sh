#!/usr/bin/env bash

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
progname=partition_data
prog_path=${app_dir}/build/tools/${progname}

dataset_path="${app_dir}/data/pubmed/"
docs_file="docword.pubmed.txt.bin.trunc.txt.bin.shuffled.train"
num_partitions=4

echo "Partition "$docs_file

GLOG_logtostderr=1 \
   $prog_path \
   --path ${dataset_path} \
   --docs ${docs_file} \
   --num_partitions ${num_partitions}
