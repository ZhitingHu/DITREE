#!/usr/bin/env bash

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
prog_path=${app_dir}/build/tools

#dataset_path="${app_dir}/data/nyt/"
#docs_file="nytimes.dat.bin.trunc.txt.bin"
#vocab_size=50000
#percent=10
#
#dataset_path="${app_dir}/data/data_pnas/pnas_heldout"
#docs_file="ALL_train.dat.bin"
#vocab_size=36901
#
dataset_path="${app_dir}/data/pubmed"
docs_file="docword.pubmed.txt.bin.trunc.txt.bin.shuffled.train"
vocab_size=70000
percent=10
#
#dataset_path="${app_dir}/data/nips"
#docs_file="nips_year_T12_train.dat.bin"
#vocab_size=14036

echo "Compute mean vector"

GLOG_logtostderr=1 \
   $prog_path/compute_mean \
   --path ${dataset_path} \
   --docs ${docs_file} \
   --vocab_size ${vocab_size}

#echo "Shuffle data"
#
#GLOG_logtostderr=1 \
#   $prog_path/shuffle_data \
#   --path ${dataset_path} \
#   --docs ${docs_file}

#echo "Split train/test data"
#
#GLOG_logtostderr=1 \
#   $prog_path/split_train_test_data\
#   --path ${dataset_path} \
#   --docs ${docs_file}.shuffled \
#   --percent ${percent}
