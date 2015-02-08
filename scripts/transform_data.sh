#!/usr/bin/env bash

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
progname=transform_data
prog_path=${app_dir}/build/tools/${progname}

#dataset_path="${app_dir}/data/nyt/"
#docs_file="nytimes.dat.bin.trunc.txt"
#
#dataset_path="${app_dir}/data/data_pnas/pnas_heldout"
#docs_file="T4_test.dat"
#
#dataset_path="${app_dir}/data/pubmed"
#docs_file="docword.pubmed.txt.bin.trunc.txt"
#
dataset_path="${app_dir}/data/nips"
docs_file="nips_year_T12_train.dat"

echo "Transform data "${dataset_path}/${docs_file}

GLOG_logtostderr=1 \
   $prog_path \
   --path ${dataset_path} \
   --docs ${docs_file}
