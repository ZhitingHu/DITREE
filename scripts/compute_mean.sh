#!/usr/bin/env bash

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
progname=compute_mean
prog_path=${app_dir}/build/tools/${progname}

#dataset_path="${app_dir}/data/nyt/"
#docs_file="nytimes.dat.bin"
#vocab_size=102660
dataset_path="${app_dir}/data/nyt/"
docs_file="nytimes.dat.bin"
vocab_size=102660

echo "Compute mean vector"

GLOG_logtostderr=1 \
   $prog_path \
   --path ${dataset_path} \
   --docs ${docs_file} \
   --vocab_size ${vocab_size}
