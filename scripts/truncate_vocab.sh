#!/usr/bin/env bash

# Figure out the paths.
script_path=`readlink -f $0`
script_dir=`dirname $script_path`
app_dir=`dirname $script_dir`
progname=truncate_vocab
prog_path=${app_dir}/build/tools/${progname}

dataset_path="${app_dir}/data/pubmed/"
docs_file="docword.pubmed.txt.bin"
vocab_file="vocab.pubmed.txt"
topk=70000

echo "Truncate vocab"

GLOG_logtostderr=1 \
   $prog_path \
   --path ${dataset_path} \
   --docs ${docs_file} \
   --vocab ${vocab_file} \
   --topk ${topk}
