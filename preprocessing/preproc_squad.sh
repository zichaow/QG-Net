#!/usr/bin/env bash

i=1 # number of sentences in a training context

mkdir ../data/train/
mkdir ../data/dev/

# process training set
python3 preprocessing.py \
-data_dir ../data/ \
-out_dir ../data/train \
-split SQuAD-v1.1-train \
-corenlp_path ../data/corenlp \
-num_sents $i

# process dev set
python3 preprocessing.py \
-data_dir ../data/ \
-out_dir ../data/dev \
-split SQuAD-v1.1-dev \
-corenlp_path ../data/corenlp \
-num_sents $i

# split processed dev set into validation and test sets
python3 split_dev.py -path ../data/dev

# move the test folder outside the dev folder
mv ../data/dev/test/ ../data/

# remove intermediate output files
rm ../data/dev/SQuAD-*
rm ../data/train/SQuAD-*

# call OpenNMT preprocess routine to output files that OpenNMT can take
cd ../OpenNMT-py
python3 preprocess.py \
-train_src ../data/train/squad.corenlp.filtered.contents.features.${i}sent.txt \
-train_tgt ../data/train/squad.corenlp.filtered.questions.txt \
-valid_src ../data/dev/squad.corenlp.filtered.contents.features.${i}sent.txt \
-valid_tgt ../data/dev/squad.corenlp.filtered.questions.txt \
-save_data ../data/data.feat.${i}sent \
-src_vocab_size 100000 -tgt_vocab_size 100000 \
-src_seq_length 10000 -tgt_seq_length 10000 \
-dynamic_dict

echo "PREPROCESSING ALL DONE!!!!!!"
cd ../

