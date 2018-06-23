#!/usr/bin/env bash
# note: you might need to change batch_size so that GPU does not run out of memory
# change the "gpu" and "i" variable to run these 8 experiments

gpu=0                   # gpu for each experiment (choose a different one for each experiment! range 0 - 7)
i=1                     # how many sentences to process (1-7 & all)
dir="results_$(date +'%y_%m_%d')"
mkdir $dir
# make a models folder to store the models
mkdir $dir/models
mkdir $dir/console.output


stdbuf -oL python3 OpenNMT-py/train.py \
-data data/data.feat.${i}sent \
-save_model $dir/models/model \
-epoch 20 -batch_size 64 \
-gpu $gpu \
-encoder_type brnn -decoder_type rnn -rnn_type LSTM -input_feed 1 -copy_attn \
-layers 2 -enc_layers 2 -dec_layers 2 -rnn_size 600 -src_word_vec_size 300 -tgt_word_vec_size 300
#> $dir/console.output/train.log
# the line above logs the training information for debugging purposes
# if you want to log training info to file, comment out line 21, and add a '\' to the end of line 20

