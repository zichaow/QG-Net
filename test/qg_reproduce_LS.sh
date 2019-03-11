#!/usr/bin/env bash

# need to change the model itself, since different folders have different models
nbest=10
num_sent=1
beam_size=15
debug='False'
selection_criterion='NLL'
model_dir="models.for.test"
#model="model_acc_53.29_ppl_13.59_e20.pt"
model="QG-Net.pt"
alpha=0.2
beta=0.2

python3 ../OpenNMT-py/generate.py \
-model $model_dir/$model \
-src input.for.test/input.txt \
-output output_questions_$model.txt \
-dynamic_dict \
-verbose -batch_size 1 -gpu 0 -beam_size $beam_size -replace_unk -n_best $nbest \
-alpha $alpha -beta $beta \
-attn_vis

