#!/usr/bin/env bash

# TODO: need to change the $dir and $model variables. Or make sure they are correct
# TODO: can play around with these variables. Explanations of these variables are in opts.py
nbest=10
num_sent=1
beam_size=15
debug='False'
selection_criterion='NLL-QA'
dir="results_18_05_20"
alpha=0.1
beta=0.1
src='data/test/squad.corenlp.filtered.contents.features.1sent.txt' # input for results L@S paper

# find the model with the highest accuracy
maxacc=0
for model in `find $dir/models/ -name model_*`; do
    acc=${model%_ppl*}
    acc=${acc##*acc_}
    if (( $(echo "$acc > $maxacc" |bc -l) )); then
        maxacc=$acc
        maxaccmodel=$model
    fi
done

# a few lines to extract the name of the model (maxaccmodel is the path to the model)
modelname=${maxaccmodel%.pt*}
modelname=${modelname##*model_}
#echo $modelname

# create directory to store generated questions
mkdir $dir/generation/

# call the question generation routine
python3 OpenNMT-py/generate.py \
-model $maxaccmodel \
-corenlp_path data/corenlp/ \
-src $src \
-output  $dir/generation/model_$modelname \
-dynamic_dict \
-verbose -batch_size 1 -gpu 0 -beam_size $beam_size -replace_unk -n_best $nbest \
-alpha $alpha -beta $beta

