#!/usr/bin/env bash

# compare sentences selected by default ranker and neural language model

cd qgevalcap/

dir="results_18_04_03"
genFile="model_acc_41.78_ppl_31.78_e1.prob.txt"

python eval.py \
-src ../data/test/squad.corenlp.filtered.contents.1sent.txt \
-tgt ../data/test/squad.corenlp.filtered.questions.txt \
-out ../$dir/generation/$genFile

cd ../
