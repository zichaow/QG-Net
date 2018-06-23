# QG-Net
QG-Net is a data-driven question generation model that found
some success generating questions from educational content
such as textbooks. This repository contains code used in the 
following two publications:

A
B

The project is built on top of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) 
and [DrQA](https://github.com/facebookresearch/DrQA).


### Dependencies
python3.5 \
pytorch \
OpenNMT-py \
DrQA (optional) \
torchtext-0.1.1 (this is important; if you use the latest 
torchtext you might encounter error when preprocessing 
corpus. Install by the command `pip3 install torchtext-
0.1.1`)


### Quick start
To quickly see the model in action and
get a sense of the quality of the generated questions,
you can use the trained model that we provide 
(in `model/` directory)
to generate questions from [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) 
test data: `./qg_squad.sh`, 
or from selected, pre-processed
[OpenStax](https://openstax.org/) textbooks: `.qg_os.sh`.


### Reproduce our results in the papers
To reproduce the results in the :
1. remove line 22 in `qg.sh`
2. repeat the above  

_Note: Question generation from a piece of text of your choice
is currently not supported because the process involves burdensome
preprocessing that is not yet streamlined. Stay tuned for this 
functionality!_

### Train your own model

#####Preprocessing
Navigate to directory `preprocessing/`, and run 
`. preproc_squad.sh` in command line. 
You should be able to run without changing any line in the bash file.
The processed `.pt` data that QG-Net uses for training will be 
stored in the `data/` directory.
It takes about 500 seconds for the processing to finish.

_make sure you have corenlp installed in `data/corenlp/` directory_

#####Trainining
Navigate to the QG-net directory, and run `. train.sh` in command
line.
You should be able to run without changing any line in the bash file.
Training results will be stored in a newly created 
`results_$DATE` directory, where `$DATE` is the current date in 
`year_month_date` format.

#####Generating
Navigate to the directory `results_$DATE/models` 

#####Evaluate


### A bit more about question generation and QG-Net
We consider question generation as a sequence-to-sequence learning
problem: the input is a sequence (a piece of context, e.g., a
sentence or paragraph in a textbook), and the output is also a 
sequence (a question).

Because several distinct questions can be generated from the same
context, depending where in the context you want to ask question,
the input sequence also needs to encode the "answer" information,
e.g., which part of the input sequence to ask question about.

QG-Net is a RNN-based, sequence-to-sequence model with attention
mechanism and copy mechanism to improve the generated question 
quality.
In particular, attention mechanism allows the model to 
access the entire history of the input sequence, and copy mechanism
allows the model to copy words/tokens directly from the input 
sequence as generated words.

