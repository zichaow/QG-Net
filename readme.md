# QG-Net
QG-Net is a data-driven question generation model that found
some success generating questions on educational content
such as textbooks. This repository contains code used in the 
following publication:

Z. Wang, A. S. Lan, W. Nie, P. Grimaldi, A. E. Waters, 
and R. G. Baraniuk. 
QG-Net: A Data-Driven Question Generation Model 
for Educational Content. 
_ACM Conference on Learning at Scale (L@S)_, 
June 2018, to appear 


![Illustration of QG-Net](https://rice.box.com/shared/static/4cjcwz25j4bll93j4wqdje06whaoez21.png) <!-- .element height="50%" width="50%" -->
*Illustration of QG-Net*

The project is built on top of 
[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) 
and uses part of the code from 
[DrQA](https://github.com/facebookresearch/DrQA).

Please note that this is
merely research code, and that there is no warrenty. 
Feel free to modify it for your work.  

### Dependencies
python3.5 \
pytorch \
OpenNMT-py \
Stanford CoreNLP (optional) \
torchtext-0.1.1 (this is important; if you use the latest 
torchtext you might encounter error when preprocessing 
corpus. Install by the command `pip3 install torchtext-
0.1.1`, or follow official installation instructions for
your python distribution.)

### Reproduce the result in the paper
1. Run the download script to download a pre-trained QG-Net model
and the test input file: `. download_QG-Net.sh`
2. Run the script in the `test` folder; first `cd test/`, then
`. qg_reproduce_LS.sh`
3. The output file is a text file starting with 
 the prefix `output_questions_`
 
If you want to reproduce the results of the baseline models,
please follow the following procedure similar to the above:
1. Run the download script to download the pre-trained baseline
models: `. download_baselines.sh`. The models are stored in 
the folder `test/models.for.test/`. We provide 2 baseline models:
LSTM + attention model and QG-Net without linguistic features.
2. modify the script in the test folder: change the `model` 
variable in line 10 to another model. The name of the model you 
entered must match with the name of the models you downloaded in
the folder `test/models.for.test/`.
3. First `cd test/` and then run the script `. qg_reproduce_LS.sh`.


### Train your own model

##### Get the data
QG-Net is trained on 
[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).
We provide both the preprocessed SQuAD dataset and the raw 
SQuAD dataset. 
- You can use our preprocessed SQuAD dataset if you want to
avoid the cumbersome data preprocessing steps. To obtain the 
preprocessed SQuAD data, run the download script 
`. download_preproc_squad.sh`.
- You can also download the raw SQuAD dataset if you want to 
perform data preprocessing on your own. To obtain the raw SQuAD
data, run the download script `. download_raw_squad.sh`.


##### Preprocessing
If you choose to download our preprocessed SQuAD dataset, you 
can skip this step. Otherwise, we provide the detailed steps 
of how we preprocessed SQuAD. If you downloaded the raw SQuAD 
dataset, you can follow the below procedure to preprocess the
dataset:
1. First you need to install Stanford CoreNLP to process
the dataset. In command line, get into the
preprocessing directory `cd preprocessing/`, then 
run the script `. install_corenlp.sh`. 
_make sure you have corenlp installed in `data/corenlp/` directory_
2. Navigate to directory `preprocessing/`, and run 
`. preproc_squad.sh` in command line. 
You should be able to run without changing any line in the bash file.
The processed `.pt` data that QG-Net uses for training will be 
stored in the `data/` directory.
It takes about 500 seconds for the processing to finish.


##### Trainining
Navigate to the QG-net directory, and run `. train.sh` in command
line.
You should be able to run without changing any line in the bash file.
Training results will be stored in a newly created 
`results_$DATE` directory, where `$DATE` is the current date in 
`year_month_date` format.

##### Generating
In the `qg.sh` script, modify the variable `dir` in line 10 to be 
the result directory `results_$DATE` you just created. 
Then run the script `. qg.sh` to generate questions on the test
split of SQuAD dataset. 

##### Evaluating
In the `eval.sh` script, modify the variable `dir` in line 10 to be 
the result directory `results_$DATE` you just created.
Then run the script `. eval.sh` to generate questions on the test
split of SQuAD dataset. Evaluation results will be printed to
the console.

_Note: Question generation from a piece of text of your choice
is currently not supported because the process involves burdensome
preprocessing that is not yet streamlined._

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

Below is a more elaborate illustration of QG-Net model:

![Illustration of QG-Net](https://rice.box.com/shared/static/9bwiyq2ly84h7rev8xvjdrocnd5v312f.png)
*Illustration of QG-Net*


