#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:21:36 2019

@author: domonique_hodge
"""
# This code preprocess pdfs for the QG-Net (Query Generation) algorithm

# Here are notes from the author for preprocessing 2\. Assemble the CoreNLP 
#output into a text document similar to `input.txt` in the path 
#`QG-Net/test/input.for.test/`. The format is, one sentence per line; 
#in each line are space-separated word-feature; each word-feature is in the
#format of `word|feature1|feature2|feature3|answer-token`. When you process 
#your custom input using CoreNLP, you will get the first three features but 
#not the last one. You will need to manually fill in the last feature for each
#word in your input. The first three features are: case (upper case or lower
#case), part of speech tag (POS), and name entity tag (NER). You can easily
#assemble CoreNLP output to the above format by looping through the CoreNLP 
#output for each feature. You may need to specify CoreNLP output format that 
#allows you to do this easily.


import slate
import pynlp
from pynlp import StanfordCoreNLP






def preprocess_pdf(pdf):     
#had to edit the next line because there was an error. 
#annotators = 'tokenize, ssplit, pos, lemma, ner, entitymentions, coref, sentiment, quote, openie'
    annotators = 'tokenize, ssplit, pos, lemma, ner'
    options = {'openie.resolve_coref': True}
    nlp = StanfordCoreNLP(annotators=annotators, options=options)
    
    with open(pdf,'rb') as f:
        doc = slate.PDF(f)
        
    doc = ' '.join([' '.join(x.split()) for x in doc])
        
    text_split = doc.split('Abstract')
    
    if len(text_split)>1:
        text_no_title = ' '.join(text_split[1:])
        text_no_title=str(text_no_title).encode('latin-1', 'ignore') 
        text_no_title=  text_no_title[0:1000]


    document = nlp(text_no_title)
    first_sentence = document[0]
#only formats the first 2 sentences in the abstract
    for token in first_sentence: 
        if str(token).isalpha():
            if str(token).islower():
                print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨-', end =" ", sep='')
            else:
                print(str(token).lower(),'￨U￨',token.pos,'￨',token.ner,'￨-', end =" ",sep='')
        else:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨-', end =" ",sep='')
    print('\n')

    second_sentence = document[1]
    for token in second_sentence: 
        if str(token).isalpha():
            if str(token).islower():
                print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨-', end =" ", sep='')
            else:
                print(str(token).lower(),'￨U￨',token.pos,'￨',token.ner,'￨-', end =" ",sep='')
        else:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨-', end =" ",sep='')
    print('\n')

#specify the location of the pdf articles
preprocess_pdf('/Users/domonique_hodge/Documents/Parse/0706.2024v3.pdf')
preprocess_pdf('/Users/domonique_hodge/Documents/Parse/0705.2105v1.pdf')
preprocess_pdf('/Users/domonique_hodge/Documents/Parse/0706.4035v1.pdf')
preprocess_pdf('/Users/domonique_hodge/Documents/Parse/0709.1039v1.pdf')

######Preprocessing your own text notes
#1. Download Stanford CoreNLP
#2. Cd Must be in the library of Stanford CoreNLP download
#
#May use a or b:
#
#A: python3 -m pynlp
#
#B: https://stanfordnlp.github.io/CoreNLP/corenlp-server.html
#
#3. java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
#
#4. from pynlp import StanfordCoreNLP
#
#5. Cd to directory and run python3 preprocessing.py > input.txt
# To  move from local machine to AWS
#
#scp -i .ssh /Users/domonique_hodge/Documents/Parse/preprocessing2.py domonique_hodge@dc-dev-server.gallupaws.com:~/.
