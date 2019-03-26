#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:21:36 2019

@author: domonique_hodge
"""

"""
This code preprocess pdfs for the QG-Net (Query Generation) algorithm
"""


import slate
import pynlp
from pynlp import StanfordCoreNLP
import warnings
import logging

import math
from textblob import TextBlob as tb
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


"""
Function to create preprocessing input.txt for QG-Net based on the first 3 sentences of an abstract included in a pdf article.
"""

def preprocess_pdf(text_no_title):     

#only formats the first 3 sentences in the abstract
    document = nlp(text_no_title)
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    
    first_sentence = document[0]
    second_sentence = document[1]
    third_sentence = document[2]
    
    #tf-idf
    top_words = []
    bloblist = [tb(str(first_sentence)),tb(str(second_sentence)),tb(str(third_sentence))]
    for i, blob in enumerate(bloblist):
        scores = {word.lower(): tfidf(word.lower(), blob, bloblist) for word in blob.words if word not in stopwords.words('english')}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #selects the word with highest tf-idf score in the dataset
        for word, score in sorted_words[:1]:
            top_words.append(word)
    
    #first sentence
    for token in first_sentence: 
        if str(token).isalpha() and str(token).islower() and str(token).lower()==top_words[0]:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨A', end =" ", sep='')
        elif str(token).isalpha() and str(token)[0].isupper() and str(token).lower()==top_words[0]:
            print(str(token).lower(),'￨U￨',token.pos,'￨',token.ner,'￨A', end =" ",sep='')
        elif str(token).isalpha() and str(token)[0].isupper()  and str(token).lower()!=top_words[0]:
            print(str(token).lower(),'￨U￨',token.pos,'￨',token.ner,'￨-', end =" ",sep='')
        elif str(token).isalpha() and str(token).islower() and str(token).lower()!=top_words[0]:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨-', end =" ", sep='')
        elif str(token).isalpha()==False and str(token)[0].replace('-', '').isupper() and str(token).lower()==top_words[0]:
            print(str(token).lower(),'￨U￨',token.pos,'￨',token.ner,'￨A', end =" ", sep='')
        elif str(token).isalpha()==False and str(token).replace('-', '').islower() and str(token).replace('-', '').lower()==top_words[0]:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨A', end =" ", sep='')
        else:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨-', end =" ",sep='')
    print('\n')
    #second sentence
    for token in second_sentence: 
        if str(token).isalpha() and str(token).islower() and str(token).lower()==top_words[1]:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨A', end =" ", sep='')
        elif str(token).isalpha() and str(token)[0].isupper() and str(token).lower()==top_words[1]:
            print(str(token).lower(),'￨U￨',token.pos,'￨',token.ner,'￨A', end =" ",sep='')
        elif str(token).isalpha() and str(token)[0].isupper()  and str(token).lower()!=top_words[1]:
            print(str(token).lower(),'￨U￨',token.pos,'￨',token.ner,'￨-', end =" ",sep='')
        elif str(token).isalpha() and str(token).islower() and str(token).lower()!=top_words[1]:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨-', end =" ", sep='')
        elif str(token).isalpha()==False and str(token)[0].replace('-', '').isupper() and str(token).lower()==top_words[1]:
            print(str(token).lower(),'￨U￨',token.pos,'￨',token.ner,'￨A', end =" ", sep='')
        elif str(token).isalpha()==False and str(token).replace('-', '').islower() and str(token).replace('-', '').lower()==top_words[1]:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨A', end =" ", sep='')
        else:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨-', end =" ",sep='')
    print('\n')
    
    #third sentence
    for token in third_sentence: 
        if str(token).isalpha() and str(token).islower() and str(token).lower()==top_words[2]:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨A', end =" ", sep='')
        elif str(token).isalpha() and str(token)[0].isupper() and str(token).lower()==top_words[2]:
            print(str(token).lower(),'￨U￨',token.pos,'￨',token.ner,'￨A', end =" ",sep='')
        elif str(token).isalpha() and str(token)[0].isupper()  and str(token).lower()!=top_words[2]:
            print(str(token).lower(),'￨U￨',token.pos,'￨',token.ner,'￨-', end =" ",sep='')
        elif str(token).isalpha() and str(token).islower() and str(token).lower()!=top_words[2]:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨-', end =" ", sep='')
        elif str(token).isalpha()==False and str(token)[0].replace('-', '').isupper() and str(token).lower()==top_words[2]:
            print(str(token).lower(),'￨U￨',token.pos,'￨',token.ner,'￨A', end =" ", sep='')
        elif str(token).isalpha()==False and str(token).replace('-', '').islower() and str(token).replace('-', '').lower()==top_words[2]:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨A', end =" ", sep='')
        else:
            print(str(token).lower(),'￨L￨',token.pos,'￨',token.ner,'￨-', end =" ",sep='')
    print('\n')
    
    
import os


"""
directory0 is the directory that includes all the pdf articles
"""
directory0 = "/Users/domonique_hodge/Documents/qgnet/QG-Net/papers/arxiv"    
directory = os.fsencode(directory0)   
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pdf"): 
        annotators = 'tokenize, ssplit, pos, lemma, ner'
        options = {'openie.resolve_coref': True}

        nlp = StanfordCoreNLP(annotators=annotators, options=options)
    
        #logging suppresses warnings in pdf import
        logging.propagate = False 
        logging.getLogger().setLevel(logging.ERROR)
        
        #print(filename)
        pdf = os.path.join(directory0,filename)
        try:
            with open(pdf,'rb') as f:
                doc = slate.PDF(f)
        except:
            print("An exception occurred")
      
        doc = ' '.join([' '.join(x.split()) for x in doc])
        
        #verify an abstract in the paper
        text_split = doc.split('Abstract.')
        if len(text_split)>1:
            text_no_title = ' '.join(text_split[1:])
            text_no_title=str(text_no_title).encode('latin-1', 'ignore') 
            text_no_title=  text_no_title[0:1000]
            preprocess_pdf(text_no_title)
        elif len(text_split)<=1:
            text_split = doc.split('ABSTRACT.')
            if len(text_split)>1:
                text_no_title = ' '.join(text_split[1:])
                text_no_title=str(text_no_title).encode('latin-1', 'ignore') 
                text_no_title=  text_no_title[0:1000]
                preprocess_pdf(text_no_title)
            elif len(text_split)<=1:
                text_split = doc.split('Abstract')
                if len(text_split)>1:
                    text_no_title = ' '.join(text_split[1:])
                    text_no_title=str(text_no_title).encode('latin-1', 'ignore') 
                    text_no_title=  text_no_title[0:1000]
                    preprocess_pdf(text_no_title)
                elif len(text_split)<=1:
                    text_split = doc.split('ABSTRACT')
                    if len(text_split)>1:
                        text_no_title = ' '.join(text_split[1:])
                        text_no_title=str(text_no_title).encode('latin-1', 'ignore') 
                        text_no_title=  text_no_title[0:1000]
                        preprocess_pdf(text_no_title)
                    else:
                        continue

