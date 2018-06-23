# steps to preprocess squad files

# 1. read squad json file, extract all context, write to file, one context per line
# 2. use corenlp to process the above file, write to a new annotated file
# 3. rea the annotated json file; for each context, create a vector of len(#words in context), indicate the sentence
#    idx of that word (sentence segmentation)
# 4. read the DrQA annotated squad file, add the sentence segmentation info into the data structure
# 5. use the answer token index vector for each question in the DrQA annotated squad file, tag each word in the context
#    to be either A (answer word) or O (not answer word)
# 6. loop through each word in the context, tag each word to be either U (upper case) or L (lower case)
#

# 7. select an appropriate truncation level. could be sentence level, or 2 sentence, or N sentence, all the way up to
#    paragraph (for input context truncation), based on
# 8. based on the truncation level, truncate all context related vectors based on inch sentence index the answer appears
#    and how many sentences to select (all based on the sentence index information produced from step 4). these vectors
#    are: ner, pos, case tag, answer position info, context itself.
# 8. for the selected DrQA annotated file, output the following files:
#    a) lower case space seperated question sequence
#    b) ner
#    c) pos
#    d) case tag
#    e) answer position info
#    f) context
#    g) answer text

# set the level of truncation
import argparse
from pdb import set_trace
import codecs
from copy import deepcopy
import json
from os import listdir
from os.path import isfile, join
from os import mkdir

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-num_sents', default='all',
                   help='number of sentences to select for the document')
parser.add_argument('-fname', default='content_txt_3.txt',
                   help='file to process')
parser.add_argument('-subject', type=str)


args = parser.parse_args()
num_sents = args.num_sents
subject = args.subject
if num_sents != 'all':
    num_sents = int(num_sents)
fname = args.fname
# set_trace()

# first load the patterns
with open('ans.patterns.txt', 'r') as f:
    patterns = json.load(f)['data']

# load the terms file
with open('/home/jack/Documents/openstaxTextbook/'+subject+'/terms.txt') as f:
    terms = json.load(f)
allTerms = []
for key in terms.keys():
    allTerms += terms[key]
terms = allTerms
terms = [term.split() for term in terms]


# 3. load the DrQA corenlp_tokenizer processed OS textbook data
mypath='/home/jack/Documents/openstaxTextbook/'+subject+'/sent_by_mo/corenlp.processed/test/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

try:
    mkdir(mypath + 'ans.augmented/'+str(num_sents))
except:
    pass

from math import floor

for fname in onlyfiles:

    data = []
    try:
        for line in codecs.open(mypath+fname, 'r', 'utf-8'):
            data.append(json.loads(line))
    except:
        continue

    filtered = data

    # select truncation level
    cat = num_sents
    catData = []
    for i in range(floor(cat/2.0), len(data)-floor(cat/2.0)):
        newData = dict.fromkeys(data[0].keys())
        document = []
        lemma = []
        sentIdx = []
        pos = []
        ner = []
        for j in range(i-floor(cat/2.0), i+floor(cat/2.0)+1):
            # set_trace()
            document += data[j]['document']
            lemma += data[j]['lemma']
            sentIdx += data[j]['sentIdx']
            pos += data[j]['pos']
            ner += data[j]['ner']
        newData['document'] = document
        newData['lemma'] = lemma
        newData['sentIdx'] = sentIdx
        newData['pos'] = pos
        newData['ner'] = ner

        catData.append(newData)
    print('number of paragraphs with ' + str(num_sents) + ' sentences per paragraph: ' + str(len(catData)))

    # use some heuristics to find answers
    # if multiple answers are found, duplicate this current datapoint

    print('start processing file: ' + fname)
    dataAns = []
    for dataIdx in range(len(catData)):
        ner = catData[dataIdx]['ner']
        pos = catData[dataIdx]['pos']
        ansInds = [] # list of lists of answer locations
        ansIdxs = []
    	
        # for idx in range(len(ner)):
        #     for pattern in patterns:
        #         if idx+len(pattern[0])<len(ner) and pos[idx:idx+len(pattern[0])]==pattern[0] and ner[idx:idx+len(pattern[1])]==pattern[1]:
        #             ansIdxs.append(list(range(idx,idx+len(pattern[0]))))	
        # set_trace()
        # # simplify
        # new_ansIdxs = []
        # for idx1 in range(len(ansIdxs)):
        #     item = ansIdxs[idx1]
        #     Add = True
        #     for idx2 in range(len(ansIdxs)):
        #         if idx1!=idx2 and set(item).issubset(set(ansIdxs[idx2])):
        #             Add = False
        #     if Add:
        #         new_ansIdxs.append(item)
        # ansIdxs = new_ansIdxs

        # find answers using special ner tags and terms
        idx = 0
        while idx < len(ner):
            # using ner
            if ner[idx] != 'O':
                k = idx+1
                while k < len(ner):
                    if ner[k] == ner[idx]:
                        k += 1
                    else:
                        break
                ansIdxs.append(list(range(idx,k)))
                idx = k
            else:
                idx += 1

        # find answers using terms
        for idx in range(len(ner)):            
            for term in terms:
                Add = True
                for i in range(len(term)):
                    if idx+i<len(ner) and catData[dataIdx]['document'][idx+i].lower() != term[i]:
                        Add = False
                if Add:
                    ansIdxs.append(list(range(idx,idx+len(term))))
        

        # set_trace()
        # simplify
        new_ansIdxs = []
        for ansIdx in ansIdxs:
            # do not include duplicate answer indices
            if ansIdx not in new_ansIdxs:
                new_ansIdxs.append(ansIdx)
        ansIdxs = new_ansIdxs
        
        # print('        found ' + str(len(ansIdxs)) + ' answers in this paragraph.')    

        for idx in range(len(ansIdxs)):
            ansInd = ['-'] * len(ner)
            for j in ansIdxs[idx]:
                ansInd[j] = 'A'
            ansInds.append(ansInd)

        for idx in range(len(ansInds)):
            ansInd = ansInds[idx]
            newData = deepcopy(catData[dataIdx])
            newData['ansInd'] = ansInd
            dataAns.append(newData)
    
    # set_trace()

    # # simplify
    # newDataAns = []
    # answers = []
    # for ex in dataAns:
    #     # extract answer
    #     ans = []
    #     for idx in range(len(ex['ansInd'])):
    #         if ex['ansInd'][idx] == 'A':
    #             ans.append(ex['document'][idx])
    #     if ans not in answers:
    #         answers.append(ans)
    #         newDataAns.append(ex)
    # dataAns = newDataAns

    print('finding answer completed.')

    # set_trace()

    # get case info
    for i in range(len(dataAns)):
        doc_case = []
        for w in dataAns[i]['document']:
            if w.isalpha():
                if w.islower():
                    doc_case.append('L')
                else:
                    doc_case.append('U')
            else:
                doc_case.append('L')
        dataAns[i]['doc_case'] = doc_case
    # set_trace()

    # append case, pos, ner, ansInd as features to context
    out_file = mypath + 'ans.augmented/'+str(num_sents)+'/' + fname
    with open(out_file, 'wb') as f:
        for ex in dataAns:
            line = u' '.join([ex['document'][idx].replace(' ', '').lower() + '￨' + ex['doc_case'][idx] + '￨' +
                            ex['pos'][idx] + '￨' + ex['ner'][idx] + '￨' + ex['ansInd'][idx]
                            for idx in range(len(ex['document']))]).encode('utf-8').strip()
            f.write(line + u'\n'.encode('utf-8'))
    f.close()

    # content content and answer
    out_file = mypath + 'ans.augmented/'+str(num_sents)+'/contentOnly' + fname
    with open(out_file, 'wb') as f:
        for ex in dataAns:
            ans = []
            for idx in range(len(ex['ansInd'])):
                if ex['ansInd'][idx] == 'A':
                    ans.append(ex['document'][idx])
            line = u' '.join(ans + [' ||| '] + [ex['document'][idx].replace(' ', '').lower() 
                            for idx in range(len(ex['document']))]).encode('utf-8').strip()
            f.write(line + u'\n'.encode('utf-8'))
    f.close()
    print('----------------------------------')


