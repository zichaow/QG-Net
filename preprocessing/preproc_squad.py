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






# ------------------------------------------------------------------------------
# tokenizer initializations
# ------------------------------------------------------------------------------

import os
from pdb import set_trace


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'corenlp':
        return CoreNLPTokenizer
    raise RuntimeError('Invalid tokenizer: %s' % name)


def get_annotators_for_args(args):
    annotators = set()
    if args.use_pos:
        annotators.add('pos')
    if args.use_lemma:
        annotators.add('lemma')
    if args.use_ner:
        annotators.add('ner')
    return annotators


def get_annotators_for_model(model):
    return get_annotators_for_args(model.args)




# ------------------------------------------------------------------------------
# Base tokenizer/tokens classes and utilities.
# ------------------------------------------------------------------------------

import copy


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5
    SENTIDX = 6

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    # add stuff to return sent index
    def sentIdx(self):
        return [t[self.SENTIDX] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """
        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """
    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()




# ------------------------------------------------------------------------------
# Simple wrapper around the Stanford CoreNLP pipeline.
#
# Serves commands to a java subprocess running the jar. Requires java 8.
# ------------------------------------------------------------------------------

import pexpect


class CoreNLPTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            classpath: Path to the corenlp directory of jars
            mem: Java heap memory
        """
        self.classpath = (kwargs.get('classpath') or
                          DEFAULTS['corenlp_classpath'])
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        self.mem = kwargs.get('mem', '2g')
        self._launch()

    def _launch(self):
        """Start the CoreNLP jar with pexpect."""
        annotators = ['tokenize', 'ssplit']
        if 'ner' in self.annotators:
            annotators.extend(['pos', 'lemma', 'ner'])
        elif 'lemma' in self.annotators:
            annotators.extend(['pos', 'lemma'])
        elif 'pos' in self.annotators:
            annotators.extend(['pos'])
        annotators = ','.join(annotators)
        options = ','.join(['untokenizable=noneDelete',
                            'invertible=true'])
        cmd = ['java', '-mx' + self.mem, '-cp', '"%s"' % self.classpath,
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators',
               annotators, '-tokenize.options', options,
               '-outputFormat', 'json', '-prettyPrint', 'false']

        # We use pexpect to keep the subprocess alive and feed it commands.
        # Because we don't want to get hit by the max terminal buffer size,
        # we turn off canonical input processing to have unlimited bytes.
        self.corenlp = pexpect.spawn('/bin/bash', maxread=100000, timeout=60)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')
        self.corenlp.sendline(' '.join(cmd))
        self.corenlp.delaybeforesend = 0
        self.corenlp.delayafterread = 0
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

    @staticmethod
    def _convert(token):
        if token == '-LRB-':
            return '('
        if token == '-RRB-':
            return ')'
        if token == '-LSB-':
            return '['
        if token == '-RSB-':
            return ']'
        if token == '-LCB-':
            return '{'
        if token == '-RCB-':
            return '}'
        return token

    def tokenize(self, text):
        # Since we're feeding text to the commandline, we're waiting on seeing
        # the NLP> prompt. Hacky!
        if 'NLP>' in text:
            raise RuntimeError('Bad token (NLP>) in text!')

        # Sending q will cause the process to quit -- manually override
        if text.lower().strip() == 'q':
            token = text.strip()
            index = text.index(token)
            data = [(token, text[index:], (index, index + 1), 'NN', 'q', 'O')]
            return Tokens(data, self.annotators)

        # Minor cleanup before tokenizing.
        clean_text = text.replace('\n', ' ')

        self.corenlp.sendline(clean_text.encode('utf-8'))
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

        # Skip to start of output (may have been stderr logging messages)
        output = self.corenlp.before
        start = output.find(b'{"sentences":')
        output = json.loads(output[start:].decode('utf-8'))

        data = []

        # collect sentence id for each token
        sentIdx = []
        for s in output['sentences']:
            sentIdx += [s['index']] * len(s['tokens'])

        tokens = [t for s in output['sentences'] for t in s['tokens']]
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i]['characterOffsetBegin']
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1]['characterOffsetBegin']
            else:
                end_ws = tokens[i]['characterOffsetEnd']

            data.append((
                self._convert(tokens[i]['word']),
                text[start_ws: end_ws],
                (tokens[i]['characterOffsetBegin'],
                 tokens[i]['characterOffsetEnd']),
                tokens[i].get('pos', None),
                tokens[i].get('lemma', None),
                tokens[i].get('ner', None),
                sentIdx[i]
            ))
        return Tokens(data, self.annotators)




# ------------------------------------------------------------------------------
# Preprocess the SQuAD dataset for training.
# ------------------------------------------------------------------------------

import argparse
import sys
import json
import time

from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial

# ----------------------
# Tokenize + annotate.
# ----------------------

TOK = None


def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
        'sentIdx': tokens.sentIdx(),
    }
    return output


# ----------------------
# Process dataset examples
# ----------------------


def load_dataset(path):
    """Load json file and store fields separately."""
    with open(path) as f:
        data = json.load(f)['data']
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    for article in data:
        for paragraph in article['paragraphs']:
            output['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                output['qids'].append(qa['id'])
                output['questions'].append(qa['question'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                if 'answers' in qa:
                    output['answers'].append(qa['answers'])
    return output


def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    assert(len(start) <= 1)
    assert(len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]


def process_dataset(data, tokenizer, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    tokenizer_class = get_class(tokenizer)
    make_pool = partial(Pool, workers, initializer=init)
    workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
    q_tokens = workers.map(tokenize, data['questions'])
    workers.close()
    workers.join()

    workers = make_pool(
        initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}})
    )
    c_tokens = workers.map(tokenize, data['contexts'])
    workers.close()
    workers.join()

    for idx in range(len(data['qids'])):

        # question tokenizations
        question = q_tokens[idx]['words']
        qlemma = q_tokens[idx]['lemma']
        qpos = q_tokens[data['qid2cid'][idx]]['pos']
        qner = q_tokens[data['qid2cid'][idx]]['ner']

        # document tokenizations
        document = c_tokens[data['qid2cid'][idx]]['words']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        lemma = c_tokens[data['qid2cid'][idx]]['lemma']
        pos = c_tokens[data['qid2cid'][idx]]['pos']
        ner = c_tokens[data['qid2cid'][idx]]['ner']
        docSentIdx = c_tokens[data['qid2cid'][idx]]['sentIdx']

        ans_tokens = []
        if len(data['answers']) > 0:
            for ans in data['answers'][idx]:
                found = find_answer(offsets,
                                    ans['answer_start'],
                                    ans['answer_start'] + len(ans['text']))
                if found:
                    ans_tokens.append(found)

        yield {
            'id': data['qids'][idx],
            'question': question,
            'qlemma': qlemma,
            'qpos': qpos,
            'qner': qner,

            'document': document,
            'offsets': offsets,
            'answers': ans_tokens,
            'lemma': lemma,
            'pos': pos,
            'ner': ner,
            'sentIdx': docSentIdx,
        }




# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
# cmd options for corenlp preprocessing step
parser.add_argument('-data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('-out_dir', type=str, help='Path to output file dir')
parser.add_argument('-split', type=str, help='Filename for train/dev split',
                    default='SQuAD-v1.1-train')
parser.add_argument('-workers', type=int, default=None)
parser.add_argument('-tokenizer', type=str, default='corenlp')
parser.add_argument('-corenlp_path', type=str, default=None)

# cmd option for processing from corenlp preprocessed data
parser.add_argument('-num_sents', default='all', type=str,
                   help='number of sentences to select for the document')

args = parser.parse_args()

t0 = time.time()


# set corenlp path
if args.tokenizer == 'corenlp':
    DEFAULTS = {
        'corenlp_classpath': args.corenlp_path + '/*'
    }
else:
    raise Exception('currently only supports CoreNLP.')


# check if data_dir already exist. if not, create it
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
else:
    print('Warning: output directory already exists.')

in_file = os.path.join(args.data_dir, args.split + '.json')
print('Loading dataset %s' % in_file, file=sys.stderr)
dataset = load_dataset(in_file)

out_file = os.path.join(
    args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer)
)
print('Will write to file %s' % out_file, file=sys.stderr)
with open(out_file, 'w') as f:
    for ex in process_dataset(dataset, args.tokenizer, args.workers):
        f.write(json.dumps(ex) + '\n')
print('Total time: %.4f (s)' % (time.time() - t0))


# set the level of truncation
# import codecs

num_sents = args.num_sents
# num_sents = 'all'
if num_sents != 'all':
    num_sents = int(num_sents)
# set_trace()

# 3. load the DrQA corenlp_tokenizer processed data
import json
data = []
for line in open(out_file, 'r'):
    data.append(json.loads(line))

# check if joined context have '\n' tokens
badidx = []
for i in range(len(data)):
    # c = ' '.join(data[i]['document'])
    # if '\n' in c:
    c = data[i]['document']
    lenc = len(c)
    lenq = len(data[i]['question'])
    if len(data[i]['ner']) != lenc or len(data[i]['pos']) != lenc or len(data[i]['lemma']) != lenc \
            or len(data[i]['sentIdx']) != lenc or len(data[i]['qlemma']) != lenq or len(data[i]['answers']) == 0:
        badidx.append(i)

# # check multiple answers
# multiidx = []
# for i in range(len(data)):
#     if len(data[i]['answers']) < 1:
#         multiidx.append(i)

# remove these bad idxs
filtered = []
for idx in range(len(data)):
    if idx not in badidx:
        filtered.append(copy.deepcopy(data[idx]))

# create answer tagging array for each data point in data
for i in range(len(filtered)):
    ansInd = ['-'] * len(filtered[i]['document'])
    ansidx = filtered[i]['answers'][0]
    # print(ansidx)
    # print(i)
    for idx in range(min(ansidx), max(ansidx)+1): # assuming there is only a single answer which can be accessed by idx 0
        ansInd[idx] = 'A'
    filtered[i]['ansInd'] = ansInd

# do the truncation
from math import floor
if num_sents == 'all':
    # get case info
    for i in range(len(filtered)):
        doc_case = []
        for w in filtered[i]['document']:
            if w.isalpha():
                if w.islower():
                    doc_case.append('L')
                else:
                    doc_case.append('U')
            else:
                doc_case.append('L')
        filtered[i]['doc_case'] = doc_case

else:
    for i in range(len(filtered)):

        # find the answer sentence index
        # note: it might happen that answer span more than one sentence
        ans_sent_idx = list(set([filtered[i]['sentIdx'][idx] for idx in filtered[i]['answers'][0]]))

        # find the selected sentence indices
        if min(ans_sent_idx) == 0:
            min_sent_idx = 0
            max_sent_idx = num_sents
        elif max(ans_sent_idx) == max(filtered[i]['sentIdx']):
            min_sent_idx = max(ans_sent_idx) - num_sents
            max_sent_idx = max(ans_sent_idx)
        elif max(ans_sent_idx) - min(ans_sent_idx) + 1 == num_sents:
            min_sent_idx = min(ans_sent_idx)
            max_sent_idx = max(ans_sent_idx)
        else:
            min_sent_idx = min(ans_sent_idx) - floor(num_sents/2)
            max_sent_idx = max(ans_sent_idx) + floor((num_sents-2)/2)

        if min_sent_idx < 0:
            min_sent_idx = 0
        if max_sent_idx > max(filtered[i]['sentIdx']):
            max_sent_idx = max(filtered[i]['sentIdx'])

        # get the first occurrence of min_sent_idx in sentIdx and last occurrence of max_sent_idx in sentIdx
        start = filtered[i]['sentIdx'].index(min_sent_idx)
        end   = max(loc for loc, val in enumerate(filtered[i]['sentIdx']) if val == max_sent_idx) + 1 # +1 to include this index

        # get answer text
        filtered[i]['answer_txt'] = [filtered[i]['document'][idx] for idx in filtered[i]['answers'][0]]

        # perform truncation
        filtered[i]['sentIdx'] = filtered[i]['sentIdx'][start:end]
        filtered[i]['ner']     = filtered[i]['ner'][start:end]
        filtered[i]['lemma']   = filtered[i]['lemma'][start:end]
        filtered[i]['pos']     = filtered[i]['pos'][start:end]
        filtered[i]['document']= filtered[i]['document'][start:end]
        filtered[i]['offsets'] = filtered[i]['offsets'][start:end]
        filtered[i]['ansInd']  = filtered[i]['ansInd'][start:end]

        # get case info
        doc_case = []
        for w in filtered[i]['document']:
            if w.isalpha():
                if w.islower():
                    doc_case.append('L')
                else:
                    doc_case.append('U')
            else:
                doc_case.append('L')
        filtered[i]['doc_case'] = doc_case


        # # do one more processing step for offset so the starting offset index is 0
        # offset = filtered[i]['offsets'][0][0]
        # for idx in range(len(filtered[i]['offsets'])):
        #     for idx2 in range(len(filtered[i]['offsets'][idx])):
        #         filtered[i]['offsets'][idx][idx2] -= offset


# # save the filtered dataset to file
# out_file = 'squad.corenlp.filtered' + str(num_sents) + '.json'
# with open(out_file, 'w') as f:
#     for ex in filtered:
#         f.write(json.dumps(ex) + '\n')
# f.close()


# append case, pos, ner, ansInd as features to context
out_file = args.out_dir + '/squad.corenlp.filtered.contents.features.' + str(num_sents) + 'sent.txt'
with open(out_file, 'wb') as f:
    for ex in filtered:
        line = u' '.join([ex['document'][idx].replace(' ', '').lower() + '￨' + ex['doc_case'][idx] + '￨' +
                        ex['pos'][idx] + '￨' + ex['ner'][idx] + '￨' + ex['ansInd'][idx]
                        for idx in range(len(ex['document']))]).encode('utf-8').strip()
        f.write(line + u'\n'.encode('utf-8'))
f.close()

out_file = args.out_dir + '/squad.corenlp.filtered.contents.' + str(num_sents) + 'sent.txt'
with open(out_file, 'wb') as f:
    for ex in filtered:
        line = u' '.join([ex['document'][idx].replace(' ', '') for idx in range(len(ex['document']))])\
            .lower().encode('utf-8').strip()
        f.write(line + u'\n'.encode('utf-8'))
f.close()

# write questions to file
out_file = args.out_dir + '/squad.corenlp.filtered.questions' + '.txt'
with open(out_file, 'wb') as f:
    for ex in filtered:
        line = u' '.join([ex['question'][idx].replace(' ', '') for idx in range(len(ex['question']))])\
            .lower().encode('utf-8').strip()
        f.write(line + u'\n'.encode('utf-8'))
f.close()
