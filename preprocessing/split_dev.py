import argparse
from pdb import set_trace
import codecs

parser = argparse.ArgumentParser(description='portion of split.')
parser.add_argument('-split', default=0.5,
                   help='portion of the dev set that goes to test set')
parser.add_argument('-seed', default=1234,
                   help='seed to translate indices for split')
parser.add_argument('-path', type=str,
                    help='path to the corenlp processed dev files')
args = parser.parse_args()
split = args.split
seed = args.seed

# get the squad files in path
import os
from os import listdir
from os.path import isfile, join

squadFiles = [f for f in listdir(args.path) if 'corenlp' in join(args.path, f) and 'squad' in join(args.path, f)]
fileDir = os.path.dirname(os.path.realpath('__file__'))
path = os.path.join(fileDir,args.path)

# make test folder
if not os.path.exists(args.path + '/test'):
    os.makedirs(args.path + '/test')


# peek the files and get the number of lines and see if they have the same number of lines
def file_len(fname):
    with codecs.open(fname, encoding='utf-8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

fileLens = []
for file in squadFiles:
    fileLens.append(file_len(path+'/'+file))
assert all(fileLens[number] == fileLens[0] for number in range(len(fileLens))), \
    'not all files have the same number of sentences/lines.'


# from seed, translate
import random
from math import floor
random.seed(seed)
testIdx = random.sample(range(fileLens[0]), floor(fileLens[0]*split))
devIdx = list(set(range(fileLens[0])) - set(testIdx))
devIdx.sort()
testIdx.sort()


# read the file and perform the split
def readLinesFromFile(path):
    with codecs.open(path, 'r', 'utf-8') as f:
        content = f.readlines()
    content = [x.rstrip('\n') for x in content]
    f.close()
    return content

for file in squadFiles:
    content = readLinesFromFile(path+'/'+file)
    test = []
    dev = []

    for idx in testIdx:
        test.append(content[idx])
    with open(path+'/test/'+file, 'wb') as f:
        for item in test:
            f.write(item.encode('utf-8') + u'\n'.encode('utf-8'))
    f.close()

    for idx in devIdx:
        dev.append(content[idx])
    with open(path+'/'+file, 'wb') as f:
        for item in dev:
            f.write(item.encode('utf-8') + u'\n'.encode('utf-8'))
    f.close()

