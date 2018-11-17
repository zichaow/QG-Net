#!/bin/bash
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

# Configure download location
DOWNLOAD_PATH="$DRQA_DATA"
if [ "$DRQA_DATA" == "" ]; then
    echo "DRQA_DATA not set; downloading to default path ('data')."
    DOWNLOAD_PATH="${path:-../data}"
fi
DATASET_PATH=$DOWNLOAD_PATH
mkdir $DOWNLOAD_PATH

# Get SQuAD train
wget -O "$DATASET_PATH/SQuAD-v1.1-train.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
python convert.py "$DATASET_PATH/SQuAD-v1.1-train.json" "$DATASET_PATH/SQuAD-v1.1-train.txt"

# Get SQuAD dev
wget -O "$DATASET_PATH/SQuAD-v1.1-dev.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
python convert.py "$DATASET_PATH/SQuAD-v1.1-dev.json" "$DATASET_PATH/SQuAD-v1.1-dev.txt"

# # Download official eval for SQuAD
# curl "https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/" >  "./scripts/reader/official_eval.py"

echo "raw squad data download done!"
