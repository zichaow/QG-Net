#!/bin/bash

# Configure download location

# first download the QG-Net model
DOWNLOAD_PATH="test/models.for.test/"
mkdir -p $DOWNLOAD_PATH

## download the baseline models
# Download qg-net
cd $DOWNLOAD_PATH
wget -v -O LSTM-attn.pt -L https://rice.box.com/shared/static/1ljzn157vg7fisilfvbsx318453sy86b.pt
wget -v -O QG-Net-nofeat.pt -L https://rice.box.com/shared/static/7v1zifdbjkpqimldjysd79uktjvfpj5m.pt
cd ../../

echo "download completed."


