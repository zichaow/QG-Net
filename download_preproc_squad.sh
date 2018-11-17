#!/bin/bash

# Configure download location
DOWNLOAD_PATH="data"
mkdir -p $DOWNLOAD_PATH
cd $DOWNLOAD_PATH

# Download the train and dev squad data
# test split already downloaded with the download_QG-Net.sh script

wget -v -O opennmt-input-data.tar.gz -L https://rice.box.com/shared/static/6haddoiep15fmdqmtdp3ccf44o1m6e4z.gz 
tar -xvf opennmt-input-data.tar.gz

#wget -v -O train.tar.gz -L https://rice.box.com/shared/static/qhyc8ytc8ikfeuv16s5vcaismum48zle.gz
#tar -xvf train.tar.gz

#wget -v -O dev.tar.gz -L https://rice.box.com/shared/static/blnqyfrfdw0tubaguslzutaa4fk1wdcn.gz
#tar -xvf dev.tar.gz

cd ../

echo "preprocessed data download completed."

