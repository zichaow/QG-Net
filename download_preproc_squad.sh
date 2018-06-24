#!/bin/bash

# Configure download location

# first download the QG-Net model
DOWNLOAD_PATH="test/models.for.test/"
mkdir -p $DOWNLOAD_PATH

# Download the train and dev squad data
# test split already downloaded with the download_QG-Net.sh script
cd data/

wget -v -O train.tar.gz -L https://rice.box.com/shared/static/qhyc8ytc8ikfeuv16s5vcaismum48zle.gz
tar -xvf train.tar.gz

wget -v -O dev.tar.gz -L https://rice.box.com/shared/static/blnqyfrfdw0tubaguslzutaa4fk1wdcn.gz

cd ../


echo "download completed."


