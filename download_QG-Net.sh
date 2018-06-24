#!/bin/bash

# Configure download location

# first download the QG-Net model
DOWNLOAD_PATH="test/models.for.test/"
mkdir -p $DOWNLOAD_PATH

# Download the test files
mkdir data/
cd data/
wget -v -O test.tar.gz -L https://rice.box.com/shared/static/o8hw9zyzm1391blwtm8lx6c38qu9shck.gz
tar -xvf test.tar.gz
cd ../

# Download qg-net
cd $DOWNLOAD_PATH
wget -v -O QG-Net.pt -L https://rice.box.com/shared/static/izhz3hasup6ekgi8jwyokt70btdq5z3j.pt
cd ../../

echo "download completed."


