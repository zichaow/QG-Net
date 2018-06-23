#!/bin/bash

# Configure download location

DOWNLOAD_PATH="data/download"
mkdir DOWNLOAD_PATH

# Download main hosted data
wget -v -O myfile.tgz -L https://rice.box.com/shared/static/utsiopkb1vsss2aqi8122q0lkqdce4z0.gz

# Untar
tar -xvf QG-Net-Downloads.tar.gz

# Remove tar ball
#rm "$DOWNLOAD_PATH_TAR"

#echo "DrQA download done!"
