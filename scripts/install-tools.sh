#!/bin/bash

# go to home
cd

# get Anaconda3 installer
wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh

# execute Anaconda3 installer
chmod +x Anaconda3-4.2.0-Linux-x86_64.sh
./Anaconda3-4.2.0-Linux-x86_64.sh

# install xgboost
sudo apt-get update
sudo apt-get -qq -y install make
sudo apt-get -qq -y install gcc
sudo apt-get -qq -y install g++
sudo apt-get -qq -y install git
sudo git clone https://github.com/dmlc/xgboost
cd xgboost
sudo git checkout tags/v0.60
sudo git submodule init
sudo git submodule update
sudo ./build.sh
cd python-package
sudo ~/anaconda3/bin/python setup.py install
cd

# install more recent libgcc
~/anaconda3/bin/conda install libgcc

# install unzip
sudo apt-get -qq -y install unzip
