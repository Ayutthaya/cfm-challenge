#!/bin/bash

# install recent libgcc
~/anaconda3/bin/conda install libgcc

# go to home
cd

# get Anaconda3 installer
wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh

# execute Anaconda3 installer
chmod +x Anaconda3-4.2.0-Linux-x86_64.sh
./Anaconda3-4.2.0-Linux-x86_64.sh -b -p ~/anaconda3

# install xgboost
sudo apt-get update
sudo apt-get -qq -y install make
sudo apt-get -qq -y install gcc
sudo apt-get -qq -y install g++
sudo git clone https://github.com/dmlc/xgboost
cd xgboost
sudo git checkout tags/v0.60
sudo git submodule init
sudo git submodule update
sudo ./build.sh
cd python-package
sudo ~/anaconda3/bin/python setup.py install
