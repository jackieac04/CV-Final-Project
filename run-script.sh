#!/bin/bash

# Generate SSH key
ssh-keygen -t rsa
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

# Clear the screen
clear

# Display public key for GitHub
echo "PUBLIC KEY: (for github.com)"
cat ~/.ssh/id_rsa.pub

# Clone the repository
git clone git@github.com:jackieac04/CV-Final-Project.git
cd CV-Final-Project

# Stash any changes
git stash

# Configure Git user
git config --global user.email "colab_bot@brown.edu"
git config --global user.name "Colab Bot"

# Reset and checkout a specific branch
git reset --hard
git checkout data-retrieval2
git pull

# Install necessary Python packages
pip install lime

# Install TensorFlow and Keras
pip install tensorflow
pip install keras

# Download VGG16 model
wget "https://browncsci1430.github.io/hw5_cnns/vgg16_imagenet.h5"

# Change directory to the code directory
cd code/model

# Run Python script
python main.py --task 3

# Restore terminal input/output
stty echo
