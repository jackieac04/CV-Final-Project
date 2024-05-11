@echo off
REM Generate SSH key
ssh-keygen -t rsa
ssh-keyscan -t rsa github.com >> \root\.ssh\known_hosts

REM Clear the screen
cls

REM Display public key for GitHub
echo PUBLIC KEY: (for github.com)
type \root\.ssh\id_rsa.pub

REM Clone the repository
git clone git@github.com:jackieac04/CV-Final-Project.git
cd CV-Final-Project

REM Stash any changes
git stash

REM Configure Git user
git config --global user.email "colab_bot@brown.edu"
git config --global user.name "Colab Bot"

REM Reset and checkout a specific branch
git reset --hard
git checkout data-retrieval2
git pull

REM Install necessary Python packages
pip install lime

REM Install TensorFlow and Keras
pip install tensorflow
pip install keras

REM Download VGG16 model
wget "https://browncsci1430.github.io/hw5_cnns/vgg16_imagenet.h5"

REM Change directory to the code directory
cd code
cd model

REM Run Python script
python main.py --task 3