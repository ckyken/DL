#!/bin/bash

wget https://raw.githubusercontent.com/circulosmeos/gdown.pl/master/gdown.pl
# perl gdown.pl https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg img_align_celeba.zip
perl gdown.pl https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing img_align_celeba.zip
unzip img_align_celeba.zip


rm gdown.pl