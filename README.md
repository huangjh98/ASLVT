# [MSAM: Multi-Semantic Adaptive Mining for Cross-Modal Drone Video-Text Retrieval]

##### Author: Jinghao Huang 
--------------------------
## Environment

Python 3.8.0  
Pytorch 1.11.0  
torchvision 0.12.0  
numpy 1.21.6  

--------------------------
## Dataset
We use the following 2 datasets: USRD Video-Text Dataset and UMCRD Video-Text Dataset. [Dataset download link](https://pan.baidu.com/)

--------------------------
## Train

We train our model on two 3090Ti GPU cards. To train on different datasets, one needs to modify the configuration file in the code and then use the following training command:

sh train.sh 

--------------------------
## Test
