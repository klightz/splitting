# Splitting Steepest Descent for Growing NeuralArchitectures
We develop a progressive training approach for neural networks which adaptively grows the network structure by splitting existing neurons to multiple off-springs. We test our algorithm on several
[[Paper]](https://arxiv.org/abs/1910.02366)     

## Overview
This code mainly implements our algorithm on CIFAR10 and CIFAR100 datasets and uses MobileNet V1 as the backbone. To use the code, simply run the train.sh script.

## Citation
Please cite this paper if you want to use it in your work,

@article{liu2019splitting,
title={Splitting Steepest Descent for Growing Neural Architectures},
author={Liu, Qiang and Wu, Lemeng and Wang, Dilin},
journal={arXiv preprint arXiv:1910.02366},
year={2019}
}

## License
MIT License
