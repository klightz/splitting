# Splitting Steepest Descent for Growing Neural Architectures
Here is the implementation of the following paper which is received by NeurIPS 2019.

[[Paper]](https://arxiv.org/abs/1910.02366)      

## Overview
This code mainly implements our algorithm on CIFAR10 and CIFAR100 datasets and uses MobileNet V1 as the backbone. To use the code, simply run

```shell 
mbv1/train.sh
```

## Citation
Please cite this paper if you want to use it in your work,

    @article{liu2019splitting,
      title={Splitting Steepest Descent for Growing Neural Architectures},
      author={Liu, Qiang and Wu, Lemeng and Wang, Dilin},
      journal={arXiv preprint arXiv:1910.02366},
      year={2019}
    }

## Energy-Aware Fast Splitting version
Here is an Energy-aware fast splitting version with more benchmarks implemented. [[Link]](https://github.com/dilinwang820/fast-energy-aware-splitting)

## License
MIT License
