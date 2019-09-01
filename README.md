# DEEMS-KDD-19

This repository holds the codes for [Dual Sequential Prediction Models Linking Sequential Recommendation and Information Dissemination. (KDD 2019)](http://delivery.acm.org/10.1145/3340000/3330959/p447-wu.pdf?ip=222.204.233.130&id=3330959&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E035EACC12F524219%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1567325423_2eda58a134223d97be5f03a70f82e404). 
The requirements are Python 3.6 and Tensorflow 1.7.0.

The utils folder contains scripts for data preparation. The DEEMS_RNN and DEEMS_ATTN folders contain implementations for dual sequential model with RNN units and self-attention units, respectively. The DREAM folder contains implementation for RNN-based sequential recommendation model, while the DIN folder for self-attention-based sequential recommendation model. The RRN folder contains implementation for Recurrent Recommender Networks.

In each folder, you can run

    python train.py
    
to train a model.

If you use this code as part of any published research, please cite the following paper:

```
@inproceedings{DEEMS-KDD-19,
  author    = {Qitian Wu and
               Yirui Gao and
               Xiaofeng Gao and
               Paul Weng and
               Guihai Chen},
  title     = {Dual Sequential Prediction Models Linking Sequential Recommendation
               and Information Dissemination},
  booktitle = {Proceedings of the 25th {ACM} {SIGKDD} International Conference on
               Knowledge Discovery {\&} Data Mining, {KDD} 2019, Anchorage, AK,
               USA, August 4-8, 2019.},
  pages     = {447--457},
  year      = {2019}
  }
```

The datasets used in our paper can be found at the following url:

http://jmcauley.ucsd.edu/data/amazon/
