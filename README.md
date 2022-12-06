# MKCGN


Meta-relation Assisted Knowledge-aware Coupled Graph Neural Network for Social Recommendation
## Environments

- python 3.8
- pytorch-1.6
- DGL 0.5.3 (https://github.com/dmlc/dgl)
## Dataset 
from KCGN (Knowledge-aware Coupled Graph Neural Network for Social Recommendation) AAAI2021
## Example to run the codes		

train model:

```
MKCGN
Dataset: Yelp, Result: test HR = 0.8315, test NDCG = 0.577
python main.py --dataset Yelp --reg 0.05 --lr 0.01 --batch 2048 --hide_dim 64 --layer [64] --slope 0.4 --time_step 360 --lam [0.1,0.001] --clear 1 --gamma=0.3 --fuse mean

```
