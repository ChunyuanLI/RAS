# RAS
AISTATS 2019: Referenced-based Adversarial Sampling (RAS) &amp; Its applications to Soft Q-learning

This repository contains source code necessary to reproduce the results presented in the paper [Adversarial Learning of a Sampler Based on an Unnormalized Distribution](https://arxiv.org) (AISTATS 2019):

```
@inproceedings{Li_RAS_2019_AISTATS,
  title={Adversarial Learning of a Sampler Based on an Unnormalized Distribution},
  author={Chunyuan Li, Ke Bai, Jianqiao Li, Guoyin Wang, Changyou Chen, Lawrence Carin},
  booktitle={AISTATS},
  year={2019}
}
```


Comparison of **RAS** and **GAN** learning scenarios to target distribution p(x)

| | RAS: Reference-based Adversarial Sampling  |   GAN 
|-------------------------|:-------------------------:|:-------------------------:
| | ![](/figs/ras_scheme.png)  |   ![](/figs/gan_scheme.png)
| Method | We propose the “reference” **p_r** to bridge neural samples **q**  and unnormalized form **u**, making the evaluations of both terms feasible. | Directly matching neural samples **q** to empirical samples **p'**
| Setup | unnormalized form **u**  | empirical samples **p'**  




## Contents
There are four steps to use this codebase to reproduce the results in the paper.

1. [Dependencies](#dependencies) 
2. [Soft Q-learning](#Soft-Q-learning) 
3. [Constrained Domain Sampling](#Constrained-Domain-Sampling)
4. [Entropy Regularization](#Entropy-regularization) 
5. [Reproduce paper figure results](#reproduce-paper-figure-results) 

## Dependencies

This code is based on Python 2.7, with the main dependencies being [TensorFlow==1.7.0](https://www.tensorflow.org/) and [Keras==2.1.5](https://keras.io/). Additional dependencies for running experiments are: `numpy`, `cPickle`, `scipy`, `math`, `gensim`. 

## Soft Q-learning

We consider the following datasets: Yahoo, AGnews, DBPedia, yelp, yelp binary. For convenience, we provide pre-processed versions of all datasets. Data are prepared in pickle format. Each `.p` file has the same fields in same order: `train text`, `val text`, `test text`, `train label`, `val label`, `test label`, `dictionary` and `reverse dictionary`.

## Constrained Domain Sampling

## Entropy Regularization

## Reproduce paper figure results
Jupyter notebooks in [`plots`](./plots) folders are used to reproduce paper figure results.

Note that without modification, we have copyed our extracted results into the notebook, and script will output figures in the paper. If you've run your own training and wish to plot results, you'll have to organize your results in the same format instead.



