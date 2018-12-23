# Adversarial Learning of a Sampler Based on an Unnormalized Distribution
RAS (Referenced-based Adversarial Sampling) and its applications to Soft Q-learning. This repository contains source code to reproduce the results presented in the paper [Adversarial Learning of a Sampler Based on an Unnormalized Distribution](https://arxiv.org) (AISTATS 2019):

```
@inproceedings{Li_RAS_2019_AISTATS,
  title={Adversarial Learning of a Sampler Based on an Unnormalized Distribution},
  author={Chunyuan Li, Ke Bai, Jianqiao Li, Guoyin Wang, Changyou Chen, Lawrence Carin},
  booktitle={AISTATS},
  year={2019}
}
```


Comparison of **RAS** and **GAN** learning scenarios: learning a neural sampler **q** to approximate the target distribution **p** (where only its unnormalized form **u** or empirical samples **p'** is available)

|**Algorithm** | RAS  |   GAN 
|-------------------------|:-------------------------:|:-------------------------:
| **Illustration** | ![](/figs/ras_scheme.png)  |   ![](/figs/gan_scheme.png)
| **Method** | We propose the “reference” **p_r** to bridge neural samples **q**  and unnormalized form **u**, making the evaluations of both F_1 and F_2 terms feasible. | Directly matching neural samples **q** to empirical samples **p'**
| **Setup** | Learning from unnormalized form **u**  | Learning from empirical samples **p'**  
| **Generator** |  ![](https://latex.codecogs.com/gif.latex?\log[\frac{u(x)}{q_{\theta}(x)}&space;]=&space;\underbrace{&space;\log&space;\big[\frac{&space;p_{r}(x)&space;}{&space;q_{\theta}&space;(x)&space;}&space;\big]}_{\mathcal{F}_1}&space;&plus;&space;\underbrace{&space;\log&space;\big[\frac{&space;u&space;(x)&space;}{p_{r}(x)&space;}\big]&space;}_{\mathcal{F}_2})  | ![](https://latex.codecogs.com/gif.latex?\log[\frac{p^{\prime}(x)}{q_{\theta}(x)}&space;])
| **Discriminator** | **q** vs **p_r** | **q** vs **p'**




## Contents
There are three steps to use this codebase to reproduce the results in the paper.

1. [Dependencies](#dependencies)

2. [Experiments](#experiments)
      (
    2.1. [Soft Q-learning](#soft-q-learning)
    2.2. [Constrained Domain Sampling](#constrained-domain-sampling)
    2.3. [Entropy Regularization](#entropy-regularization) 
    )
    
3. [Reproduce paper figure results](#reproduce-paper-figure-results) 

## Dependencies

This code is based on Python 2.7, with the main dependencies being [TensorFlow==1.7.0](https://www.tensorflow.org/) and [Keras==2.1.5](https://keras.io/). Additional dependencies for running experiments are: `numpy`, `cPickle`, `scipy`, `math`, `gensim`. 

## Soft Q-learning

We consider the following environments: `Hopper`, `Half-cheetah`, `Ant`, `Walker`, `Swimmer` and `Humanoid`. All soft q-learning code is at [sql]: 

To run:

    python --env Hopper --method ras

It takes the following options (among others) as arguments:

- `--env` It specifies the Mojoco/RLlab environment; default `Hopper`.   
- `--method`: To apply the sampling method. default 'ras'. It supports [`ras`, `svgd`].

## Constrained Domain Sampling

To show that RAS can draw samples when the support is bounded, we apply it to sample from the distributions with the support [a,b].

## Entropy Regularization

An entropy term **H(x)** is approximated to stablize adversarial training. We consider to regulize the following GAN variants: `GAN`, [`SN-GAN`](https://arxiv.org/abs/1802.05957), `D2GAN` and `Unrolled-GAN`.

All entropy-regularization code is at [entropy]: 

To run:

    python --baseline GAN --entropy
    
It takes the following options (among others) as arguments:

- The `baseline` specifies the GAN variant to apply the entropy regularizer. It supports [`GAN`, `SN-GAN`, `D2GAN`, `Unrolled-GAN`]; default `GAN`.   
- `--entropy`: To apply entropy regularizer or not.
  

## Reproduce paper figure results
Jupyter notebooks in [`plots`](./plots) folders are used to reproduce paper figure results.

Note that without modification, we have copyed our extracted results into the notebook, and script will output figures in the paper. If you've run your own training and wish to plot results, you'll have to organize your results in the same format instead.


## Questions?
Please drop us ([Chunyuan](http://chunyuan.li/), [Ke](https://github.com/beckybai), [Jianqiao](https://github.com/jianqiaol) or [Guoyin](https://github.com/guoyinwang)) a line if you have any questions.


