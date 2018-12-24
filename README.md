# Adversarial Learning of a Sampler Based on an Unnormalized Distribution

(Sorry, Still Under Construction...)
<img src="figs/under_construction.png" width="200">

--------------

The RAS (Referenced-based Adversarial Sampling) algorithm is proposed to enable adversarial learning applicable to general unnormalized distribution sampling, with demonstrations on constrained domain sampling and soft Q-learning. This repository contains source code to reproduce the results presented in the paper [Adversarial Learning of a Sampler Based on an Unnormalized Distribution](https://arxiv.org) (AISTATS 2019):

```
@inproceedings{Li_RAS_2019_AISTATS,
  title={Adversarial Learning of a Sampler Based on an Unnormalized Distribution},
  author={Chunyuan Li, Ke Bai, Jianqiao Li, Guoyin Wang, Changyou Chen, Lawrence Carin},
  booktitle={AISTATS},
  year={2019}
}
```

## Introduction

### Comparison of **RAS** and **GAN** learning scenarios

Learning a neural sampler **q** to approximate the target distribution **p**, where only the latter's unnormalized form **u** or empirical samples **p'** is available, respectively.

|**Algorithm** | RAS  |   GAN 
|-------------------------|:-------------------------:|:-------------------------:
| **Illustration** | ![](/figs/scheme/ras_scheme.png)  |   ![](/figs/scheme/gan_scheme.png)
| **Method** | We propose the “reference” **p_r** to bridge neural samples **q**  and unnormalized form **u**, making the evaluations of both F_1 and F_2 terms feasible. | Directly matching neural samples **q** to empirical samples **p'**
| **Setup** | Learning from unnormalized form **u**  | Learning from empirical samples **p'**  
| **Generator** |  ![](https://latex.codecogs.com/gif.latex?\log[\frac{u(x)}{q_{\theta}(x)}&space;]=&space;\underbrace{&space;\log&space;\big[\frac{&space;p_{r}(x)&space;}{&space;q_{\theta}&space;(x)&space;}&space;\big]}_{\mathcal{F}_1}&space;&plus;&space;\underbrace{&space;\log&space;\big[\frac{&space;u&space;(x)&space;}{p_{r}(x)&space;}\big]&space;}_{\mathcal{F}_2})  | ![](https://latex.codecogs.com/gif.latex?\log[\frac{p^{\prime}(x)}{q_{\theta}(x)}&space;])
| **Discriminator** | **q** vs **p_r** | **q** vs **p'**

### Discussion
1. In many applications (e.g. Soft Q-learining), only **u** is known, and we are inerested in drawing its samples efficiently
2. The choice of **p_r** has an effect on learning; It should be carefully chosen.


## Contents
There are three steps to use this codebase to reproduce the results in the paper.

1. [Dependencies](#dependencies)
2. [Experiments](#experiments)

    2.1. [Adversarial Soft Q-learning](#adversarial-soft-q-learning)
    
    2.2. [Constrained Domain Sampling](#constrained-domain-sampling)
    
    2.3. [Entropy Regularization](#entropy-regularization) 

3. [Reproduce paper figure results](#reproduce-paper-figure-results) 

## Dependencies

This code is based on Python 2.7, with the main dependencies being [TensorFlow==1.7.0](https://www.tensorflow.org/) and [Keras==2.1.5](https://keras.io/). Additional dependencies for running experiments are: `numpy`, `cPickle`, `scipy`, `math`, `gensim`. 

## Adversarial Soft Q-learning

We consider the following environments: `Hopper`, `Half-cheetah`, `Ant`, `Walker`, `Swimmer` and `Humanoid`. All soft q-learning code is at [sql](/sql): 

To run:

    python --env Hopper --method ras

It takes the following options (among others) as arguments:

- `--env` It specifies the Mojoco/RLlab environment; default `Hopper`.   
- `--method`: To apply the sampling method. default 'ras'. It supports [`ras`, `svgd`].


| Swimmer (rllab) | Humanoid (rllab) |  Hopper-v1 |  Half-cheetah-v1 |  Ant-v1 |  Walker-v1
|-------------------------|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
| ![](/figs/sql/Swimmer-rllab.png) | ![](/figs/sql/Humanoid-rllab.png)  |  ![](/figs/sql/Hopper-v1.png) |  ![](/figs/sql/Half-cheetah-v1.png) |  ![](/figs/sql/Ant-v1.png) |  ![](/figs/sql/Walker-v1.png)
| ![](https://i.makeagif.com/media/3-27-2018/u2cewJ.gif) | ![](https://outlookseries.com/A0972/Infrastructure/image7.gif) | ![](https://devblogs.nvidia.com/wp-content/uploads/2016/04/hopper-1.gif) | ![](https://ask.qcloudimg.com/http-save/yehe-1326493/xzyqmpribu.gif) | ![](https://blog.openai.com/content/images/2017/02/running_bug.gif) | ![](https://user-images.githubusercontent.com/306655/28396526-d4ce6334-6cb0-11e7-825c-63a85c8ff533.gif)

TODO: Replace the gif files with RAS results

## Constrained Domain Sampling

To show that RAS can draw samples when the support is bounded, we apply it to sample from the distributions with the support [a,b]. Please see the code in [constrained_sampling](./constrained_sampling) 


| RAS: Beta reference | Gaussian reference  | SVGD | Amortized SVGD  
|-------------------------|:-------------------------:|:-------------------------:|:-------------------------:
| ![](/figs/constrained/cons1_beta.png) | ![](/figs/constrained/cons1_gaussian.png) | ![](/figs/constrained/cons1_svgd_teacher.png) | ![](/figs/constrained/cons1_svgd_student.png)
| ![](/figs/constrained/cons1_beta_2mode.png) | ![](/figs/constrained/cons1_gaussian_2mode.png) | ![](/figs/constrained/cons1_svgd_teacher_2mode.png) | ![](/figs/constrained/cons1_svgd_student_2mode.png)

## Entropy Regularization

An entropy term **H(x)** is approximated to stablize adversarial training. As examples, we consider to regulize the following GAN variants: [`GAN`](https://arxiv.org/abs/1406.2661), [`SN-GAN`](https://arxiv.org/abs/1802.05957), [`D2GAN`](https://arxiv.org/abs/1709.03831) and [`Unrolled-GAN`](https://arxiv.org/abs/1611.02163). All entropy-regularization code is at [entropy](entropy): 

To run:

    python --baseline GAN --entropy
    
It takes the following options (among others) as arguments:

- The `baseline` specifies the GAN variant to apply the entropy regularizer. It supports [`GAN`, `SN-GAN`, `D2GAN`, `Unrolled-GAN`]; default `GAN`.   
- `--entropy`: To apply entropy regularizer or not.
  

| Entropy regularizer on 8-GMM toy dataset | SN-GAN  |   SN-GAN + Entropy  
|-------------------------|:-------------------------:|:-------------------------:
| ![](/figs/entropy/Symmetric_KL_Divergence_iclr.png) | ![](/figs/entropy/sn_gan_8gmm.png)  |   ![](/figs/entropy/sn_gan_entropy_8gmm.png)


## Reproduce paper figure results
Jupyter notebooks in [`plots`](./plots) folders are used to reproduce paper figure results.

Note that without modification, we have copyed our extracted results into the notebook, and script will output figures in the paper. If you've run your own training and wish to plot results, you'll have to organize your results in the same format instead.


## Questions?
Please drop us ([Chunyuan](http://chunyuan.li/), [Ke](https://github.com/beckybai), [Jianqiao](https://github.com/jianqiaol) or [Guoyin](https://github.com/guoyinwang)) a line if you have any questions.


