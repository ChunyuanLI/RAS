import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import scipy.io as scio

EPSILON = 10e-14


def KL_divergence(h1, h2):
    h1_flatten = h1.flatten()
    h2_flatten = h2.flatten()
    h1_log = np.log(np.maximum(EPSILON, h1_flatten))
    h2_log = np.log(np.maximum(EPSILON, h2_flatten))
    # h1_log = np.log(h1_flatten)
    # h2_log = np.log(h2_flatten)
    h1_h2 = np.sum(np.multiply(h1_flatten, h1_log - h2_log))
    h2_h1 = np.sum(np.multiply(h2_flatten, h2_log - h1_log))
    return h1_h2, h2_h1


# How to use KL_divergence function to compute KL_sym
# h1 and h2 are twp 2D histograms
# kl12, kl21 = KL_divergence(h1, h2)
# KL_sym[j + 1, i] = (kl12 + kl21) / 2.0

#  For KL divergence, I attached a part of our code to compute it. You just pass two probabilities p and q which actually are histograms but normalized to be summed to 1. The output are KL(p||q) and KL(q||p). Our final symmetric KL metric is the average of these KLs. 
# 2. For Wasserstein metric, we used the Matlab code published by Marco Cuturi. Please see at
# http://marcocuturi.net/SI.html
# You have to define a cost matrix (here we used Euclidean distance between data points) and pass probabilities to sinkhornTransport function provided by that code. The second attached file is an example saying you how we used sinkhornTransport function.
