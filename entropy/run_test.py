from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import tensorflow as tf

from tqdm import tqdm

import os
import scipy.io
from argparse import ArgumentParser

from GAN_models import GAN
import time


GPUID = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)


slim = tf.contrib.slim
ds = tf.contrib.distributions
#st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace



# build parser
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default='gan_cc')
    parser.add_argument('--anneal', type=int, required=True, default=0)
    parser.add_argument('--sample_n',type=int,required=True, default=5000)
    parser.add_argument('--n_seed',type=int,required=False,default=10)
    parser.add_argument('--alpha', type=float,required=False,default=1.0)
    parser.add_argument('--beta', type=float,required=False,default=1.0)
    return parser

def test(model_name,anneal,sample_n,alpha,beta,number_of_random_seed=1):
	num_mixtures =8
	radius = 0.5
	thetas = np.linspace(0, 2 * np.pi, num_mixtures + 1)[:num_mixtures]
	xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
	std = 0.05

	random_seed=np.arange(number_of_random_seed)*10
	for i in range(len(random_seed)):
	    start=time.time()
	    rs=random_seed[i]
	    np.random.seed(rs)
	    model =GAN(
	    	sample_n=sample_n,
	        model_name=model_name,
	        anneal=anneal,
	        num_z=2,
	        hidden_size=128,
	        alpha=alpha,
	        beta=beta,
	        mix_coeffs=tuple([1 / num_mixtures] * num_mixtures),
	        mean=tuple(zip(xs, ys)),
	        cov=tuple([(std**2, std**2)] * num_mixtures),
	        batch_size=1024,
	        learning_rate=0.0002,
	        num_epochs=50000,
	        disp_freq=5000,
	        random_seed=rs)

	    model.fit()
	    model.save_point()
	    print(time.time()-start)



if  __name__ =='__main__':
    # Parse argument
    parser = build_parser()
    options = parser.parse_args()

    test(options.model,options.anneal,options.sample_n,options.alpha,options.beta,options.n_seed)

