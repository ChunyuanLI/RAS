from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from utils.data_gmm import GMM_distribution, sample_GMM, plot_GMM
from utils.data_utils import shuffle, iter_data
from tqdm import tqdm
from six.moves import urllib
import tensorflow.contrib.slim as slim


slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace


print ("PACKAGES LOADED")

""" Create dataset """
n_epoch = 100
batch_size  = 128
dataset_size_x = 512*4
dataset_size_z = 512*4

dataset_size_x_test = 512*2
dataset_size_z_test = 512*2

input_dim = 2
latent_dim = 2
eps_dim = 2


# create X dataset
means_x = map(lambda x:  np.array(x), [[0, 0],
                                     [2, 2],
                                     [-2, -2],
                                     [2, -2],
                                     [-2, 2]])
means_x = list(means_x)
std_x = 0.04
variances_x = [np.eye(2) * std_x for _ in means_x]

priors_x = [1.0/len(means_x) for _ in means_x]

gaussian_mixture = GMM_distribution(means=means_x,
                                               variances=variances_x,
                                               priors=priors_x)
dataset_x = sample_GMM(dataset_size_x, means_x, variances_x, priors_x, sources=('features', ))
result_dir='result/'
save_path_x = result_dir + 'X_gmm_data_train.pdf'
# plot_GMM(dataset, save_path)

##  reconstruced x
X_dataset  = dataset_x.data['samples']
X_targets = dataset_x.data['label']

fig_mx, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))

ax.scatter(X_dataset[:, 0], X_dataset[:, 1], c=cm.Set1(X_targets.astype(float)/input_dim/2.0),
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.axis('on')
plt.savefig(save_path_x, transparent=True, bbox_inches='tight')


# create Z dataset
means_z = map(lambda x:  np.array(x), [[0, 0]])
# means = map(lambda x:  np.array(x), [[-1, -1],[1, 1]])
means_z = list(means_z)
std_z = 1.0
variances_z = [np.eye(2) * std_z for _ in means_z]
priors_z = [1.0/len(means_z) for _ in means_z]

dataset_z = sample_GMM(dataset_size_z, means_z, variances_z, priors_z, sources=('features', ))
save_path_z = result_dir + 'Z_gmm_data_train.pdf'

##  reconstruced x
Z_dataset = dataset_z.data['samples']
Z_labels  = dataset_z.data['label']

fig_mx, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
ax.scatter(Z_dataset[:, 0], Z_dataset[:, 1], c=cm.Set1(Z_labels.astype(float)/input_dim/2.0),
           edgecolor='none', alpha=0.5)
ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('$z_1$'); ax.set_ylabel('$z_2$')
ax.axis('on')
plt.savefig(save_path_z, transparent=True, bbox_inches='tight')

# CONFIGURATION
MODEL_DIRECTORY   = "model/model.ckpt"
LOGS_DIRECTORY    = "logs/train"
training_epochs   = 10
TRAIN_BATCH_SIZE  = 50
display_step      = 500
validation_step   = 500
TEST_BATCH_SIZE   = 5000    