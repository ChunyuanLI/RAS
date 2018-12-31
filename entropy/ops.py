from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from sn import spectral_normed_weight

def linear(input, output_dim, scope='linear', stddev=0.01,use_SN=False, update_collection=None):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('biases', [output_dim], initializer=const)
        if use_SN:
            print('using SN')
            return tf.matmul(input,spectral_normed_weight(w, update_collection=update_collection))+b
        else:
            return tf.matmul(input, w) + b


def gmm_sample(num_samples, mix_coeffs, mean, cov):
    # z = np.random.multinomial(num_samples, mix_coeffs)
    z=[int(num_samples/8.0)]*len(mean)
    samples = np.zeros(shape=[num_samples, len(mean[0])])
    i_start = 0
    for i in range(len(mix_coeffs)):
        i_end = i_start + z[i]
        samples[i_start:i_end, :] = np.random.multivariate_normal(
            mean=np.array(mean)[i, :],
            cov=np.diag(np.array(cov)[i, :]),
            size=z[i])
        i_start = i_end
    return samples
