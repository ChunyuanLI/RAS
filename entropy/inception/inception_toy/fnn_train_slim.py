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


def CNN(inputs, _is_training=True):
    x   = tf.reshape(inputs, [-1, 2])
    batch_norm_params = {'is_training': _is_training, 'decay': 0.9, 'updates_collections': None}
    net = slim.fully_connected(x, 1024
                    , activation_fn       = tf.nn.relu
                    , weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
                    , normalizer_fn       = slim.batch_norm
                    , normalizer_params   = batch_norm_params
                    , scope='fc4')

    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 1024
                    , activation_fn       = tf.nn.relu
                    , weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
                    , normalizer_fn       = slim.batch_norm
                    , normalizer_params   = batch_norm_params
                    , scope='fc4')
    net = slim.dropout(net, keep_prob=0.7, is_training=_is_training, scope='dropout4')  
    out = slim.fully_connected(net, 5, activation_fn=None, normalizer_fn=None, scope='fco')
    return out



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

pdb.set_trace()


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
# plot_GMM(dataset, save_path)

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



NUM_LABELS=5


# EXTRACT LABELS
def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding

# PREPARE MNIST DATA
def prepare_MNIST_data(use_data_augmentation=True):
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)
    validation_data = train_data[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE,:]
    train_data = train_data[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:,:]
    if use_data_augmentation:
        train_total_data = expend_training_data(train_data, train_labels)
    else:
        train_total_data = np.concatenate((train_data, train_labels), axis=1)
    train_size = train_total_data.shape[0]

    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels

# CONFIGURATION
MODEL_DIRECTORY   = "model/model.ckpt"
LOGS_DIRECTORY    = "logs/train"
training_epochs   = 10
TRAIN_BATCH_SIZE  = 50
display_step      = 500
validation_step   = 500
TEST_BATCH_SIZE   = 5000    


# PREPARE MNIST DATA
batch_size = TRAIN_BATCH_SIZE # BATCH SIZE (50)
num_labels = NUM_LABELS       # NUMBER OF LABELS (10)
train_total_data, train_size, validation_data, validation_labels \
    , test_data, test_labels = prepare_MNIST_data(True)


# DEFINE MODEL
# PLACEHOLDERS
x  = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 5]) #answer
is_training = tf.placeholder(tf.bool, name='MODE')
# CONVOLUTIONAL NEURAL NETWORK MODEL 
y = CNN(x, is_training)
# DEFINE LOSS
with tf.name_scope("LOSS"):
    loss = slim.losses.softmax_cross_entropy(y, y_)
# DEFINE ACCURACY
with tf.name_scope("ACC"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# DEFINE OPTIMIZER
with tf.name_scope("ADAM"):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        1e-4,               # LEARNING_RATE
        batch * batch_size, # GLOBAL_STEP
        train_size,         # DECAY_STEP
        0.95,               # DECAY_RATE
        staircase=True)     # LR = LEARNING_RATE*DECAY_RATE^(GLOBAL_STEP/DECAY_STEP)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=batch)
    # 'batch' IS AUTOMATICALLY UPDATED AS WE CALL 'train_step'

# SUMMARIES
saver = tf.train.Saver()
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('loss', loss)
tf.summary.scalar('acc', accuracy)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())
print ("MODEL DEFINED.")


# OPEN SESSION
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})


# MAXIMUM ACCURACY
max_acc = 0.
# LOOP
for epoch in range(training_epochs): # training_epochs: 10
    # RANDOM SHUFFLE
    np.random.shuffle(train_total_data)
    train_data_   = train_total_data[:, :-num_labels]
    train_labels_ = train_total_data[:, -num_labels:]
    # ITERATIONS
    total_batch = int(train_size / batch_size)
    for iteration in range(total_batch):
        # GET CURRENT MINI-BATCH
        offset = (iteration * batch_size) % (train_size)
        batch_xs = train_data_[offset:(offset + batch_size), :]
        batch_ys = train_labels_[offset:(offset + batch_size), :]
        # OPTIMIZE
        _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op]
                                    , feed_dict={x: batch_xs, y_: batch_ys, is_training: True})
        # WRITE LOG
        summary_writer.add_summary(summary, epoch*total_batch + iteration)

        # DISPLAY
        if iteration % display_step == 0:
            print("Epoch: [%3d/%3d] Batch: [%04d/%04d] Training Acc: %.5f" 
                  % (epoch + 1, training_epochs, iteration, total_batch, train_accuracy))

        # GET ACCURACY FOR THE VALIDATION DATA
        if iteration % validation_step == 0:
            validation_accuracy = sess.run(accuracy,
            feed_dict={x: validation_data, y_: validation_labels, is_training: False})
            print("Epoch: [%3d/%3d] Batch: [%04d/%04d] Validation Acc: %.5f" 
                  % (epoch + 1, training_epochs, iteration, total_batch, validation_accuracy))
        # SAVE THE MODEL WITH HIGEST VALIDATION ACCURACY
        if validation_accuracy > max_acc:
            max_acc = validation_accuracy
            save_path = saver.save(sess, MODEL_DIRECTORY)
            print("  MODEL UPDATED TO [%s] VALIDATION ACC IS %.5f" 
                  % (save_path, validation_accuracy))
print("OPTIMIZATION FINISHED")




# RESTORE SAVED NETWORK
saver.restore(sess, MODEL_DIRECTORY)

# COMPUTE ACCURACY FOR TEST DATA
test_size   = test_labels.shape[0]
total_batch = int(test_size / batch_size)
acc_buffer  = []
for i in range(total_batch):
    offset = (i * batch_size) % (test_size)
    batch_xs = test_data[offset:(offset + batch_size), :]
    batch_ys = test_labels[offset:(offset + batch_size), :]
    y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
    correct_prediction = np.equal(np.argmax(y_final, 1), np.argmax(batch_ys, 1))
    acc_buffer.append(np.sum(correct_prediction.astype(float)) / batch_size)
print("TEST ACCURACY IS: %.4f" % np.mean(acc_buffer))



