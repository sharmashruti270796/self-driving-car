import tensorflow.compat.v1 as tf
import numpy
import scipy
import pandas as pd
import imageio
from PIL import Image






tf.disable_v2_behavior()

def weights(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def cnn(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = x

w_cnn1 = weights([5, 5, 3, 24])
b_cnn1 = bias([24])
cnn1 = tf.nn.relu(cnn(x_image, w_cnn1, 2) + b_cnn1)


w_cnn2 = weights([5, 5, 24, 36])
b_cnn2 = bias([36])
cnn2 = tf.nn.relu(cnn(cnn1, w_cnn2, 2) + b_cnn2)

w_cnn3 = weights([5, 5, 36, 48])
b_cnn3 = bias([48])
cnn3 = tf.nn.relu(cnn(cnn2, w_cnn3, 2) + b_cnn3)

w_cnn4 = weights([3, 3, 48, 64])
b_cnn4 = bias([64])
cnn4 = tf.nn.relu(cnn(cnn3, w_cnn4, 1) + b_cnn4)

w_cnn5 = weights([3, 3, 64, 64])
b_cnn5 = bias([64])
cnn5 = tf.nn.relu(cnn(cnn4, w_cnn5, 1) + b_cnn5)
w_full1 = weights([1152, 1164])
b_full1 = bias([1164])
cnn5_flat = tf.reshape(cnn5, [-1, 1152])


full_cnn1 = tf.nn.relu(tf.matmul(cnn5_flat, w_full1) + b_full1)
keep_prob = tf.placeholder(tf.float32)

full_cnn1_drop = tf.nn.dropout(full_cnn1, rate = 1 - keep_prob)

w_full2 = weights([1164, 100])
b_full2 = bias([100])
full_cnn2 = tf.nn.relu(tf.matmul(full_cnn1_drop, w_full2) + b_full2)
full_cnn2_drop = tf.nn.dropout(full_cnn2, rate = 1 - keep_prob)

w_full3 = weights([100, 50])
b_full3 = bias([50])
full_cnn3 = tf.nn.relu(tf.matmul(full_cnn2_drop, w_full3) + b_full3)
full_cnn3_drop = tf.nn.dropout(full_cnn3, rate = 1 - keep_prob)

w_full4 = weights([50, 10])
b_full4 = bias([10])
full_cnn4 = tf.nn.relu(tf.matmul(full_cnn3_drop, w_full4) + b_full4)
full_cnn4_drop = tf.nn.dropout(full_cnn4, rate= 1 - keep_prob)




w_full5 = weights([10, 1])
b_full5 = bias([1])

output = tf.multiply(tf.atan(tf.matmul(full_cnn4_drop, w_full5) + b_full5), 2)
