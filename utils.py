import tensorflow as tf


def fc_layer(input, size, name):
  with tf.name_scope(name):
    w = tf.Variable(tf.random_normal(size), name="W")
    b = tf.Variable(tf.random_normal([size[-1]]), name="B")
    return tf.nn.relu(tf.matmul(input, W) + b)