from baselines.ddpg.models import Actor, Critic
import numpy as np
import tensorflow as tf

sess = tf.Session()
a = Actor(3)
# o = np.ones((4,4))
# print (o)
x = tf.placeholder(tf.float32, shape=(None,4), name="obs")
act = a(x)
sess.run(tf.global_variables_initializer())
# print (sess.run(act, feed_dict={x: o}))
ss = tf.summary.FileWriter("/tmp/ddpg", sess.graph)
ss.close()