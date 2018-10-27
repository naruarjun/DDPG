import tensorflow as tf
from tensorflow import float32 as f32
from FCNN import FCNN

class Model:
  def __init__(self, inputs):
    self.inputs = inputs

  def get_tf_var(self, name):
    pass


class Actor(Model):
  def __init__(self, inputs, n_layers, n_units):
    super(Actor, self).__init__(inputs)
    self._input = tf.concat([inputs["c_obs"], inputs["goal"]], axis=1)
    self.pi = FCNN(self._input, inputs["u"].shape[-1], n_layers, n_units, tf.nn.relu, tf.nn.tanh, name="Actor")
    self.actor_op = self.pi.network_op

  def __call__(self, x):
    return tf.get_default_session().run(self.actor_op, feed_dict=x)

class Critic(Model):
  def __init__(self, inputs, actor, n_layers, n_units):
    super(Critic, self).__init__(inputs)
    self._input = tf.concat([inputs["c_obs"], inputs["goal"], actor], axis=1)
    self.Q  = FCNN(self._input, 1, n_layers, n_units, tf.nn.relu, None, "Critic")
    self.critic_op = self.Q.network_op

  def __call__(self, x):
    return tf.get_default_session().run(self.critic_op, feed_dict=x)

  def get_loss(self, x):
    # apply some loss function here
    pass




# PREP FOR STAGING AREA
from collections import OrderedDict
var = {"c_obs": (None, 18), "n_obs": (None, 18), "u": (None, 2), "r": (None, 1), "goal": (None, 4)}
ovar = OrderedDict()
for i in sorted(var.keys()):
  ovar[i] = var[i] 
print (ovar)

# STAGING AREA
sa = tf.contrib.staging.StagingArea(dtypes=[f32 for _ in var], 
                                    shapes=[shape for shape in ovar.values()])
sa.put([tf.placeholder(f32, shape=shape, name=name) for name, shape in ovar.items()])

# GETTING STUFF FROM STAGING AREA
b = sa.get()
inputs = {k: b[i] for i, k in enumerate(ovar.keys())}

a = Actor(inputs, 2, 2)
c = Critic(inputs, a.actor_op, 2, 2)


# TESTS
import numpy as np
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  some_act = a({inputs["c_obs"]: np.random.normal(size=18).reshape((1,18)), inputs["goal"]: np.random.normal(size=4).reshape((1,4))})
  some_val = c({inputs["c_obs"]: np.random.normal(size=18).reshape((1,18)), inputs["goal"]: np.random.normal(size=4).reshape((1,4))})
  print(some_act)
  print(some_val)


writer = tf.summary.FileWriter("/tmp/a", tf.get_default_graph())
# print (op)
writer.close()
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))