import tensorflow as tf
from tensorflow.layers import dense as FCLayer
from tensorflow.initializers import truncated_normal

class FCNN:
  def __init__(self, _input, op_dim, n_layers, n_units, activation, op_act, 
               name="FCNN", logdir="/tmp/ddpg/", w_init=None):
    self.op_dim     = op_dim
    self.n_layers   = n_layers
    self.n_units    = n_units
    self.scope      = name
    self.logdir     = logdir + self.scope
    self.w_init     = w_init if w_init != None else truncated_normal(stddev=1e-3)
    self.activation = activation
    self.op_act     = op_act
    self._input     = _input
    self.make()

  def make(self):
    with tf.variable_scope(self.scope):
      _input   = self._input
      for i in range(0, self.n_layers-1):
        output = FCLayer(_input, self.n_units, kernel_initializer=self.w_init, 
                         name="layer_{}".format(i))
        _input = self.activation(output)
      output   = FCLayer(_input, self.op_dim, kernel_initializer=self.w_init, 
                         name="layer_{}".format(i+1))

    self.network_op = output if self.op_act == None else self.op_act(output)

# sess = tf.Session()
# import numpy as np
# obs = tf.Variable(np.array([[0, 0]], "f"), name="obs")
# nn  = FCNN(obs, 2, 1, 3, 5, tf.nn.relu, tf.nn.tanh)
# writer = tf.summary.FileWriter("/tmp/test", tf.get_default_graph())
# writer.close()
