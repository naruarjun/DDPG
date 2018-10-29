import tensorflow as tf
from tensorflow.layers import dense as FCLayer
from tensorflow.initializers import truncated_normal as TN


class FCNN:
    def __init__(self, _input, op_dim, n_layers, n_units, activation, op_act,
                 name="FCNN", logdir="/tmp/ddpg/", w_init=None):
        self.op_dim = op_dim
        self.n_layers = n_layers
        self.n_units = n_units
        self.scope = name
        self.logdir = logdir + self.scope
        self.w_init = w_init if w_init is not None else TN(stddev=1e-3)
        self.activation = activation
        self.op_act = op_act
        self._input = _input
        self.name = name
        self.make()

    def make(self):
        with tf.variable_scope(self.scope):
            _input = self._input
        for i in range(0, self.n_layers-1):
            op = FCLayer(_input, self.n_units,
                         kernel_initializer=self.w_init,
                         name="layer_{}".format(i))
            _input = self.activation(op)
        op = FCLayer(_input, self.op_dim, kernel_initializer=self.w_init,
                     name="layer_{}".format(i+1))
        self.network_op = op if self.op_act is None else self.op_act(op)
