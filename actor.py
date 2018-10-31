import tensorflow as tf

from tensorflow import multiply as mul

from tensorflow.train import AdamOptimizer as Adam

from FCNN import FCNN


class Actor:
    def __init__(self, sess, input_t, **params):
        self.session = sess
        self.__dict__.update(params)
        self.__dict__.update(input_t)
        self.generate_networks()
        self.define_operations()

    def generate_networks(self):
        # MAIN ACTOR NETWORK
        self.pi = FCNN(self.x, self.x.shape[-1], self.n_layers, self.n_units,
                       tf.nn.relu, tf.nn.tanh, name="pi")
        # TARGET ACTOR NETWORK
        self.PI = FCNN(self.x, self.x.shape[-1], self.n_layers, self.n_units,
                       tf.nn.relu, tf.nn.tanh, name="t_pi")

    def define_operations(self):
        with tf.name_scope("actor_ops"):
            # GRADIENT OF ACTIONS WRT ACTOR PARAMS TIMES NEGATIVE GRADIENT OF
            # VALUE FUNCTION WRT ACTIONS
            grads = tf.gradients(self.pi.nn, self.pi.net_params, -self.g)
            # APPLY GRADIENTS TO ACTOR NETWORK
            self.optimize = Adam(self.lr, name="pi_adam")\
                .apply_gradients(zip(grads, self.pi.net_params))
            # UPDATE TARGET OP
            net_param_pairs = zip(self.pi.net_params, self.PI.net_params)
            self.update_PI = [j.assign(mul(self.tau, i)+mul((1-self.tau), j))
                              for i, j in net_param_pairs]

    def predict(self, x, u):
        return self.session.run(self.pi.nn,
                                feed_dict={self.x: x, self.u: u})

    def predict_target(self, x, u):
        return self.session.run(self.PI.nn,
                                feed_dict={self.x: x, self.u: u})

    def train(self, x, u, g):
        return self.session.run(self.optimize,
                                feed_dict={self.x: x, self.i: u,
                                           self.g: g})

    def update_target(self):
        self.session.run(self.update_target)
