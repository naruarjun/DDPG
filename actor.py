import tensorflow as tf


from tensorflow import float32 as f
from tensorflow import multiply as mul
from tensorflow.train import AdamOptimizer as Adam
from tensorflow.initializers import truncated_normal as TN

from FCNN import FCNN


class Actor:
    def __init__(self, sess, input_t, **params):
        self.session = sess
        self.inputs = input_t
        self.__dict__.update(params)
        self.generate_networks()
        self.define_operations()

    def generate_networks(self, load_from_ckpt=False):
        # Concatenate state and goals for policy input
        a_input = [self.inputs[k] for k in sorted(self.inputs.keys())]
        a_input = tf.concat(a_input, axis=1)
        # MAIN ACTOR NETWORK
        self.pi = FCNN(a_input, self.dim_action, self.n_layers,
                       self.n_units, tf.nn.relu, tf.nn.tanh, name="pi")
        # TARGET ACTOR NETWORK
        self.PI = FCNN(a_input, self.dim_action, self.n_layers,
                       self.n_units, tf.nn.relu, tf.nn.tanh, name="t_pi")
        self.grads = tf.placeholder(f, shape=(None, self.dim_action),
                                    name="grad")

    def define_operations(self):
        with tf.name_scope("actor_ops"):
            # GRADIENT OF ACTIONS WRT ACTOR PARAMS TIMES NEGATIVE GRADIENT OF
            # VALUE FUNCTION WRT ACTIONS
            grads = tf.gradients(self.pi.nn, self.pi.net_params, -self.grads)
            # APPLY GRADIENTS TO ACTOR NETWORK
            self.optimize = Adam(self.lr, name="pi_adam")\
                .apply_gradients(zip(grads, self.pi.net_params))
            # UPDATE TARGET OP
            net_param_pairs = zip(self.pi.net_params, self.PI.net_params)
            with tf.name_scope("update_target_pi"):
                self.updt_PI = [j.assign(mul(self.tau, i)+mul((1-self.tau), j))
                                for i, j in net_param_pairs]

    def predict(self, inputs):
        values = {v: inputs[k] for k, v in self.inputs.items()}
        return self.session.run(self.pi.nn, feed_dict=values)

    def predict_target(self, inputs):
        values = {v: inputs[k] for k, v in self.inputs.items()}
        return self.session.run(self.PI.nn, feed_dict=values)

    def train(self, inputs, grads):
        values = {v: inputs[k] for k, v in self.inputs.items()}
        values[self.grads] = grads
        return self.session.run(self.optimize, feed_dict=values)

    def update_target(self):
        self.session.run(self.updt_PI)
