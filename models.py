import tensorflow as tf
from FCNN import FCNN


class Model:
    def __init__(self, inputs):
        self.inputs = inputs

    def get_tf_var(self, name):
        pass


class Actor(Model):
    def __init__(self, inputs, n_layers=2, n_units=64, name="Actor"):
        super(Actor, self).__init__(inputs)
        self._input = tf.concat([inputs["c_obs"], inputs["goal"]], axis=1)
        self.pi = FCNN(self._input, inputs["u"].shape[-1], n_layers,
                       n_units, tf.nn.relu, tf.nn.tanh, name=name)
        self.actor_op = self.pi.network_op

    def __call__(self, x):
        return tf.get_default_session().run(self.actor_op, feed_dict=x)


class Critic(Model):
    def __init__(self, inputs, actor, n_layers=2, n_units=64, name="Critic"):
        super(Critic, self).__init__(inputs)
        self._input = tf.concat([inputs["c_obs"], inputs["goal"],
                                actor.actor_op], axis=1)
        self.Q = FCNN(self._input, 1, n_layers, n_units, tf.nn.relu,
                      None, name=name)
        self.critic_op = self.Q.network_op

    def __call__(self, x):
        return tf.get_default_session().run(self.critic_op, feed_dict=x)
