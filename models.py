import tensorflow as tf
from tensorflow import reduce_mean as RM, square as SQ, stop_gradient as SG
from FCNN import FCNN
from tensorflow.train import AdamOptimizer


class Actor:
    def __init__(self, inputs, n_layers=2, n_units=64, **params):
        # _input = tf.concat([inputs["c_obs"], inputs["goal"]], axis=1)
        _input = inputs["c_obs"]
        self.pi = FCNN(_input, inputs["u"].shape[-1], n_layers,
                       n_units, tf.nn.tanh, tf.nn.tanh, name="actor")
        self.t_pi = FCNN(_input, inputs["u"].shape[-1], n_layers,
                         n_units, tf.nn.tanh, tf.nn.tanh, name="t_actor")
        self.pi_params = self.pi.network_params
        self.t_pi_params = self.t_pi.network_params
        self.params_pair = zip(self.t_pi_params, self.pi_params)
        with tf.name_scope("t_actor_update"):
            self.update_target_op = [i.assign(tf.multiply(i, 1-params["tau"]) +
                                     tf.multiply(j, params["tau"]))
                                     for i, j in self.params_pair]
        self.optimizer = AdamOptimizer(learning_rate=1e-4)

    def update_target(self, sess):
        return sess.run(self.update_target_op)

    def __call__(self, sess, x, use_target=False):
        if use_target:
            return sess.run(self.t_pi.nn, feed_dict=x)
        return sess.run(self.pi.nn, feed_dict=x)


class Critic:
    def __init__(self, inputs, actor, n_layers=2, n_units=64, **params):
        _input = tf.concat([inputs["c_obs"], actor.pi.nn], axis=1)
        self.Q = FCNN(_input, 1, n_layers, n_units, tf.nn.relu,
                      None, name="critic")

        _input = tf.concat([inputs["n_obs"], actor.t_pi.nn], axis=1)
        self.t_Q = FCNN(_input, 1, n_layers, n_units, tf.nn.relu,
                        None, name="t_critic")

        self.Q_params = self.Q.network_params
        self.t_Q_params = self.t_Q.network_params
        self.params_pair = zip(self.t_Q_params, self.Q_params)
        with tf.name_scope("t_critic_update"):
            self.update_target_op = [i.assign(tf.multiply(i,
                                     1 - params["tau"]) +
                                     tf.multiply(j, params["tau"]))
                                     for i, j in self.params_pair]
        with tf.name_scope("c_pred"):
            c_pred = inputs["r"] + params["gamma"]*self.t_Q.nn
        with tf.name_scope("critic_loss"):
            self.loss = RM(SQ(SG(c_pred) - self.Q.nn))
        with tf.name_scope("critic_opt"):
            self.optimize = AdamOptimizer(learning_rate=1e-4)\
                .minimize(self.loss, var_list=self.Q_params)
        self.configure_actor(actor, params)

    def configure_actor(self, actor, params):
        actor.loss = -RM(self.Q.nn)
        # with tf.name_scope("dQ_dPI"):
        #     actor.a_grads = tf.gradients(self.Q.nn,
        #                                  actor.pi.network_params,
        #                                  name="q_wrt_pi")
        # actor.a_grads = [tf.negative(i) for i in actor.a_grads]
        # with tf.name_scope("dQ_dU"):
        #     actor.d_awt = tf.gradients(actor.pi.nn,
        #                                actor.pi.network_params,
        #                                actor.a_grads,
        #                                name="pi_wrt_net_params")
        # with tf.name_scope("grad_scaling"):
        #     actor.d_awt = map(lambda x: tf.div(x, params["b_size"]),
        #                       actor.d_awt)
        # actor.grad_param_pair = zip(actor.a_grads, actor.pi.network_params)
        # with tf.name_scope("actor_opt"):
        actor.optimize = actor.optimizer\
                              .minimize(actor.loss, var_list=actor.pi_params)

    def train(self, sess, inputs):
        return sess.run(self.optimize, feed_dict=inputs)

    def update_target(self, sess):
        return sess.run(self.update_target_op)

    def a_grads(self, sess, inputs):
        return sess.run(self.a_grads, feed_dict={inputs})

    def __call__(self, x):
        return tf.get_default_session().run(self.critic_op, feed_dict=x)
