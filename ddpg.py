import numpy as np
import tensorflow as tf


from actor import Actor
from noise import Noise
from critic import Critic
from memory import Memory


class DDPG:

    def __init__(self, sess, params):
        self.sess = sess
        self.__dict__.update(params)
        # create placeholders
        self.create_input_placeholders()
        # create actor/critic models
        self.actor = Actor(self.sess, self.inputs, **self.actor_params)
        self.critic = Critic(self.sess, self.inputs, **self.critic_params)
        self.sess.run(tf.global_variables_initializer())
        self.noise = Noise(**self.noise_params)
        self.ou_level = 0.
        self.memory = Memory(self.n_mem_objects,
                             self.memory_size)

    def create_input_placeholders(self):
        self.inputs = {}
        with tf.name_scope("inputs"):
            for ip_name, dim in self.dimensions.items():
                self.inputs[ip_name] = tf.placeholder(tf.float32,
                                                      shape=(None, dim),
                                                      name=ip_name)
            self.inputs["g"] = tf.placeholder(tf.float32,
                                              shape=self.inputs["u"].shape,
                                              name="a_grad")
            self.inputs["p"] = tf.placeholder(tf.float32,
                                              shape=(None, 1),
                                              name="pred_q")

    def step(self, x, is_u_discrete, explore=True):
        x = x.reshape(-1, self.dimensions["x"])
        u = self.actor.predict(x)
        if explore and np.random.random() < self.random_eps:
            self.ou_level = self.noise.ornstein_uhlenbeck_level(self.ou_level)
            u = u + self.ou_level
        q = self.critic.predict(x, u)
        if is_u_discrete:
            return [np.argmax(u), u[0], q[0]]
        return [u[0], u, q[0]]

    def remember(self, experience):
        self.memory.add(experience)

    def train(self):
        # check if the memory contains enough experiences
        if self.memory.size < self.b_size:
            return
        x, u, r, nx, t = self.get_batch()
        nu = self.actor.predict_target(nx)
        tq = r + self.gamma*self.critic.predict_target(nx, nu)*(1-t)
        self.critic.train(x, u, tq)
        g = self.critic.get_action_grads(x, u)
        self.actor.train(x, g)
        self.update_targets()

    def get_batch(self):
        return self.memory.sample(3)
        return self.memory.sample(self.b_size)

    def update_targets(self):
        self.critic.update_target()
        self.actor.update_target()