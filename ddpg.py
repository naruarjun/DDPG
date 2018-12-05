import numpy as np
import tensorflow as tf
from copy import deepcopy
from tensorflow import float32 as f

from actor import Actor
from noise import Noise
from critic import Critic
from memory import Memory


class DDPG:

    def __init__(self, sess, params):
        self.sess = sess
        self.__dict__.update(params)
        self.createInputs()
        self.actor = Actor(self.sess, self.actor_ips, **self.actor_params)
        self.critic = Critic(self.sess, self.inputs, **self.critic_params)

        # # create actor/critic models
        # self.actor = Actor(self.sess, self.inputs, **self.actor_params)
        # self.critic = Critic(self.sess, self.inputs, **self.critic_params)
        # self.noise_params = {k: np.array(list(map(float, v.split(","))))
        #                      for k, v in self.noise_params.items()}
        # self.noise = Noise(**self.noise_params)
        # self.ou_level = np.zeros(self.dimensions["u"])
        # self.memory = Memory(self.n_mem_objects,
        #                      self.memory_size)

    def createInputs(self):
        def PH(k, v): return tf.placeholder(f, shape=(None, v), name=k)
        input_specs = deepcopy(self.inputs)
        self.inputs = {}
        # Create input placeholders for single inputs
        for k, v in input_specs.items():
            if k == "multi_agent":
                continue
            self.inputs[k] = PH(k, v)
        
        # Create input placeholders for multiagent inputs
        for i in range(self.n_agents):
            for key, v in input_specs["multi_agent"].items():
                k = "{}{}".format(key, i)
                self.inputs[k] = PH(k, v)
        # Seperate inputs for actor
        self.actor_ips = {x: y for x, y in self.inputs.items()
                          if x in ["state0", "f_goal", "t_goal0"]}


    def step(self, x, explore=True):
        x = x.reshape(-1, self.dimensions["x"])
        if explore:
            u = self.actor.predict(x)
            self.ou_level = self.noise.ornstein_uhlenbeck_level(self.ou_level)
            u = u + self.ou_level
            q = self.critic.predict(x, u)
        else:
            u = self.actor.predict_target(x)
            q = self.critic.predict_target(x, u)
        return [self.scale_u(u[0]), u, q[0]]

    def remember(self, experience):
        self.memory.add(experience)

    def train(self):
        # check if the memory contains enough experiences
        if self.memory.size < 3*self.b_size:
            return
        x, g, ag, u, r, nx, ng, t = self.get_batch()
        # for her transitions
        her_idxs = np.where(np.random.random(self.b_size) < 0.80)[0]
        # print("{} of {} selected for HER transitions".
        # format(len(her_idxs), self.b_size))
        g[her_idxs] = ag[her_idxs]
        r[her_idxs] = 1
        t[her_idxs] = 1
        x = np.hstack([x, g])
        nx = np.hstack([nx, ng])
        nu = self.actor.predict_target(nx)
        tq = r + self.gamma*self.critic.predict_target(nx, nu)*(1-t)
        self.critic.train(x, u, tq)
        grad = self.critic.get_action_grads(x, u)
        # print("Grads:\n", g)
        self.actor.train(x, grad)
        self.update_targets()

    def get_batch(self):
        return self.memory.sample(self.b_size)

    def update_targets(self):
        self.critic.update_target()
        self.actor.update_target()
