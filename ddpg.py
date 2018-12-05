import numpy as np
import tensorflow as tf
from copy import deepcopy
from tensorflow import float32 as f

from actor import Actor
from noise import Noise
from critic import Critic
from memory import Memory
from util import info, log, error


class DDPG:

    def __init__(self, sess, params):
        self.sess = sess
        self.__dict__.update(params)
        self.createInputs()
        self.actor = Actor(self.sess, self.actor_ips, **self.actor_params)
        self.critic = Critic(self.sess, self.inputs, **self.critic_params)
        self.noise_params = {k: np.array(list(map(float, v.split(","))))
                             for k, v in self.noise_params.items()}
        self.noise = Noise(**self.noise_params)
        self.ou_level = np.zeros((self.n_agents,
                                  self.actor_params["dim_action"]))
        self.OU = self.noise.ornstein_uhlenbeck_level
        self.memory = Memory(self.n_mem_objects, self.memory_size)

    def createInputs(self):
        def PH(k, v): return tf.placeholder(f, shape=(None, v), name=k)
        self.input_specs = deepcopy(self.inputs)
        self.inputs = {}
        # Create input placeholders for single inputs
        for k, v in self.input_specs.items():
            if k == "multi":
                continue
            self.inputs[k] = PH(k, v)

        # Create input placeholders for multiagent inputs
        for i in range(self.n_agents):
            for key, v in self.input_specs["multi"].items():
                k = "{}{}".format(key, i)
                self.inputs[k] = PH(k, v)
        # Seperate inputs for actor
        self.actor_ips = {x: y for x, y in self.inputs.items()
                          if x in ["state0", "f_goal", "t_goal0"]}

    def step(self, states, goals, explore=True):
        states = np.matrix(states)
        t_goal = np.matrix(goals[:, :2])
        f_goal = np.matrix(goals[:, 2:])
        x = {"state0": states, "t_goal0": t_goal, "f_goal": f_goal}
        if explore:
            action = self.actor.predict(x)
            self.ou_level = np.array([self.OU(self.ou_level[i])
                                      for i in range(self.n_agents)])
            action += self.ou_level
        else:
            action = self.actor.predict_target(x)
        if "scale_action" in self.__dict__.keys():
            return self.scale_action(action)
        return action

    def remember(self, experience):
        self.memory.add(experience)

    def train(self):
        # check if the memory contains enough experiences
        if self.memory.size < 3*self.b_size:
            error.out("Not enough {} : {}".format(self.memory.size, 3*self.b_size))
            return
        x, g, ag, u, r, nx, ng, t = self.get_batch()
        # for her transitions
        her_idxs = np.where(np.random.random(self.b_size) < 0.80)[0]
        g[her_idxs] = ag[her_idxs]
        r[her_idxs] = 1.0
        t[her_idxs] = 1
        ip_a = self.compile_inputs_for_actor(nx, ng)
        nu = self.actor.predict_target(ip_a).reshape((self.b_size, -1, 2))
        ip_c = self.compile_inputs_for_critic(nx, ng, nu, self.b_size)
        tq = r + self.gamma*np.multiply(self.critic.predict_target(ip_c), (1-t))
        log.out(self.critic.predict_target(ip_c))
        ip_a = self.compile_inputs_for_actor(x, g)
        ip_c = self.compile_inputs_for_critic(x, g, u, self.b_size)
        self.critic.train(ip_c, tq)
        grads = self.critic.get_action_grads(ip_c)
        grads = np.mean(np.swapaxes(np.array(grads), 0, 1), axis=1)
        error.out(np.array(grads))
        print("-"*100)
        error.out(np.swapaxes(np.array(grads), 0, 1))
        print("-"*100)
        error.out(np.mean(np.swapaxes(np.array(grads), 0, 1), axis=1))
        print("-"*100)
        error.out(np.array(grads).reshape((-1, 2)))

        self.actor.train(ip_a, grads)
        self.update_targets()

        # ip_c = self.compile_inputs_for_critic(nx, u)


        # ip_c = {"state{}".format(i): j for i, j in
        #         enumerate(np.swapaxes(nx, 0, 1))}
        # ip_c.update({"t_goal{}".format(i): j for i, j in
        #              enumerate(np.swapaxes(g[:, :, :2], 0, 1))})
        # ip_c.["f_goal"] = g[:, 0, :].reshape(-1, 3)
        # ip_a = {"state0": nx.reshape(-1, nx.shape[-1]),
        #         "t_goal0": ng.reshape(-1, ng.shape[-1])[:2],
        #         "f_goal": ng.reshape(-1, ng.shape[-1])[2:]}

        # ip_c.update({"state{}".format(i): j for i, j in
        #              enumerate(np.swapaxes(nx, 0, 1))})
        # nu = self.actor.predict_target(ip_a).reshape(u.shape)
        # ip_c.update({"action{}".format(i): j for i, j in
        #              enumerate(np.swapaxes(nu, 0, 1))})
        # tq = r + self.gamma*self.critic.predict_target(ip_c)*(1-t)
        # self.critic.train(ip_c, tq)
        # grad = self.critic.get_action_grads(x, u)
        # # print("Grads:\n", g)
        # self.actor.train(x, grad)
        # self.update_targets()
        pass

    def compile_inputs_for_critic(self, x, g, u, b_size):
        info.out(u)
        feed = {"f_goal": g[:, 0, 2:]}
        feed.update({"state{}".format(i): j
                     for i, j in enumerate(np.swapaxes(x, 0, 1))})
        feed.update({"action{}".format(i): j
                     for i, j in enumerate(np.swapaxes(u, 0, 1))})
        feed.update({"t_goal{}".format(i): j[:, :2]
                     for i, j in enumerate(np.swapaxes(g, 0, 1))})
        info.out(feed)
        return feed

    def compile_inputs_for_actor(self, x, g):
        feed = {}
        feed["state0"] = x.reshape((-1, 8))
        feed["f_goal"] = g[:,:,2:].reshape((-1, 3))
        feed["t_goal0"] = g[:,:,:2].reshape((-1, 2))
        return feed

    def get_batch(self):
        return self.memory.sample(self.b_size)

    def update_targets(self):
        self.critic.update_target()
        self.actor.update_target()
