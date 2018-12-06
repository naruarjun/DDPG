import numpy as np
import tensorflow as tf
from copy import deepcopy
from tensorflow import float32 as f
from tensorflow.summary import FileWriter

from actor import Actor
from noise import Noise
from critic import Critic
from memory import Memory
from util import info, log, error, warn


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
        self.summ_writer = FileWriter("__tensorboard/f", self.sess.graph)
        self.merged_summary = tf.summary.merge_all()
        self.t_c = 0

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
            return self.scale_action(action), action
        return action, action

    def remember(self, experience):
        self.memory.add(experience)

    def train(self):
        # check if the memory contains enough experiences
        if self.memory.size < 3*self.b_size:
            # error.out("Not enough {} : {}".format(self.memory.size, 
            # 3*self.b_size))
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
        error.out("TARGET Q: \n{}".format(tq))
        ip_a = self.compile_inputs_for_actor(x, g)
        ip_c = self.compile_inputs_for_critic(x, g, u, self.b_size)
        self.critic.train(ip_c, tq)
        grads = self.critic.get_action_grads(ip_c)
        grads = np.mean(np.swapaxes(np.array(grads), 0, 1), axis=1)
        warn.out("grads------------------:\n{}\n\n".format(grads))
        if np.isnan(grads).any():
            log.out("REWARDS: \n{}\n".format(r))
            log.out("TERMINAL: \n{}\n".format(t))
            log.out("NEXT_U: \n{}\n".format(nu))
            log.out("NEXT_Q: \n{}\n".format(np.multiply(self.critic.predict_target(ip_c), (1-t))))
            warn.out("NEXT_X: \n{0}\n".format(nx))
            warn.out("NEXT_G: \n{0}\n".format(ng))
            # warn.out(nx)
            raise ValueError("NAN encountered in grads")
        log.out("Befor actor.train()"+"*"*80)
        log.out([np.isnan(i).any() for i in self.sess.run(self.actor.pi.net_params)])
        log.out([np.isnan(i).any() for i in self.sess.run(self.actor.PI.net_params)])
        log.out("*"*80)
        self.actor.train(ip_a, grads)
        log.out("After actor.train()"+"*"*80)
        log.out([np.isnan(i).any() for i in self.sess.run(self.actor.pi.net_params)])
        log.out([np.isnan(i).any() for i in self.sess.run(self.actor.PI.net_params)])
        log.out("*"*80)
        self.update_targets()
        log.out("After actor.update()"+"*"*80)
        log.out([np.isnan(i).any() for i in self.sess.run(self.actor.pi.net_params)])
        log.out([np.isnan(i).any() for i in self.sess.run(self.actor.PI.net_params)])
        log.out("*"*80)
        self.summ_writer.add_summary(self.sess.run(self.merged_summary), self.t_c)
        self.t_c += 1
        log.out("TRAIN SUCCESSFUL")

    def compile_inputs_for_critic(self, x, g, u, b_size):
        feed = {"f_goal": g[:, 0, 2:]}
        feed.update({"state{}".format(i): j
                     for i, j in enumerate(np.swapaxes(x, 0, 1))})
        feed.update({"action{}".format(i): j
                     for i, j in enumerate(np.swapaxes(u, 0, 1))})
        feed.update({"t_goal{}".format(i): j[:, :2]
                     for i, j in enumerate(np.swapaxes(g, 0, 1))})
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
