import yaml
import sys
from collections import OrderedDict

import tensorflow as tf
from tensorflow.summary import FileWriter
from tensorflow.contrib.staging import StagingArea
from tensorflow import GraphKeys
from tensorflow import float32 as f32
from models import Actor, Critic

import gym
import numpy as np
from ReplayBuffer import ReplayBuffer
TVARS = GraphKeys.TRAINABLE_VARIABLES


class DDPG:

    def __init__(self, sess, params):
        self.sess = sess
        self.params = params
        # Create shape dict for inputs
        self.create_shape_dict()
        # create Staging area
        self.create_stageing_area()
        # create actor/critic models
        self.actor = Actor(self.inputs, **params["actor"])
        self.critic = Critic(self.inputs, self.actor, **params["critic"])
        self.sess.run(tf.global_variables_initializer())
        self.summarizer = FileWriter("__tensorboard/cp3", sess.graph)
        self.eps = params["random_eps"]
        self.buffer = ReplayBuffer(500000)

    def create_shape_dict(self):
        self.shape_dict = OrderedDict()
        for key, shape in sorted(self.params["dimensions"].items()):
            self.shape_dict[key] = (None, shape)
        print("SHAPE DICT\n", self.shape_dict)

    def create_stageing_area(self):
        self.staging_area = StagingArea(dtypes=[f32 for _ in self.shape_dict],
                                        shapes=self.shape_dict.values())
        self.staging_area.put([tf.placeholder(f32, shape=shape, name=name)
                              for name, shape in self.shape_dict.items()])
        self.inputs = self.staging_area.get()
        self.inputs = {k: self.inputs[i] for i, k
                       in enumerate(self.shape_dict.keys())}

    def get_action(self, obs, explore=True):
        if explore:
            action = self.actor(self.sess, {self.inputs["c_obs"]: obs})
            print(action)
            if np.random.random() < self.eps:
                return np.random.randint(2), True
        else:
            action = self.actor(self.sess, {self.inputs["c_obs"]: obs}, True)
        print(action)
        return action, False

    def train(self, inputs):
        self.critic.train(self.sess, inputs)
        self.sess.run(self.actor.optimize, feed_dict=inputs)

    def update_target(self):
        self.actor.update_target(self.sess)
        self.critic.update_target(self.sess)


def load_config(filename):
    with open(filename) as f:
        config = yaml.load(f.read())
    return config


def build_summaries():
    episode_reward = tf.Variable(0., name="reward")
    eval_reward = tf.Variable(0., name="eval_reward")
    tf.summary.scalar("Reward", episode_reward)
    tf.summary.scalar("Eval Reward", eval_reward)
    # episode_ave_max_q = tf.Variable(0.)
    # tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, eval_reward]

    return summary_vars


summary_vars = build_summaries()
dd = DDPG(tf.Session(), load_config("config.yaml"))
summary_ops = tf.summary.merge_all()


s_dim = dd.params["dimensions"]["c_obs"]
env = gym.make("CartPole-v1")
input()
for i in range(100000):
    ep_reward = 0
    eval_reward = 0
    obs = env.reset()
    # for j in range():
    for j in range(env._max_episode_steps):
        env.render()
        action = dd.get_action(obs.reshape(-1, s_dim))[0]
        # print(action, "----<<")
        n_obs, r, done, _ = env.step(int(action > 0))
        ep_reward += r
        dd.buffer.add(obs, action, r, done, n_obs)
        obs = n_obs
        if done:
            fd = {i.name.split(":")[0]: i for i in tf.get_collection(TVARS)}
            fd = {summary_vars[0]: ep_reward}
            # summary_str = dd.sess.run(summary_ops, feed_dict=fd)

            # dd.summarizer.add_summary(summary_str, i)
            # dd.summarizer.flush()
            break
    if dd.eps > 0.3:
        dd.eps = dd.eps*0.99
    for k in range(10):
        batch = dd.buffer.sample_batch(200)
        batch = [i.reshape((-1, j)) for i, j in zip(batch, [4, 1, 1, 1, 4])]
        d_ip = {dd.inputs["c_obs"]: batch[0],
                dd.inputs["u"]: batch[1],
                dd.inputs["r"]: batch[2],
                dd.inputs["n_obs"]: batch[4]}
        # for i, v in d_ip.items():
        #     print(i)
        #     print(v)

        dd.train(d_ip)
        # sys.exit(0)
    dd.update_target()
    eval_reward = 0
    for eval_run in range(10):
        obs = env.reset()
        for fbf in range(500):
            action = dd.get_action(obs.reshape(-1, s_dim), False)[0]
            n_obs, r, done, _ = env.step(int(action>0))
            eval_reward += r
            obs = n_obs
            if done:
                break
    # print(eval_reward/10.)
    fd.update({summary_vars[1]: eval_reward/10.})
    summary_str = dd.sess.run(summary_ops, feed_dict=fd)

    dd.summarizer.add_summary(summary_str, i)
    dd.summarizer.flush()
    # fd = {summary_vars[1]: eval_reward}
    # summary_str = dd.sess.run(summary_ops, feed_dict=fd)
    # dd.summarizer.add_summary(summary_str, fbf)
    # dd.summarizer.flush()

    # print([i.reshape((-1, j)) for i, j in zip(batch, [6, 1, 1, 1, 6])])

# params = load_config("config.yaml")
# print(params)
# ddpg = DDPG(params)
# a, c = ddpg.actor, ddpg.critic
# ovar = create_input_dict(params)
# sa = create_stageing_area(ovar)

# b = sa.get()
# inputs = {k: b[i] for i, k in enumerate(ovar.keys())}


# s = DDPG(inputs, params)
# a = s.actor
# c = s.critic


# writer = tf.summary.FileWriter("__tensorboard/", tf.get_default_graph())
# # print (op)
# writer.close()
# print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="T_ACTOR"))
# print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ACTOR"))
# tav = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="T_ACTOR")
# print([i.name for i in tav])
# with tf.variable_scope("ACTOR", reuse=True):
#     print(tf.get_variable("layer_0/kernel"))
# print(tf.trainable_variables())
