import gym
import yaml
import numpy as np
import tensorflow as tf

from tensorflow.summary import FileWriter
from tensorflow import GraphKeys

from actor import Actor
from critic import Critic
from ReplayBuffer import ReplayBuffer


class DDPG:

    def __init__(self, sess, params):
        self.sess = sess
        self.params = params
        # create placeholders
        self.create_input_placeholders()
        # create actor/critic models
        self.actor = Actor(self.sess, self.inputs, **params["actor"])
        self.critic = Critic(self.sess, self.inputs, **params["critic"])
        self.sess.run(tf.global_variables_initializer())
        self.summarizer = FileWriter("__tensorboard/ac3", sess.graph)

    def create_input_placeholders(self):
        self.inputs = {}
        with tf.name_scope("inputs"):
            for ip_name, dim in self.params["dimensions"].items():
                self.inputs[ip_name] = tf.placeholder(tf.float32,
                                                      shape=(None, dim),
                                                      name=ip_name)
            self.inputs["g"] = tf.placeholder(tf.float32,
                                              shape=self.inputs["u"].shape,
                                              name="a_grad")
            self.inputs["p"] = tf.placeholder(tf.float32,
                                              shape=self.inputs["x"].shape,
                                              name="pred_q")

    def train(self):
        pass

    def get_batch(self):
        pass

    def update_targets(self):
        pass


def load_config(filename):
    with open(filename) as f:
        config = yaml.load(f.read())
    return config


dd = DDPG(tf.Session(), load_config("config.yaml"))
