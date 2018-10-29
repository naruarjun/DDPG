import yaml
from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from tf.GraphKeys import TRAINABLE_VARIABLES as TVARS
from tensorflow import float32 as f32

from models import Actor, Critic


class DDPG:

    def __init__(self, params):
        self.params = params
        # Create shape dict for inputs
        self.create_shape_dict()
        # create Staging area
        self.create_stageing_area()
        # create actor/critic models
        self.actor = Actor(self.inputs, **params["actor"], name="ACTOR")
        self.critic = Critic(self.inputs, self.actor,
                             **params["critic"], name="CRITIC")
        # create their corresponding target models
        self.t_actor = Actor(self.inputs, **params["actor"], target=True,
                             name="T_ACTOR")
        self.t_critic = Critic(self.inputs, self.t_actor, **params["critic"],
                               target=True, name="T_CRITIC")
        # self.create_losses()
        # Training rules
        # for Critic
        # for Actor
        # self.setup_trainers()
        # create target update rules
        self.create_target_update_op()

    def create_shape_dict(self):
        self.shape_dict = OrderedDict()
        for key, shape in sorted(params["dimensions"].items()):
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

    def create_losses(self):
        c_pred = self.inputs["r"] + \
                        self.params["gamma"]*self.t_critic(self.inputs)
        self.critic_loss = tf.reduce_mean(tf.square(tf.stop_gradient(c_pred)
                                          - self.critic(self.inputs)))

    def create_target_update_op(self):
        actor_wt_pairs = zip(tf.get_collection(TVARS, scope="T_ACTOR"),
                             tf.get_collection(TVARS,  scope="ACTOR"))
        critic_wt_pairs = zip(tf.get_collection(TVARS, scope="T_CRITIC"),
                              tf.get_collection(TVARS,  scope="CRITIC"))
        self.act_update_op = [i.assign(tf.mul(i, self.params["tau"]) +
                              tf.mul(j, 1 - self.params["tau"]))
                              for i, j in actor_wt_pairs]
        self.cri_update_op = [i.assign(tf.mul(i, self.params["tau"]) +
                              tf.mul(j, 1 - self.params["tau"]))
                              for i, j in critic_wt_pairs]


def load_config(filename):
    with open(filename) as f:
        config = yaml.load(f.read())
    return config


params = load_config("config.yaml")
print(params)
ddpg = DDPG(params)
a, c = ddpg.actor, ddpg.critic
# ovar = create_input_dict(params)
# sa = create_stageing_area(ovar)

# b = sa.get()
# inputs = {k: b[i] for i, k in enumerate(ovar.keys())}


# s = DDPG(inputs, params)
# a = s.actor
# c = s.critic


writer = tf.summary.FileWriter("__tensorboard/", tf.get_default_graph())
# print (op)
writer.close()
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="T_ACTOR"))
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ACTOR"))
tav = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="T_ACTOR")
print([i.name for i in tav])
with tf.variable_scope("ACTOR", reuse=True):
    print(tf.get_variable("layer_0/kernel"))
print(tf.trainable_variables())
