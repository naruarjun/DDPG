import tensorflow as tf
import tf.float32 as f32
import yaml
from baselines.ddpg.models import Actor, Critic

class DDPG:
  def __init__(self, config):
    self.params = yaml.loads(config)
    self.actor  = Actor(**params["actor"])
    self.critic = Critic(**params["critic"])
    self.buffer = ReplayBuffer(**params["buffer"])
    self.create_tf_vars()

  def create_tf_vars(self)
    self.obs0   = tf.placeholder(f32, shape=(None, self.params["d_obs"]), name="obs0")
    self.obs1   = tf.placeholder(f32, shape=(None, self.params["d_obs"]), name="obs1")
    self.reward = tf.placeholder(f32, shape=(None, self.params["d_ret"]), name="rets")
    self.action = tf.placeholder(f32, shape=(None, self.params["d_act"]), name="actn")
    self.goal   = tf.placeholder(f32, shape=(None, self.params["d_gol"]), name="goal")
    self.a_goal = tf.placeholder(f32, shape=(None, self.params["d_gol"]), name="a_gl")