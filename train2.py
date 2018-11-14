import gym
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.summary import FileWriter

from ddpg import DDPG
import go2goal
from rollout import RolloutGenerator


def load_config(filename):
    with open(filename) as f:
        config = yaml.load(f.read())
    return config


def scale_action_gen(env, u_min, u_max):
    def scale_action(u):
        u = np.clip(u, u_min, u_max)
        # print("clipped ", u)
        zo = (u - u_min)/(u_max - u_min)
        return zo * (env.action_high - env.action_low) + env.action_low
    return scale_action


def train(config):
    from_ckpt = "__checkpoints/P_2900"
    tf_session = tf.Session()
    env = gym.make("Go2Goal-v0")
    scale_action = scale_action_gen(env, np.ones(2)*-1, np.ones(2))
    ddpg_agent = DDPG(tf_session, scale_action, config)
    if from_ckpt is not None:
        new_saver = tf.train.Saver()
        new_saver.restore(tf_session, from_ckpt)
    else:
        tf_session.run(tf.global_variables_initializer())
    train_rollouts = RolloutGenerator(env, ddpg_agent, config["train_rollout"])
    eval_rollouts = RolloutGenerator(env, ddpg_agent, config["eval_rollout"],
                                     eval=True)

    while not train_rollouts.done():
        train_rollouts.generate_rollout()
        if (train_rollouts.episode) % config["eval_after"] == 0:
            eval_rollouts.reset()
            while not eval_rollouts.done():
                eval_rollouts.generate_rollout()


def exit():
    import sys
    sys.exit(0)


def main():
    config = load_config("config_g2g.yaml")
    print()
    print(train(config))


if __name__ == "__main__":
    main()
