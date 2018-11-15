import gym
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.summary import FileWriter

from ddpg import DDPG
import go2goal
from rollout import RolloutGenerator
from train2 import load_config, scale_action_gen, create_parser


def evaluate(config, from_ckpt):
    tf_session = tf.Session()
    env = gym.make("Go2Goal-v0")
    scale_action = scale_action_gen(env, np.ones(2)*-1, np.ones(2))
    ddpg_agent = DDPG(tf_session, scale_action, config)
    new_saver = tf.train.Saver()
    new_saver.restore(tf_session, from_ckpt)
    eval_rollouts = RolloutGenerator(env, ddpg_agent, config["eval_rollout"],
                                     _eval=True)
    eval_rollouts.reset()
    while not eval_rollouts.done():
        eval_rollouts.generate_rollout()


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    # print(config)
    evaluate(config, args.from_ckpt)


if __name__ == "__main__":
    main()
