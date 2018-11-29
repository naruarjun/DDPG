import os
import gym
import yaml
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.summary import FileWriter

from ddpg import DDPG
import go2goal
from rollout import RolloutGenerator
from util import info

EXAMPLE_USAGE = """
Train agent using DDPG + HER
Example Usage:
* Train agent using a specific configuration
(See config for more details)

python train.py --config some_config.yaml

* Resume training from a specific checkpoint
(Only specify the name. No extensions needed)

python train.py --config some_config.yaml --from-ckpt __checkpoints/some-ckpt
"""


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


def create_parser(parser_creator=None):
    parser = parser = argparse.ArgumentParser(
             formatter_class=argparse.RawDescriptionHelpFormatter,
             description="Train a reinforcement learning agent.",
             epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="Config file.")
    parser.add_argument(
        "--from-ckpt",
        default=None,
        type=str,
        help="Checkpoint file if training is to"+\
             " be resumed from checkpoint")
    return parser


def get_summarizer(session, expr_name=None):
    if expr_name is None:
        expr_name = datetime.now().strftime("%d%h%y_%H%M")
    log_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(log_dir, "__tensorboard", expr_name)
    tf_summary_writer = FileWriter(summary_path, session.graph)
    return tf_summary_writer


def train(config, from_ckpt=None):
    tf_session = tf.Session()
    env = gym.make("Go2Goal-v0")
    scale_action = scale_action_gen(env, np.ones(2)*-1, np.ones(2))
    ddpg_agent = DDPG(tf_session, scale_action, config)

    summary_writer = get_summarizer(tf_session)

    if from_ckpt not in [None, "None"]:
        info.out("USING PRETRAINED MODEL: {}".format(from_ckpt))
        new_saver = tf.train.Saver()
        new_saver.restore(tf_session, from_ckpt)
    else:
        tf_session.run(tf.global_variables_initializer())
    train_rollouts = RolloutGenerator(env, ddpg_agent, config["train_rollout"],
                                      summary_writer=summary_writer)
    eval_rollouts = RolloutGenerator(env, ddpg_agent, config["eval_rollout"],
                                     _eval=True, summary_writer=summary_writer)

    while not train_rollouts.done():
        train_rollouts.generate_rollout()
        # train_rollouts.write_summary(summary_writer)
        if (train_rollouts.episode) % config["eval_after"] == 0:
            eval_rollouts.reset()
            while not eval_rollouts.done():
                eval_rollouts.generate_rollout()
                eps = train_rollouts.episode//config["eval_after"]
                # eval_rollouts.write_summary(summary_writer, eps)

def exit():
    import sys
    sys.exit(0)


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    print(train(config, args.from_ckpt))


if __name__ == "__main__":
    main()
