import gym
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.summary import FileWriter

from ddpg2 import DDPG
from reward import Reward


def load_config(filename):
    with open(filename) as f:
        config = yaml.load(f.read())
    return config


def train(config):
    reward_normalizer = Reward(0.1, config["gamma"])
    env = gym.make(config["env_name"])
    env.seed(1234)
    is_u_discrete = len(env.action_space.shape) == 0
    tf_session = tf.Session()
    ddpg_agent = DDPG(tf_session, config)
    summarizer = FileWriter("__tensorboard/ac98", tf_session.graph)
    log_str = "| [{}] Episode: {:4} | Reward: {:3d} | Q: {:8.3f} |"
    summary_op = tf.summary.merge_all()

    for episode in range(config["n_episodes"]):
        episodic_r = 0.
        episodic_q = 0.
        obs = env.reset()
        episode_batch = []
        for i in range(env._max_episode_steps):
            # print(obs)
            action, u, q = ddpg_agent.step(obs, is_u_discrete, episode < 100)
            episodic_q += q
            new_obs, r, done, _ = env.step(action)
            # print("REWARD: ", r, type(r), type(episodic_r))
            episode_batch.append([obs, u, r, new_obs, int(done)])
            obs = new_obs
            if "render" in config.keys() and config["render"]:
                env.render()
            episodic_r += r
            ddpg_agent.train()
            if done:
                n_batch = reward_normalizer.discount(episode_batch)
                for experience in n_batch:
                    ddpg_agent.remember(experience)
                break
        print(log_str.format("T", episode+1, int(episodic_r),
                             float(episodic_q)/(i+1)))
        summarizer.add_summary(tf_session.run(summary_op), episode)
        # eval_score = 0.
        # eval_q = 0.
        # if (episode+1) % 10 != 0:
        #     continue
        # for eval_run in range(10):
        #     obs = env.reset()
        #     for j in range(env._max_episode_steps):
        #         u, _, q = ddpg_agent.step(obs, is_u_discrete, explore=False)
        #         obs, r, done, _ = env.step(u)
        #         eval_score += r
        #         eval_q += q
        #         if done:
        #             break
        # print(log_str.format("E", int((episode+1)/10), int(eval_score/10),
        #                      float(eval_q)/(10)))


def main():
    config = load_config("config.yaml")
    print()
    print(train(config))


if __name__ == "__main__":
    main()
