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
    if from_ckpt is not None:
        new_saver = tf.train.import_meta_graph("{}.meta".format(from_ckpt))
        new_saver.restore(tf_session, from_ckpt)
        ip_tensor_names = ["r", "x", "u"]
        config["inputs"] = {i: tf_session.graph.get_tensor_by_name(
                            "inputs/{}:0".format(i)) for i in ip_tensor_names}
        config["inputs"]["g"] = tf_session.graph.get_tensor_by_name("inputs/a_grad:0")
        config["inputs"]["p"] = tf_session.graph.get_tensor_by_name("inputs/pred_q:0")
        config["actor_params"]["load_from_ckpt"] = True
        config["critic_params"]["load_from_ckpt"] = True
    else:
        tf_session.run(tf.global_variables_initializer())

    env = gym.make("Go2Goal-v0")
    scale_action = scale_action_gen(env, np.ones(2)*-1, np.ones(2))
    ddpg_agent = DDPG(tf_session, scale_action, config)
    train_rollouts = RolloutGenerator(env, ddpg_agent, config["train_rollout"])
    eval_rollouts = RolloutGenerator(env, ddpg_agent, config["eval_rollout"], _eval=True)

    while not train_rollouts.done():
        train_rollouts.generate_rollout()
        if (train_rollouts.episode) % config["eval_after"] == 0:
            eval_rollouts.reset()
            while not eval_rollouts.done():
                eval_rollouts.generate_rollout()
    # saver = tf.train.Saver()
    # summarizer = FileWriter("__tensorboard/her", tf_session.graph)
    # s_summary = tf.Summary()
    # log_str = "| [{}] Episode: {:4} | Reward: {:7.3f} | Q: {:8.3f} | T: {:3d} | MIND: {:4.3f} |"

    # summary_op = tf.summary.merge_all()

    # # for testing purposes!!!
    # current_best_eval_score = 0
    # for episode in range(config["n_episodes"]):
    #     episodic_r = 0.
    #     episodic_q = 0.
    #     obs = env.reset()
    #     episode_batch = []
    #     min_d2goal = env.distance_from_goal()
    #     for i in range(env._max_episode_steps):
    #         if "render" in config.keys() and config["render"]:
    #             env.render()
    #         action, u, q = ddpg_agent.step(np.hstack([obs["observation"],
    #                                        obs["desired_goal"]]),
    #                                        is_u_discrete)
    #         episodic_q += q
    #         # action = scale_action(action)
    #         new_obs, r, done, info = env.step(action)
    #         ogag = [obs[k] for k in ["observation", "desired_goal", "achieved_goal"]]
    #         episode_batch.append([*ogag, u, r, new_obs["observation"],
    #                               new_obs["desired_goal"], int(done)])
    #         if (info["dist"] < min_d2goal).all():
    #             min_d2goal = info["dist"]
    #         obs = new_obs
    #         episodic_r += r
    #         for epoch in range(5):
    #             ddpg_agent.train()
    #         if done:
    #             break
    #         s_summary.value.add(tag="run/l_velocity", simple_value=(action)[0])
    #         s_summary.value.add(tag="run/a_velocity", simple_value=(action)[1])
    #         s_summary.value.add(tag="run/meanQ",
    #                             simple_value=float(episodic_q/(i+1)))
    #         summarizer.add_summary(s_summary, episode*env._max_episode_steps+i)
    #     # n_batch = reward_normalizer.discount(episode_batch)
    #     for experience in episode_batch:
    #         ddpg_agent.remember(experience)
    #     print(log_str.format("T", episode+1, episodic_r,
    #                          float(episodic_q), i+1, np.linalg.norm(min_d2goal)))
    #     summarizer.add_summary(tf_session.run(summary_op), episode)
    #     summarizer.flush()
    #     # To run or not to run evaluations on current target policy...
    #     if (episode+1) % 20 != 0:
    #         continue
    #     m_eval_score = 0.
    #     m_eval_q = 0.
    #     print()
    #     for eval_run in range(5):
    #         eval_score = 0.
    #         eval_q = 0.
    #         obs = env.reset()
    #         for j in range(env._max_episode_steps):
    #             u, _, q = ddpg_agent.step(np.hstack([obs["observation"], obs["desired_goal"]]),
    #                                       is_u_discrete, explore=False)
    #             obs, r, done, _ = env.step(u)
    #             eval_score += r
    #             eval_q += q
    #             if done:
    #                 break
    #         m_eval_q += eval_q
    #         m_eval_score += eval_score
    #         print(log_str.format("E", eval_run+1, m_eval_score,
    #                              float(m_eval_q), j+1, -1))
    #     print()
    #     # save the model checkpoints if they are the current best...
    #     if m_eval_score > current_best_eval_score:
    #         print("New best policy found with eval score of: ", m_eval_score)
    #         print("old best policy's eval score: ", current_best_eval_score)
    #         current_best_eval_score = m_eval_score
    #         saver.save(tf_session, "__checkpoints/nb_policy", episode)


def exit():
    import sys
    sys.exit(0)


def main():
    config = load_config("config_g2g.yaml")
    print()
    print(train(config))


if __name__ == "__main__":
    main()
