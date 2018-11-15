import numpy as np
import tensorflow as tf
from util import info


class RolloutGenerator:
    """
    Class for generating a rollout of trajectory by the agent
    args:
    env: gym env
    agent: agent for performing rollout
    config: rollout configuration
    checkpoint(opt): perform rollout from a saved policy
    """

    def __init__(self, env, agent, config: dict, _eval=False, summarizer=None):
        self.name = "{}_ROLLOUT".format("*EVAL" if _eval else "TRAIN")
        self.env = env
        self.agent = agent
        self.eval = _eval
        self.best_score = 0.
        self.__dict__.update(config)
        self.log_str = "| [{}] Episode: {:4} | Reward: {:7.3f} | "
        self.log_str += "Q: {:8.3f} | T: {:3d} |"
        self.p_ckpt = "__checkpoints/{}_{}"
        self.saver = tf.train.Saver()
        if "periodic_ckpt" not in self.__dict__:
            self.periodic_ckpt = False
        if "save_best" not in self.__dict__:
            self.save_best = False
        # if load_checkpt is not None:
        #     self.restore_checkpt(load_checkpt)
        self.reset()
        info.out("Initialized {}".format(self.name))

    def reset(self):
        self.q_total = 0.
        self.r_total = 0.
        self.t_steps = 0
        self.episode = 0

    def generate_rollout(self):
        t = 0
        done = False
        episodic_q = 0.
        episodic_r = 0.
        x = self.env.reset()
        while not done and t < self.env.max_episode_steps:
            a, u, q = self.agent.step(np.hstack([x["observation"],
                                      x["desired_goal"]]),
                                      explore=(not self.eval))
            x2, r, done, info = self.env.step(a)
            self.agent.remember([x["observation"], x["desired_goal"],
                                 x["achieved_goal"], u, r, x2["observation"],
                                 x2["desired_goal"], int(done)])
            x = x2

            # Render if required
            if self.render:
                self.env.render()

            # Update stats
            t += 1
            episodic_r += float(r)
            episodic_q += float(q)

            # Train agent if required
            if not self.eval:
                assert "train_cycles_per_ts"
                for i in range(self.train_cycles_per_ts):
                    self.agent.train()
        self.episode += 1
        self.update_stats(episodic_q, episodic_r, t)
        if not self.eval:
            self.log(episodic_q, episodic_r, t)
        self.create_checkpoint()

    def create_checkpoint(self):
        if self.periodic_ckpt and self.episode % self.periodic_ckpt == 0:
            info.out("Creating periodic checkpoint")
            self.saver.save(self.agent.sess,
                            self.p_ckpt.format("P", self.episode))
        if self.eval and self.save_best and self.mean_er > self.best_score:
            info.out("New best score: {}".format(self.mean_er))
            self.best_score = self.mean_er
            self.saver.save(self.agent.sess,
                            self.p_ckpt.format("B", self.episode))

    def update_stats(self, eps_q, eps_r, t):
        self.q_total += eps_q
        self.r_total += eps_r
        self.t_steps += t
        self.mean_eq = self.q_total/self.episode
        self.mean_er = self.r_total/self.episode

    def log(self, q, r, t):
        if not self.eval:
            print(self.log_str.format(self.name, self.episode, r, q, t))
        else:
            evalstr = "\nEVAL RESULT:\nMEAN REWARD: {}\nMEAN QVALUE: {}"
            evalstr += "\nTIMESTEPS: {}\n"
            print()
            info.out(evalstr.format(r, q, t))
            print()

    def done(self):
        done = self.n_episodes <= self.episode
        if done and self.eval:
            self.log(self.mean_eq, self.mean_er, self.t_steps)
        return done

    # def summarize(self):
    #     if self.summarizer is None:
    #         return
    #     summarizer.value.add(tag="{}/")
