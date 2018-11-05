import gym
from gym.spaces import Box
from gym.envs.registration import EnvSpec

import numpy as np
from numpy.linalg import norm
from numpy.random import random

from PointEnvironment.Agent import Agent
from PointEnvironment.Pose import Pose

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env


class Go2Goal(gym.Env):

    def __init__(self, config):
        """
        Simple go to goal environment where a non-holonomic agent is
        required to move to a goal. The goal can be random or single,
        however it is advised that during training we provide it a
        random goal everytime. Rewards are binary.

        config includes:
        max_episode_steps (int): maximum number of timesteps in an episode
        reward_max (int): reward when goal is achieved.
        seed (int): seed of the random numpy process.
        her (bool): whether to use the HER compatible variant or not
        dt (float): dt in kinematic update equation
        num_iter (int): num of iterations of kinematic update equation

        """
        # Default values. Will be overridden if specified in config
        self.dt = 1e-2
        self.her = False
        self.seed = None
        self.d_clip = 1
        self.thresh = 0.1
        self.num_iter = 50
        self.reward_max = 10
        self.step_penalty = 0.0
        self.max_episode_steps = 120
        self._max_episode_steps = 120

        self.action_low = np.array([-0.1, -np.pi/4])
        self.action_high = np.array([0.3, np.pi/4])
        self.action_space = Box(self.action_low, self.action_high, dtype="f")

        self.observation_space = Box(low=-1, high=1, shape=(5,), dtype="f")

        self.limits = np.array([2.5, 2.5, 2*np.pi])
        self.agent = Agent(0)
        self.__dict__.update(config)

        if self.seed is not None:
            np.random.seed(self.seed)

        self.goal = None
        if not self.her:
            self.dMax = self.action_high[0]*self.dt*self.num_iter
            self.dRange = 2*self.dMax
        self._spec = EnvSpec("Go2Goal-v0")

    def reset(self):
        self.agent.reset(self.sample_pose())
        self.goal = self.sample_pose()

        # print("Agent spawned at: ", self.agent.pose)
        # print("Goalpoint set to: ", self.goal)
        return self.compute_obs(self.agent.pose)

    def step(self, action):
        assert self.goal is not None, "Call reset before calling step"
        prev_pose = Pose(*self.agent.pose.tolist())
        prev_dist = self.get_current_distance()
        self.agent.step(action)
        reward, done = self.get_reward(prev_dist)
        # print("reward: ", reward)
        return self.compute_obs(prev_pose), reward, done,\
            {"dist": self.current_distance}

    def get_reward(self, prev_distance):
        did = prev_distance - self.get_current_distance()
        reached = self.get_current_distance() < self.thresh
        reward = self.reward_max if reached else 0.0
        if not self.her:
            reward = ((did + self.dMax)/self.dRange*2 - 1)*self.reward_max
        # print("p: {}\tc: {}\td: {}".format(prev_distance, self.get_current_distance(), did))
        # print(reward)
        return reward-self.step_penalty, reached

    def get_current_distance(self):
        self.current_distance = norm((self.agent.pose - self.goal)[:-1])
        return self.current_distance

    def sample_pose(self):
        return Pose(*(random(3)*self.limits - self.limits/2))

    def compute_obs(self, prev_pose=None):
        # our observations are going to be the distance to be moved in
        # the direction of the goal in the vector form...
        # to be brief obs =  pos vec of goal relative to agent...
        # but what about angles? lets ignore them for now...
        goal_vec, angle = np.split((self.goal - self.agent.pose), [-1])
        distance = np.linalg.norm(goal_vec)
        unit_vec = goal_vec / distance if distance != 0 else goal_vec
        c_dist = min(distance, self.d_clip)/self.d_clip
        sc = np.hstack([np.cos(angle), np.sin(angle)])
        goal = np.hstack([unit_vec, c_dist])
        if not self.her:
            return np.hstack([sc, goal])
        if prev_pose is None:
            ag = np.zeros(3)
        else:
            ag = (self.agent.pose - prev_pose)[:-1]
            ag = np.hstack([ag/norm(ag), norm(ag)])
        return {"obs": sc,
                "goal": goal,
                "ag": ag}


if __name__ == "__main__":
    register_env("Go2Goal-v0", lambda config: Go2Goal(config))
    ray.init()
    run_experiments({
        "demo": {
            "run": "DDPG",
            "env": "Go2Goal-v0",
            "config": {
                "learning_starts": 1000,
                "horizon": 120,
                "actor_hiddens": [150, 150],
                "critic_hiddens": [150, 150],
                "schedule_max_timesteps": 1000000,
                "timesteps_per_iteration": 120,
                "exploration_fraction": 0.2,
                "gpu": True,
                "num_workers": 1,
                # "env_config": {
                #     "seed": 5,
                # },
            },
        },
    })
