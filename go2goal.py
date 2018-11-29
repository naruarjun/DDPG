import os
import sys
import rospy
import subprocess as sp

import gym
from gym.spaces import Box
from gym.envs.registration import register

import numpy as np
from numpy.linalg import norm
from numpy.random import random

from AgentStage import AgentStage as Agent
from util import error, warn, info


class Go2Goal(gym.Env):

    cossin = staticmethod(lambda x: np.array([np.cos(x), np.sin(x)]))

    def __init__(self, config=None):
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
        self.launch_stage()
        self.her = True
        self.seed = None
        self.thresh = np.array([0.15, 0.15])
        self.reward_max = 10
        self.max_episode_steps = 250
        self._max_episode_steps = 250
        self.step_penalty = .5#/(self.max_episode_steps)
        self.stall_penalty = 20

        self.action_low = np.array([0.0, -np.pi/4])
        self.action_high = np.array([0.3, np.pi/4])
        self.action_space = Box(self.action_low, self.action_high, dtype="f")
        self.observation_space = Box(low=-1, high=1, shape=(5,), dtype="f")
        
        if config is not None:
            self.__dict__.update(config)
        
        self.put_seed(self.seed)
        self.agent = Agent(0)
        self.goal = None
        rospy.init_node("stageGo2Goal")

    def reset(self):
        self.agent.reset(self.sample_goal())
        self.goal = self.sample_goal()
        info.out("Goal set to: {}".format(self.goal.tolist()))
        return self.compute_obs()

    def step(self, action):
        assert self.goal is not None, error.format("Call reset before step")
        prev_pose = self.agent.pose[:]
        self.agent.step(action)
        reward, done = self.get_reward()
        return self.compute_obs(prev_pose), reward, done,\
            {"dist": self.current_distance, "is_success": reward>0}

    def get_reward(self):
        reached = (self.distance_from_goal() < self.thresh).all()
        if self.agent.stall:
            return -self.stall_penalty, True
        if reached:
            return self.reward_max, True
        return -self.step_penalty, False

    def distance_from_goal(self):
        self.current_distance = np.abs((self.agent.pose[:-1] - self.goal))
        info.out("Dist: {}".format(norm(self.current_distance)))
        return self.current_distance

    def compute_obs(self, prev_pose=None):
        goal_vec = self.goal - self.agent.pose[:-1]
        distance = np.linalg.norm(goal_vec)
        unit_vec = goal_vec / distance if distance != 0 else goal_vec

        ##### This has been added to make sure that the inputs are ########
        ##### similar to the ones that the g2g model was trained for..#####
        distance = np.min([distance, 1.])
        ###################################################################

        goal = np.hstack([unit_vec, distance])
        if not self.her:
            return np.hstack([sc, goal])
        if prev_pose is None:
            ag = np.zeros(3)
        else:
            ag = (self.agent.pose - prev_pose)[:-1]
            if norm(ag) < 1e-8:
                ag += 1e-8

            ag = np.hstack([ag/norm(ag), norm(ag)])
        obs = np.hstack([Go2Goal.cossin(self.agent.pose[-1])])
        self.obs = {"observation": obs,
                    "desired_goal": goal,
                    "achieved_goal": ag}
        return self.obs

    def put_seed(self, seed):
        if seed is not None:
            np.random.seed(seed)

    def launch_stage(self, world=None):
        current_path = os.getcwd()
        world_file = os.path.join(current_path, "__stage", "simple.world")
        info.out("USING WORLD FILE: {}".format(world_file))
        args = ["rosrun", "stage_ros", "stageros", world_file]
        try:
            sp.Popen(args)
        except Exception as e:
            error.out(e)

    def sample_goal(self):
        bounds = [(np.array([-7, -7]), np.array([-2, -5])),
                  (np.array([+3, -5]), np.array([+7, -2])),
                  (np.array([+2, +4]), np.array([+7, +7])),
                  (np.array([-7, +0]), np.array([-6, +6]))]
        bmin, bmax = bounds[np.random.choice(4)]
        goal = np.random.random(2)*(bmax-bmin) + bmin
        return goal

    def render(self):
        pass


register(
    id="Go2Goal-v0",
    entry_point="go2goal:Go2Goal",
)
