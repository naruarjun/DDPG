import gym
from gym.spaces import Box
from gym.envs.registration import EnvSpec
from gym.envs.registration import register
from gym.envs.classic_control import rendering

import numpy as np
from numpy.linalg import norm
from numpy.random import random

from PointEnvironment.Agent import Agent
from PointEnvironment.Pose import Pose
from util import error, log


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
        self.dt = 1e-2
        # self.her = True
        self.her = True
        # self.seed = None
        self.thresh = np.array([0.05, 0.05, 0.1])[:-1]
        self.num_iter = 50
        self.reward_max = 1
        self.max_episode_steps = 25
        self._max_episode_steps = 25
        self.step_penalty = 1.0  # (self.max_episode_steps)

        self.action_low = np.array([0.0, -np.pi/4])
        self.action_high = np.array([0.3, np.pi/4])
        self.action_space = Box(self.action_low, self.action_high, dtype="f")

        self.observation_space = Box(low=-1, high=1, shape=(5,), dtype="f")
        # World Limits
        self.w_limits = np.array([10, 10])
        # Screen Limits
        self.s_limits = np.array([600, 600])
        self.scale = self.s_limits/self.w_limits
        assert self.scale.tolist().count(self.scale[0]) == len(self.scale),\
            error.format("Scale for both axis must be equal...")
        self.scale = self.scale[0]
        self.agent_radius = 0.15  # in meters
        self.agents = [Agent(i) for i in range(3)]
        if config is not None:
            self.__dict__.update(config)

        # if self.seed is not None:
        #     np.random.seed(self.seed)

        self.goal = None
        if not self.her:
            self.dMax = self.action_high[0]*self.dt*self.num_iter
            self.dRange = 2*self.dMax
        self.viewer = None
        self._spec = EnvSpec("Go2Goal-v0")

    def reset(self):
        [agent.reset(self.sample_pose()) for agent in self.agents]
        self.goal = self.sample_pose()
        self.goal_changed = True
        # print("Agent spawned at: ", self.agent.pose)
        # print("Goalpoint set to: ", self.goal)
        return self.compute_obs()

    def step(self, action):
        assert self.goal is not None, "Call reset before calling step"
        prev_poses = [Pose(*agent.pose.tolist()) for agent in self.agents]
        [agent.step(action[i]) for i, agent in enumerate(self.agents)
            for j in range(self.num_iter)]
        for i in range(self.num_iter):
            self.agent.step(action)
        reward, done = self.get_reward()
        return self.compute_obs(prev_poses), reward, done,\
            {"dist": self.current_distance, "is_success": done}

    def get_reward(self):
        reached = (self.distance_from_goal() < self.thresh).all()
        if reached:
            return self.reward_max, True
        return -self.step_penalty, False
        # return -self.step_penalty if not reached else self.reward_max
        # reward = self.reward_max if reached else 0.0
        # return reward-self.step_penalty, reached

    def distance_from_goal(self):
        self.current_distance = np.abs((self.agent.pose - self.goal))[:-1]
        return self.current_distance

    def sample_pose(self):
        x, y = random(2)*self.w_limits - self.w_limits/2
        theta = (random()*2 - 1)*np.pi
        return Pose(x=x, y=y, t=theta)

    def init_viewer(self):
        self.viewer = rendering.Viewer(*self.s_limits)
        # GOAL MARKER
        circle = rendering.make_circle()
        circle.set_color(0.3, 0.82, 0.215)
        self.goal_tf = rendering.Transform()
        circle.add_attr(self.goal_tf)
        self.viewer.add_geom(circle)
        # AGENT MARKERS
        self.agent_tfs = []
        a_rad_px = self.agent_radius * self.scale
        verx = [a_rad_px*Go2Goal.cossin(np.radians(i)) for i in [0, 140, -140]]
        for i in self.agents:
            agent = rendering.FilledPolygon([tuple(j) for j in verx])
            agent.set_color(0.15, 0.235, 0.459)
            agent_tf = rendering.Transform()
            agent.add_attr(agent_tf)
            self.agent_tfs.append(agent_tf)
            self.viewer.add_geom(agent)
        log.out(self.agent_tfs)
        # GOAL VECTORS
        # self.goal_vex = [rendering.Line((0, 0), (1, 1)) for i in self.agents]
        # [i.set_color(1., 0.01, 0.02) for i in self.goal_vex]
        # [self.viewer.add_geom(i) for i in self.goal_vex]

    def render(self, mode='human'):
        # We also add some offset = self.s_limit//2 to center the origin
        if self.goal is None:
            return None
        if self.viewer is None:
            self.init_viewer()
        if self.goal_changed:
            self.goal_changed = False
            self.goal_tf.set_translation(*(self.goal.tolist()[:-1] +
                                         self.w_limits//2)*self.scale)
        for agent, agent_tf in zip(self.agents, self.agent_tfs):
            agent_tf.set_translation(*(agent.pose.tolist()[:-1] +
                                     self.w_limits//2)*self.scale)
            agent_tf.set_rotation(agent.pose.theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def compute_obs(self, prev_poses=None):
        # our observations are going to be the distance to be moved in
        # the direction of the goal in the vector form...
        # to be brief obs =  pos vec of goal relative to agent...
        # but what about angles? lets ignore them for now...
        goal_vex = [np.array((self.goal - agent.pose)[:-1])
                    for agent in self.agents]
        distance = [norm(goal_vec) for goal_vec in goal_vex]
        unit_vex = [(goal_vec / dist if distance != 0 else goal_vec)
                    for goal_vec, dist in zip(goal_vex, distance)]
        sc = [Go2Goal.cossin(agent.pose.theta) for agent in self.agents]
        goal = [np.hstack(i) for i in zip(unit_vex, distance)]
        if not self.her:
            return np.hstack([sc, goal])
        ag = [np.zeros(3) for i in self.agents]
        if prev_poses is not None:
            for idx, agent in enumerate(self.agents):
                agx = (agent.pose - prev_poses[idx])[:-1]
                if norm(ag) < 1e-8:
                    ag = 1e-8

                ag[idx] = np.hstack([agx/norm(agx), norm(agx)])
        self.obs = {"observation": sc,
                    "desired_goal": goal,
                    "achieved_goal": ag}
        return self.obs

    def close(self):
            if self.viewer:
                self.viewer.close()
                self.viewer = None

    def seed(self, seed):
        np.random.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # print(achieved_goal.shape)
        # print(desired_goal.shape)
        r = norm(achieved_goal - desired_goal, axis=1)
        # print(r, r.shape, r.astype('i'))
        return (r < self.thresh).astype('i')


register(
    id='Go2Goal-v0',
    entry_point='go2goal:Go2Goal',
)
