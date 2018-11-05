import numpy as np
from numpy.linalg import norm
from numpy.random import random

from PointEnvironment.Environment import PointEnvironment
from PointEnvironment.Agent import Agent
from PointEnvironment.Pose import Pose

np.set_printoptions(4)

REWARD = 1


class G2Goal:
    def __init__(self, seed=None):
        self.env = PointEnvironment(num_iterations=50, dt=.01)
        self.agent = Agent(0)
        self.env.addAgent(self.agent)
        self.limits = np.array([10, 10, 2*np.pi])
        self.thresh = 0.1
        self.d_clip = 1
        # just to make sure its read as a continuous action space...
        self.action_space = np.array(np.ones((2, 2)))
        self._max_episode_steps = 150
        if seed is not None:
            self.seed(seed)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        # self.agent.reset(self.sample_pose())
        self.agent.reset(Pose())
        # self.goal = self.sample_pose()
        self.goal = Pose(2, 3, 0)
        # print("Agent spawned at: ", self.agent.pose)
        # print("Goalpoint set to: ", self.goal)
        return self.compute_obs()

    def step(self, u):
        prev_pose = Pose(*self.agent.pose.tolist())
        self.env.step({self.agent.id: u})
        reward = self.reward()
        return self.compute_obs(prev_pose), reward, reward > 0,\
            {"dist": self.current_distance}

    def reward(self):
        reached = self.get_current_distance() < self.thresh
        return REWARD if reached else 0.0

    def sample_pose(self):
        return Pose(*(random(3)*self.limits - self.limits/2))

    def get_current_distance(self):
        self.current_distance = norm((self.agent.pose - self.goal)[:-1])
        return self.current_distance

    def compute_obs(self, prev_pose=None):
        # our observations are going to be the distance to be moved in
        # the direction of the goal in the vector form...
        # to be brief obs =  pos vec of goal relative to agent...
        # but what about angles? lets ignore them for now...
        goal_vec, angle = np.split((self.goal - self.agent.pose), [-1])
        distance = np.linalg.norm(goal_vec)
        unit_vec = goal_vec / distance
        c_dist = min(distance, self.d_clip)/self.d_clip

        assert np.abs(norm(unit_vec)-1) < 1e-4, norm(unit_vec)

        sc = np.cos(angle), np.sin(angle)

        goal = np.hstack([unit_vec, c_dist])
        if prev_pose is None:
            ag = np.zeros(3)
        else:
            ag = (self.agent.pose - prev_pose)[:-1]
            ag = np.hstack([ag/norm(ag), norm(ag)])
        return {"obs": np.hstack(sc),
                "goal": goal,
                "ag": ag}
