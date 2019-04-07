import gym
import go2goal
import numpy as np
from PointEnvironment.Pose import Pose
import copy


h = 2000 #length of random action sequence
k = 100 #number of sequences
thresh = np.array([0.05, 0.05, 0.1])[:-1]
env = gym.make("Go2Goal-v0");

def MPC_dist_from_goal(env,pose):
	current_distance = np.abs((pose - env.goal))[:-1]
	return current_distance

def MPC_done(env,current_distance):
	reached = (current_distance < thresh).all()
	if reached:
		return env.reward_max,True
	return -env.step_penalty,False



def cost(dist):
	cost_temp = dist[0]**2+dist[1]**2;
	return cost_temp
mno = 0
"""
TODO: Try different sampling methods
      Try different cost metrics
      Try a different state space
"""
while(True):
	obs = env.reset()
	done = False
	length=0
	while(done==False and length<25):
		final_action = 0
		action_sequence = []
		minimum = 100
		cost_current=0
		for i in range(k):
			current_pose = copy.deepcopy(env.agent.pose)
			cost_current=0
			action_seq1 = np.random.uniform(low=0.0, high=0.3, size=(h,1));
			action_seq2 = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(h,1));
			action_seq = np.hstack([action_seq1,action_seq2])
			cur_action = action_seq[0] 
			for j in action_seq:
				current_pose.updateHolonomic(j,0.01)
				current_distance = MPC_dist_from_goal(env,current_pose)
				rew, done_temp = MPC_done(env,current_distance)
				cost_current += cost(current_distance) #Using the distance as the cost metric
				if(done_temp):
					break
			if cost_current<minimum or i==0:
				minimum = cost_current
				final_action = cur_action
			cost_current = 0
		obs = env.step(cur_action)
		done = obs[2]
		env.render()
		length = length+1
	mno = mno+1
	print("Episode "+str(mno)+":"+str(length))







			





