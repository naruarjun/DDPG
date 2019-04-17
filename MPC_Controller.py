import tensorflow as tf
from tensorflow import multiply as mu
from tensorflow import GraphKeys
from tensorflow.train import AdamOptimizer as Adam
from tensorflow.layers import dense
from tensorflow.initializers import truncated_normal as TN
import gym
import go2goal
import numpy as np
from PointEnvironment.Pose import Pose
import copy
from Model_approx import Model_Approx
from tqdm import tqdm
import pickle


h = 2000 #length of random action sequence
k = 100 #number of sequences
thresh = np.array([0.05, 0.05, 0.1])[:-1]
env = gym.make("Go2Goal-v0");

def gather_random_trajectories(num_traj, env):
    '''
    Run num_traj random trajectories to gather information about the next state and reward.
    Data used to train the dynamics models in a supervised way.
    '''
    dataset_random = []

    game_rewards = []
    for n in range(num_traj):
        print("Generating Trajectory number {}".format(n))
        length = 0
        obs = env.reset()
        while True:
            sampled_action = env.action_space.sample()
            #print(sampled_action)
            old_pose = copy.deepcopy(env.agent.pose)
            obs = env.step(sampled_action)
            new_pose = copy.deepcopy(env.agent.pose)
            done = obs[2]
            dataset_random.append([old_pose, new_pose, obs[1], obs[2], sampled_action])
            #game_rewards.append(reward)
            if done or length>25:
                break
            length = length+1
            env.render()
    #print('Mean R:',np.round(np.sum(game_rewards)/num_traj,2), 'Max R:', np.round(np.max(game_rewards),2), np.round(len(game_rewards)/num_traj))

    return np.array(dataset_random)

def load_config(filename):
    with open(filename) as f:
        config = yaml.load(f.read())
    return config

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

normalize_x = []
normalize_y = []
normalize_theta = []
normalize_act0 = []
normalize_act1 = []

def train_dynamics_model(net,env):
    # trajs = gather_random_trajectories(1000,env)
    # print(len(trajs))
    # rand = np.arange(len(trajs))
    # np.random.shuffle(rand)
    # random_trajs = trajs[rand]
    # print(random_trajs)
    with open("X_train.pkl","rb") as f:
        X_train = pickle.load(f)
        print(X_train.shape)
    with open("y_train.pkl","rb") as f:
        y_env_train = pickle.load(f)
    for it in range(100):
        # rand = np.arange(len(trajs))
        # np.random.shuffle(rand)
        # random_trajs = trajs[rand]
        # x = np.array([obs.x for obs,_,_,_,act in random_trajs]).reshape((-1,1))
        # print(x.shape)
        # y = np.array([obs.y for obs,_,_,_,act in random_trajs]).reshape((-1,1))
        # theta = np.array([obs.theta for obs,_,_,_,act in random_trajs]).reshape((-1,1))
        # act0 = np.array([act[0] for obs,_,_,_,act in random_trajs]).reshape((-1,1))
        # act1 = np.array([act[1] for obs,_,_,_,act in random_trajs]).reshape((-1,1))
        # normalize_x.append(np.mean(x))
        # normalize_x.append(np.max(x))
        # normalize_x.append(np.min(x))
        # normalize_y.append(np.mean(y))
        # normalize_y.append(np.max(y))
        # normalize_y.append(np.min(y))
        # normalize_theta.append(np.mean(theta))
        # normalize_theta.append(np.max(theta))
        # normalize_theta.append(np.min(theta))
        # normalize_act0.append(np.mean(act0))
        # normalize_act0.append(np.max(act0))
        # normalize_act0.append(np.min(act0))
        # normalize_act1.append(np.mean(act1))
        # normalize_act1.append(np.max(act1))
        # normalize_act1.append(np.min(act1))
        # x = (x-np.mean(x))/(np.max(x)-np.min(x))
        # y = (y-np.mean(y))/(np.max(y)-np.min(y))
        # theta = (theta-np.mean(theta))/(np.max(theta)-np.min(theta))
        # act0 = (act0-np.mean(act0))/(np.max(act0)-np.min(act0))
        # act1 = (act1-np.mean(act1))/(np.max(act1)-np.min(act1))

        # X_train = np.hstack([x,y,theta,act0,act1])
        # print(X_train.shape)
        # x = np.array([new_obs.x-obs.x for obs,new_obs,_,_,act in random_trajs]).reshape((-1,1))
        # x = (x-normalize_x[0])/(normalize_x[1]-normalize_x[2])
        # #print(x.shape)
        # y = np.array([new_obs.y-obs.y for obs,new_obs,_,_,act in random_trajs]).reshape((-1,1))
        # y = (y-normalize_y[0])/(normalize_y[1]-normalize_y[2])
        # theta = np.array([new_obs.theta-obs.theta for obs,new_obs,_,_,act in random_trajs]).reshape((-1,1))
        # theta = (theta-normalize_theta[0])/(normalize_theta[1]-normalize_theta[2])
        # y_env_train = np.hstack([x,y,theta])

        # with open('X_train.pkl','wb') as f:
        #     pickle.dump(X_train,f)
        # with open('y_train.pkl','wb') as f:
        #     pickle.dump(y_env_train,f)
        # break
        #print(y_env_train.shape)
        np.random.shuffle(X_train)
        np.random.shuffle(y_env_train)
        loss = 0
        batch_size=16
        BATCH_SIZE=16
        for mb in tqdm(range(0, len(X_train), batch_size)):
            if len(X_train) > mb+BATCH_SIZE:
                X_mb = X_train[mb:mb+BATCH_SIZE]
                y_env_mb = y_env_train[mb:mb+BATCH_SIZE]
                net.train(X_mb,y_env_mb)
                loss = loss + net.evaluate(X_mb,y_env_mb)
        print("###############################")
        print("Iteration:{}    Loss:{}".format(it,loss))
        print("###############################")
        if(it==99):
            print(net.predict(X_train[0].reshape((1,5))))
            print(y_env_train[0].reshape(1,3))



input_shape = [None,5]
output_shape = [None,3]
action_shape = [None,env.action_space.shape[0]]
scope = "Approx_Network"
n_layers = 4
n_units = 128
activation = tf.nn.tanh
output_act = tf.nn.tanh
lr = 1e-3
# Training part for neural network
with tf.Session() as sess:
	net = Model_Approx(sess, input_shape, action_shape, output_shape, scope, n_layers, n_units, activation, output_act, lr)
	sess.run(tf.global_variables_initializer()) 
	train_dynamics_model(net,env)
	# obs = env.reset()
	# prev_pose = copy.deepcopy(env.agent.pose)
	# current_pose = copy.deepcopy(env.agent.pose)
	# for i in range(300):
	# 	obs = env.

	# while(True):
	# 	obs = env.reset()
	# 	done = False
	# 	length=0
	# 	while(done==False and length<25):
	# 		final_action = 0
	# 		action_sequence = []
	# 		minimum = 100
	# 		cost_current=0
	# 		for i in range(k):
	# 			current_pose = copy.deepcopy(env.agent.pose)
	# 			cost_current=0
	# 			action_seq1 = np.random.uniform(low=0.0, high=0.3, size=(h,1));
	# 			action_seq2 = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(h,1));
	# 			action_seq = np.hstack([action_seq1,action_seq2])
	# 			cur_action = action_seq[0] 
	# 			for j in action_seq:
	# 				current_pose.updateHolonomic(j,0.01)
	# 				current_distance = MPC_dist_from_goal(env,current_pose)
	# 				rew, done_temp = MPC_done(env,current_distance)
	# 				cost_current += cost(current_distance) #Using the distance as the cost metric
	# 				if(done_temp):
	# 					break
	# 			if cost_current<minimum or i==0:
	# 				minimum = cost_current
	# 				final_action = cur_action
	# 		obs = env.step(cur_action)
	# 		done = obs[2]
	# 		env.render()
	# 		length = length+1
	# 	mno = mno+1
	# 	print("Episode "+str(mno)+":"+str(length))







			





