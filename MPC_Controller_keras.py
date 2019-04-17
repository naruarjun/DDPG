import tensorflow as tf
from tensorflow import multiply as mu
from tensorflow import GraphKeys
from tensorflow.train import AdamOptimizer as Adam
from tensorflow.layers import dense
from tensorflow.initializers import truncated_normal as TN
import gym
import go2goal
import numpy as np
from keras import optimizers
from PointEnvironment.Pose import Pose
import copy
from Model_approx import Model_Approx
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pickle
import tensorflow as tf
from keras import backend as k
tf.enable_eager_execution()
# fix random seed for reproducibility
numpy.random.seed(7)
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
h = 25 #length of random action sequence
k = 50 #number of sequences
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
TODO: Convert output of dynamics model to difference in states 
"""

normalize_x = []
normalize_y = []
normalize_theta = []
normalize_act0 = []
normalize_act1 = []

def get_data(env):
	# trajs = gather_random_trajectories(500,env)
	# random_trajs = trajs
	# #print(random_trajs)
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
	# x = np.array([new_obs.x-obs.x for obs,new_obs,_,_,act in random_trajs]).reshape((-1,1))
	# x = (x-normalize_x[0])/(normalize_x[1]-normalize_x[2])
	# #print(x.shape)
	# y = np.array([new_obs.y-obs.y for obs,new_obs,_,_,act in random_trajs]).reshape((-1,1))
	# y = (y-normalize_y[0])/(normalize_y[1]-normalize_y[2])
	# theta = np.array([new_obs.theta-obs.theta for obs,new_obs,_,_,act in random_trajs]).reshape((-1,1))
	# theta = (theta-normalize_theta[0])/(normalize_theta[1]-normalize_theta[2])
	# y_env_train = np.hstack([x,y,theta])
	with open("X_train.pkl","rb") as f:
		X_train = pickle.load(f)
		print(X_train.shape)
	with open("y_train.pkl","rb") as f:
		y_env_train = pickle.load(f)
	return X_train,y_env_train


input_shape = [None,5]
output_shape = [None,3]
action_shape = [None,env.action_space.shape[0]]
scope = "Approx_Network"
n_layers = 1
n_units = 50
activation = tf.nn.relu
output_act = tf.nn.tanh
lr = 1e-5
dynamics_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation=tf.nn.relu,input_shape = (5,)),
    tf.keras.layers.Dense(64, activation=tf.nn.tanh),
    tf.keras.layers.Dense(64, activation=tf.nn.tanh),
    tf.keras.layers.Dense(3)
])
dynamics_model.compile(loss='mean_squared_error', optimizer=tf.train.AdamOptimizer())
X_train,y_env_train = get_data(env)
dynamics_model.fit(X_train, y_env_train, epochs=20, batch_size=16,shuffle = True)
print(dynamics_model.summary())
print(dynamics_model.predict(X_train[0].reshape(1,5)))
print(y_env_train[0])
print(dynamics_model.predict(X_train[1].reshape(1,5)))
# print(y_env_train[1])
# print(dynamics_model.predict(X_train[2].reshape(1,5)))
# print(y_env_train[2])
# print(dynamics_model.predict(X_train[3].reshape(1,5)))
# print(y_env_train[3])
# print(dynamics_model.predict(X_train[4].reshape(1,5)))
# print(y_env_train[4])
obs = env.reset()
prev_pose = copy.deepcopy(env.agent.pose)
current_pose = copy.deepcopy(env.agent.pose)


policy_model = tf.keras.Sequential([
    tf.keras.layers.Dense(256,activation=tf.nn.relu,input_shape = (6,)),
    tf.keras.layers.Dense(256, activation=tf.nn.tanh),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(2)
])

def dist(inp_state,goal_state):
	current_distance = tf.slice(tf.abs((tf.square(inp_state) - tf.square(goal_state))),[0,0],[-1,2])**(1/2)
	return current_distance

def loss_grad_with_dynamics_model(model1, model2, dense4, input_state, goal_state):
	dense4 = tf.convert_to_tensor(dense4,tf.float32)
	goal_state = tf.convert_to_tensor(goal_state,tf.float32)
	input_state = tf.convert_to_tensor(input_state,tf.float32)	
	with tf.GradientTape() as tape:
		dense4 = model2(tf.concat([input_state,goal_state], 1))
		init_state = tf.concat([input_state,dense4], 1)
		final_states = model1(init_state)
		distance1 = dist(final_states,goal_state)
		distance2 = dist(input_state,goal_state)
		loss = tf.reduce_mean(tf.subtract(distance1,distance2))
	return loss,tape.gradient(loss,model2.trainable_variables)

def gradient_update(model, optimizer, input_grad):
	optimizer.apply_gradients(zip(input_grad, model.trainable_variables))

optimizer = tf.train.AdamOptimizer()
while(True):
	obs = env.reset()
	done = False
	length=0
	loss_final=0
	while(done==False and length<25):
		train_input_state = []
		train_goal_state = []
		train_dense = []
		train_output = []
		current_pose = copy.deepcopy(env.agent.pose)
		current_pose_temp = np.array([current_pose.x,current_pose.y,current_pose.theta])
		current_pose_new = copy.deepcopy(np.array([[current_pose.x,current_pose.y,current_pose.theta]])) 
		current_pose_goal = np.array([env.goal.x,env.goal.y,env.goal.theta]) 
		distance_old = MPC_dist_from_goal(env,current_pose)
		curr_act = np.zeros([2,])
		for j in range(h):
			current_pose_temp = current_pose_new
			current_input = np.hstack([np.squeeze(current_pose_temp,axis=0),current_pose_goal])
			current_output = policy_model.predict(np.expand_dims(current_input,axis=0)) #Getting the action to perform
			if(j==0):
				cur_act = current_output
			
			current_input_dynamics = np.hstack([np.squeeze(current_pose_temp,axis=0),np.squeeze(current_output,axis=0)])
			
			current_pose_new = dynamics_model.predict(np.expand_dims(current_input_dynamics,axis=0)) #Getting next state
			#training data addition
			train_input_state.append(current_pose_temp)
			train_goal_state.append(current_pose_goal)
			train_dense.append(current_output)
		obs = env.step(cur_act.reshape((2,)))
		done = obs[2]
		train_input_state = np.squeeze(train_input_state,axis=1)
		train_dense = np.squeeze(train_dense,axis=1)
		#print(np.array(train_input_state).shape)
		loss,grads = loss_grad_with_dynamics_model(dynamics_model,policy_model,np.array(train_dense),np.array(train_input_state),np.array(train_goal_state))
		loss_final = loss_final+loss
		gradient_update(policy_model,optimizer,grads)
		env.render()
		length = length+1
	mno = mno+1
	print("Episode "+str(mno)+":"+str(length))
	print("##############")
	print("Loss:"+str(loss_final))







			





