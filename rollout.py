import numpy as np
import tensorflow as tf


class RolloutGenerator:
	"""
	Class for generating a rollout of trajectory by the agent
	args:
	env: gym env
	agent: agent for performing rollout
	config: rollout configuration 
	checkpoint(opt): perform rollout from a saved policy
	"""

	def __init__(self, env, agent, config: dict, eval=False, load_checkpt=None,
				 summarizer=None):
		self.env = env
		self.agent = agent
		self.train = train
		self.checkpt = checkpt
		self.current_best_score = 0.
		self.__dict__.update(self.config)
		self.log_str = "| [{}] Episode: {:4} | Reward: {:7.3f} | "
		self.log_str += "Q: {:8.3f} | T: {:3d} | MIN_D: {:4.3f} |"
		self.p_ckpt = "__checkpoint/{}"
		self.saver = tf.train.Saver()
		if load_checkpt is not None:
			self.restore_checkpt(load_checkpt)
		self.reset()

	def reset():
		self.episode = 1

	def restore_checkpt(self, load_checkpt):
		pass

	def generate_rollout(self):
		t = 0
		done = False
		episodic_q = 0.
		episodic_r = 0.
		x = self.env.reset()
		min_dist_achieved = self.env.distance_from_goal()
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
			t +=1
			episodic_r += float(r)
			episodic_q += float(q)

			if info["dist"] < min_dist_achieved:
				min_dist_achieved = info["dist"]

			# Train agent if required
			if self.train:
				assert "train_cycles_per_ts"
				for i in range(self.train_cycles_per_ts):
					self.agent.train()
		if "periodic_checkpt" in self.__dict__:
			self.create_checkpoint()
		self.episode += 1
		self.log(episodic_q/t, episodic_r/t, t, min_dist_achieved)

	def create_checkpoint(self):
		if self.episode % periodic_checkpt == 0:
			print("Creating periodic checkpoint")
			self.saver.save(self.agent.sess, self.p_ckpt.format(self.episode))

	def log(self, mean_q, mean_r, t, min_d):
		print(self.log_str.format(self.episode, mean_r, mean_q, t, min_d))

	def done(self):
		return self.n_episodes < self.episode

	def should_eval(self):
		return self.episode % self.eval_after == 0

	# def summarize(self):
	# 	if self.summarizer is None:
	# 		return
	# 	summarizer.value.add(tag="{}/")

