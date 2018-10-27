import numpy as np
from collections import deque

class ReplayBuffer(deque):

  def __init__(self, storing_order: dict, maxlen=None):
    self.storing_order = 
    super(ReplayBuffer, self).__init__(maxlen=maxlen)

  def store_episode(self, episode):
    assert len(set([len(episode[k].shape) == 2 for k in episode.keys])) == 1,\
           "Only 2 dim arrays can pass through here !!"
    assert set([episode[k].shape[0] for k in episode.keys])[0] == True, \
           "All data holders must be of the same timestep length !!"
    for i in range(episode[0].shape[0]):
      self.append(tuple(episode[k][i] for k in self.storing_order))
    print ("Stored Episode...\nCurrent length of replay buffer is {}".format(len(self)))

  def sample(self, batch_size):
    return np.random.choice(self, size=batch_size, replace=True)
