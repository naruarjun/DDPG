import numpy as np
from collections import deque


class Memory:

    def __init__(self, dims, maxlen, seed=None):
        self.dims = dims
        self.buffer = deque(maxlen=maxlen)
        if seed is not None:
            np.random.seed(seed)

    def add(self, experience):
        assert type(experience) == tuple, "req: (s, a, r, ns)"
        assert len(experience) == self.dims
        self.buffer.append(np.array(experience))

    def sample(self, num):
        if num > len(self.buffer):
            raise ValueError("Memory size less than required batch size")
        samples = np.random.choice(len(self.buffer), num, False)
        batches = [np.vstack(np.array(self.buffer)[samples, i])
                   for i in range(self.dims)]
        return batches
