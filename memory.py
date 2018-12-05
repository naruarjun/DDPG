import numpy as np
from collections import deque


class Memory:

    def __init__(self, n_objects, maxlen, seed=None):
        self.n_objects = n_objects
        self.buffer = [deque(maxlen=maxlen) for _ in range(self.n_objects)]
        if seed is not None:
            np.random.seed(seed)

    def add(self, experience):
        assert type(experience) == list, "req: (s, a, r, ns, t)"
        assert len(experience) == self.n_objects
        [self.buffer[i].append(j) for i, j in enumerate(experience)]

    def sample(self, num):
        if num > len(self.buffer):
            raise ValueError("Memory size less than required batch size")
        samples = np.random.choice(len(self.buffer), num, False)
        batches = [np.array([self.buffer[i][j] for j in samples])
                   for i in range(self.n_objects)]
        return batches

    @property
    def size(self):
        return len(self.buffer[0])
