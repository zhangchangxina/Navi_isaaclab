import random
from operator import itemgetter
import numpy as np


class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer: list[tuple] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch: list[tuple]) -> None:
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)
        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[: len(self.buffer) - self.position]
            self.buffer[: len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position :]
            self.position = len(batch) - len(self.buffer) + self.position

    def push_batch_tensors(self, s, a, r, ns, d) -> None:
        # s,a,r,ns,d are numpy arrays or array-like with batch dim first
        batch = [(s[i], a[i], r[i], ns[i], d[i]) for i in range(len(s))]
        self.push_batch(batch)

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size: int):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done


