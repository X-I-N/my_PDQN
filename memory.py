import random


class ReplayBuffer:
    def __init__(self, capacity=1e5):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def push(self, state, action_with_param, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action_with_param, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)
