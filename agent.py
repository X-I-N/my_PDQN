import numpy as np
import torch


class RandomAgent:
    def __init__(self, action_space, seed):
        self.action_space = action_space
        self.num_actions = self.action_space.spaces[0].n
        self.action_parameter_sizes = np.array(
            [self.action_space.spaces[i].shape[0] for i in range(1, 1 + self.num_actions)])
        self.np_random = np.random.RandomState(seed)
        self.action_parameter_min_numpy = np.concatenate(
            [action_space.spaces[i].low for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_max_numpy = np.concatenate(
            [action_space.spaces[i].high for i in range(1, self.num_actions + 1)]).ravel()

    def choose_action(self, state):
        action = self.np_random.choice(self.num_actions)
        all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                                   self.action_parameter_max_numpy))

        offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
        action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]

        return action, action_parameters, all_action_parameters

    def update(self):
        pass
