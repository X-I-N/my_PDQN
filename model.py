import torch
import torch.nn as nn
import torch.nn.functional as F


class Q_Actor(nn.Module):
    def __init__(self, n_states, n_actions, action_parameter_size, hidden_layers=(100, )):
        super(Q_Actor, self).__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]
        self.action_parameter_size = action_parameter_size
        self.n_states = n_states
        self.n_actions = n_actions

        self.layers = nn.ModuleList()
        inputSize = self.n_states + action_parameter_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            num_hl = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, num_hl):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[num_hl - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.n_actions))
        # initialize the weights

    def forward(self, state, action_parameter):
        x = torch.cat((state, action_parameter), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            x = F.relu(self.layers[i](x))
        Q_value = self.layers[-1](x)
        return Q_value


class ParamNet(nn.Module):

    def __init__(self, n_state, n_action, action_parameter_size, hidden_layers):
        super(ParamNet, self).__init__()
        self.action_parameter_size = action_parameter_size
        self.n_action = n_action
        self.n_state = n_state
        self.layers = nn.ModuleList()
        inputSize = self.n_state
        lastHiddenSize = inputSize
        if hidden_layers is not None:
            num_hl = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, num_hl):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenSize = hidden_layers[num_hl - 1]
        self.layers.append(nn.Linear(lastHiddenSize, action_parameter_size))

    def forward(self, state):
        x = state
        num_hl = len(self.layers)
        for i in range(0, num_hl - 1):
            x = F.relu(self.layers[i](x))

        action_params = self.layers[-1](x)

        return action_params
