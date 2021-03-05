import numpy as np


def pad_action(act, act_param):
    action = np.zeros((7,))
    action[0] = act
    if act == 0:
        action[[1, 2]] = act_param
    elif act == 1:
        action[3] = act_param
    elif act == 2:
        action[[4, 5]] = act_param
    elif act == 3:
        action[[6]] = act_param
    else:
        raise ValueError("Unknown action index '{}'".format(act))
    return action
