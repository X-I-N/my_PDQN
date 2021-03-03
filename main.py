import os
import numpy as np
import gym
import torch
import datetime
import argparse
import gym_soccer
from gym.wrappers import Monitor
from agent import RandomAgent, PDQNAgent
from utils import pad_action


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, type=bool, help="Train mode or not")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_dir', default='results/soccer', type=str)
    parser.add_argument('--max_steps', default=15000, type=int)
    parser.add_argument('--train_eps', default=20000, type=int)
    parser.add_argument('--eval_eps', default=1000, type=int)

    config = parser.parse_args()
    return config


def train(cfg):
    env = gym.make('SoccerScoreGoal-v0')
    env = Monitor(env, directory=os.path.join(cfg.save_dir), video_callable=False, write_upon_reset=False, force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = RandomAgent(action_space=env.action_space, seed=cfg.seed)
    state = env.reset()
    state = np.array(state, dtype=np.float32)
    act, act_param, all_action_param = agent.choose_action(state)
    action = pad_action(act, act_param)
    rewards = []
    tot_reward = 0.
    # todo : the std RL train mode && tenserboard plot
    for i in range(5000):
        next_state, reward, done, info = env.step(action)
        tot_reward += reward
        rewards.append(reward)
        act, act_param, all_action_param = agent.choose_action(state)
        action = pad_action(act, act_param)
        print("step:{} info: {} reward:{}".format(i, info, reward))
        if done:
            break
        # if i % 100 == 0:
        #     print('step:{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i + 1), tot_reward, np.array(rewards[-100:]).mean()))
    print("Total reward = {}".format(tot_reward))
    env.close()


def evaluation(cfg, saved_model):
    pass


if __name__ == '__main__':
    cfg = get_args()
    if cfg.train:
        train(cfg)
        # evaluation(cfg)
    else:
        pass
        # evaluation(cfg, saved_model=model_path)
