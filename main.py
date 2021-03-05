import os
import numpy as np
import gym
import torch
import datetime
import argparse
import gym_soccer
from gym.wrappers import Monitor
from torch.utils.tensorboard import SummaryWriter
from agent import PDQNAgent, RandomAgent
from utils import pad_action

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, type=bool, help="Train mode or not")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_dir', default='results/soccer', type=str)
    parser.add_argument('--max_steps', default=5000, type=int)
    parser.add_argument('--train_eps', default=10000, type=int)
    parser.add_argument('--eval_eps', default=1000, type=int)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")

    config = parser.parse_args()
    return config


def train(cfg):
    env = gym.make('SoccerScoreGoal-v0')
    env = Monitor(env, directory=os.path.join(cfg.save_dir), video_callable=False, write_upon_reset=False, force=True)
    agent = RandomAgent(action_space=env.action_space, seed=cfg.seed)

    rewards = []
    moving_avg_rewards = []
    eps_steps = []
    log_dir = os.path.split(os.path.abspath(__file__))[0] + "/logs/train/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    for i_eps in range(1, 1 + cfg.train_eps):
        info = {'status': 'NOT_SET'}
        state = env.reset()
        state = np.array(state, dtype=np.float32)

        episode_reward = 0.
        transitions = []
        for i_step in range(cfg.max_steps):
            act, act_param, all_action_param = agent.choose_action(state)
            action = pad_action(act, act_param)
            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            transitions.append(
                [state, np.concatenate(([act], all_action_param.data)).ravel(), reward, next_state, done])
            episode_reward += reward
            state = next_state
            agent.update()
            if done:
                break
        rewards.append(episode_reward)
        eps_steps.append(i_step)
        if i_eps == 1:
            moving_avg_rewards.append(episode_reward)
        else:
            moving_avg_rewards.append(episode_reward * 0.1 + moving_avg_rewards[-1] * 0.9)
        writer.add_scalars('rewards', {'raw': rewards[-1], 'moving_average': moving_avg_rewards[-1]}, i_eps)
        writer.add_scalar('steps_of_each_trials', eps_steps[-1], i_eps)

        if i_eps % 100 == 0:
            print('Episode: ', i_eps, 'R100: ', moving_avg_rewards[-1], 'n_steps: ', np.array(eps_steps[-100]).mean())

    writer.close()


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
